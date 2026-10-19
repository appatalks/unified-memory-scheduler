#!/usr/bin/env python3
"""Measure Linux backend physical reads across load and streaming generation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time
import urllib.error

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vrambridge.app import AppError, Runtime, read_hybrid_profile

SPEC = importlib.util.spec_from_file_location(
    "io_study_sweep", Path(__file__).with_name("vrambridge-execution-sweep.py"))
assert SPEC and SPEC.loader
sweep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sweep)


def snapshot(pid: int, proc: Path = Path("/proc")) -> dict:
    root = proc / str(pid)
    fields = (root / "stat").read_text().rsplit(")", 1)[1].split()
    status = {parts[0].rstrip(":"): int(parts[1]) * 1024
              for line in (root / "status").read_text().splitlines()
              if (parts := line.split()) and parts[0] in
              ("VmRSS:", "RssAnon:", "RssFile:", "RssShmem:", "VmSwap:")}
    memory = {parts[0].rstrip(":"): int(parts[1]) * 1024
              for line in (proc / "meminfo").read_text().splitlines()
              if (parts := line.split()) and parts[0] in
              ("MemTotal:", "MemFree:", "MemAvailable:", "Cached:", "Buffers:", "Shmem:",
               "SwapTotal:", "SwapFree:")}
    io = {key: int(value) for line in (root / "io").read_text().splitlines()
          for key, value in [line.split(":", 1)]}
    return {
        "pid": pid, "monotonic_seconds": time.monotonic(),
        "physical_read_bytes": io["read_bytes"], "read_syscall_bytes": io["rchar"],
        "minor_faults": int(fields[7]), "major_faults": int(fields[9]),
        "cpu_seconds": (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK"),
        "rss_bytes": status["VmRSS"], "anonymous_bytes": status["RssAnon"],
        "file_backed_bytes": status["RssFile"], "shared_bytes": status["RssShmem"],
        "process_swap_bytes": status["VmSwap"], "system_memory": memory,
    }


def interval(before: dict, after: dict) -> dict:
    if before["pid"] != after["pid"]:
        raise ValueError("Backend PID changed inside a measurement interval.")
    elapsed = after["monotonic_seconds"] - before["monotonic_seconds"]
    if elapsed <= 0:
        raise ValueError("Measurement interval must have positive duration.")
    result = {"elapsed_seconds": elapsed}
    for key in ("physical_read_bytes", "read_syscall_bytes", "minor_faults", "major_faults", "cpu_seconds"):
        delta = after[key] - before[key]
        if delta < 0:
            raise ValueError(f"Backend counter decreased: {key}")
        result[key] = delta
    result["physical_read_bytes_s"] = result["physical_read_bytes"] / elapsed
    return result


def runtime_environment(pid: int, keys: set[str], kind: str) -> dict:
    requested = {key: os.environ[key] for key in keys if key in os.environ}
    observed = {}
    for item in (Path("/proc") / str(pid) / "environ").read_bytes().split(b"\0"):
        key, separator, value = item.partition(b"=")
        if separator and key.decode(errors="replace") in keys:
            observed[key.decode()] = value.decode()
    if requested != observed:
        raise RuntimeError(f"Backend {kind} runtime environment differs from the requested controls.")
    return observed


def cpu_runtime_environment(pid: int) -> dict:
    return runtime_environment(pid, {"GOMP_SPINCOUNT", "OMP_WAIT_POLICY", "OMP_PROC_BIND",
                                    "OMP_PLACES", "GOMP_CPU_AFFINITY"}, "CPU")


def cuda_runtime_environment(pid: int) -> dict:
    return runtime_environment(pid, {"GGML_CUDA_GRAPH_OPT", "GGML_CUDA_GRAPH_DEBUG",
                                    "GGML_CUDA_DISABLE_GRAPHS", "GGML_CUDA_DISABLE_FUSION",
                                    "GGML_CUDA_MMQ_PREC"}, "CUDA")


def cpu_backend_modules(pid: int) -> list[str]:
    return sorted({fields[5] for line in (Path("/proc") / str(pid) / "maps").read_text().splitlines()
                   if len(fields := line.split(maxsplit=5)) == 6 and "/libggml-cpu" in fields[5]})


def backend_library_modules(pid: int) -> list[str]:
    return sorted({fields[5] for line in (Path("/proc") / str(pid) / "maps").read_text().splitlines()
                   if len(fields := line.split(maxsplit=5)) == 6
                   and ("/libggml" in fields[5] or "/libllama" in fields[5])})


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(path) + ".part")
    with partial.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    os.link(partial, path)
    partial.unlink()


def measure(args: argparse.Namespace) -> tuple[int, dict]:
    model = args.model.resolve(strict=True)
    prompt = sweep.PROMPT
    if args.prompt_file is not None:
        if args.prompt_file.stat().st_size > 512 * 1024:
            raise ValueError("Prompt context file exceeds 512 KiB.")
        context = args.prompt_file.read_text(encoding="utf-8")
        if not context.strip():
            raise ValueError("Prompt context file is empty.")
        prompt = context if getattr(args, "verbatim_prompt", False) else (
            "Read the following material carefully. Explain its design, correctness constraints, "
            "performance tradeoffs, and practical recommendations in detail.\n\n" + context)
    profile = read_hybrid_profile(args.hybrid_profile) if args.hybrid_profile else {}
    report = {
        "schema_version": 1, "status": "running", "model": model.name,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "sweep_source_sha256": hashlib.sha256(Path(sweep.__file__).read_bytes()).hexdigest(),
        "execution_mode": "hybrid" if profile else "standard",
        "requested_context": args.context, "cpu_threads": args.threads, "batch_threads": args.batch_threads,
        "requested_cpu_poll": getattr(args, "cpu_poll", 50),
        "output_tokens": args.tokens, "repetitions": args.repetitions,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "prompt_source": args.prompt_file.name if args.prompt_file is not None else "fixed_technical_prompt",
        "prompt_kind": ("verbatim_file" if getattr(args, "verbatim_prompt", False) else
                        "wrapped_context" if args.prompt_file is not None else "fixed_technical_prompt"),
        "raw_prompt_recorded": False,
        "ramdisk_used": False, "cache_eviction_requested": False,
        "instrumented": True, "full_model_acceleration_claim": False,
        "samples": [],
        "scope": {
            "physical_reads": "Linux /proc/<backend-pid>/io read_bytes, not read syscall traffic or GPU PCIe transfers.",
            "generation_interval": "First received content/reasoning text event to response completion; not an exact backend decode-only boundary.",
            "memory": "Endpoint snapshots, not continuous peak measurements. File-backed RSS and OS page cache overlap.",
            "limitations": "Small instrumented control; no user UI trace, cold-cache forcing, RAM-disk comparison, or model-quality qualification.",
        },
    }
    requested_batches = (args.batch_size, args.ubatch_size) if args.batch_size is not None else None
    runtime = Runtime(args.llama_server, load_timeout=args.load_timeout,
                      research_batch_sizes=requested_batches, **profile)
    first_load_snapshot = None
    try:
        load_started = time.monotonic()
        runtime.load({"model": str(model), "context": args.context, "cpu_threads": args.threads,
                      "batch_threads": args.batch_threads, "execution_mode": report["execution_mode"],
                      "cpu_poll": getattr(args, "cpu_poll", 50)})
        with runtime.lock:
            process = runtime.process
        if process is None:
            raise RuntimeError("Launcher did not create a backend process.")
        first_load_snapshot = snapshot(process.pid)
        deadline = time.monotonic() + args.load_timeout
        heartbeat = threading.Event()
        while True:
            with runtime.lock:
                state, error = runtime.state, runtime.error
            if state == "error":
                raise RuntimeError(error or "Backend failed without a diagnostic.")
            if state == "ready":
                break
            if time.monotonic() >= deadline:
                raise RuntimeError("Backend did not become ready before the study deadline.")
            heartbeat.wait(0.1)
        loaded = snapshot(process.pid)
        report["model_load_wall_seconds"] = time.monotonic() - load_started
        report["load"] = {"start": first_load_snapshot, "ready": loaded,
                          "interval": interval(first_load_snapshot, loaded)}
        with runtime.lock:
            report["actual_context"] = runtime.context
            report["hybrid_active"] = runtime.hybrid_active
            report["hybrid_markers"] = sorted(runtime.hybrid_markers)
            report["backend_batch_sizes"] = {
                "requested": requested_batches, "observed": dict(runtime.observed_batch_sizes)}
            report["applied_settings"] = {
                "execution_mode": runtime.execution_mode, "placement": runtime.placement,
                "load_mode": runtime.load_mode, "context": runtime.context,
                "cpu_threads": runtime.cpu_threads, "batch_threads": runtime.batch_threads,
                "expert_cache_mib": runtime.expert_cache_mib,
                "hybrid_plan_sha256": runtime.hybrid_plan_sha256,
                "hybrid_transfer": runtime.active_hybrid_transfer if runtime.hybrid_active else None,
                "decode_pipeline": runtime.hybrid_decode_pipeline if runtime.hybrid_active else False,
                "async_ids": runtime.hybrid_async_ids if runtime.hybrid_active else False}
        if report["actual_context"] != args.context:
            raise RuntimeError("Backend did not preserve the requested context.")
        if requested_batches is not None and report["backend_batch_sizes"]["observed"] != {
                "n_batch": args.batch_size, "n_ubatch": args.ubatch_size}:
            raise RuntimeError("Backend logs did not confirm the exact requested batch sizes.")
        report["cpu_runtime_environment"] = cpu_runtime_environment(process.pid)
        report["cuda_runtime_environment"] = cuda_runtime_environment(process.pid)
        report["mapped_cpu_backend_modules"] = cpu_backend_modules(process.pid)
        report["mapped_backend_libraries"] = backend_library_modules(process.pid)
        for repetition in range(args.repetitions):
            before = snapshot(process.pid)
            first = None

            def observe_first_token() -> None:
                nonlocal first
                first = snapshot(process.pid)

            timing = sweep.sample(f"http://127.0.0.1:{runtime.port}", args.tokens,
                                  headers={"Authorization": f"Bearer {runtime.key}"},
                                  on_first_token=observe_first_token, prompt=prompt)
            after = snapshot(process.pid)
            if first is None:
                raise RuntimeError("No first output-text event was observed.")
            report["samples"].append({
                "repetition": repetition, "before_request": before, "first_text_event": first,
                "response_complete": after, "request_interval": interval(before, after),
                "before_first_text_interval": interval(before, first),
                "generation_interval": interval(first, after), "timing": timing,
            })
        report["status"] = "complete"
    except (AppError, OSError, ValueError, RuntimeError, urllib.error.URLError) as exc:
        report["status"] = "failed"
        report["error"] = str(exc).replace(str(model), "/models/" + model.name).replace(runtime.key, "[redacted]") \
            if runtime.key else str(exc).replace(str(model), "/models/" + model.name)
        print(f"vrambridge-io-study: {report['error']}", file=sys.stderr)
    finally:
        with runtime.lock:
            report["backend_log_tail"] = [line.replace(str(model), "/models/" + model.name)
                                          for line in runtime.logs]
        runtime.stop(closing=True)
        report["owned_backend_stopped"] = runtime.process is None
    return (0 if report["status"] == "complete" else 1), report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-server", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hybrid-profile", type=Path)
    parser.add_argument("--prompt-file", type=Path, help="Bounded UTF-8 context file for a real-model technical review.")
    parser.add_argument("--verbatim-prompt", action="store_true",
                        help="Use --prompt-file as the exact user message, without the technical-review wrapper.")
    parser.add_argument("--batch-size", type=int, help="Research-only logical batch size; supply with --ubatch-size.")
    parser.add_argument("--ubatch-size", type=int, help="Research-only physical batch size; supply with --batch-size.")
    parser.add_argument("--context", type=int, default=16384)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch-threads", type=int, default=16)
    parser.add_argument("--cpu-poll", type=int, default=50,
                        help="Native backend CPU-pool polling level, 0-100; not GNU OpenMP spin count.")
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--load-timeout", type=float, default=600)
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("This research probe requires Linux /proc counters.")
    if not (512 <= args.context <= 65536 and 1 <= args.threads <= 64 and 1 <= args.batch_threads <= 64
            and 16 <= args.tokens <= 1024 and 1 <= args.repetitions <= 10
            and 0 <= args.cpu_poll <= 100
            and 0 < args.load_timeout <= 1200):
        parser.error("Invalid bounded study parameters.")
    if args.output.exists() or Path(str(args.output) + ".part").exists():
        parser.error("Refusing to overwrite an existing report or partial.")
    if args.verbatim_prompt and args.prompt_file is None:
        parser.error("--verbatim-prompt requires --prompt-file.")
    if (args.batch_size is None) != (args.ubatch_size is None):
        parser.error("Supply both --batch-size and --ubatch-size.")
    if args.batch_size is not None and not 32 <= args.ubatch_size <= args.batch_size <= min(2048, args.context):
        parser.error("Invalid bounded logical/physical batch sizes.")
    try:
        status, report = measure(args)
        write_report(args.output, report)
    except (AppError, OSError, ValueError) as exc:
        print(f"vrambridge-io-study: {exc}", file=sys.stderr)
        return 2
    print(args.output)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
