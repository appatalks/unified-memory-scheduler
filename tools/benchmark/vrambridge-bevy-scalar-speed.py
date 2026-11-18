#!/usr/bin/env python3
"""Measure an isolated scalar-token interleave control, not Bevy runahead."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("bevy_numerics", Path(__file__).with_name("vrambridge-agent-numerics.py"))
assert spec is not None and spec.loader is not None
numerics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(numerics)
bench = numerics.bench


def checked_sample(path: Path, log_path: Path, level: int, tokens: int, context: int,
                   disable_graph_reuse: bool = False) -> dict:
    raw = json.loads(path.read_text())
    sequences = raw.get("tokens")
    if (raw.get("concurrency") != level or raw.get("requests") != 8
            or raw.get("output_tokens_per_request") != tokens or raw.get("context_per_sequence") != context
            or raw.get("warmup_decode_tokens") != 32 * level or raw.get("serial_tokens") is not True
            or not isinstance(sequences, list) or len(sequences) != 8
            or any(not isinstance(seq, list) or len(seq) != tokens
                   or any(type(token) is not int or token < 0 for token in seq) for seq in sequences)):
        raise ValueError("Incomplete or mismatched timing output")
    for key in ("aggregate_decode_tok_s", "aggregate_end_to_end_tok_s", "cohort_seconds"):
        if type(raw.get(key)) not in (int, float) or not math.isfinite(raw[key]) or raw[key] <= 0:
            raise ValueError("Invalid timing result")
    for key in ("first_token_seconds", "completion_seconds"):
        values = raw.get(key)
        if (not isinstance(values, list) or len(values) != 8 or
                any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in values)):
            raise ValueError("Invalid per-request timing")
    if any(first > last + 1e-6 or last > raw["cohort_seconds"] + 1e-6 for first, last in
           zip(raw["first_token_seconds"], raw["completion_seconds"])):
        raise ValueError("Inconsistent request timing")
    log = log_path.read_text(errors="replace")
    if ("graph reuse disabled" in log) != disable_graph_reuse:
        raise ValueError("Unexpected graph-reuse configuration")
    coverage = re.search(r"tensor-derived hybrid coverage=(\d+)/(\d+) routed layers", log)
    dispatch = re.findall(r"hybrid dispatches=(\d+) CPU experts=(\d+) GPU experts=(\d+) prior hits=(\d+)", log)
    if ("EXPERIMENTAL HYBRID enabled" not in log or "hybrid batch limit=1" not in log
            or coverage is None or int(coverage[1]) <= 0 or coverage[1] != coverage[2]
            or not dispatch or not all(int(x) > 0 for x in dispatch[-1][:3])):
        raise ValueError("Hybrid coverage and mixed CPU/GPU execution were not confirmed")
    result = {
        "level": level,
        "aggregate_decode_tok_s": raw["aggregate_decode_tok_s"],
        "aggregate_end_to_end_tok_s": raw["aggregate_end_to_end_tok_s"],
        "median_first_token_seconds": statistics.median(raw["first_token_seconds"]),
        "median_completion_seconds": statistics.median(raw["completion_seconds"]),
        "request_token_sha256": [hashlib.sha256(json.dumps(seq, separators=(",", ":")).encode()).hexdigest()
                                  for seq in sequences],
        "dispatch": dict(zip(("calls", "cpu_experts", "gpu_experts", "prior_hits"), map(int, dispatch[-1]))),
        "routed_layers": int(coverage[1]),
        "raw_output_sha256": bench.sha256(path), "backend_log_sha256": bench.sha256(log_path),
    }
    for key in ("decode_graph_reuses", "decode_graph_reallocs_same", "decode_graph_reallocs_diff"):
        if key in raw:
            if type(raw[key]) is not int or raw[key] < 0:
                raise ValueError("Invalid graph diagnostic counter")
            result[key] = raw[key]
    return result


def summarize(samples: list[dict], repetitions: int) -> dict:
    summary = {}
    for level in (1, 2):
        selected = [s for s in samples if s["level"] == level]
        row = {"completed_runs": len(selected)}
        if len(selected) == repetitions:
            for key in ("aggregate_decode_tok_s", "aggregate_end_to_end_tok_s",
                        "median_first_token_seconds", "median_completion_seconds"):
                row[key] = statistics.median(s[key] for s in selected)
        summary[str(level)] = row
    by_repetition = {s["repetition"]: {} for s in samples}
    for sample in samples:
        by_repetition[sample["repetition"]][sample["level"]] = sample["request_token_sha256"]
    paired = [value[1] == value[2] for value in by_repetition.values() if 1 in value and 2 in value]
    all_hashes = [tuple(s["request_token_sha256"]) for s in samples]
    return {"levels": summary, "paired_token_matches": sum(paired), "paired_runs": len(paired),
            "all_outputs_repeat_across_runs": len(set(all_hashes)) == 1 if all_hashes else False}


def memory_kib(text: str, field: str) -> int:
    match = re.search(rf"^{re.escape(field)}:\s+(\d+)\s+kB\s*$", text, re.MULTILINE)
    if match is None:
        raise RuntimeError(f"Missing {field} memory measurement")
    return int(match[1])


def process_swap_kib(process: subprocess.Popen) -> int | None:
    try:
        return memory_kib(Path(f"/proc/{process.pid}/status").read_text(), "VmSwap")
    except (FileNotFoundError, RuntimeError) as error:
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            raise error
        return None


def run_owned(command: list[str], env: dict, output: Path, log: Path, libraries: list[Path], timeout: int) -> dict:
    bench.check_gpu_processes(bench.gpu_processes())
    peak = 0
    min_available = memory_kib(Path("/proc/meminfo").read_text(), "MemAvailable")
    if min_available < 16 * 1024 * 1024:
        raise RuntimeError("Host memory headroom fell below 16 GiB")
    max_swap = 0
    mapped = False
    process = None
    try:
        with output.open("x") as out, log.open("x") as err:
            process = subprocess.Popen(command, stdout=out, stderr=err, env=env)
            deadline = time.monotonic() + timeout
            while process.poll() is None:
                available = memory_kib(Path("/proc/meminfo").read_text(), "MemAvailable")
                min_available = min(min_available, available)
                if available < 16 * 1024 * 1024:
                    raise RuntimeError("Host memory headroom fell below 16 GiB")
                swap = process_swap_kib(process)
                if swap is None:
                    break
                max_swap = max(max_swap, swap)
                if swap:
                    raise RuntimeError("Owned timing probe entered swap")
                processes = bench.gpu_processes()
                if any(pid != process.pid for pid in processes):
                    raise RuntimeError("Another GPU compute process appeared during the screen")
                if process.pid in processes:
                    peak = max(peak, processes[process.pid])
                    if not mapped:
                        numerics.check_mapped_libraries(process.pid, libraries)
                        mapped = True
                if time.monotonic() >= deadline:
                    raise RuntimeError("Owned timing probe exceeded its deadline")
                time.sleep(1)
            if process.returncode != 0:
                raise RuntimeError(f"Owned timing probe exited with status {process.returncode}")
        if not mapped:
            raise RuntimeError("Owned timing probe GPU and library mappings were not confirmed")
        bench.check_gpu_processes(bench.gpu_processes())
        return {"sampled_peak_gpu_mib": peak, "library_mappings_verified": True,
                "sampled_min_host_available_kib": min_available, "sampled_peak_process_swap_kib": max_swap}
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def execute(args) -> int:
    probe, model = args.probe.resolve(strict=True), args.model.resolve(strict=True)
    backend_dir = args.backend_dir.resolve(strict=True)
    cpu_plugin, cuda_plugin, plan = (p.resolve(strict=True) for p in (args.cpu_plugin, args.cuda_plugin, args.plan))
    output = args.output.resolve()
    if output.exists() or output.with_suffix(output.suffix + ".tmp").exists():
        raise ValueError("Refusing to overwrite an existing report")
    output.parent.mkdir(parents=True, exist_ok=True)
    env = {key: value for key, value in bench.comparison.clean_environment().items()
           if not key.startswith("LLAMA_")}
    env.update(LD_LIBRARY_PATH=str(backend_dir), VRAMBRIDGE_PROBE_CPU_PLUGIN=str(cpu_plugin),
               VRAMBRIDGE_PROBE_CUDA_PLUGIN=str(cuda_plugin), VRAMBRIDGE_HYBRID_PLAN=str(plan),
               VRAMBRIDGE_HYBRID_ADAPT="0", VRAMBRIDGE_HYBRID_ASYNC="0", VRAMBRIDGE_HYBRID_DIRECT="0",
               VRAMBRIDGE_HYBRID_STAGE_INPUT="1", VRAMBRIDGE_HYBRID_OVERLAP="1", VRAMBRIDGE_HYBRID_BOUNDARY="gpu",
               VRAMBRIDGE_HYBRID_FUSE_GATE_UP="1", VRAMBRIDGE_HYBRID_ASYNC_IDS="1", VRAMBRIDGE_HYBRID_BATCH="1",
               VRAMBRIDGE_SPEED_HYBRID="1", VRAMBRIDGE_SPEED_SERIAL_TOKENS="1", VRAMBRIDGE_PROBE_SLOTS="2",
               VRAMBRIDGE_PROBE_CONTEXT=str(args.context), VRAMBRIDGE_PROBE_MOE_CACHE_MIB=str(args.cache_mib))
    if args.disable_graph_reuse:
        env["LLAMA_GRAPH_REUSE_DISABLE"] = "1"
    shards = numerics.model_shards(model)
    report = {
        "schema_version": 1, "kind": "hybrid-scalar-token-interleave-control", "status": "running",
        "model": model.name, "model_shards": [{"name": p.name, "sha256": bench.sha256(p)} for p in shards],
        "plan_sha256": bench.sha256(plan), "probe_sha256": bench.sha256(probe),
        "runner_sha256": bench.sha256(Path(__file__)),
        "source_sha256": {"sequence-speed-probe.cpp": bench.sha256(args.speed_source.resolve(strict=True)),
                          "sequence-decode-probe.cpp": bench.sha256(ROOT / "tools/trace/sequence-decode-probe.cpp")},
        "backend_sha256": {p.name: bench.sha256(p) for p in (backend_dir / "libllama.so", cpu_plugin, cuda_plugin)},
        "context_per_slot": args.context, "slots": 2, "expert_cache_mib": args.cache_mib,
        "requests": 8, "output_tokens": args.tokens, "repetitions": args.repetitions,
        "threads": 8, "batch_threads": 16, "warmup_decode_steps_per_active_sequence": 32,
        "graph_reuse_disabled": args.disable_graph_reuse,
        "limitations": ["Scalar token-level alternation, not Bevy's layer-runahead scheduler or HTTP",
                        "Synthetic short prompts and fixed synthetic seed, not adaptive or full-context inference",
                        "Raw first-token and completion timings include an eight-request arrival burst and queue",
                        "Output token identity does not establish full-logit, long-stream or quality equivalence",
                        "One-second GPU and memory sampling may miss transient events or disk reads"],
        "samples": [],
    }
    bench.sweep.save_report(output, report)
    try:
        for repetition in range(args.repetitions):
            for level in ((1, 2) if repetition % 2 == 0 else (2, 1)):
                print(f"START repetition={repetition + 1} clients={level}", flush=True)
                prefix = output.with_name(f"{output.stem}-{repetition + 1}-{level}")
                raw, log = prefix.with_suffix(".json"), prefix.with_suffix(".log")
                command = [str(probe), str(model), str(level), str(args.tokens), "gpu-cpu-experts"]
                metadata = run_owned(command, env, raw, log, [backend_dir / "libllama.so", cpu_plugin, cuda_plugin], args.timeout)
                sample = checked_sample(raw, log, level, args.tokens, args.context, args.disable_graph_reuse)
                sample.update(repetition=repetition + 1, **metadata)
                report["samples"].append(sample)
                report["summary"] = summarize(report["samples"], args.repetitions)
                bench.sweep.save_report(output, report)
                print(f"RATE repetition={repetition + 1} clients={level} decode={sample['aggregate_decode_tok_s']:.2f}", flush=True)
                if len(report["samples"]) % 2 == 0 and report["summary"]["paired_token_matches"] != (repetition + 1):
                    raise RuntimeError("Paired scalar output tokens differ; stopping timing screen")
        if not report["summary"]["all_outputs_repeat_across_runs"]:
            raise RuntimeError("Output tokens differ across fresh runs; stopping timing screen")
        report["status"] = "completed"
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        report.update(status="failed", error=str(exc).replace(str(Path.home()), "/home/benchmark-user"))
    finally:
        bench.sweep.save_report(output, report)
    print(json.dumps({"status": report["status"], "summary": report.get("summary"), "error": report.get("error")}))
    return 0 if report["status"] == "completed" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("probe", "backend-dir", "model", "plan", "cpu-plugin", "cuda-plugin", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--speed-source", type=Path, default=ROOT / "tools/trace/sequence-speed-probe.cpp")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--cache-mib", type=int, default=4096)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--disable-graph-reuse", action="store_true")
    args = parser.parse_args()
    if not (2 <= args.tokens <= 512 and 1 <= args.repetitions <= 10 and 512 <= args.context <= 16384
            and 1 <= args.cache_mib <= 8192 and args.timeout > 0):
        parser.error("Invalid bounded timing settings")
    return execute(args)


if __name__ == "__main__":
    os.umask(0o077)
    raise SystemExit(main())
