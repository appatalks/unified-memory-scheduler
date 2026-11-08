#!/usr/bin/env python3
"""Screen multiple independent requests against owned, isolated llama-server processes."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import secrets
import socket
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sweep = load_module("concurrency_sweep", ROOT / "tools/benchmark/vrambridge-execution-sweep.py")
comparison = load_module("concurrency_environment", ROOT / "tools/benchmark/vrambridge-engine-comparison.py")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_startup(log: str, slots: int, context: int) -> dict:
    matches = re.findall(r"n_slots\s*=\s*(\d+),\s*n_ctx_slot\s*=\s*'?(\d+)", log)
    if not matches or tuple(map(int, matches[-1])) != (slots, context):
        raise RuntimeError("Backend did not confirm the requested slot count and per-slot context.")
    buffers = re.findall(r"load_tensors:\s+(\S+)\s+model buffer size\s*=\s*([\d.]+)\s+MiB", log)
    if not buffers:
        raise RuntimeError("Backend did not report model buffer placement.")
    placement = {}
    for name, amount in buffers:
        placement[name] = placement.get(name, 0.0) + float(amount)
    if not any(name.startswith("CUDA") and not name.startswith("CUDA_Host") and size > 0
               for name, size in placement.items()):
        raise RuntimeError("GPU model placement was not confirmed.")
    if not re.search(r"K\s*\(f16\).*V\s*\(f16\)", log):
        raise RuntimeError("F16 key and value cache precision was not confirmed.")
    return {"actual_slots": slots, "actual_context_per_slot": context,
            "kv_cache": "f16", "model_buffer_mib": placement}


def parse_peak_slots(log: str) -> int:
    event = re.compile(
        r"slot\s+(launch_slot_|release):\s+id\s+(\d+)\s*\|\s*task\s+(\d+)\s*\|\s*"
        r"(processing task|stop processing)"
    )
    active = {}
    peak = 0
    for kind, slot, task, action in event.findall(log):
        if kind == "launch_slot_" and action == "processing task":
            if slot in active:
                raise RuntimeError("Backend slot activity log is inconsistent.")
            active[slot] = task
            peak = max(peak, len(active))
        elif kind == "release" and action == "stop processing":
            if active.get(slot) != task:
                raise RuntimeError("Backend slot release log is inconsistent.")
            del active[slot]
    return peak


def gpu_processes() -> dict[int, int]:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=10,
    )
    processes = {}
    for line in result.stdout.splitlines():
        fields = line.split(",")
        if len(fields) != 2 or not all(field.strip().isdigit() for field in fields):
            raise RuntimeError("Cannot parse GPU process memory.")
        pid, memory = (int(field.strip()) for field in fields)
        processes[pid] = processes.get(pid, 0) + memory
    return processes


def check_gpu_processes(processes: dict[int, int], owner: int | None = None) -> int:
    if any(pid != owner for pid in processes):
        raise RuntimeError("Another GPU compute process is active; refusing a contaminated measurement.")
    if owner is not None and owner not in processes:
        raise RuntimeError("The owned server did not appear as a GPU compute process.")
    return processes.get(owner, 0)


class GpuMonitor:
    def __init__(self, owner: int):
        self.owner = owner
        self.stop_event = threading.Event()
        self.readings: list[int] = []
        self.error: Exception | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stop_event.wait(1):
            try:
                self.readings.append(check_gpu_processes(gpu_processes(), self.owner))
            except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                self.error = exc
                return

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join()
        if self.error is not None:
            raise RuntimeError(f"GPU monitoring failed: {self.error}") from self.error


def run_cohort(base: str, key: str, corpus: list[dict], requests: int, concurrency: int, tokens: int) -> dict:
    if not corpus or requests < concurrency or concurrency < 1:
        raise ValueError("A cohort requires a corpus and at least one request per active client.")
    headers = {"Authorization": f"Bearer {key}"}
    started = time.monotonic()

    def run_one(index: int, submitted: float):
        row = corpus[index % len(corpus)]
        queue_seconds = time.monotonic() - submitted
        sample = sweep.sample(base, tokens, headers, prompt=row["prompt"])
        if sample["first_token_seconds"] is None or sample["timings"].get("cache_n") != 0:
            raise RuntimeError("Missing streamed text or unexpected prompt reuse.")
        sample.update(
            request=index,
            category=row["category"],
            prompt_sha256=hashlib.sha256(row["prompt"].encode()).hexdigest(),
            queue_seconds=queue_seconds,
            agent_ttft_seconds=queue_seconds + sample["first_token_seconds"],
            agent_latency_seconds=queue_seconds + sample["elapsed_seconds"],
        )
        return sample

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(run_one, i, time.monotonic()) for i in range(requests)]
        samples = [future.result() for future in futures]
    elapsed = time.monotonic() - started
    if elapsed <= 0 or len(samples) != requests:
        raise RuntimeError("Incomplete cohort.")
    return {
        "cohort_wall_seconds": elapsed,
        "aggregate_end_to_end_output_tok_s": requests * tokens / elapsed,
        "median_agent_ttft_seconds": statistics.median(x["agent_ttft_seconds"] for x in samples),
        "max_agent_ttft_seconds": max(x["agent_ttft_seconds"] for x in samples),
        "median_agent_latency_seconds": statistics.median(x["agent_latency_seconds"] for x in samples),
        "max_agent_latency_seconds": max(x["agent_latency_seconds"] for x in samples),
        "median_active_agent_decode_tok_s": statistics.median(x["wall_decode_tok_s"] for x in samples),
        "samples": samples,
    }


def summarize(runs: list[dict], concurrency: list[int], repetitions: int) -> list[dict]:
    summary = []
    for count in concurrency:
        valid = [run for run in runs if run["concurrency"] == count and run["status"] == "completed"]
        row = {"concurrency": count, "completed_runs": len(valid), "expected_runs": repetitions}
        if len(valid) == repetitions:
            for key in ("aggregate_end_to_end_output_tok_s", "median_agent_ttft_seconds",
                        "max_agent_ttft_seconds", "median_agent_latency_seconds",
                        "max_agent_latency_seconds", "median_active_agent_decode_tok_s"):
                row[key] = statistics.median(run["cohort"][key] for run in valid)
        summary.append(row)
    return summary


def wait_ready(process: subprocess.Popen, base: str, key: str, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Owned backend exited before readiness (code {process.returncode}).")
        try:
            request = urllib.request.Request(base + "/health", headers={"Authorization": f"Bearer {key}"})
            with urllib.request.urlopen(request, timeout=1) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            time.sleep(0.25)
    raise RuntimeError("Owned backend exceeded its readiness deadline.")


def execute(args) -> int:
    model = args.model.resolve(strict=True)
    backend = args.backend.resolve(strict=True)
    corpus = [json.loads(line) for line in args.corpus.read_text().splitlines()]
    corpus = [row for row in corpus if row["split"] == args.split]
    if not corpus:
        raise ValueError("No matching corpus requests.")
    if args.output.exists() or args.output.with_suffix(args.output.suffix + ".tmp").exists():
        raise ValueError("Refusing to overwrite an existing report.")
    check_gpu_processes(gpu_processes())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    libraries = [backend, backend.parent / "libllama.so", backend.parent / "libggml-cuda.so"]
    report = {
        "schema_version": 1,
        "kind": "owned-agent-concurrency-screen",
        "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model": model.name,
        "model_sha256": sha256(model),
        "backend_sha256": {path.name: sha256(path) for path in libraries},
        "runner_sha256": sha256(Path(__file__)),
        "corpus_sha256": sha256(args.corpus),
        "concurrency": args.concurrency, "backend_slots": args.backend_slots, "repetitions": args.repetitions,
        "per_slot_context": args.context, "requests": args.requests, "output_tokens": args.tokens,
        "threads": args.threads, "batch_threads": args.batch_threads, "fit_target_mib": args.fit_target,
        "gpu_placement": args.gpu_placement,
        "limitations": [
            "Direct isolated server, not the current serialized Synapse app or its hybrid backend.",
            "All client arrivals are queued by the harness when active-client concurrency is smaller.",
            "Aggregate output rate includes prefill, queueing and response overhead.",
            "Forced-length short prompts are not natural answers or a filled-context test.",
            "Automatic fit can change tensor placement; logged per-load placement must be compared.",
            "Output hashes are not logit equivalence or an answer-quality qualification.",
            "Verbose startup logging is enabled to record actual placement and can affect timings.",
            "One-second GPU sampling may miss transient memory peaks or short-lived competing processes.",
        ],
        "runs": [], "summary": [],
    }
    sweep.save_report(args.output, report)
    failed = False
    for repetition in range(args.repetitions):
        order = args.concurrency if repetition % 2 == 0 else list(reversed(args.concurrency))
        for concurrency in order:
            run = {"repetition": repetition, "concurrency": concurrency, "status": "starting"}
            report["runs"].append(run)
            sweep.save_report(args.output, report)
            print(f"START repetition={repetition} concurrency={concurrency}", flush=True)
            process = None
            try:
                check_gpu_processes(gpu_processes())
                with socket.socket() as reservation:
                    reservation.bind(("127.0.0.1", 0))
                    port = reservation.getsockname()[1]
                key = secrets.token_hex(32)
                base = f"http://127.0.0.1:{port}"
                command = [
                    str(backend), "--model", str(model), "--alias", "local-model",
                    "--host", "127.0.0.1", "--port", str(port), "--api-key", key,
                    "--ctx-size", str(args.context * args.backend_slots), "--parallel", str(args.backend_slots),
                    "--threads", str(args.threads), "--threads-batch", str(args.batch_threads),
                    "--n-gpu-layers", "auto", "--fit", "on", "--fit-target", str(args.fit_target),
                    "--cache-type-k", "f16", "--cache-type-v", "f16",
                    "--poll", "50", "--slots", "--cont-batching", "--spec-type", "none",
                    "--log-verbosity", "4",
                ]
                if args.gpu_placement == "cpu-experts":
                    gpu_index = command.index("--n-gpu-layers")
                    command[gpu_index + 1] = "all"
                    fit_index = command.index("--fit")
                    command[fit_index + 1] = "off"
                    command += ["--cpu-moe"]
                environment = comparison.clean_environment()
                environment["GGML_OP_OFFLOAD_MIN_BATCH"] = "32"
                log_path = args.output.with_name(f"{args.output.stem}-{repetition}-{concurrency}.log")
                with log_path.open("x") as log:
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
                    wait_ready(process, base, key, args.load_timeout)
                    run["startup"] = parse_startup(log_path.read_text(), args.backend_slots, args.context)
                    run["gpu_memory_loaded_mib"] = check_gpu_processes(gpu_processes(), process.pid)
                    monitor = GpuMonitor(process.pid)
                    monitor.start()
                    try:
                        sweep.sample(base, 64, {"Authorization": f"Bearer {key}"}, prompt=corpus[0]["prompt"])
                        run["status"] = "running"
                        run["cohort"] = run_cohort(base, key, corpus, args.requests, concurrency, args.tokens)
                        run["gpu_memory_after_mib"] = check_gpu_processes(gpu_processes(), process.pid)
                    finally:
                        monitor.stop()
                    run["observed_peak_active_slots"] = parse_peak_slots(log_path.read_text())
                    if run["observed_peak_active_slots"] != concurrency:
                        raise RuntimeError("Requested simultaneous backend slot activity was not observed.")
                    run["gpu_memory_sampled_peak_mib"] = max(
                        [run["gpu_memory_loaded_mib"], run["gpu_memory_after_mib"], *monitor.readings]
                    )
                    run["status"] = "completed"
                    print(f"RATE concurrency={concurrency} aggregate={run['cohort']['aggregate_end_to_end_output_tok_s']:.2f}", flush=True)
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                failed = True
                run.update(status="failed", error=str(exc).replace(str(model.parent), "/models")
                           .replace(str(Path.home()), "/home/benchmark-user"))
                print(f"FAILED concurrency={concurrency}: {run['error']}", flush=True)
            finally:
                if process is not None and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                report["summary"] = summarize(report["runs"], args.concurrency, args.repetitions)
                sweep.save_report(args.output, report)
            if failed:
                break
        if failed:
            break
    print(json.dumps(report["summary"], indent=2), flush=True)
    return int(failed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=ROOT / "benchmarks/agent-concurrency-corpus.jsonl")
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--backend-slots", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--context", type=int, default=16384)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch-threads", type=int, default=16)
    parser.add_argument("--fit-target", type=int, default=1024)
    parser.add_argument("--gpu-placement", choices=("auto", "cpu-experts"), default="auto")
    parser.add_argument("--load-timeout", type=int, default=120)
    args = parser.parse_args()
    if (len(set(args.concurrency)) != len(args.concurrency) or not args.concurrency or min(args.concurrency) < 1
            or not max(args.concurrency) <= args.backend_slots <= args.requests or not 1 <= args.repetitions <= 10
            or not 512 <= args.context <= 131072 or not 1 <= args.requests <= 64
            or not 2 <= args.tokens <= 4096 or min(args.threads, args.batch_threads, args.fit_target,
                                                     args.load_timeout) < 1):
        parser.error("Invalid concurrency, slot, context, request, token or timing setting.")
    return execute(args)


if __name__ == "__main__":
    os.umask(0o077)
    raise SystemExit(main())
