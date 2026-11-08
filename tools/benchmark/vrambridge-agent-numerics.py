#!/usr/bin/env python3
"""Compare sequential and concurrent first-token full-vocabulary log probabilities."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import secrets
import socket
import subprocess
import time
import urllib.request


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("agent_concurrency", Path(__file__).with_name("vrambridge-agent-concurrency.py"))
assert spec and spec.loader
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def probability_vector(event: dict, vocabulary: int) -> dict[int, float | None]:
    positions = event.get("completion_probabilities")
    if not isinstance(positions, list) or len(positions) != 1:
        raise ValueError("Expected one probability position.")
    values = positions[0].get("top_logprobs")
    if not isinstance(values, list) or len(values) != vocabulary:
        raise ValueError("Full vocabulary was not returned.")
    result = {}
    for item in values:
        if not isinstance(item, dict):
            raise ValueError("Invalid vocabulary entry.")
        token, value = item.get("id"), item.get("logprob")
        if type(token) is not int or not 0 <= token < vocabulary or token in result:
            raise ValueError("Invalid or duplicate vocabulary ID.")
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
            raise ValueError("Invalid log probability.")
        result[token] = value
    return result


def compare_probabilities(first: dict, second: dict, vocabulary: int) -> dict:
    for event in (first, second):
        tokens = event.get("tokens")
        if (not isinstance(tokens, list) or len(tokens) != 1
                or type(tokens[0]) is not int or not 0 <= tokens[0] < vocabulary):
            raise ValueError("Invalid generated token.")
    before = probability_vector(first, vocabulary)
    after = probability_vector(second, vocabulary)
    if any((before[token] is None) != (after[token] is None) for token in before):
        raise ValueError("Vocabulary probability masks differ.")
    pairs = [(value, after[token]) for token, value in before.items() if value is not None]
    if not pairs:
        raise ValueError("No finite vocabulary probabilities.")
    maximum = max(abs(a - b) for a, b in pairs)
    reference = sum(a * a for a, _ in pairs)
    relative = math.sqrt(sum((a - b) ** 2 for a, b in pairs) / max(reference, 1e-30))
    return {
        "vocabulary": vocabulary, "finite_entries": len(pairs),
        "max_absolute_logprob_error": maximum, "relative_l2_logprob_error": relative,
        "passes": maximum <= 0.01 and relative <= 0.001,
        "first_token_matches": first.get("tokens") == second.get("tokens"),
    }


def cpu_startup(log: str, slots: int, context: int) -> dict:
    contexts = re.findall(r"n_slots\s*=\s*(\d+),\s*n_ctx_slot\s*=\s*'?(\d+)", log)
    if not contexts or tuple(map(int, contexts[-1])) != (slots, context):
        raise RuntimeError("CPU server did not confirm the requested slot count and context.")
    if (not re.search(r"load_tensors:\s+CPU\S*\s+model buffer size", log)
            or re.search(r"load_tensors:\s+CUDA\d+\s+model buffer size", log)
            or not re.search(r"K\s*\(f16\).*V\s*\(f16\)", log)):
        raise RuntimeError("CPU model placement or F16 KV precision was not confirmed.")
    return {"actual_slots": slots, "actual_context_per_slot": context, "placement": "CPU", "kv_cache": "f16"}


def complete(base: str, key: str, prompt: str, vocabulary: int) -> dict:
    payload = {
        "prompt": prompt, "n_predict": 1, "temperature": 0, "seed": 42,
        "cache_prompt": False, "ignore_eos": True, "return_tokens": True,
        "n_probs": vocabulary, "post_sampling_probs": False,
    }
    request = urllib.request.Request(base + "/completion", json.dumps(payload).encode(),
        {"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(request, timeout=180) as response:
        body = response.read(64 * 1024 * 1024 + 1)
    if len(body) > 64 * 1024 * 1024:
        raise ValueError("Probability response exceeds 64 MiB.")
    event = json.loads(body)
    tokens = event.get("tokens")
    if (event.get("error") or not isinstance(tokens, list) or len(tokens) != 1
            or type(tokens[0]) is not int or tokens[0] < 0):
        raise ValueError("Incomplete probability request.")
    if event.get("timings", {}).get("cache_n") != 0:
        raise ValueError("Unexpected prompt reuse.")
    return event


def execute(args) -> int:
    backend = args.backend.resolve(strict=True)
    model = args.model.resolve(strict=True)
    rows = [json.loads(line) for line in args.corpus.read_text().splitlines()]
    corpus = [row for row in rows if row["split"] == args.split]
    if len(corpus) < max(args.levels):
        raise ValueError("Corpus needs at least one distinct request per concurrent client.")
    if len({row["prompt"] for row in corpus}) != len(corpus):
        raise ValueError("Mixed-prompt corpus contains duplicates.")
    log_path = args.output.with_name(args.output.stem + ".log")
    if args.output.exists() or args.output.with_suffix(args.output.suffix + ".tmp").exists() or log_path.exists():
        raise ValueError("Refusing to overwrite an existing report or log.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.placement != "cpu":
        bench.check_gpu_processes(bench.gpu_processes())
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
        "--cache-type-k", "f16", "--cache-type-v", "f16", "--poll", "50",
        "--slots", "--cont-batching", "--spec-type", "none", "--log-verbosity", "4",
    ]
    if args.placement == "auto":
        command += ["--n-gpu-layers", "auto", "--fit", "on", "--fit-target", str(args.fit_target)]
    elif args.placement == "gpu-cpu-experts":
        command += ["--n-gpu-layers", "all", "--cpu-moe", "--fit", "off"]
    else:
        command += ["--n-gpu-layers", "0", "--device", "none", "--fit", "off"]
    environment = bench.comparison.clean_environment()
    environment["GGML_OP_OFFLOAD_MIN_BATCH"] = "32"
    report = {
        "schema_version": 1, "kind": "agent-first-token-logprob-diagnostic",
        "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model": model.name, "model_sha256": bench.sha256(model),
        "backend_sha256": {path.name: bench.sha256(path) for path in
                           (backend, backend.parent / "libllama.so", backend.parent / "libggml-cpu.so",
                            backend.parent / "libggml-cuda.so")},
        "corpus_sha256": bench.sha256(args.corpus), "runner_sha256": bench.sha256(Path(__file__)),
        "placement": args.placement, "context_per_slot": args.context, "backend_slots": args.backend_slots,
        "levels": args.levels, "vocabulary": args.vocabulary, "requests": [],
        "status": "starting",
        "limitations": [
            "Raw-completion prompts and one output token; not the chat throughput workload.",
            "Full-vocabulary pre-sampling log probabilities, not raw logits or broad answer quality.",
            "Backend slot overlap does not prove that every request was evaluated in the same graph batch.",
            "One-second GPU sampling may miss transient competing processes.",
        ],
    }
    bench.sweep.save_report(args.output, report)
    process = None
    monitor = None
    try:
        with log_path.open("x") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
            bench.wait_ready(process, base, key, args.load_timeout)
            text = log_path.read_text()
            report["startup"] = (bench.parse_startup(text, args.backend_slots, args.context)
                                 if args.placement != "cpu" else cpu_startup(text, args.backend_slots, args.context))
            if args.placement != "cpu":
                bench.check_gpu_processes(bench.gpu_processes(), process.pid)
                monitor = bench.GpuMonitor(process.pid)
                monitor.start()
            complete(base, key, corpus[0]["prompt"], 0)
            first = [complete(base, key, row["prompt"], args.vocabulary) for row in corpus]
            second = [complete(base, key, row["prompt"], args.vocabulary) for row in corpus]
            report["requests"] = [
                {"id": row["id"], "sequential_repeat": compare_probabilities(first[i], second[i], args.vocabulary)}
                for i, row in enumerate(corpus)
            ]
            del second
            report["peak_active_slots"] = {}
            for level in args.levels:
                time.sleep(0.5)
                offset = log_path.stat().st_size
                with ThreadPoolExecutor(max_workers=level) as executor:
                    concurrent = list(executor.map(lambda row: complete(base, key, row["prompt"], args.vocabulary), corpus))
                time.sleep(0.5)
                log_slice = log_path.read_bytes()[offset:].decode("utf-8", errors="replace")
                report["peak_active_slots"][str(level)] = bench.parse_peak_slots(log_slice)
                if report["peak_active_slots"][str(level)] != level:
                    raise RuntimeError("Requested simultaneous backend slots were not observed.")
                for i, event in enumerate(concurrent):
                    report["requests"][i][str(level)] = compare_probabilities(first[i], event, args.vocabulary)
                del concurrent
            if monitor is not None:
                monitor.stop()
                monitor = None
                bench.check_gpu_processes(bench.gpu_processes(), process.pid)
            report["numerical_pass"] = all(
                row[key]["passes"] and row[key]["first_token_matches"]
                for row in report["requests"] for key in ["sequential_repeat", *map(str, args.levels)]
            )
            report["status"] = "completed"
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        report.update(status="failed", error=str(exc).replace(str(model.parent), "/models")
                      .replace(str(Path.home()), "/home/benchmark-user"))
    finally:
        if monitor is not None:
            try:
                monitor.stop()
            except RuntimeError as exc:
                report.update(status="failed", error=str(exc))
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        bench.sweep.save_report(args.output, report)
    print(json.dumps({"status": report["status"], "numerical_pass": report.get("numerical_pass"),
                      "peak_active_slots": report.get("peak_active_slots"), "error": report.get("error")}))
    return 0 if report["status"] == "completed" and report["numerical_pass"] else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vocabulary", type=int, required=True)
    parser.add_argument("--corpus", type=Path, default=ROOT / "benchmarks/agent-concurrency-corpus.jsonl")
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--levels", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--backend-slots", type=int, default=8)
    parser.add_argument("--context", type=int, default=16384)
    parser.add_argument("--placement", choices=("auto", "cpu", "gpu-cpu-experts"), default="auto")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch-threads", type=int, default=16)
    parser.add_argument("--fit-target", type=int, default=1024)
    parser.add_argument("--load-timeout", type=int, default=120)
    args = parser.parse_args()
    if (not args.levels or len(set(args.levels)) != len(args.levels) or min(args.levels) < 2
            or not max(args.levels) <= args.backend_slots <= 64 or not 512 <= args.context <= 131072
            or not 1 <= args.vocabulary <= 1000000
            or min(args.threads, args.batch_threads, args.fit_target, args.load_timeout) < 1):
        parser.error("Invalid level, context, vocabulary or timing setting.")
    return execute(args)


if __name__ == "__main__":
    os.umask(0o077)
    raise SystemExit(main())
