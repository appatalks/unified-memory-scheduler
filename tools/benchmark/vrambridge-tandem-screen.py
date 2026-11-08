#!/usr/bin/env python3
"""Owned single-user tandem/reference screens; raw reports remain private."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import secrets
import socket
import statistics
import subprocess
import tempfile
import urllib.request


ROOT = Path(__file__).resolve().parents[2]
import importlib.util

spec = importlib.util.spec_from_file_location(
    "tandem_agent_numerics", Path(__file__).with_name("vrambridge-agent-numerics.py"))
assert spec and spec.loader
numerics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(numerics)
bench = numerics.bench
COMPONENTS = {
    "tandem-packed": ({"GGML_SCHED_BATCH_INPUTS": "3"}, ["EXPERIMENT_COMPONENT_PACKED_INPUTS"]),
    "tandem-graphs": ({"GGML_CUDA_GRAPH_KEY_SHAPE": "1", "LLAMA_GRAPH_POOL": "6"},
                      ["EXPERIMENT_COMPONENT_GRAPH_SHAPE", "EXPERIMENT_COMPONENT_GRAPH_POOL"]),
    "tandem-q8": ({"GGML_CUDA_Q8_CACHE": "1"}, ["EXPERIMENT_COMPONENT_Q8_REUSE"]),
    "tandem-launch": ({"GGML_SCHED_BATCH_INPUTS": "3", "GGML_CUDA_GRAPH_KEY_SHAPE": "1", "LLAMA_GRAPH_POOL": "6"},
                      ["EXPERIMENT_COMPONENT_PACKED_INPUTS", "EXPERIMENT_COMPONENT_GRAPH_SHAPE",
                       "EXPERIMENT_COMPONENT_GRAPH_POOL"]),
    "tandem-safe": ({"GGML_SCHED_BATCH_INPUTS": "3", "GGML_CUDA_GRAPH_KEY_SHAPE": "1",
                     "LLAMA_GRAPH_POOL": "6", "GGML_CUDA_Q8_CACHE": "1"},
                    ["EXPERIMENT_COMPONENT_PACKED_INPUTS", "EXPERIMENT_COMPONENT_GRAPH_SHAPE",
                     "EXPERIMENT_COMPONENT_GRAPH_POOL", "EXPERIMENT_COMPONENT_Q8_REUSE"]),
}
CACHE_ONLY_ARMS = {"reference-cache", "tandem-cache", *COMPONENTS}


def environment(arm: str, threads: int, stats: Path) -> dict:
    env = {key: value for key, value in bench.comparison.clean_environment().items()
           if not key.startswith("LLAMA_") and key != "LD_LIBRARY_PATH"}
    env.update(LLAMA_MOE_NO_DEFAULTS="0" if arm == "tandem-defaults" else "1",
               LLAMA_MOE_TAIL_SKIP="0", LLAMA_MOE_PIN_EXPS="0",
               LLAMA_PLE_GPU_COPY="0", LLAMA_MOE_CPU_THREADS=str(threads))
    if not arm.startswith("reference"):
        env["LLAMA_MOE_STATS_FILE"] = str(stats)
    if arm == "tandem-fixed":
        env["LLAMA_MOE_CPU"] = "0.2"
    if arm in COMPONENTS:
        env.update(COMPONENTS[arm][0], VRAMBRIDGE_COMPONENT_TRACE="1")
    return env


def activation(log: str, arm: str) -> dict:
    if arm == "reference":
        return {"mechanism": "unpatched automatic fit"}
    caches = re.findall(r"MoE cache size\s*=\s*([\d.]+)\s+MiB", log)
    if not caches or sum(map(float, caches)) <= 0 or re.search(
            r"MoE cache is disabled|the cache is disabled|not supported by the CPU path|"
            r"missed experts are all uploaded|approximate, not the model", log):
        raise RuntimeError("Requested expert-cache/hybrid mechanism absent or unsupported.")
    if arm not in CACHE_ONLY_ARMS and not re.search(r"hybrid mode:.*misses are uploaded", log):
        raise RuntimeError("Hybrid CPU/upload split was not confirmed.")
    if arm == "tandem-defaults" and "tandem.cpp defaults" not in log:
        raise RuntimeError("The tandem defaults bundle was not confirmed.")
    return {"mechanism": arm, "expert_cache_mib": sum(map(float, caches))}


def component_activation(log: str, arm: str) -> None:
    missing = [marker for marker in COMPONENTS.get(arm, ({}, []))[1] if marker not in log]
    if missing:
        raise RuntimeError("Requested components did not activate: " + ", ".join(missing))


def summarize(runs: list[dict], arms: list[str], repetitions: int, categories: list[str]) -> list[dict]:
    results = []
    for arm in arms:
        completed = [run for run in runs if run["arm"] == arm and run["status"] == "completed"]
        row = {"arm": arm, "completed_runs": len(completed), "expected_runs": repetitions}
        if len(completed) == repetitions:
            row["workloads"] = [
                {"category": category, **{
                    key: statistics.median(sample[key] for run in completed
                                           for sample in run["samples"] if sample["category"] == category)
                    for key in ("wall_decode_tok_s", "end_to_end_tok_s", "first_token_seconds")
                }} for category in categories
            ]
        results.append(row)
    return results


def execute(args) -> int:
    if args.output.exists() or args.output.with_suffix(args.output.suffix + ".tmp").exists():
        raise ValueError("Refusing to overwrite prior evidence.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    corpus = [json.loads(line) for line in args.corpus.read_text().splitlines()]
    held_out = [row for row in corpus if row["split"] == "test"]
    warmup = next(row for row in corpus if row["split"] == "train")
    models = json.loads(args.manifest.read_text())
    if args.labels:
        models = [model for model in models if model["label"] in args.labels]
    if not models or any(not re.fullmatch(r"[A-Za-z0-9_-]+", model["label"]) for model in models):
        raise ValueError("No selected models or unsafe report labels.")
    identities = {}
    for name, server in (("reference", args.reference), ("candidate", args.candidate)):
        identities[name] = {path.name: bench.sha256(path) for path in
                            (server, server.parent / "libllama.so", server.parent / "libggml-cpu.so",
                             server.parent / "libggml-cuda.so")}
    report = {
        "schema_version": 1, "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "kind": "tandem-owned-single-user-screen", "backend_sha256": identities,
        "runner_sha256": bench.sha256(Path(__file__)), "corpus_sha256": bench.sha256(args.corpus),
        "context": args.context, "slots": 1, "kv_cache": "f16", "tokens": args.tokens,
        "threads": args.threads, "batch_threads": args.batch_threads, "margin_mib": args.margin,
        "repetitions": args.repetitions, "arms": args.arms,
        "fixed_cache_mib": args.cache_mib,
        "sampling": "greedy, seed 42, reasoning low, forced length, prompt reuse off",
        "limitations": [
            "Single-user synthetic short prompts, not full context or multi-agent qualification.",
            "Different expert-placement mechanisms share a free-memory margin, not identical tensor placement.",
            "Decode rate is (outputs - 1) / (request wall - backend prompt time).",
            "First-token full-vocabulary log probabilities are diagnostic, not raw logits or broad quality.",
            "Fixed share is a preserving control, not a guarantee of numerical equivalence.",
            "Defaults may change rounding; faster rates do not qualify a failed numerical path.",
            "Pinned-weight copies and drafts disabled; no installed state changed.",
        ],
        "models": [],
    }
    bench.sweep.save_report(args.output, report)
    control_arm = "reference" if "reference" in args.arms else "reference-cache"
    for model in models:
        entry = {"label": model["label"], "files": [], "runs": [], "summary": []}
        report["models"].append(entry)
        for shard in model["files"]:
            path = Path(shard["path"]).resolve(strict=True)
            actual = bench.sha256(path)
            if actual != shard["sha256"]:
                raise RuntimeError("Qualification model hash differs from the manifest.")
            entry["files"].append({"name": path.name, "sha256": actual})
        model_path = Path(model["files"][0]["path"])
        references = {}
        for repetition in range(args.repetitions):
            order = args.arms if repetition % 2 == 0 else list(reversed(args.arms))
            for arm in order:
                run = {"arm": arm, "repetition": repetition, "status": "starting", "samples": []}
                entry["runs"].append(run)
                bench.sweep.save_report(args.output, report)
                print(f"START {model['label']} {arm} repetition={repetition}", flush=True)
                server = args.reference if arm.startswith("reference") else args.candidate
                process = monitor = None
                log_path = args.output.with_name(f"{args.output.stem}-{model['label']}-{repetition}-{arm}.log")
                stats_path = log_path.with_suffix(".stats.jsonl")
                try:
                    bench.check_gpu_processes(bench.gpu_processes())
                    with socket.socket() as reservation:
                        reservation.bind(("127.0.0.1", 0))
                        port = reservation.getsockname()[1]
                    key = secrets.token_hex(32)
                    base = f"http://127.0.0.1:{port}"
                    command = [
                        str(server), "--model", str(model_path), "--alias", "local-model",
                        "--host", "127.0.0.1", "--port", str(port), "--api-key", key,
                        "--ctx-size", str(args.context), "--parallel", "1", "--threads", str(args.threads),
                        "--threads-batch", str(args.batch_threads), "--cache-type-k", "f16",
                        "--cache-type-v", "f16", "--poll", "50", "--spec-type", "none",
                        "--cache-ram", "0", "--slots", "--log-verbosity", "4",
                    ]
                    if arm == "reference":
                        command += ["--n-gpu-layers", "auto", "--fit", "on", "--fit-target", str(args.margin)]
                    elif arm in CACHE_ONLY_ARMS:
                        command += ["--n-gpu-layers", "all", "--fit", "off", "--cpu-moe",
                                    "--moe-cache-mib", str(args.cache_mib)]
                    else:
                        command += ["--n-gpu-layers", "all", "--fit", "off", "--cpu-moe",
                                    "--moe-cache-mib", "auto", "--moe-cache-margin-mib", str(args.margin)]
                    env = environment(arm, args.threads, stats_path)
                    env["LD_LIBRARY_PATH"] = str(server.parent)
                    with tempfile.TemporaryDirectory(prefix="tandem-runtime-", dir=args.output.parent) as config:
                        env["XDG_CONFIG_HOME"] = config
                        with log_path.open("x") as log:
                            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
                            bench.wait_ready(process, base, key, args.load_timeout)
                            text = log_path.read_text()
                            run["startup"] = bench.parse_startup(text, 1, args.context)
                            run["activation"] = activation(text, arm)
                            run["gpu_loaded_mib"] = bench.check_gpu_processes(bench.gpu_processes(), process.pid)
                            monitor = bench.GpuMonitor(process.pid)
                            monitor.start()
                            bench.sweep.sample(base, 64, {"Authorization": f"Bearer {key}"}, prompt=warmup["prompt"])
                            for row in held_out:
                                sample = bench.sweep.sample(base, args.tokens, {"Authorization": f"Bearer {key}"},
                                                            prompt=row["prompt"])
                                if sample["timings"].get("cache_n") != 0 or sample["first_token_seconds"] is None:
                                    raise RuntimeError("Missing first output or unexpected prompt reuse.")
                                sample["category"] = row["category"]
                                run["samples"].append(sample)
                                print(f"RATE {model['label']} {arm} {row['category']} "
                                      f"decode={sample['wall_decode_tok_s']:.2f} "
                                      f"whole={sample['end_to_end_tok_s']:.2f}", flush=True)
                            request = urllib.request.Request(base + "/v1/models",
                                                             headers={"Authorization": f"Bearer {key}"})
                            with urllib.request.urlopen(request, timeout=20) as response:
                                vocabulary = json.load(response)["data"][0]["meta"]["n_vocab"]
                            events = [numerics.complete(base, key, row["prompt"], vocabulary) for row in held_out]
                            references[(repetition, arm)] = events
                            if not arm.startswith("reference"):
                                if not stats_path.exists():
                                    raise RuntimeError("Hybrid execution counters were not emitted.")
                                stats = [json.loads(line) for line in stats_path.read_text().splitlines()]
                                if not stats or stats[-1]["hits"] + stats[-1]["misses"] <= 0:
                                    raise RuntimeError("Configured expert cache did not execute lookups.")
                                if arm in CACHE_ONLY_ARMS and stats[-1]["cpu_experts"] != 0:
                                    raise RuntimeError("CPU expert computation appeared in the GPU-only cache arm.")
                                if arm not in CACHE_ONLY_ARMS and (
                                        stats[-1]["cpu_experts"] <= 0 or stats[-1]["calls"] <= 0):
                                    raise RuntimeError("Configured hybrid path did not execute CPU experts.")
                                run["execution_counters"] = stats[-1]
                            component_activation(log_path.read_text(), arm)
                            monitor.stop()
                            monitor = None
                            run["gpu_after_mib"] = bench.check_gpu_processes(bench.gpu_processes(), process.pid)
                            run["status"] = "completed"
                except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as exc:
                    run.update(status="failed", error=str(exc).replace(str(model_path.parent), "/models")
                               .replace(str(Path.home()), "/home/benchmark-user"))
                    if log_path.exists():
                        lines = log_path.read_text(errors="replace").splitlines()
                        run["failure_tail"] = [line.replace(str(model_path.parent), "/models")
                                               .replace(str(Path.home()), "/home/benchmark-user") for line in lines[-16:]]
                    print(f"FAILED {model['label']} {arm}: {run['error']}", flush=True)
                finally:
                    try:
                        if monitor is not None:
                            monitor.stop()
                    finally:
                        if process is not None and process.poll() is None:
                            process.terminate()
                            try:
                                process.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait()
                    entry["summary"] = summarize(entry["runs"], args.arms, args.repetitions,
                                                 [row["category"] for row in held_out])
                    bench.sweep.save_report(args.output, report)
            for arm in args.arms:
                if arm != control_arm and (repetition, arm) in references and (repetition, control_arm) in references:
                    run = next(row for row in entry["runs"] if row["arm"] == arm and row["repetition"] == repetition)
                    run["first_token_logprob_comparisons"] = [
                        numerics.compare_probabilities(ref, cand, vocabulary)
                        for ref, cand in zip(references[(repetition, control_arm)], references[(repetition, arm)])
                    ]
                    control = next(row for row in entry["runs"] if row["arm"] == control_arm
                                   and row["repetition"] == repetition)
                    run["output_hash_matches"] = [
                        left["generated_text_sha256"] == right["generated_text_sha256"]
                        for left, right in zip(control["samples"], run["samples"])
                    ]
            references.clear()
            bench.sweep.save_report(args.output, report)
            if all(run["status"] == "failed" for run in entry["runs"]):
                break
    report["completed_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    bench.sweep.save_report(args.output, report)
    return int(any(run["status"] == "failed" for model in report["models"] for run in model["runs"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--corpus", type=Path, default=ROOT / "benchmarks/cpu-decode-corpus.jsonl")
    parser.add_argument("--arms", nargs="+", choices=("reference", "reference-cache", "tandem-cache",
                                                     "tandem-fixed", "tandem-defaults", *COMPONENTS),
                        default=["reference", "tandem-fixed", "tandem-defaults"])
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--context", type=int, default=16384)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch-threads", type=int, default=16)
    parser.add_argument("--margin", type=int, default=1024)
    parser.add_argument("--cache-mib", type=int, default=4096)
    parser.add_argument("--load-timeout", type=int, default=180)
    args = parser.parse_args()
    if (sum(arm.startswith("reference") for arm in args.arms) != 1 or len(set(args.arms)) != len(args.arms)
            or not 1 <= args.repetitions <= 3 or not 2 <= args.tokens <= 512
            or not 256 <= args.cache_mib <= 8192
            or min(args.context, args.threads, args.batch_threads, args.margin, args.load_timeout) <= 0):
        parser.error("Invalid arms, repetitions, token count or resource settings.")
    return execute(args)


if __name__ == "__main__":
    os.umask(0o077)
    raise SystemExit(main())
