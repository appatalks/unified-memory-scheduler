#!/usr/bin/env python3
"""Measure owned Synapse runtimes without changing installed profiles or saved settings."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import statistics
import sys
import tempfile
import threading
import time


ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ("standard", "classic", "candidate-classic", "hosted")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot import benchmark dependency: {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def trial_settings(model: Path, context: int, variant: str, cache_mib: int) -> dict:
    if variant not in VARIANTS:
        raise ValueError("Unknown execution variant.")
    if type(context) is not int or context < 512 or context % 256:
        raise ValueError("Use an explicit context capacity aligned to 256 tokens.")
    hybrid = variant != "standard"
    return {
        "model": str(model), "execution_mode": "hybrid" if hybrid else "standard",
        "context": context, "placement": "auto", "cpu_threads": 8, "batch_threads": 16,
        "load_mode": "auto", "expert_cache_mib": cache_mib if hybrid else 0,
        "fit_target_mib": 1024, "host_op_offload": not hybrid, "cpu_poll": 50,
        "op_offload_min_batch": 32, "kv_cache_type": "f16", "draft_type": "none",
        "hybrid_variant": "hosted" if variant == "hosted" else "classic",
    }


def verify_activation(state: dict, settings: dict, variant: str) -> None:
    for key in ("context", "cpu_threads", "batch_threads", "kv_cache_type",
                "expert_cache_mib", "execution_mode"):
        if state.get(key) != settings[key]:
            raise RuntimeError(f"Backend did not apply the requested {key}.")
    hybrid = variant != "standard"
    if state.get("hybrid_active") is not hybrid:
        raise RuntimeError("Unexpected execution engine; fallback is not a valid measurement.")
    if hybrid and not state.get("hybrid_verified"):
        raise RuntimeError("Required hybrid activation markers were not confirmed.")
    if hybrid and state.get("hybrid_hosted") is not (variant == "hosted"):
        raise RuntimeError("Unexpected hybrid variant.")
    if variant == "hosted" and not state.get("hybrid_hosted_verified"):
        raise RuntimeError("Hosted execution and LFU activation were not confirmed.")
    if state.get("draft_model"):
        raise RuntimeError("Scalar baseline must not enable drafting.")


def wait_ready(runtime, settings: dict, variant: str) -> dict:
    deadline = time.monotonic() + runtime.load_timeout + 20
    while time.monotonic() < deadline:
        state = runtime.status()
        if state["state"] == "error":
            raise RuntimeError(f"{state['error']}\n" + "\n".join(state["logs"][-12:]))
        if state["state"] == "ready":
            verify_activation(state, settings, variant)
            return state
        if state["state"] in ("stopped", "stopping"):
            raise RuntimeError("Owned backend stopped before becoming ready.")
        threading.Event().wait(0.25)
    raise RuntimeError("Owned backend exceeded its loading deadline.")


def trial_order(variants: list[str], repetition: int) -> list[str]:
    return list(reversed(variants)) if repetition % 2 else list(variants)


def track_runtime_marker(runtime, marker: str):
    observed = threading.Event()
    original = runtime._read_logs

    def tracked(process, key):
        source = process.stdout

        class Stream:
            def __iter__(self):
                for line in source:
                    if marker in line:
                        observed.set()
                    yield line

            def close(self):
                source.close()

        process.stdout = Stream()
        return original(process, key)

    runtime._read_logs = tracked
    return observed


def wait_runtime_marker(runtime, marker: str, timeout: float = 10, observed=None) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = runtime.status()
        if (observed is not None and observed.is_set()) or any(marker in line for line in state["logs"]):
            return
        if state["state"] != "ready":
            raise RuntimeError("Backend stopped before confirming its required marker.")
        time.sleep(0.1)
    raise RuntimeError("Required backend activation marker was not observed.")


def verify_causal_singleton_log(log: str, context: int) -> None:
    batches = re.findall(r"n_ubatch\s*=\s*(\d+)\b", log)
    causal = re.findall(r"causal_attn\s*=\s*(\d+)\b", log)
    slots = re.findall(r"n_slots\s*=\s*(\d+),\s*n_ctx_slot\s*=\s*(\d+)\b", log)
    caches = re.findall(r"K \(([^)]+)\):[^\n]*V \(([^)]+)\):", log)
    gpu = re.findall(r"CUDA\d+ model buffer size\s*=\s*(\d+(?:\.\d+)?)\s+MiB", log)
    if (not batches or set(batches) != {"1"} or not causal or set(causal) != {"1"} or
            not slots or set(slots) != {("1", str(context))} or
            not caches or set(caches) != {("f16", "f16")} or not any(float(size) > 0 for size in gpu)):
        raise RuntimeError("The required single-slot, causal, F16, GPU and microbatch contract was not confirmed.")


def capacity_probe(server, runtime, context: int, timeout: float, progress_log: Path | None = None) -> dict:
    expired = threading.Event()
    done = threading.Event()

    def cancel():
        expired.set()
        server.tuner.cancelled.set()

    timer = threading.Timer(timeout, cancel)
    timer.start()
    monitor = None
    if progress_log is not None:
        def watch_progress():
            import re
            offset = progress_log.stat().st_size if progress_log.exists() else 0
            while not done.wait(3):
                try:
                    with progress_log.open("r", encoding="utf-8", errors="replace") as stream:
                        stream.seek(offset)
                        text = stream.read(1024 * 1024)
                        offset = stream.tell()
                    for line in text.splitlines():
                        match = re.search(r"prompt processing, n_tokens =\s*(\d+), progress =\s*([\d.]+)", line)
                        if match:
                            print(f"CAPACITY processed={match[1]} progress={match[2]}", flush=True)
                except OSError:
                    pass
        monitor = threading.Thread(target=watch_progress, daemon=True)
        monitor.start()
    try:
        try:
            result = server.tuner._probe(runtime, context)
        except RuntimeError as exc:
            if expired.is_set():
                raise RuntimeError(f"Near-capacity probe exceeded its {timeout:g}-second deadline; capacity remains unverified.") from exc
            raise
        if expired.is_set():
            raise RuntimeError("Near-capacity probe completed after its deadline; capacity remains unverified.")
        return result
    finally:
        done.set()
        if monitor is not None:
            monitor.join(timeout=5)
        timer.cancel()
        timer.join()


def summarize(runs: list[dict]) -> list[dict]:
    groups = {}
    for run in runs:
        if run["status"] != "completed":
            continue
        for sample in run["samples"]:
            key = run["context"], run["variant"], sample["category"]
            groups.setdefault(key, []).append(sample)
    controls = {(context, category): {s["generated_text_sha256"] for s in samples}
                for (context, variant, category), samples in groups.items() if variant == "standard"}
    return [
        {"context": context, "variant": variant, "category": category,
         "samples": len(samples),
         "median_wall_decode_tok_s": statistics.median(s["wall_decode_tok_s"] for s in samples),
         "median_end_to_end_tok_s": statistics.median(s["end_to_end_tok_s"] for s in samples),
         "repeated_text_hashes_match": len({s["generated_text_sha256"] for s in samples}) == 1,
         "matches_standard_text": (
             len(controls[context, category]) == 1
             and {s["generated_text_sha256"] for s in samples} == controls[context, category]
         ) if (context, category) in controls else None}
        for (context, variant, category), samples in groups.items()
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def execute(args) -> int:
    sys.path.insert(0, str(args.synapse_source.resolve(strict=True)))
    from synapse_app import app

    sweep = load_module("baseline_sweep", ROOT / "tools/benchmark/vrambridge-execution-sweep.py")
    matrix = load_module("baseline_matrix", ROOT / "tools/benchmark/vrambridge-model-matrix.py")
    bindings = (load_module("baseline_bindings", ROOT / "tools/benchmark/vrambridge-serial-reuse-check.py")
                if args.require_causal_singleton else None)
    profiles = {"installed": app.read_hybrid_profile(args.profile.resolve(strict=True))}
    if args.candidate_profile:
        profiles["candidate"] = app.read_hybrid_profile(args.candidate_profile.resolve(strict=True))
    if any(v in ("candidate-classic", "hosted") for v in args.variants) and "candidate" not in profiles:
        raise ValueError("Candidate variants require an explicit separate candidate profile.")
    model = args.model.resolve(strict=True)
    metadata = app.model_metadata(model)
    corpus = [json.loads(line) for line in args.corpus.read_text().splitlines()]
    corpus = [row for row in corpus if row["split"] == args.split]
    if not corpus:
        raise ValueError("No matching corpus requests.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() or args.output.with_suffix(args.output.suffix + ".tmp").exists():
        raise ValueError("Refusing to overwrite an existing report.")
    report = {
        "schema_version": 1, "kind": "owned-synapse-engine-baseline",
        "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model": model.name, "metadata": metadata, "output_tokens": args.tokens,
        "repetitions": args.repetitions, "split": args.split,
        "corpus_sha256": sha256(args.corpus),
        "runner_sha256": sha256(Path(__file__)),
        "synapse_runtime_sha256": sha256(args.synapse_source / "synapse_app/app.py"),
        "backend_identities": {
            name: {"executable_sha256": sha256(Path(profile["hybrid_executable"])),
                   "libllama_sha256": sha256(Path(profile["hybrid_executable"]).parent / "libllama.so"),
                   "cuda_plugin_sha256": sha256(profile["hybrid_cuda_plugin"])}
            for name, profile in profiles.items()
        },
        "standard_backend_identity": {
            "executable_sha256": sha256(args.backend),
            "libllama_sha256": sha256(args.backend.parent / "libllama.so"),
        },
        "runs": [], "summary": [], "performance_target_met": False,
        "limitations": [
            "Context capacity is allocated; short-prompt rates do not establish near-capacity input performance.",
            "Long-context hybrid profile overrides exist only in the owned runtime, not the installed profile.",
            "Forced-length throughput includes possible output after an end marker.",
            "Rate is (output_tokens - 1) / (request wall time - reported prompt time).",
            "Hash and bounded answer checks do not establish full-logit or broad quality equivalence.",
            "Installed versus candidate binaries differ; compare candidate-classic with hosted for attribution.",
            "Warmup is excluded; adaptive expert residency continues across workload requests within a load.",
        ],
    }
    if args.require_causal_singleton:
        directory = args.backend.resolve().parent
        report["model_sha256"] = sha256(model)
        report["standard_backend_identity"]["components_sha256"] = {
            name: sha256(directory / name) for name in ("llama-server", *bindings.LOCAL_LIBRARY_NAMES)}
    failed = False
    sweep.save_report(args.output, report)
    for context in args.contexts:
        for repetition in range(args.repetitions):
            for variant in trial_order(args.variants, repetition):
                settings = trial_settings(model, context, variant, args.cache_mib)
                profile = dict(profiles["candidate" if variant in ("candidate-classic", "hosted") else "installed"])
                profile["hybrid_context"] = context
                runtime = app.Runtime(str(args.backend), execution_mode="standard", **profile)
                marker_observed = track_runtime_marker(runtime, args.require_runtime_marker) if args.require_runtime_marker else None
                server = app.AppServer(("127.0.0.1", 0), runtime, update_checks=False)
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                base = f"http://127.0.0.1:{server.server_port}"
                run = {"context": context, "variant": variant, "repetition": repetition,
                       "status": "running", "samples": [], "answer_checks": [],
                       "settings": {key: value for key, value in settings.items() if key != "model"}}
                report["runs"].append(run)
                sweep.save_report(args.output, report)
                print(f"START context={context} variant={variant} repetition={repetition}", flush=True)
                try:
                    started = time.monotonic()
                    runtime.load(settings, remember=False)
                    state = wait_ready(runtime, settings, variant)
                    run.update(load_seconds=time.monotonic() - started,
                               hybrid_verified=state["hybrid_verified"],
                               hybrid_hosted_verified=state["hybrid_hosted_verified"],
                               hybrid_async_ids_verified=state["hybrid_async_ids_verified"],
                               execution_profile=state["execution_profile"],
                               selection_reason=state["selection_reason"])
                    sweep.sample(base, 64, prompt=corpus[0]["prompt"])
                    if args.require_runtime_marker:
                        wait_runtime_marker(runtime, args.require_runtime_marker, observed=marker_observed)
                        run["required_runtime_marker_verified"] = args.require_runtime_marker
                    if args.require_causal_singleton:
                        verify_causal_singleton_log(args.progress_log.read_text(), context)
                        bindings.verify_local_libraries(Path(f"/proc/{runtime.process.pid}/maps").read_text(),
                                                        args.backend.resolve().parent)
                        run["causal_singleton_contract_verified"] = True
                    for row in corpus:
                        sample = sweep.sample(base, args.tokens, prompt=row["prompt"])
                        sample.update(category=row["category"], prompt_sha256=hashlib.sha256(row["prompt"].encode()).hexdigest())
                        run["samples"].append(sample)
                        print(f"RATE {variant} {row['category']} {sample['wall_decode_tok_s']:.2f}", flush=True)
                    if args.answer_checks and repetition == 0:
                        for name, prompt in (
                            ("arithmetic", "What is 17 times 19? Return only the integer."),
                            ("code", "Return only this Python function, without Markdown: def clamp(value, low, high): return min(high, max(low, value))"),
                        ):
                            run["answer_checks"].append(matrix.quality_sample(base, name, prompt))
                    if args.capacity_check and repetition == 0:
                        run["capacity_check"] = capacity_probe(server, runtime, context, args.capacity_timeout,
                                                               args.progress_log)
                    if args.require_causal_singleton:
                        bindings.verify_local_libraries(Path(f"/proc/{runtime.process.pid}/maps").read_text(),
                                                        args.backend.resolve().parent)
                    run.update(status="completed", resources=runtime.status()["resources"])
                except KeyboardInterrupt:
                    run.update(status="cancelled", error="Owned trial interrupted; unfinished capacity and throughput checks are not qualified.")
                    raise
                except (OSError, RuntimeError, ValueError, app.AppError) as exc:
                    failed = True
                    message = str(exc).replace(str(model.parent), "/models").replace(str(Path.home()), "/home/benchmark-user")
                    run.update(status="failed", error=message)
                    print(f"FAILED {variant}: {message}", file=sys.stderr, flush=True)
                finally:
                    runtime.stop(closing=True)
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=10)
                    report["summary"] = summarize(report["runs"])
                    sweep.save_report(args.output, report)
    print(json.dumps(report["summary"], indent=2), flush=True)
    return int(failed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synapse-source", type=Path, required=True)
    parser.add_argument("--backend", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--candidate-profile", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--contexts", type=int, nargs="+", required=True)
    parser.add_argument("--variants", choices=VARIANTS, nargs="+", default=["standard", "classic"])
    parser.add_argument("--cache-mib", type=int, default=8192)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--corpus", type=Path, default=ROOT / "benchmarks/cpu-decode-corpus.jsonl",
                        help="Explicit synthetic corpus for matched short- or long-input requests.")
    parser.add_argument("--answer-checks", action="store_true")
    parser.add_argument("--capacity-check", action="store_true",
                        help="Also run Synapse's near-capacity input and anchor-retrieval probe.")
    parser.add_argument("--capacity-timeout", type=int, default=300,
                        help="Maximum seconds per capacity probe; expiration is a reported trial failure.")
    parser.add_argument("--require-runtime-marker", help="Require a bounded backend log marker before benchmarking.")
    parser.add_argument("--progress-log", type=Path, help="Read backend prompt-progress counters for this owned run.")
    parser.add_argument("--require-causal-singleton", action="store_true",
                        help="Verify actual single-slot causal GPU/F16 placement and local library bindings.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 2 <= args.tokens <= 4096 or not 1 <= args.repetitions <= 10 or not 1 <= args.cache_mib <= 65536:
        parser.error("Invalid output length, repetitions, or expert-cache budget.")
    if not 1 <= args.capacity_timeout <= 2700:
        parser.error("Capacity timeout must be between 1 and 2700 seconds.")
    if args.require_runtime_marker is not None and (not args.require_runtime_marker or len(args.require_runtime_marker) > 160):
        parser.error("The activation marker must contain 1-160 characters.")
    if args.require_causal_singleton and (sys.platform != "linux" or args.progress_log is None or
                                          args.require_runtime_marker != "EXPERIMENT_CAUSAL_SINGLE_TOKEN prefill executed" or
                                          args.variants != ["standard"] or args.repetitions != 1):
        parser.error("The causal singleton contract requires Linux, its execution marker, one standard run and a private log.")
    if len(set(args.variants)) != len(args.variants) or len(set(args.contexts)) != len(args.contexts):
        parser.error("Duplicate variants or contexts are not independent trials.")
    for context in args.contexts:
        trial_settings(args.model, context, "standard", args.cache_mib)
    comparison = load_module("baseline_environment", ROOT / "tools/benchmark/vrambridge-engine-comparison.py")
    environment = comparison.clean_environment()
    os.environ.clear()
    os.environ.update(environment)
    with tempfile.TemporaryDirectory(prefix="vrambridge-baseline-") as config:
        os.environ["XDG_CONFIG_HOME"] = config
        return execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
