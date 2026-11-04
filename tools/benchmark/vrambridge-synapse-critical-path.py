#!/usr/bin/env python3
"""Capture an owned Synapse decode request for CPU/CUDA critical-path analysis."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[2]


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot import diagnostic dependency: {filename}")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


baseline = module("critical_baseline", "vrambridge-synapse-baseline.py")
io = module("critical_io", "vrambridge-io-study.py")
sweep = module("critical_sweep", "vrambridge-execution-sweep.py")


def validate_directory(path: Path) -> None:
    info = path.stat()
    if not path.is_dir() or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError("Diagnostic output requires an owned private directory with mode 0700.")


def capture(args, app):
    model = args.model.resolve(strict=True)
    profile = app.read_hybrid_profile(args.profile.resolve(strict=True))
    profile["hybrid_context"] = args.context
    settings = baseline.trial_settings(model, args.context, args.variant, args.cache_mib)
    corpus = [json.loads(line) for line in (ROOT / "benchmarks/cpu-decode-corpus.jsonl").read_text().splitlines()]
    row = next(row for row in corpus if row["split"] == "train" and row["category"] == args.category)
    report = {
        "schema_version": 1, "study": "owned_synapse_decode_critical_path",
        "status": "running", "performance_measurement": False,
        "full_model_acceleration_claim": False, "model": model.name,
        "requested_context": args.context, "output_tokens": args.tokens,
        "variant": args.variant, "category": args.category,
        "source_sha256": baseline.sha256(Path(__file__)),
        "cpu_profiler_sha256": baseline.sha256(args.cpu_profiler),
        "cpu_library_sha256": baseline.sha256(args.cpu_library),
        "synapse_runtime_sha256": baseline.sha256(args.synapse_source / "synapse_app/app.py"),
        "prompt_sha256": hashlib.sha256(row["prompt"].encode()).hexdigest(),
        "samples": [],
        "limitations": [
            "Profiler-perturbed diagnostic; not an uninstrumented throughput benchmark.",
            "Allocated context is preserved; the short request does not establish filled-context behavior.",
            "First received text bounds a partial token interval, not exact backend decode boundaries.",
            "Adaptive causal residency continues through excluded warmup; selected experts and F16 KV are unchanged.",
        ],
    }
    runtime = app.Runtime(str(args.backend), execution_mode="standard", **profile)
    try:
        started = time.monotonic()
        runtime.load(settings, remember=False)
        state = baseline.wait_ready(runtime, settings, args.variant)
        process = runtime.process
        if process is None:
            raise RuntimeError("Owned diagnostic backend disappeared after readiness.")
        report.update(actual_context=state["context"], model_load_wall_seconds=time.monotonic() - started,
                      applied_settings={key: value for key, value in settings.items() if key != "model"},
                      hybrid_active=state["hybrid_active"],
                      hybrid_markers=sorted(runtime.hybrid_markers),
                      cuda_runtime_environment=io.cuda_runtime_environment(process.pid),
                      mapped_backend_libraries=io.backend_library_modules(process.pid))
        if str(args.cpu_library.resolve(strict=True)) not in io.cpu_backend_modules(process.pid):
            raise RuntimeError("The profiler's CPU library does not match the actual mapped CPU backend.")
        url = f"http://127.0.0.1:{runtime.port}"
        headers = {"Authorization": f"Bearer {runtime.key}"}
        sweep.sample(url, 64, headers=headers, prompt=row["prompt"])
        before, first = io.snapshot(process.pid), None

        def first_text():
            nonlocal first
            first = io.snapshot(process.pid)

        timing = sweep.sample(url, args.tokens, headers=headers, prompt=row["prompt"],
                              on_first_token=first_text)
        after = io.snapshot(process.pid)
        if first is None:
            raise RuntimeError("No first output text was observed; the diagnostic interval is unavailable.")
        report["samples"].append({
            "timing": timing, "before_request": before, "first_text_event": first,
            "response_complete": after, "generation_interval": io.interval(first, after),
        })
        report["status"] = "complete"
    except (OSError, ValueError, RuntimeError, app.AppError) as exc:
        report.update(status="failed", error=str(exc).replace(str(model), "/models/" + model.name))
    finally:
        try:
            runtime.stop(closing=True)
            report["owned_backend_stopped"] = runtime.process is None
        except (OSError, RuntimeError) as exc:
            report.update(status="failed", cleanup_error=str(exc))
        report["backend_log_tail"] = [line.replace(str(model), "/models/" + model.name) for line in runtime.logs]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synapse-source", type=Path, required=True)
    parser.add_argument("--backend", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--cache-mib", type=int, default=4096)
    parser.add_argument("--variant", choices=("standard", "classic", "hosted"), default="classic")
    parser.add_argument("--category", choices=("coding", "reasoning", "prose"), default="coding")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--cpu-profiler", type=Path, required=True)
    parser.add_argument("--cpu-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != "linux" or not 16 <= args.tokens <= 128:
        parser.error("Linux and 16-128 diagnostic outputs are required.")
    if not 512 <= args.context <= 262144 or args.context % 256 or not 1 <= args.cache_mib <= 65536:
        parser.error("Invalid context alignment/range or bank budget.")
    validate_directory(args.output.parent)
    cpu_output = args.output.with_suffix(".cpu.jsonl")
    if args.output.exists() or cpu_output.exists():
        parser.error("Refusing to overwrite an existing diagnostic.")
    profiler = args.cpu_profiler.resolve(strict=True)
    cpu = args.cpu_library.resolve(strict=True)
    os.environ.update(
        LD_PRELOAD=":".join(filter(None, (os.environ.get("LD_PRELOAD"), str(profiler)))),
        VRAMBRIDGE_CPU_PROFILE_LIBRARY=str(cpu),
        VRAMBRIDGE_CPU_PROFILE_OUT=str(cpu_output), VRAMBRIDGE_CPU_PROFILE_EXPERT_IDS="1")
    sys.path.insert(0, str(args.synapse_source.resolve(strict=True)))
    from synapse_app import app

    report = capture(args, app)
    io.write_report(args.output, report)
    print(json.dumps({key: report[key] for key in ("status", "variant", "requested_context")}), flush=True)
    return int(report["status"] != "complete")


if __name__ == "__main__":
    raise SystemExit(main())
