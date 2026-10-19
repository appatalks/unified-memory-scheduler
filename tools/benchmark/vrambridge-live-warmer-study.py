#!/usr/bin/env python3
"""Run matched native/noop/read live-warmer research requests, never alter app settings."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import tempfile


def run(args):
    root = args.output_dir.resolve()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if root.stat().st_uid != os.getuid() or root.stat().st_mode & 0o077:
        raise ValueError("Research output directory must be owned and private.")
    profile = json.loads(args.profile.read_text())
    if profile.get("context") != 16384 or profile.get("cache_mib") != 12000:
        raise ValueError("This matched research study requires the existing16K/12000MiB profile.")
    records = [json.loads(line) for line in args.corpus.read_text().splitlines() if line.strip()]
    requests = [r for r in records if r["split"] == "test" and r["category"] == args.category]
    if len(requests) != 1:
        raise ValueError("Choose exactly one held-out request for the category.")
    prompt = requests[0]["prompt"]
    library, cpu_library = args.warmer.resolve(strict=True), args.cpu_library.resolve(strict=True)
    variants = ["native", "noop", "read"] if args.pair % 2 == 0 else ["read", "noop", "native"]
    names = [f"{args.label}-{args.category}-pair{args.pair}-{variant}" for variant in variants]
    for name in names:
        for suffix in ("-request.json", "-request.json.part", "-warm.json", ".log"):
            if (root / (name + suffix)).exists():
                raise ValueError("Refusing to overwrite a prior run: " + name)
    base_env = {key: value for key, value in os.environ.items()
                if key != "LD_PRELOAD" and not key.startswith(("VRAMBRIDGE_CPU_PROFILE", "VRAMBRIDGE_CPU_WARM_"))
                and key not in ("GOMP_SPINCOUNT", "OMP_PLACES", "OMP_PROC_BIND", "OMP_WAIT_POLICY",
                                "GOMP_CPU_AFFINITY", "GGML_CUDA_DISABLE_FUSION", "GGML_CUDA_MMQ_PREC")}
    base_env["TZ"] = "UTC"
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    observed, mapped_libraries = [], []
    with tempfile.TemporaryDirectory(prefix="user-prompt-", dir=root) as temporary:
        prompt_file = Path(temporary) / "prompt.txt"
        prompt_file.write_text(prompt)
        prompt_file.chmod(0o600)
        for name, variant in zip(names, variants):
            environment = dict(base_env)
            warm_output = root / (name + "-warm.json")
            if variant != "native":
                environment.update(
                    LD_PRELOAD=str(library), VRAMBRIDGE_CPU_WARM_MODE=variant,
                    VRAMBRIDGE_CPU_WARM_LIBRARY=str(cpu_library), VRAMBRIDGE_CPU_WARM_OUT=str(warm_output),
                    VRAMBRIDGE_CPU_WARM_CPU=str(args.helper_cpu), VRAMBRIDGE_CPU_WARM_KIB=str(args.warm_kib),
                    VRAMBRIDGE_CPU_WARM_MAX_US=str(args.maximum_us))
            command = [
                sys.executable, str(Path(__file__).with_name("vrambridge-io-study.py")),
                "--llama-server", profile["executable"], "--hybrid-profile", str(args.profile.resolve()),
                "--model", str(args.model.resolve(strict=True)), "--context", "16384", "--threads", "8",
                "--batch-threads", "16", "--batch-size", "2048", "--ubatch-size", "512",
                "--tokens", str(args.tokens), "--repetitions", "1", "--verbatim-prompt",
                "--prompt-file", str(prompt_file), "--output", str(root / (name + "-request.json")),
            ]
            with (root / (name + ".log")).open("x") as log:
                process = subprocess.Popen(command, env=environment, stdout=log, stderr=log, start_new_session=True)
                try:
                    returncode = process.wait(timeout=180)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    raise RuntimeError("Timed out and stopped the owned request process group: " + name)
            if returncode:
                raise RuntimeError(f"Owned request failed: {name}; see its retained log/report.")
            report = json.loads((root / (name + "-request.json")).read_text())
            if (report["status"] != "complete" or not report["owned_backend_stopped"]
                    or report["prompt_sha256"] != hashlib.sha256(prompt.encode()).hexdigest()
                    or report["actual_context"] != 16384
                    or report["applied_settings"]["expert_cache_mib"] != 12000):
                raise RuntimeError("Request control/shape acknowledgement failed: " + name)
            sample = report["samples"][0]
            modules = report["mapped_backend_libraries"]
            if str(cpu_library) not in modules or not any("/libggml-base.so" in path for path in modules):
                raise RuntimeError("Required native CPU/base library mapping missing: " + name)
            if mapped_libraries and modules != mapped_libraries:
                raise RuntimeError("Baseline and candidate backend library mappings differ: " + name)
            mapped_libraries = modules
            if variant != "native":
                warm = json.loads(warm_output.read_text())
                if (warm["complete"] is not True or warm["mode"] != variant
                        or warm["pid"] != sample["before_request"]["pid"]
                        or warm["helper_cpu"] != args.helper_cpu
                        or warm["initial_compute_affinity"] != sorted(os.sched_getaffinity(0))
                        or warm["warm_budget_bytes"] != args.warm_kib * 1024):
                    raise RuntimeError("Live helper identity/configuration acknowledgement failed: " + name)
            observed.append(sample["timing"]["generated_text_sha256"])
            print(json.dumps({"run": name, "decode_tok_s": sample["timing"]["timings"]["predicted_per_second"],
                              "request_seconds": sample["timing"]["elapsed_seconds"],
                              "generation_process_cpu_seconds": sample["generation_interval"]["cpu_seconds"]}),
                  flush=True)
    if len(set(observed)) != 1:
        raise RuntimeError("Matched request output hashes differ; results retained but not qualified.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--warmer", type=Path, required=True)
    parser.add_argument("--cpu-library", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=Path("benchmarks/cpu-decode-corpus.jsonl"))
    parser.add_argument("--category", choices=("coding", "reasoning", "prose"), required=True)
    parser.add_argument("--pair", type=int, choices=range(3), required=True)
    parser.add_argument("--helper-cpu", type=int, required=True)
    parser.add_argument("--warm-kib", type=int, default=1024)
    parser.add_argument("--maximum-us", type=int, default=200)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--label", default="matched")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if (not 0 <= args.warm_kib <= 16384 or not 1 <= args.maximum_us <= 5000
            or not 0 <= args.helper_cpu < 1024 or not 16 <= args.tokens <= 1024
            or not args.label or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in args.label)):
        parser.error("Invalid bounded live-warmer study options.")
    try:
        run(args)
        return 0
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"vrambridge-live-warmer-study: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
