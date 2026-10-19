#!/usr/bin/env python3
"""Compare bounded handoff candidates using the existing owned-request runner."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile


SPEC = importlib.util.spec_from_file_location("handoff_io", Path(__file__).with_name("vrambridge-io-study.py"))
assert SPEC and SPEC.loader
io = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(io)


def run(args):
    root = args.output_dir.resolve()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if root.stat().st_uid != os.getuid() or root.stat().st_mode & 0o077:
        raise ValueError("Owned private output directory required.")
    profile = json.loads(args.profile.read_text())
    if profile.get("context") != 16384 or profile.get("cache_mib") != 12000:
        raise ValueError("Matched study requires the checked16K/12000MiB profile.")
    corpus = [json.loads(line) for line in args.corpus.read_text().splitlines() if line.strip()]
    requests = [r for r in corpus if r["split"] == "test" and r["category"] in args.categories]
    if len(requests) != len(args.categories) or len({r["category"] for r in requests}) != len(requests):
        raise ValueError("One unique held-out request per category is required.")
    if args.variant == "cpu-one" and (args.library is None or args.cpu_library is None):
        raise ValueError("CPU control experiment requires explicit library and native CPU library.")
    original = dict(os.environ)
    baseline_modules = None
    hashes = {}
    try:
        with tempfile.TemporaryDirectory(prefix="prompts-", dir=root) as directory:
            for pair in range(args.pairs):
                for request in (requests if pair % 2 == 0 else list(reversed(requests))):
                    category = request["category"]
                    prompt = Path(directory) / (category + ".txt")
                    prompt.write_text(request["prompt"])
                    prompt.chmod(0o600)
                    for variant in (("native", args.variant) if pair % 2 == 0 else (args.variant, "native")):
                        name = f"{args.label}-{category}-pair{pair}-{variant}"
                        output = root / (name + ".json")
                        metrics = root / (name + "-metrics.json")
                        if any(p.exists() for p in (output, Path(str(output) + ".part"), metrics)):
                            raise ValueError("Refusing to overwrite an existing run: " + name)
                        env = {k: v for k, v in original.items() if k != "LD_PRELOAD"
                            and not k.startswith(("VRAMBRIDGE_CPU_PROFILE", "VRAMBRIDGE_CPU_WARM_", "VRAMBRIDGE_CPU_CONTROL_"))
                            and k not in ("GOMP_SPINCOUNT", "OMP_PLACES", "OMP_PROC_BIND", "OMP_WAIT_POLICY",
                                          "GOMP_CPU_AFFINITY", "GGML_CUDA_GRAPH_DEBUG", "GGML_CUDA_DISABLE_GRAPHS",
                                          "GGML_CUDA_DISABLE_FUSION", "GGML_CUDA_MMQ_PREC")}
                        env.update(TZ="UTC", GGML_CUDA_GRAPH_OPT="1" if variant == "qkv" else "0")
                        if variant == "cpu-one":
                            env.update(LD_PRELOAD=str(args.library.resolve(strict=True)),
                                VRAMBRIDGE_CPU_CONTROL_THREADS="one",
                                VRAMBRIDGE_CPU_CONTROL_LIBRARY=str(args.cpu_library.resolve(strict=True)),
                                VRAMBRIDGE_CPU_CONTROL_OUT=str(metrics))
                        os.environ.clear()
                        os.environ.update(env)
                        settings = SimpleNamespace(model=args.model, hybrid_profile=args.profile, context=16384,
                            threads=8, batch_threads=16, tokens=args.tokens, repetitions=1,
                            llama_server=profile["executable"], load_timeout=600, prompt_file=prompt,
                            verbatim_prompt=True, batch_size=2048, ubatch_size=512, cpu_poll=50)
                        status, report = io.measure(settings)
                        report["handoff_study_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
                        report["handoff_variant"] = variant
                        io.write_report(output, report)
                        if status:
                            raise RuntimeError("Owned request failed: " + report.get("error", name))
                        if report["actual_context"] != 16384 or report["applied_settings"]["expert_cache_mib"] != 12000:
                            raise RuntimeError("Context or residency changed: " + name)
                        expected_cuda = {"GGML_CUDA_GRAPH_OPT": "1" if variant == "qkv" else "0"}
                        if report["cuda_runtime_environment"] != expected_cuda:
                            raise RuntimeError("CUDA environment mismatch: " + name)
                        modules = report["mapped_backend_libraries"]
                        if baseline_modules is None:
                            baseline_modules = modules
                        if modules != baseline_modules:
                            raise RuntimeError("Backend library set mismatch: " + name)
                        sample = report["samples"][0]
                        if variant == "cpu-one":
                            control = json.loads(metrics.read_text())
                            if (control["complete"] is not True or control["pid"] != sample["before_request"]["pid"]
                                    or control["changed_calls"] <= 0 or control["expert_plan_threads_changed"] is not False):
                                raise RuntimeError("Control-plan acknowledgement failed: " + name)
                        digest = sample["timing"]["generated_text_sha256"]
                        if category in hashes and hashes[category] != digest:
                            raise RuntimeError("Output hashes differ; results retained but not qualified: " + category)
                        hashes[category] = digest
                        print(json.dumps({"run": name,
                            "decode_tok_s": sample["timing"]["timings"]["predicted_per_second"],
                            "request_seconds": sample["timing"]["elapsed_seconds"]}), flush=True)
    finally:
        os.environ.clear()
        os.environ.update(original)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=Path("benchmarks/cpu-decode-corpus.jsonl"))
    parser.add_argument("--variant", choices=("cpu-one", "qkv"), default="cpu-one")
    parser.add_argument("--library", type=Path)
    parser.add_argument("--cpu-library", type=Path)
    parser.add_argument("--categories", choices=("coding", "reasoning", "prose"), nargs="+",
                        default=["coding", "reasoning", "prose"])
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--label", default="paired")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if (not 1 <= args.pairs <= 5 or not 16 <= args.tokens <= 1024
            or len(set(args.categories)) != len(args.categories) or not args.label
            or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in args.label)):
        parser.error("Invalid bounded handoff study parameters.")
    try:
        run(args)
        return 0
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"vrambridge-handoff-study: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
