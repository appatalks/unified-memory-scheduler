#!/usr/bin/env python3
"""Compare standard and model-selected execution on existing GGUFs; restore the app afterward."""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time
from urllib.parse import urlsplit
import urllib.request


ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot import benchmark dependency: {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def answer_check(name: str, answer: str) -> bool:
    if name == "arithmetic":
        return answer.strip() == "323"
    expected = ast.parse("def clamp(value, low, high):\n    return min(high, max(low, value))")
    try:
        return ast.dump(ast.parse(answer.strip()), include_attributes=False) == ast.dump(expected, include_attributes=False)
    except SyntaxError:
        return False


def quality_sample(base: str, name: str, prompt: str, headers: dict[str, str] | None = None) -> dict:
    payload = {"model": "local-model", "messages": [{"role": "user", "content": prompt}],
               "max_tokens": 1024, "temperature": 0, "seed": 42, "reasoning_effort": "low",
               "cache_prompt": False, "stream": False}
    request = urllib.request.Request(base + "/v1/chat/completions", json.dumps(payload).encode(),
                                     {"Content-Type": "application/json", **(headers or {})})
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=120) as response:
        event = json.load(response)
    if "error" in event:
        raise RuntimeError(event["error"])
    choice = event["choices"][0]
    answer = choice["message"].get("content") or ""
    return {"workload": name, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "answer_sha256": hashlib.sha256(answer.encode()).hexdigest(),
            "finish_reason": choice["finish_reason"], "usage": event.get("usage"),
            "elapsed_seconds": time.monotonic() - started,
            "check_passed": choice["finish_reason"] == "stop" and answer_check(name, answer),
            "check": "exact integer" if name == "arithmetic" else "AST-only; model code is never executed"}


def execute(args) -> int:
    sweep = load_module("matrix_execution_sweep", ROOT / "tools/benchmark/vrambridge-execution-sweep.py")
    app = load_module("matrix_app", ROOT / "tools/vrambridge/app.py")
    initial = sweep.request_json(args.base_url, "/api/v1/status")
    if initial["state"] != "ready" or initial["chat_busy"]:
        raise ValueError("App must be idle and ready before a model matrix.")
    models = []
    seen = set()
    for path in args.models:
        path = path.expanduser().resolve(strict=True)
        if path in seen:
            raise ValueError("Duplicate model path.")
        seen.add(path)
        metadata = app.model_metadata(path)
        if metadata["general.architecture"] == "clip":
            raise ValueError("A CLIP vision projector is not a standalone language model.")
        models.append((path, metadata, app.supports_hybrid(metadata)))
    original = {key: initial[key] for key in sweep.SETTING_KEYS}
    original["execution_mode"] = initial.get("execution_mode", "standard")
    report = {"schema_version": 1, "kind": "downloaded-model-matrix",
              "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "backend_commit": app.PINNED_LLAMA_COMMIT,
              "runner_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "app_source_sha256": hashlib.sha256((ROOT / "tools/vrambridge/app.py").read_bytes()).hexdigest(),
              "hybrid_patch_sha256": hashlib.sha256(
                  (ROOT / "third_party/patches/llama-b11669-hybrid-experts.patch").read_bytes()).hexdigest(),
              "plan_sha256": initial.get("hybrid_plan_sha256"),
              "context": args.context, "cpu_threads": args.threads, "batch_threads": args.batch_threads,
              "output_tokens": args.tokens, "repetitions": args.repetitions,
              "throughput_prompt": sweep.PROMPT,
              "throughput_prompt_sha256": hashlib.sha256(sweep.PROMPT.encode()).hexdigest(),
              "runs": [], "restore": {"status": "pending"}, "performance_target_met": False,
              "limitations": [
                  "One technical-explanation throughput prompt, not sustained multi-category qualification.",
                  "Forced output length can include output after an end marker; not 512 natural answer tokens.",
                  "Arithmetic and AST-only checks do not establish broad answer quality.",
                  "Only supported GPT-OSS-120B MXFP4 uses custom expert dispatch; other architectures use standard execution.",
                  "Do not compare model token rates as quality rankings; standard/automatic use identical thread/context settings.",
                  "Strict CPU-reference logit gates and an earlier quantitative answer screen remain failed.",
              ]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    failed = False
    try:
        for path, metadata, supported in models:
            control_hash = None
            for mode in ("standard", "auto"):
                policy = app.select_hybrid_policy(
                    metadata, path.stat().st_size, mode=mode,
                    bound_layout=initial.get("hybrid_model_layout"),
                    generic=initial.get("hybrid_generic", False),
                    automatic_plan=initial.get("hybrid_initialization") == "auto",
                    cache_mib=initial.get("hybrid_cache_mib", 10240))
                expect_hybrid = mode == "auto" and policy["hybrid"] and initial.get("hybrid_available", False)
                settings = {"model": str(path), "execution_mode": mode, "context": args.context,
                            "cpu_threads": args.threads, "batch_threads": args.batch_threads,
                            "placement": "auto", "load_mode": "auto", "fit_target_mib": 1024,
                            "expert_cache_mib": initial.get("hybrid_cache_mib", 10240) if expect_hybrid else 0,
                            "host_op_offload": not expect_hybrid, "cpu_poll": 50,
                            "op_offload_min_batch": 32}
                run = {"model": path.name, "first_shard_size_bytes": path.stat().st_size,
                       "metadata": metadata, "hybrid_layout_supported": supported, "requested_mode": mode,
                       "status": "running", "samples": [], "answer_checks": []}
                report["runs"].append(run)
                sweep.save_report(args.output, report)
                try:
                    started = time.monotonic()
                    state = sweep.load_profile(args.base_url, settings)
                    run.update(load_seconds=time.monotonic() - started,
                               execution_profile=state["execution_profile"],
                               hybrid_verified=state["hybrid_verified"],
                               hybrid_decode_pipeline=state.get("hybrid_decode_pipeline", False),
                               hybrid_async_ids=state.get("hybrid_async_ids", False),
                               selection_reason=state["selection_reason"],
                               backend=Path(state["backend"]).name,
                               settings={key: value for key, value in settings.items() if key != "model"})
                    if state["execution_profile"].startswith("experimental-hybrid-") != expect_hybrid:
                        raise RuntimeError("Selected execution profile differs from metadata-based expectation.")
                    if expect_hybrid and not state["hybrid_verified"]:
                        raise RuntimeError("Hybrid execution not verified.")
                    for repetition in range(args.repetitions):
                        result = sweep.sample(args.base_url, args.tokens)
                        run["samples"].append(result)
                        sweep.save_report(args.output, report)
                        print(json.dumps({"model": path.name, "mode": mode, "repetition": repetition + 1,
                                          "wall_decode_tok_s": result["wall_decode_tok_s"]}), flush=True)
                    if mode == "standard":
                        control_hash = run["samples"][0]["generated_text_sha256"]
                    run.update(sweep.summarize(run["samples"], control_hash))
                    for name, prompt in (
                            ("arithmetic", "Compute 17 times 19. Return only the integer."),
                            ("coding", "Return only Python source code, without Markdown fences, implementing "
                             "clamp(value, low, high). It must return min(high, max(low, value)). "
                             "No imports, tests, comments, annotations, or explanation.")):
                        run["answer_checks"].append(quality_sample(args.base_url, name, prompt))
                    run["status"] = "ok" if all(item["check_passed"] for item in run["answer_checks"]) else "answer_check_failed"
                    failed |= run["status"] != "ok"
                except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
                    failed = True
                    run["status"] = "failed"
                    run["error"] = str(exc).replace(str(path), "/models/" + path.name)
                    print(f"{path.name}/{mode}: {run['error']}", file=sys.stderr, flush=True)
                sweep.save_report(args.output, report)
    finally:
        try:
            sweep.load_profile(args.base_url, original)
            report["restore"] = {"status": "ready", "original_settings_restored": True}
        except (OSError, ValueError, RuntimeError) as exc:
            failed = True
            report["restore"] = {"status": "failed", "error": str(exc).replace(original["model"], "/models/" + Path(original["model"]).name)}
        report["completed_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        sweep.save_report(args.output, report)
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--allow-reloads", action="store_true")
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--context", type=int, default=16384)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch-threads", type=int, default=16)
    args = parser.parse_args()
    url = urlsplit(args.base_url)
    if (not args.allow_reloads or url.scheme != "http" or url.hostname not in ("127.0.0.1", "localhost") or
            url.username or url.password or url.path not in ("", "/") or url.query or url.fragment or
            args.tokens < 512 or args.tokens > 8192 or not 1 <= args.repetitions <= 10 or
            not 512 <= args.context <= 65536 or args.threads < 1 or args.batch_threads < 1):
        parser.error("Require --allow-reloads, a loopback HTTP endpoint, >=512 tokens, and valid run sizes.")
    if args.output.exists() or args.output.with_suffix(args.output.suffix + ".tmp").exists():
        parser.error("Refusing to overwrite a result or partial result.")
    args.base_url = args.base_url.rstrip("/")
    try:
        return execute(args)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"model matrix: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
