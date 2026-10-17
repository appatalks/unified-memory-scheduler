#!/usr/bin/env python3
"""Screen official-backend execution profiles through the local app, restoring its original model settings."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import time
from urllib.parse import urlsplit
import urllib.request


PROFILES = {
    "control": {},
    "host-cpu": {"host_op_offload": False},
    "poll0": {"cpu_poll": 0},
    "host-cpu-poll0": {"host_op_offload": False, "cpu_poll": 0},
    "margin512": {"fit_target_mib": 512},
    "margin512-host-cpu": {"fit_target_mib": 512, "host_op_offload": False},
    "batch128": {"op_offload_min_batch": 128},
}
SETTING_KEYS = (
    "model", "context", "placement", "cpu_threads", "batch_threads", "load_mode",
    "expert_cache_mib", "fit_target_mib", "host_op_offload", "cpu_poll", "op_offload_min_batch",
)
PROMPT = ("Explain how a CPU and GPU can cooperate to execute a mixture-of-experts model. "
          "Discuss memory bandwidth, expert placement, synchronization, and correctness in detail.")


def request_json(base: str, path: str, data: dict | None = None) -> dict:
    body = json.dumps(data).encode() if data is not None else None
    request = urllib.request.Request(base + path, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.load(response)


def sanitize(message: str, original: dict) -> str:
    return message.replace(original["model"], "/models/" + Path(original["model"]).name)


def load_profile(base: str, settings: dict) -> dict:
    status = request_json(base, "/api/v1/status")
    if status["chat_busy"]:
        raise RuntimeError("A chat is running; refusing to interrupt it.")
    request_json(base, "/api/v1/model/unload", {})
    request_json(base, "/api/v1/model/load", settings)
    deadline = time.monotonic() + 620
    while True:
        status = request_json(base, "/api/v1/status")
        if status["state"] == "error":
            raise RuntimeError(status["error"] + "\n" + "\n".join(status["logs"]))
        if status["state"] == "ready":
            for key, value in settings.items():
                if status.get(key) != value:
                    raise RuntimeError(f"Setting not applied: {key}")
            contexts = [int(value) for line in status["logs"]
                        for value in re.findall(r"n_ctx_slot\s*=\s*'?(\d+)", line)]
            if not contexts or contexts[-1] != settings["context"]:
                raise RuntimeError("Backend output did not confirm the required actual slot context.")
            return status
        if time.monotonic() >= deadline:
            raise RuntimeError("Model did not become ready before the load deadline.")
        time.sleep(0.25)


def validate_sample(timings: dict, usage: dict, tokens: int, elapsed: float, done: bool) -> dict:
    if not done or usage.get("completion_tokens") != tokens or timings.get("predicted_n") != tokens:
        raise ValueError("Incomplete generation or output token counts do not match the requested length.")
    for name in ("prompt_ms", "predicted_ms", "predicted_per_second"):
        value = timings.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Missing or invalid backend timing: {name}")
    if timings["predicted_ms"] == 0 or timings["predicted_per_second"] == 0:
        raise ValueError("Completed generation must have positive decode duration and rate.")
    decode = elapsed - timings["prompt_ms"] / 1000
    if not math.isfinite(elapsed) or decode <= 0 or elapsed <= 0:
        raise ValueError("Invalid wall-clock timing interval.")
    return {
        "timings": timings, "usage": usage, "elapsed_seconds": elapsed,
        "wall_decode_tok_s": (tokens - 1) / decode,
        "end_to_end_tok_s": tokens / elapsed,
    }


def sample(base: str, tokens: int, headers: dict[str, str] | None = None, *,
           prompt: str = PROMPT, ignore_eos: bool = True, reasoning_effort: str | None = "low") -> dict:
    if reasoning_effort not in (None, "low", "medium", "high"):
        raise ValueError("Reasoning effort must be low, medium, high, or None for the model default.")
    payload = {
        "model": "local-model", "messages": [{"role": "user", "content": prompt}],
        "max_tokens": tokens, "ignore_eos": ignore_eos, "temperature": 0, "seed": 42,
        "cache_prompt": False, "stream": True,
        "stream_options": {"include_usage": True}, "timings_per_token": True,
    }
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort
    request = urllib.request.Request(base + "/v1/chat/completions", json.dumps(payload).encode(),
                                     {"Content-Type": "application/json", **(headers or {})})
    started = time.monotonic()
    first = None
    done = False
    timings, usage = {}, {}
    digest = hashlib.sha256()
    with urllib.request.urlopen(request, timeout=120) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            value = line[6:].strip()
            if value == b"[DONE]":
                done = True
                break
            event = json.loads(value)
            if "error" in event:
                raise RuntimeError(event["error"])
            timings = event.get("timings") or timings
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                text = (delta.get("reasoning_content") or "") + (delta.get("content") or "")
                if text and first is None:
                    first = time.monotonic() - started
                digest.update(text.encode())
    result = validate_sample(timings, usage, tokens, time.monotonic() - started, done)
    result.update(first_token_seconds=first, generated_text_sha256=digest.hexdigest())
    return result


def summarize(samples: list[dict], control_hash: str | None) -> dict:
    hashes = {item["generated_text_sha256"] for item in samples}
    return {
        "median_wall_decode_tok_s": statistics.median(item["wall_decode_tok_s"] for item in samples),
        "median_end_to_end_tok_s": statistics.median(item["end_to_end_tok_s"] for item in samples),
        "repeated_text_hashes_match": len(hashes) == 1,
        "matches_control_text": len(hashes) == 1 and next(iter(hashes)) == control_hash,
        "not_a_quality_or_logit_equivalence_test": True,
    }


def save_report(destination: Path, report: dict) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, destination)


def execute(base: str, profiles: list[str], tokens: int, repetitions: int, output: Path) -> int:
    initial = request_json(base, "/api/v1/status")
    if initial["state"] != "ready" or initial["chat_busy"] or initial["backend_kind"] != "llama.cpp":
        raise ValueError("The official backend must have an idle, ready model before a sweep.")
    if initial.get("draft_model") or initial["expert_cache_mib"] or initial["placement"] != "auto":
        raise ValueError("Sweep requires automatic placement without speculation or an expert cache.")
    original = {key: initial[key] for key in SETTING_KEYS}
    original["execution_mode"] = initial.get("execution_mode", "standard")
    report = {
        "schema_version": 1, "model": Path(original["model"]).name,
        "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "context": original["context"], "output_tokens": tokens, "repetitions": repetitions,
        "initial_settings": {key: value for key, value in original.items() if key != "model"},
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "runner_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runs": [], "restore": {"status": "pending"}, "performance_target_met": False,
        "limitations": [
            "Fixed-length screen, not a sustained natural-answer or quality qualification.",
            "Auto-fit may change expert placement when available VRAM changes.",
            "CPU polling may include busy waiting; high CPU use is not productive-work evidence.",
            "Text hashes do not establish per-token logit tolerance.",
        ],
    }
    control_hash = None
    failed = False
    try:
        for name in profiles:
            configuration = {**original, "execution_mode": "standard", **PROFILES[name]}
            run = {"profile": name, "overrides": PROFILES[name], "samples": [], "status": "running"}
            report["runs"].append(run)
            save_report(output, report)
            try:
                started = time.monotonic()
                status = load_profile(base, configuration)
                run["load_seconds"] = time.monotonic() - started
                run["actual_context_confirmed"] = configuration["context"]
                run["resource_snapshot"] = status.get("resources")
                for repetition in range(repetitions):
                    result = sample(base, tokens)
                    run["samples"].append(result)
                    save_report(output, report)
                    print(json.dumps({"profile": name, "repetition": repetition + 1,
                                      "wall_decode_tok_s": result["wall_decode_tok_s"],
                                      "end_to_end_tok_s": result["end_to_end_tok_s"]}), flush=True)
                if name == "control":
                    control_hash = run["samples"][0]["generated_text_sha256"]
                run.update(summarize(run["samples"], control_hash))
                run["status"] = "ok"
            except (OSError, ValueError, RuntimeError) as exc:
                failed = True
                run["status"] = "failed"
                run["error"] = sanitize(str(exc), original)
                print(f"{name}: {run['error']}", file=sys.stderr, flush=True)
            save_report(output, report)
    finally:
        try:
            load_profile(base, original)
            report["restore"] = {"status": "ready", "original_settings_restored": True}
        except (OSError, ValueError, RuntimeError) as exc:
            failed = True
            report["restore"] = {"status": "failed", "error": sanitize(str(exc), original)}
            print("RESTORE FAILED: " + report["restore"]["error"], file=sys.stderr)
        report["completed_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        save_report(output, report)
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--profiles", nargs="+", choices=tuple(PROFILES), default=list(PROFILES))
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-reloads", action="store_true", help="acknowledge that model reloads interrupt app availability")
    args = parser.parse_args()
    parsed = urlsplit(args.base_url)
    if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost") or parsed.path not in ("", "/") \
            or parsed.username or parsed.password or parsed.query or parsed.fragment:
        parser.error("--base-url must be a plain loopback HTTP app URL")
    if not args.allow_reloads:
        parser.error("--allow-reloads is required; do not run while another person uses the app")
    if not 2 <= args.tokens <= 8192 or not 1 <= args.repetitions <= 10:
        parser.error("tokens must be 2-8192 and repetitions 1-10")
    if args.profiles[0] != "control" or len(set(args.profiles)) != len(args.profiles):
        parser.error("profiles must be unique and start with control")
    if args.output.exists() or args.output.with_suffix(args.output.suffix + ".tmp").exists():
        parser.error("Refusing to overwrite an existing sweep or partial report")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        return execute(args.base_url.rstrip("/"), args.profiles, args.tokens, args.repetitions, args.output)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"vrambridge-execution-sweep: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
