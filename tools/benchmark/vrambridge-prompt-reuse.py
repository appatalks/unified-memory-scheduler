#!/usr/bin/env python3
"""Measure exact prompt replay on isolated native Hybrid servers."""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import secrets
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("reuse_environment", Path(__file__).with_name("vrambridge-engine-comparison.py"))
assert spec and spec.loader
environment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(environment)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def token_hash(tokens):
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def summarize_response(event, elapsed, requested):
    timings = event.get("timings")
    tokens = event.get("tokens")
    if not isinstance(timings, dict) or not isinstance(tokens, list) or len(tokens) != requested:
        raise ValueError("Missing complete output tokens or native timings.")
    if any(type(token) is not int or token < 0 for token in tokens):
        raise ValueError("Native output IDs must be nonnegative integers.")
    if timings.get("predicted_n") != requested:
        raise ValueError("Native output count differs from requested length.")
    for key in ("cache_n", "prompt_n", "prompt_ms", "predicted_ms", "predicted_per_second"):
        value = timings.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Missing/invalid native timing: {key}")
    if elapsed <= 0 or (requested > 1 and (timings["predicted_ms"] <= 0 or timings["predicted_per_second"] <= 0)):
        raise ValueError("Completed request has no positive generation interval.")
    return {"timings": timings, "wall_seconds": elapsed,
            "performance_measurement": requested > 1,
            "output_token_sha256": token_hash(tokens), "output_tokens": requested,
            "prompt_cache_tokens": timings["cache_n"], "processed_tokens": timings["prompt_n"]}


def replay_cases(tokens, suffix, replacement):
    if len(tokens) < 4096 or not suffix or type(replacement) is not int or replacement < 0:
        raise ValueError("Replay requires a long exact token prefix and valid suffix/replacement.")
    middle = len(tokens) // 2 + 257
    if tokens[middle] == replacement:
        raise ValueError("The edit must change the prompt token.")
    edited = list(tokens)
    edited[middle] = replacement
    return [("cold", tokens), ("repeat", tokens), ("append", tokens + suffix),
            ("middle-edit", edited), ("return-original", tokens)]


def compare_probabilities(first, second, vocabulary):
    def vector(event):
        positions = event.get("completion_probabilities")
        if not isinstance(positions, list) or len(positions) != 1:
            raise ValueError("Expected one full-vocabulary probability position.")
        values = positions[0].get("top_logprobs")
        if not isinstance(values, list) or len(values) != vocabulary:
            raise ValueError("Full vocabulary was not returned; top-k is not a numerical gate.")
        result = {}
        for item in values:
            token, value = item.get("id"), item.get("logprob")
            if type(token) is not int or not 0 <= token < vocabulary or token in result:
                raise ValueError("Invalid/duplicate vocabulary ID.")
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
                raise ValueError("Invalid log probability.")
            result[token] = value
        return result
    before, after = vector(first), vector(second)
    if any((before[token] is None) != (after[token] is None) for token in before):
        raise ValueError("Vocabulary probability masks differ.")
    pairs = [(value, after[token]) for token, value in before.items() if value is not None]
    if not pairs:
        raise ValueError("No finite vocabulary probabilities.")
    maximum = max(abs(a - b) for a, b in pairs)
    reference = sum(a * a for a, _ in pairs)
    relative = math.sqrt(sum((a - b) ** 2 for a, b in pairs) / max(reference, 1e-30))
    return {"vocabulary": vocabulary, "finite_entries": len(pairs),
            "max_absolute_logprob_error": maximum, "relative_l2_logprob_error": relative,
            "passes": maximum <= 0.01 and relative <= 0.001,
            "not_a_raw_logit_or_broad_quality_gate": True}


class OwnedServer:
    def __init__(self, args, log, variant):
        self.args = args
        self.variant = variant
        self.key = secrets.token_hex(32)
        self.process = None
        self.thread = None
        self.lines = deque(maxlen=400)
        self.file = log.open("x")
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        profile = json.loads(args.profile.read_text())
        env = environment.clean_environment()
        env.update(GGML_BACKEND_PATH=profile["cuda_plugin"], GGML_CUDA_REGISTER_HOST="1",
                   VRAMBRIDGE_HYBRID_PLAN="auto", VRAMBRIDGE_HYBRID_ADAPT="1",
                   VRAMBRIDGE_HYBRID_ASYNC="1", VRAMBRIDGE_HYBRID_DIRECT="1",
                   VRAMBRIDGE_HYBRID_STAGE_INPUT="1", VRAMBRIDGE_HYBRID_OVERLAP="1",
                   VRAMBRIDGE_HYBRID_BOUNDARY="gpu", VRAMBRIDGE_HYBRID_FUSE_GATE_UP="1",
                   VRAMBRIDGE_HYBRID_ASYNC_IDS="1")
        if args.freeze_adaptation:
            env.update(VRAMBRIDGE_HYBRID_ADAPT="0", VRAMBRIDGE_HYBRID_ASYNC="0",
                       VRAMBRIDGE_HYBRID_DIRECT="0")
            env.pop("GGML_CUDA_REGISTER_HOST", None)
        grid = variant == "grid"
        binary = args.baseline if variant == "baseline" else args.candidate
        if grid:
            env["VRAMBRIDGE_PROMPT_CHECKPOINT_GRID"] = "1"
        spacing = args.spacing if variant in ("dense-boundary", "grid") else 8192
        command = [str(binary), "--model", str(args.model), "--host", "127.0.0.1",
                   "--port", str(port), "--api-key", self.key, "--alias", "research",
                   "--ctx-size", str(args.context), "--parallel", "1",
                   "--threads", "8", "--threads-batch", "16", "--n-gpu-layers", "all",
                   "--cpu-moe", "--fit", "off", "--no-repack", "--no-host", "--no-op-offload",
                   "--moe-cache-mib", str(args.cache_mib), "--cache-type-k", "f16", "--cache-type-v", "f16",
                   "--ctx-checkpoints", "32", "--checkpoint-min-step", str(spacing),
                   "--cache-ram", str(args.ram_cache_mib), "--log-verbosity", str(args.verbosity)]
        self.info = {"binary_sha256": digest(binary), "libllama_sha256": digest(binary.parent / "libllama.so"),
                     "server_impl_sha256": digest(binary.parent / "libllama-server-impl.so"),
                     "context": args.context, "expert_bank_mib": args.cache_mib,
                     "checkpoint_spacing": spacing, "checkpoint_count": 32, "cache_ram_mib": args.ram_cache_mib,
                     "frozen_empty_bank": args.freeze_adaptation}
        try:
            self.process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT, text=True)
            self.thread = threading.Thread(target=self._read, daemon=True)
            self.thread.start()
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(f"Owned {variant} server exited before readiness.")
                try:
                    self.request("/v1/models")
                except (OSError, ValueError, urllib.error.URLError):
                    threading.Event().wait(0.2)
                    continue
                text = log.read_text()
                required = ["EXPERIMENTAL HYBRID enabled", "VRAMBridge decode pipeline configured",
                            "asynchronous GPU ID handoff", f"n_ctx_slot = {args.context}"]
                if not args.freeze_adaptation:
                    required.append("direct asynchronous admissions; immutable registered sources")
                self.info["ready_log_markers"] = [marker for marker in required if marker in text]
                coverage = re.search(r"tensor-derived hybrid coverage=(\d+)/(\d+) routed layers", text)
                if (len(self.info["ready_log_markers"]) != len(required) or coverage is None
                        or int(coverage[1]) <= 0 or coverage[1] != coverage[2]):
                    raise RuntimeError("Owned server did not confirm required Hybrid controls and full coverage.")
                if grid and "VRAMBridge prompt checkpoint grid active" not in text:
                    raise RuntimeError("Candidate did not acknowledge the checkpoint grid.")
                if not grid and "VRAMBridge prompt checkpoint grid active" in text:
                    raise RuntimeError("Control unexpectedly enabled the checkpoint grid.")
                return
            raise RuntimeError("Owned server readiness deadline exceeded.")
        except (OSError, ValueError, RuntimeError):
            self.close()
            raise

    def _read(self):
        for line in self.process.stdout:
            self.file.write(line)
            self.file.flush()
            self.lines.append(line.rstrip())

    def request(self, path, body=None, timeout=600):
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(self.url + path, data, {
            "Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read(64 * 1024 * 1024 + 1)
            if len(raw) > 64 * 1024 * 1024:
                raise ValueError("Native diagnostic response exceeds64MiB.")
            event = json.loads(raw)
        if not isinstance(event, dict) or event.get("error"):
            raise ValueError("Native endpoint rejected the diagnostic request.")
        return event

    def complete(self, prompt, outputs, cache=True, n_probs=0):
        started = time.monotonic()
        event = self.request("/completion", {"prompt": prompt, "n_predict": outputs,
            "temperature": 0, "seed": 42, "cache_prompt": cache, "ignore_eos": True,
            "return_tokens": True, "n_probs": n_probs, "post_sampling_probs": False})
        summary = summarize_response(event, time.monotonic() - started, outputs)
        return summary, event

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
            if self.thread is not None:
                self.thread.join(timeout=10)
            if self.process.stdout is not None:
                self.process.stdout.close()
        self.file.close()


def run(args):
    info = args.output_dir.stat()
    if info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError("The output directory must be owned and private (0700).")
    report_path = args.output_dir / "report.json"
    if report_path.exists():
        raise ValueError("Refusing to overwrite an existing report.")
    report = {"schema_version": 1, "study": "prompt_reuse_checkpoint_replay",
              "status": "running", "context": args.context, "output_tokens": args.tokens,
              "source_sha256": digest(Path(__file__)), "runs": [],
              "limitations": [
                  "Synthetic raw completion requests, not a reconstruction of private CLI prompts.",
                  "Checkpoint changes are research-only and not full-vocabulary numerical qualification.",
                  "Output hash matches are bounded; adaptive residency can change arithmetic placement.",
                  "Allocated96K context is not occupied prompt length; input length is reported separately.",
                  "Verbose private instrumentation may perturb timing; confirmation requires reduced-log controls.",
              ]}
    try:
        for repetition in range(args.repetitions):
            variants = args.variants if repetition % 2 == 0 else list(reversed(args.variants))
            for variant in variants:
                entry = {"variant": variant, "repetition": repetition, "status": "running", "requests": []}
                report["runs"].append(entry)
                print(f"START {variant} repetition={repetition}", flush=True)
                server = OwnedServer(args, args.output_dir / f"{variant}-{repetition}.private.log", variant)
                try:
                    entry["server"] = server.info
                    body = "".join(f"Row {index:06d}: amber birch silver; item {index * 7919 % 1000003:07d}.\n" for index in range(args.prompt_tokens // 10 + 1024))
                    tokens = server.request("/tokenize", {"content": body, "add_special": True})["tokens"]
                    suffix = server.request("/tokenize", {"content": "\nContinue explaining the engineering tradeoffs in detail.", "add_special": False})["tokens"]
                    replacement = server.request("/tokenize", {"content": "quartz", "add_special": False})["tokens"][0]
                    prompt = tokens[:args.prompt_tokens]
                    if len(prompt) != args.prompt_tokens:
                        raise RuntimeError("Tokenizer produced insufficient synthetic input.")
                    server.complete(prompt[:128], 64, cache=False)
                    entry["memory_before_replay"] = environment.io.snapshot(server.process.pid)
                    for name, ids in ([] if args.numerical_only else replay_cases(prompt, suffix, replacement)):
                        summary, event = server.complete(ids, args.tokens, cache=name != "cold")
                        summary.update(case=name, input_tokens=len(ids), prompt_sha256=token_hash(ids))
                        entry["requests"].append(summary)
                        print(f"{variant} {name}: cached={summary['prompt_cache_tokens']} processed={summary['processed_tokens']} prompt_ms={summary['timings']['prompt_ms']:.2f} decode={summary['timings']['predicted_per_second']:.2f}", flush=True)
                    # Preserve a cold reference for the exact edited prompt in the same execution contract.
                    edited = replay_cases(prompt, suffix, replacement)[3][1]
                    if not args.numerical_only:
                        cold, _ = server.complete(edited, args.tokens, cache=False)
                        cached = next(r for r in entry["requests"] if r["case"] == "middle-edit")
                        entry["edited_vs_uncached_output_matches"] = cached["output_token_sha256"] == cold["output_token_sha256"]
                        entry["uncached_edited_control"] = cold
                    entry["memory_after_replay"] = environment.io.snapshot(server.process.pid)
                    if args.probability_check and repetition == 0:
                        vocabulary = server.request("/v1/models")["data"][0]["meta"]["n_vocab"]
                        if type(vocabulary) is not int or not 1 <= vocabulary <= 262144:
                            raise ValueError("Invalid backend vocabulary size for bounded capture.")
                        for label, target in (("middle-edit", edited), ("append", prompt + suffix)):
                            server.complete(prompt, 1, cache=False)
                            cached_summary, cached_event = server.complete(target, 1, n_probs=vocabulary)
                            cold_summary, cold_event = server.complete(target, 1, cache=False, n_probs=vocabulary)
                            entry.setdefault("probability_checks", []).append({
                                "case": label, "cached_tokens": cached_summary["prompt_cache_tokens"],
                                "cached_processed_tokens": cached_summary["processed_tokens"],
                                "cold_processed_tokens": cold_summary["processed_tokens"],
                                **compare_probabilities(cold_event, cached_event, vocabulary)})
                    entry["status"] = "complete"
                finally:
                    server.close()
                    entry["owned_backend_stopped"] = server.process.poll() is not None
                    numeric = [line for line in server.lines if any(marker in line for marker in (
                        "restored context checkpoint", "forcing full prompt re-processing",
                        "created context checkpoint", "erasing old context checkpoint"))]
                    entry["checkpoint_diagnostics"] = numeric[-40:]
                report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        report["status"] = "complete"
    except (OSError, ValueError, RuntimeError) as exc:
        if report["runs"] and report["runs"][-1]["status"] == "running":
            report["runs"][-1]["status"] = "failed"
        report.update(status="failed", error=str(exc).replace(str(args.model), "/models/" + args.model.name))
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--context", type=int, default=98304)
    parser.add_argument("--cache-mib", type=int, default=6144)
    parser.add_argument("--ram-cache-mib", type=int, default=8192)
    parser.add_argument("--spacing", type=int, default=1024)
    parser.add_argument("--prompt-tokens", type=int, default=8192)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--verbosity", type=int, choices=(3, 5), default=3)
    parser.add_argument("--probability-check", action="store_true")
    parser.add_argument("--freeze-adaptation", action="store_true",
                        help="Numerical attribution only: empty fixed expert bank, no adaptive transfers.")
    parser.add_argument("--numerical-only", action="store_true")
    parser.add_argument("--variants", nargs="+", choices=("baseline", "off", "dense-boundary", "grid"),
                        default=["baseline", "off", "dense-boundary", "grid"])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not (4096 <= args.prompt_tokens < args.context <= 262144 and args.context % 256 == 0
            and 16 <= args.tokens <= 1024 and 1 <= args.repetitions <= 3
            and 1 <= args.cache_mib <= 65536 and 1 <= args.ram_cache_mib <= 32768
            and 1 <= args.spacing <= 32768):
        parser.error("Invalid bounded study parameters.")
    if len(set(args.variants)) != len(args.variants):
        parser.error("Duplicate variants are not independent measurements.")
    if args.numerical_only and not args.probability_check:
        parser.error("--numerical-only requires --probability-check.")
    os.umask(0o077)
    run(args)


if __name__ == "__main__":
    main()
