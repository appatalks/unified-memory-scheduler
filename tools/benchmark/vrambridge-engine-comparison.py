#!/usr/bin/env python3
"""Compare owned engines on a private, tokenizer-verified common token corpus."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
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


SPEC = importlib.util.spec_from_file_location("comparison_io", Path(__file__).with_name("vrambridge-io-study.py"))
assert SPEC and SPEC.loader
io = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(io)


def require(value, message):
    if not value:
        raise RuntimeError(message)


def clean_environment():
    env = {k: v for k, v in os.environ.items() if k != "LD_PRELOAD"
           and not k.startswith(("STRATA_", "VRAMBRIDGE_", "LLAMA_ARG_", "GGML_", "OMP_", "GOMP_"))}
    env.update(TZ="UTC", PYTHONDONTWRITEBYTECODE="1")
    return env


def token_digest(tokens):
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def build_identity(library, source, header):
    trace = Path(__file__).resolve().parents[1] / "trace"
    return {"source_sha256": hashlib.sha256((trace / source).read_bytes()).hexdigest(),
            "header_sha256": hashlib.sha256((trace / header).read_bytes()).hexdigest(),
            "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest()}


class NativeEngine:
    def __init__(self, args, log):
        self.key = secrets.token_hex(32)
        self.ignore_eos = not args.natural_eos
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        self.process = None
        self.file = log.open("x")
        self.window_metrics_path = None
        self.window_identity = None
        self.packed_metrics_path = None
        self.packed_identity = None
        self.repack_ram_guard = None
        self.kv_metrics_path = None
        self.kv_identity = None
        env = clean_environment()
        if args.managed_memory:
            env["GGML_CUDA_ENABLE_UNIFIED_MEMORY"] = "1"
        env.update(GGML_BACKEND_PATH=str(args.cuda_plugin.resolve(strict=True)),
            VRAMBRIDGE_HYBRID_PLAN=str(args.native_plan.resolve(strict=True)),
            VRAMBRIDGE_HYBRID_ADAPT="0", VRAMBRIDGE_HYBRID_ASYNC="0", VRAMBRIDGE_HYBRID_DIRECT="0",
            VRAMBRIDGE_HYBRID_STAGE_INPUT="1", VRAMBRIDGE_HYBRID_OVERLAP="1",
            VRAMBRIDGE_HYBRID_BOUNDARY="gpu", VRAMBRIDGE_HYBRID_FUSE_GATE_UP="1", VRAMBRIDGE_HYBRID_ASYNC_IDS="1")
        if args.cpu_repack:
            spec = importlib.util.spec_from_file_location("comparison_staging", Path(__file__).with_name("vrambridge-model-staging.py"))
            require(spec is not None and spec.loader is not None, "Model-shard discovery unavailable.")
            staging = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(staging)
            source_bytes = sum(p.stat().st_size for p in staging.discover(args.model))
            available = io.snapshot(os.getpid())["system_memory"]["MemAvailable"]
            require(available >= source_bytes + 16 * 1024**3,
                    "CPU repacking requires a conservative source-size RAM allowance plus16GiB host headroom.")
            self.repack_ram_guard = {"source_size_allowance_bytes": source_bytes,
                "available_before_load_bytes": available, "host_headroom_bytes": 16 * 1024**3}
            env["VRAMBRIDGE_HYBRID_REPACK"] = "1"
        if args.mode in ("canonical", "mtp") and args.attention_window is None:
            env.update(VRAMBRIDGE_HYBRID_SCALAR_LINEAR="1", VRAMBRIDGE_HYBRID_SCALAR_EXPERT="1",
                       VRAMBRIDGE_HYBRID_SCALAR_ATTN="1", VRAMBRIDGE_HYBRID_BATCH=str(args.window))
        if args.attention_window is not None:
            library = args.attention_window.resolve(strict=True)
            self.window_metrics_path = log.with_suffix(".window.json")
            require(not self.window_metrics_path.exists(), "Refusing to overwrite window diagnostics.")
            self.window_identity = build_identity(library, "attention_window.cpp", "attention_window.h")
            env.update(LD_PRELOAD=str(library), VRAMBRIDGE_ATTN_WINDOW="split",
                VRAMBRIDGE_ATTN_WINDOW_PRODUCTS="columns", VRAMBRIDGE_ATTN_WINDOW_OUT=str(self.window_metrics_path),
                VRAMBRIDGE_ATTN_WINDOW_LIBRARY=str(args.native_binary.resolve(strict=True).parent / "libllama.so"),
                VRAMBRIDGE_HYBRID_BATCH=str(args.window))
            env["VRAMBRIDGE_ATTN_WINDOW_EXPERT_PRODUCTS"] = args.expert_products or "columns"
            if args.verification_threads is not None:
                env["VRAMBRIDGE_ATTN_WINDOW_THREADS"] = args.verification_threads
        if args.exact_packed_cpu is not None:
            library = args.exact_packed_cpu.resolve(strict=True)
            native_cpu = args.cpu_native_library.resolve(strict=True)
            self.packed_metrics_path = log.with_suffix(".packed.json")
            require(not self.packed_metrics_path.exists(), "Refusing to overwrite packed diagnostics.")
            self.packed_identity = build_identity(library, "exact_packed_cpu.cpp", "exact_packed_cpu.h")
            self.packed_identity["native_cpu_sha256"] = hashlib.sha256(native_cpu.read_bytes()).hexdigest()
            env.update(LD_PRELOAD=":".join(filter(None, (env.get("LD_PRELOAD"), str(library)))),
                VRAMBRIDGE_CPU_PACKED_MODE="exact", VRAMBRIDGE_CPU_PACKED_LIBRARY=str(native_cpu),
                VRAMBRIDGE_CPU_PACKED_OUT=str(self.packed_metrics_path))
        if args.kv_cold_tier is not None:
            library = args.kv_cold_tier.resolve(strict=True)
            self.kv_metrics_path = log.with_suffix(".kv-cold.json")
            require(not self.kv_metrics_path.exists(), "Refusing to overwrite KV diagnostics.")
            self.kv_identity = build_identity(library, "kv_cold_tier.cpp", "kv_cold_tier.h")
            env.update(LD_PRELOAD=":".join(filter(None, (env.get("LD_PRELOAD"), str(library)))),
                VRAMBRIDGE_KV_COLD_MODE="stage", VRAMBRIDGE_KV_COLD_OUT=str(self.kv_metrics_path),
                VRAMBRIDGE_KV_COLD_BASE_LIBRARY=str(args.native_binary.resolve(strict=True).parent / "libggml-base.so"))
        command = [str(args.native_binary.resolve(strict=True)), "--model", str(args.model.resolve(strict=True)),
            "--host", "127.0.0.1", "--port", str(port), "--api-key", self.key, "--alias", "research",
            "--ctx-size", "16384", "--parallel", "1", "--threads", "8", "--threads-batch", "16",
            "--batch-size", "2048", "--ubatch-size", "512", "--n-gpu-layers", "all", "--cpu-moe",
            "--moe-cache-mib", str(args.cache_mib), "--fit", "off", "--no-repack", "--no-host", "--no-op-offload",
            "--cache-type-k", "f16", "--cache-type-v", "f16", "--verbosity", "4"]
        if args.mode == "mtp":
            require(args.native_draft is not None, "Native MTP requires a verified derived draft.")
            env["VRAMBRIDGE_HYBRID_MTP"] = "1"
            command += ["--model-draft", str(args.native_draft.resolve(strict=True)),
                        "--spec-type", "draft-mtp", "--spec-draft-n-max", str(args.window - 1),
                        "--spec-draft-p-min", "0", "--spec-draft-ngl", "all"]
        if args.cpu_repack:
            command.remove("--no-repack")
        if args.mode == "dflash":
            require(args.native_draft is not None and args.attention_window is not None,
                    "DFlash requires a verified existing draft and the optimized-preserving window prototype.")
            env["VRAMBRIDGE_HYBRID_DFLASH"] = "1"
            command += ["--model-draft", str(args.native_draft.resolve(strict=True)), "--spec-type", "draft-dflash",
                        "--spec-draft-n-max", str(args.window - 1), "--spec-draft-p-min", "0",
                        "--spec-draft-ngl", "all", "--spec-draft-n-ubatch", "64"]
        if args.mode == "lookup":
            command += ["--spec-type", "ngram-simple", "--spec-ngram-simple-size-n", str(args.lookup_n),
                        "--spec-ngram-simple-size-m", str(args.window - 1), "--spec-ngram-simple-min-hits", "1"]
        if args.mode in ("mtp", "dflash") and args.draft_threads is not None:
            command += ["--spec-draft-threads", str(args.draft_threads),
                        "--spec-draft-threads-batch", str(args.draft_threads)]
        start = time.monotonic()
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=self.file, stderr=self.file, env=env)
            deadline = start + 600
            heartbeat = threading.Event()
            while True:
                require(self.process.poll() is None, "Owned native engine exited before READY; see its private log.")
                try:
                    request = urllib.request.Request(self.url + "/health", headers={"Authorization": "Bearer " + self.key})
                    with urllib.request.urlopen(request, timeout=2) as response:
                        if response.status == 200:
                            break
                except (urllib.error.URLError, TimeoutError):
                    pass
                require(time.monotonic() < deadline, "Owned native engine load timeout.")
                heartbeat.wait(.1)
            self.load_seconds = time.monotonic() - start
            startup = log.read_text()
            residency = re.search(r"EXPERIMENTAL HYBRID enabled: (\d+) selected experts; frozen, no decode weight migration", startup)
            require(residency is not None and int(residency[1]) == args.experts,
                    "Native did not acknowledge the shared frozen expert count.")
            repacked = re.search(r"CPU-repacked expert tensors=(\d+)", startup)
            if args.cpu_repack:
                require(repacked is not None and int(repacked[1]) > 0, "CPU expert repacking was not acknowledged.")
            if args.mode == "lookup":
                require("adding speculative implementation 'ngram-simple'" in startup, "Lookup implementation not acknowledged.")
            if args.mode == "dflash":
                require("draft-dflash" in startup, "DFlash implementation not acknowledged.")
            self.info = {"context": 16384, "cache_payload_budget_mib": args.cache_mib, "frozen_plan": True,
                         "expert_slots": int(residency[1]), "native_threads": 8, "native_batch_threads": 16,
                         "cache_allocation_log": [line.strip() for line in startup.splitlines() if "MoE cache size =" in line],
                         "cpu_weight_storage": ("lossless CPU-repacked host buffers; originals unchanged" if args.cpu_repack
                                                else "original GGUF mmap; decode physical reads measured")}
            self.info["cpu_repacked_expert_tensors"] = int(repacked[1]) if repacked is not None else 0
            if self.repack_ram_guard is not None:
                available = io.snapshot(self.pid)["system_memory"]["MemAvailable"]
                require(available >= self.repack_ram_guard["host_headroom_bytes"], "CPU-repack ready state violates host headroom.")
                self.repack_ram_guard["available_at_ready_bytes"] = available
                self.info["cpu_repack_ram_guard"] = self.repack_ram_guard
        except BaseException:
            self.close()
            raise

    @property
    def pid(self):
        return self.process.pid

    def generate(self, ids, maximum):
        payload = {"prompt": ids, "n_predict": maximum, "temperature": 0, "seed": 42,
                   "ignore_eos": self.ignore_eos, "cache_prompt": False, "stream": True, "return_tokens": True}
        request = urllib.request.Request(self.url + "/completion", json.dumps(payload).encode(),
            {"Authorization": "Bearer " + self.key, "Content-Type": "application/json"})
        self.last = {}
        self.stream_positions = []
        with urllib.request.urlopen(request, timeout=300) as response:
            for line in response:
                if not line.startswith(b"data: "):
                    continue
                value = line[6:].strip()
                if value == b"[DONE]":
                    break
                event = json.loads(value)
                if "error" in event:
                    raise RuntimeError("Native completion failed: " + str(event["error"]))
                if event.get("stop"):
                    self.last = {"generated": event["tokens_predicted"], "prompt_tokens": event["tokens_evaluated"],
                        "prompt_ms": event["timings"]["prompt_ms"], "decode_ms": event["timings"]["predicted_ms"],
                        "finish": event.get("stop_type", "length"), "raw_timings": event["timings"],
                        "drafts_offered": event["timings"].get("draft_n", 0),
                        "drafts_accepted": event["timings"].get("draft_n_accepted", 0),
                        "generation_settings": event.get("generation_settings", {})}
                else:
                    tokens = event.get("tokens", [])
                    require(len(tokens) <= 1, "Native partial token grouping changed; requalify collection.")
                    for token in tokens:
                        position = event["tokens_predicted"] - 1
                        require(0 <= position < maximum and
                                (not self.stream_positions or position > self.stream_positions[-1]),
                                "Native streamed output positions are invalid.")
                        self.stream_positions.append(position)
                        yield int(token)

    def audit(self, ids, maximum, streamed, positions, expected_count=None):
        payload = {"prompt": ids, "n_predict": maximum, "temperature": 0, "seed": 42,
                   "ignore_eos": self.ignore_eos, "cache_prompt": False, "stream": False, "return_tokens": True}
        request = urllib.request.Request(self.url + "/completion", json.dumps(payload).encode(),
            {"Authorization": "Bearer " + self.key, "Content-Type": "application/json"})
        start = time.monotonic()
        with urllib.request.urlopen(request, timeout=300) as response:
            result = json.load(response)
        tokens = result["tokens"]
        expected_count = maximum if expected_count is None else expected_count
        require(len(tokens) == result["tokens_predicted"] == expected_count
                and result["tokens_evaluated"] == len(ids),
                "Native nonstream audit count mismatch.")
        require(len(positions) == len(streamed) and all(0 <= p < len(tokens) for p in positions),
                "Native audit stream positions are invalid.")
        require(all(tokens[p] == token for p, token in zip(positions, streamed)),
                "Native audit output differs at a streamed token position.")
        return tokens, time.monotonic() - start

    def close(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=15)
        if not self.file.closed:
            self.file.close()


class StrataAdapter:
    def __init__(self, args, log):
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(args.strata_source.resolve(strict=True)))
        from serve.server import StrataEngine
        env = clean_environment()
        env.update(STRATA_ROUTE_TAIL_SKIP="0", STRATA_IQ_MT_MIN="1", STRATA_DROP_CACHE_ON_EXIT="0",
                   STRATA_PREFILL_CPU_SHARE="0")
        params = ["--pack", str(args.pack.resolve(strict=True)), "--native", str(args.model),
            "--ple-gguf", str(args.model.with_name(args.model.name.replace("-00001-", "-00002-"))),
            "--ple-io", "mmap", "--expert-profile", str(args.strata_profile.resolve(strict=True)),
            "--expert-cache", str(args.experts), "--prefill", "512", "--spec", str(args.window),
            "--max-context", "16384", "--kv", "fp16", "--suffix-draft", "0", "--lookup-chain", "0",
            "--prompt-cache", "0", "--adapt-every", "0", "--adapt-swaps", "0", "--pcie-frac", "0",
            "--pool-workers", "7", "--gpu", "0"]
        if args.mode == "mtp":
            params += ["--mtp", str(args.mtp.resolve(strict=True)), "--spec-min-p", "0"]
        if args.natural_eos:
            params.append("--stop-eos")
        start = time.monotonic()
        self.engine = StrataEngine(str(args.strata_binary.resolve(strict=True)), params,
                                  cwd=str(log.parent), log=str(log), env=env)
        self.load_seconds = time.monotonic() - start
        self.info = self.engine.info
        try:
            require(self.info["context"] == 16384 and self.info["kv"] == "fp16", "Strata context/KV mismatch.")
            require(self.info["expert_slots"] == args.experts and self.info["lookup"] == 0,
                    "Strata resident count or lookup differs from frozen control.")
            require(self.info["pcie_frac"] in ("0.00", 0), "Strata unexpectedly offloads cold experts to GPU.")
        except BaseException:
            self.close()
            raise

    @property
    def pid(self):
        return self.engine.proc.pid

    @property
    def last(self):
        return self.engine.last

    def generate(self, ids, maximum):
        yield from self.engine.generate(ids, maximum, {}, threading.Event())

    def close(self):
        self.engine.close()


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("native", "strata"), required=True)
    parser.add_argument("--mode", choices=("scalar", "canonical", "mtp", "lookup", "dflash"), default="scalar")
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--experts", type=int, default=3067)
    parser.add_argument("--cache-mib", type=int, default=6144)
    parser.add_argument("--attention-window", type=Path)
    parser.add_argument("--managed-memory", action="store_true")
    parser.add_argument("--verification-threads", choices=("off", "decode"))
    parser.add_argument("--draft-threads", type=int)
    parser.add_argument("--cpu-repack", action="store_true")
    parser.add_argument("--exact-packed-cpu", type=Path)
    parser.add_argument("--cpu-native-library", type=Path)
    parser.add_argument("--kv-cold-tier", type=Path)
    parser.add_argument("--expert-products", choices=("columns", "native"))
    parser.add_argument("--lookup-n", type=int, default=3)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--pack", type=Path)
    parser.add_argument("--mtp", type=Path)
    parser.add_argument("--strata-source", type=Path)
    parser.add_argument("--strata-binary", type=Path)
    parser.add_argument("--strata-profile", type=Path)
    parser.add_argument("--native-binary", type=Path)
    parser.add_argument("--native-plan", type=Path)
    parser.add_argument("--native-draft", type=Path)
    parser.add_argument("--cuda-plugin", type=Path)
    parser.add_argument("--token-corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--natural-eos", action="store_true")
    parser.add_argument("--category", choices=("coding", "reasoning", "prose"), nargs="+",
                        default=["coding", "reasoning", "prose"])
    args = parser.parse_args()
    if not 2 <= args.window <= 4 or not 16 <= args.tokens <= 512 or args.output.exists():
        parser.error("Invalid bounded parameters or existing output.")
    if args.engine == "strata" and args.mode == "canonical":
        parser.error("Native canonical reductions are not a Strata capability.")
    if not 1 <= args.cache_mib <= 131072 or not 2 <= args.lookup_n <= 8:
        parser.error("Invalid cache budget or causal lookup key length.")
    if args.engine == "strata" and (args.mode in ("lookup", "dflash") or args.attention_window or
                                   args.managed_memory or args.cache_mib != 6144):
        parser.error("These native window/cache controls are not Strata controls.")
    if args.mode == "lookup" and (args.attention_window is None or args.window not in (2, 4)):
        parser.error("Lookup requires the explicit qualified native window prototype at width two or four.")
    if args.mode == "dflash" and (args.attention_window is None or args.native_draft is None or args.window not in (2, 4)):
        parser.error("DFlash requires the explicit native window prototype, verified draft and width two or four.")
    if args.attention_window and args.mode == "canonical":
        parser.error("Optimized-preserving windows are not the older canonical attention contract.")
    if args.verification_threads is not None and args.attention_window is None:
        parser.error("Verification thread policy requires the explicit window prototype.")
    if args.expert_products is not None and args.attention_window is None:
        parser.error("Expert-product ablation requires the explicit window prototype.")
    if args.draft_threads is not None and (args.mode not in ("mtp", "dflash") or not 1 <= args.draft_threads <= 128):
        parser.error("Draft threads require a draft-model mode and a positive bounded thread count.")
    if args.engine != "native" and (args.cpu_repack or args.exact_packed_cpu or args.cpu_native_library):
        parser.error("Packed CPU controls are native-only.")
    if args.exact_packed_cpu and (not args.cpu_repack or args.cpu_native_library is None):
        parser.error("Exact packed CPU requires lossless repacking and an explicit matching native CPU plugin.")
    if args.kv_cold_tier and (args.engine != "native" or args.managed_memory):
        parser.error("Selective cold KV staging is native-only and requires global managed mode off.")
    report = {"schema_version": 1, "engine": args.engine, "mode": args.mode, "window_capacity": args.window,
        "split": args.split,
        "natural_eos": args.natural_eos,
        "cache_budget_mib": args.cache_mib, "lookup_key_tokens": args.lookup_n,
        "managed_memory_requested": args.managed_memory,
        "kv_cold_tier_requested": bool(args.kv_cold_tier),
        "expert_products_requested": (args.expert_products or "columns") if args.attention_window else None,
        "cpu_repack_requested": args.cpu_repack,
        "status": "running", "full_model_acceleration_claim": False,
        "raw_prompts_or_tokens_in_report": False, "requests": [],
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    engine = None
    gpu_probe = None
    try:
        require(args.output.parent.stat().st_mode & 0o077 == 0, "Private output parent required.")
        if args.managed_memory or args.kv_cold_tier:
            from vrambridge.telemetry import NvidiaProbe
            gpu_probe = NvidiaProbe()
            report["physical_gpu_before_load"] = gpu_probe.sample()
        corpus = json.loads(args.token_corpus.read_text())
        requests = [r for r in corpus["records"] if r["split"] == args.split and r["category"] in args.category]
        require(len(requests) == len(args.category), "One prompt per category in the selected split required.")
        log = args.output.with_suffix(".private.log")
        require(not log.exists(), "Refusing to append to an old log.")
        engine = StrataAdapter(args, log) if args.engine == "strata" else NativeEngine(args, log)
        report["load_to_ready_seconds"], report["info"], report["pid"] = engine.load_seconds, engine.info, engine.pid
        report["mapped_backend_libraries"] = io.backend_library_modules(engine.pid)
        if gpu_probe is not None:
            report["physical_gpu_ready"] = gpu_probe.sample()
            require(bool(report["physical_gpu_ready"]) and all(g["vram_total_bytes"] and g["vram_used_bytes"] is not None
                        and g["vram_used_bytes"] <= g["vram_total_bytes"] for g in report["physical_gpu_ready"]),
                    "Physical NVML memory counters unavailable or invalid.")
        report["observed_control_environment"] = {
            key.decode(): value.decode() for entry in (Path("/proc") / str(engine.pid) / "environ").read_bytes().split(b"\0")
            if (parts := entry.partition(b"="))[1] and
            (key := parts[0]).startswith((b"VRAMBRIDGE_", b"STRATA_", b"GGML_", b"OMP_", b"GOMP_"))
            for value in [parts[2]]}
        native_audits = []
        for request in requests:
            before = io.snapshot(engine.pid)
            start = time.monotonic()
            first = None
            first_snapshot = None
            output = []
            for token in engine.generate(request["tokens"], args.tokens):
                if token is None:
                    continue
                if first is None:
                    first = time.monotonic()
                    first_snapshot = io.snapshot(engine.pid)
                output.append(token)
            end = time.monotonic()
            after = io.snapshot(engine.pid)
            last = dict(engine.last)
            report["pending_request"] = {"id": request["id"], "streamed_token_ids": len(output), "engine_last": last}
            expected_count = last.get("generated")
            require(isinstance(expected_count, int) and 1 <= expected_count <= args.tokens, "Invalid engine output count.")
            if args.natural_eos:
                require(last.get("finish") == ("stop" if args.engine == "strata" else "eos"),
                        "Sanity request did not stop at natural EOS.")
            else:
                require(expected_count == args.tokens, "Engine output count does not match the cap.")
            if args.engine == "strata":
                require(len(output) == expected_count, "Strata token stream is incomplete.")
            else:
                require(bool(engine.stream_positions) and engine.stream_positions[0] == 0,
                        "Native first-token boundary is unavailable.")
                require(len(output) == len(engine.stream_positions) <= args.tokens,
                        "Native streamed token positions/counts do not agree.")
            require(last.get("prompt_tokens") == len(request["tokens"]), "Engine changed prompt tokenization.")
            require(last.get("reused", 0) == 0, "Prompt reuse violates frozen comparison.")
            require(first is not None and end > first and last.get("finish") not in ("nonfinite", "cancel"),
                    "Incomplete/nonfinite response.")
            if args.mode not in ("mtp", "lookup", "dflash"):
                require(last.get("drafts_offered", 0) == 0, "Scalar arm unexpectedly drafted tokens.")
            report["requests"].append({"id": request["id"], "category": request["category"],
                "prompt_tokens": len(request["tokens"]), "output_tokens": last["generated"],
                "streamed_token_ids": len(output), "stream_id_gaps": last["generated"] - len(output),
                "first_token_seconds": first - start, "request_seconds": end - start,
                "accepted_output_after_first_tok_s": (last["generated"] - 1) / (end - first),
                "engine_native_times_not_cross_engine_normalized": last,
                "process_interval": io.interval(before, after),
                "generation_interval": io.interval(first_snapshot, after),
                "rss_bytes_at_end": after["rss_bytes"], "anonymous_bytes_at_end": after["anonymous_bytes"],
                "file_backed_bytes_at_end": after["file_backed_bytes"],
                "host_memory_before_request": before["system_memory"],
                "host_memory_after_request": after["system_memory"],
                "process_swap_bytes_at_end": after["process_swap_bytes"]})
            private = args.output.with_name(args.output.stem + "-" + request["category"] + "-private-output.json")
            if args.engine == "native":
                native_audits.append((request, output, list(engine.stream_positions), private, report["requests"][-1]))
            else:
                with private.open("x") as stream:
                    stream.write(json.dumps({"id": request["id"], "tokens": output}) + "\n")
                report["requests"][-1]["output_token_sha256"] = token_digest(output)
            print(args.engine, args.mode, request["category"],
                  round(report["requests"][-1]["accepted_output_after_first_tok_s"], 3), "accepted wall tok/s", flush=True)
            if gpu_probe is not None:
                report["requests"][-1]["physical_gpu_after_request"] = gpu_probe.sample()
            del report["pending_request"]
        for request, streamed, positions, private, record in native_audits:
            output, audit_seconds = engine.audit(request["tokens"], args.tokens, streamed, positions, record["output_tokens"])
            with private.open("x") as stream:
                stream.write(json.dumps({"id": request["id"], "tokens": output,
                                         "streamed_positions": positions, "streamed_tokens": streamed}) + "\n")
            record.update(output_token_sha256=token_digest(output), audit_seconds_not_timed=audit_seconds,
                          audit_all_streamed_positions_match=True)
            if record["stream_id_gaps"]:
                print("native output audit:", request["category"], record["stream_id_gaps"],
                      "UTF-8-buffered IDs absent from stream; full replay checked", flush=True)
        report["status"] = "complete"
    except (OSError, ValueError, RuntimeError, KeyError, urllib.error.URLError) as error:
        report["status"], report["error"] = "failed", str(error)
        print("vrambridge-engine-comparison:", error, file=sys.stderr)
    finally:
        try:
            if engine is not None:
                engine.close()
                report["owned_backend_stopped"] = True
                if args.attention_window is not None:
                    metrics = json.loads(engine.window_metrics_path.read_text())
                    require(metrics["complete"] and metrics["pid"] == engine.pid and metrics["calls"] > 0
                            and metrics["window_scoped_products"], "Window prototype acknowledgement incomplete.")
                    require(all(metrics[key] == engine.window_identity[key] for key in ("source_sha256", "header_sha256")),
                            "Window prototype source/header identity mismatch.")
                    if any(r["engine_native_times_not_cross_engine_normalized"].get("drafts_offered", 0) > 0
                           for r in report["requests"]):
                        require(metrics["split_calls"] > 0 and metrics["linear_split_calls"] > 0
                                and metrics["expert_split_calls"] > 0, "Speculative window rewrites were not observed.")
                    report["attention_window"] = metrics
                    report["attention_window_build_identity"] = engine.window_identity
                if args.exact_packed_cpu is not None:
                    metrics = json.loads(engine.packed_metrics_path.read_text())
                    require(metrics["complete"] and metrics["pid"] == engine.pid and metrics["gemv_calls"] > 0,
                            "Exact packed CPU execution not acknowledged.")
                    require(all(metrics[key] == engine.packed_identity[key] for key in ("source_sha256", "header_sha256")),
                            "Packed CPU source/header identity mismatch.")
                    report["exact_packed_cpu"] = metrics
                    report["exact_packed_cpu_build_identity"] = engine.packed_identity
                if args.kv_cold_tier is not None:
                    metrics = json.loads(engine.kv_metrics_path.read_text())
                    require(metrics["complete"] and metrics["pid"] == engine.pid and metrics["kv_allocations"] > 0
                            and metrics["managed_allocation_calls"] > 0 and metrics["cpu_stage_requests"] > 0
                            and metrics["buffers_still_tracked"] == 0, "Cold KV staging/owner drain not acknowledged.")
                    require(all(metrics[key] == engine.kv_identity[key] for key in ("source_sha256", "header_sha256")),
                            "Cold KV source/header identity mismatch.")
                    report["kv_cold_tier"] = metrics
                    report["kv_cold_build_identity"] = engine.kv_identity
        except BaseException as error:
            report["status"] = "failed"
            report["cleanup_error"] = str(error)
            raise
        finally:
            try:
                if gpu_probe is not None:
                    gpu_probe.close()
            finally:
                io.write_report(args.output, report)
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
