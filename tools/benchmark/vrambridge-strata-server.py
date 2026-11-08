#!/usr/bin/env python3
"""Research-only Synapse CLI/OpenAI adapter for a reviewed Strata installation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import threading


def verified_engine(engine, context, experts):
    info = engine.info
    if (info.get("context") != context or info.get("kv") != "fp16" or
            info.get("expert_slots") != experts or info.get("drafting") != "mtp" or
            info.get("lookup") != 0 or float(info.get("pcie_frac", -1)) != 0):
        engine.close()
        raise ValueError("Strata did not activate the reviewed full-routing/F16/MTP contract.")
    return engine


def raw_completion(svc, request):
    prompt, maximum = request.get("prompt"), request.get("n_predict")
    if (not isinstance(prompt, str) or type(maximum) is not int or not 1 <= maximum <= 8192 or
            request.get("stream", False) or request.get("cache_prompt", False)):
        raise ValueError("The native tuning endpoint requires a bounded uncached text completion.")
    ids = svc.tok.encode(prompt, parse_special=False)
    if len(ids) + maximum > svc.reported_ctx():
        raise ValueError("Completion exceeds the actual provider context.")
    reservation = svc.reserve_inference()
    if reservation is not True:
        raise RuntimeError("The provider is busy or draining.")
    try:
        cancel = threading.Event()
        sampling = {key: request[key] for key in ("temperature", "seed", "top_k", "top_p") if key in request}
        with svc._turn(cancel):
            tokens = []
            stream = svc.engine.generate(ids, maximum, sampling, cancel)
            try:
                for token in stream:
                    if token is None:
                        continue
                    tokens.append(token)
                    if not request.get("ignore_eos", False) and token in svc.stop_ids:
                        break
            finally:
                stream.close()
            last = dict(svc.engine.last)
        generated, milliseconds = last.get("generated"), last.get("decode_ms")
        if (type(generated) is not int or generated != len(tokens) or
                type(milliseconds) not in (int, float) or milliseconds <= 0):
            raise ValueError("Provider did not report complete accepted-token timing.")
        return {"content": svc.tok.decode(tokens), "tokens_predicted": generated,
                "timings": {"predicted_n": generated, "predicted_ms": milliseconds,
                            "predicted_per_second": 1000 * generated / milliseconds,
                            "prompt_n": len(ids), "prompt_ms": last.get("prompt_ms", 0)}}
    finally:
        svc.release_inference()


def install_native_endpoints(server):
    original = server.make_handler

    def make_handler(svc):
        base = original(svc)

        class CompatibleHandler(base):
            def do_POST(self):
                path = self.path.split("?", 1)[0].rstrip("/")
                if path not in ("/tokenize", "/detokenize", "/completion"):
                    return super().do_POST()
                if not self._authorized():
                    return
                try:
                    body = json.loads(self._body() or b"{}")
                    if not isinstance(body, dict):
                        raise ValueError("Request must be a JSON object.")
                    if path == "/tokenize":
                        if not isinstance(body.get("content"), str):
                            raise ValueError("Tokenizer content must be text.")
                        result = {"tokens": svc.tok.encode(body["content"],
                                                          parse_special=body.get("parse_special") is True)}
                    elif path == "/detokenize":
                        tokens = body.get("tokens")
                        if not isinstance(tokens, list) or any(type(token) is not int or token < 0 for token in tokens):
                            raise ValueError("Detokenizer tokens must be nonnegative integer IDs.")
                        result = {"content": svc.tok.decode(tokens)}
                    else:
                        result = raw_completion(svc, body)
                    self._json(200, result)
                except (ValueError, TypeError, IndexError) as exc:
                    self._json(400, {"error": {"message": str(exc)}})
                except RuntimeError as exc:
                    self._json(503, {"error": {"message": str(exc)}})

        return CompatibleHandler

    server.make_handler = make_handler


def configuration(args, root):
    path = Path(os.environ["VRAMBRIDGE_STRATA_CONFIG"]).resolve(strict=True)
    if path.stat().st_size > 65536:
        raise ValueError("Strata bridge configuration exceeds 64 KiB.")
    supplied = json.loads(path.read_text())
    required = {"source", "binary", "model", "ple", "pack", "mtp", "profile", "experts"}
    if set(supplied) not in (required, required | {"expected_slots"}):
        raise ValueError("Strata bridge configuration has unsupported fields.")
    if not os.path.samefile(args.model, supplied["model"]):
        raise ValueError("This model has no reviewed Strata pack/profile binding.")
    binary = Path(supplied["binary"]).resolve(strict=True)
    component = json.loads((binary.parent / "engine-component.json").read_text())
    for item in component["files"]:
        name = Path(item["path"])
        if name.is_absolute() or ".." in name.parts:
            raise ValueError("Invalid engine component path.")
        with (binary.parent / name).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != item["sha256"]:
                raise ValueError("Strata component checksum mismatch.")
    profile = root / "profile.bin"
    profile.write_bytes(Path(supplied["profile"]).read_bytes())
    env = {"STRATA_ROUTE_TAIL_SKIP": "0", "STRATA_ROUTE_PRIOR": "0", "STRATA_ROUTE_RESIDENT": "0",
           "STRATA_ROUTE_RESIDENT_MTP": "0", "STRATA_DROP_CACHE_ON_EXIT": "0",
           "STRATA_PREFILL_CPU_SHARE": "0", "STRATA_IQ_MT_MIN": "1", "STRATA_SPEC_PROB": "1"}
    engine_args = ["--pack", supplied["pack"], "--native", supplied["model"], "--ple-gguf", supplied["ple"],
                   "--ple-io", "mmap", "--expert-profile", str(profile), "--expert-cache", str(supplied["experts"]),
                   "--prefill", "512", "--spec", "2", "--max-context", str(args.ctx_size), "--kv", "fp16",
                   "--suffix-draft", "0", "--lookup-chain", "0", "--prompt-cache", "0",
                   "--adapt-every", "0", "--adapt-swaps", "0", "--pcie-frac", "0",
                   "--pool-workers", str(max(0, args.threads - 1)), "--gpu", "0",
                   "--mtp", supplied["mtp"], "--spec-min-p", "0"]
    return supplied, {"exe": str(binary), "args": engine_args, "env": env,
                      "lib_dirs": [str(binary.parent)], "cwd": str(root),
                      "log": str(root / "engine.log"), "tokenizer": str(Path(supplied["pack"]) / "tokenizer"),
                      "model_name": args.alias, "api_key": args.api_key}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--alias", default="local-model")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--ctx-size", type=int, default=16384)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--threads-batch", type=int, default=16)
    parser.add_argument("--n-gpu-layers", default="all")
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--slots", action="store_true")
    parser.add_argument("--fit-target")
    parser.add_argument("--fit", nargs="?", const="on")
    parser.add_argument("--load-mode", default="auto")
    parser.add_argument("--poll", type=int, default=50)
    parser.add_argument("--no-op-offload", action="store_true")
    args = parser.parse_args()
    if (args.host != "127.0.0.1" or not 512 <= args.ctx_size <= 16384 or
            args.n_gpu_layers not in ("all", "auto") or args.parallel != 1 or
            args.load_mode != "auto" or not 1 <= args.threads <= 32):
        parser.error("The reviewed bridge supports loopback, automatic placement and contexts up to 16K only.")
    os.umask(0o077)
    sys.dont_write_bytecode = True
    for key in list(os.environ):
        if key.startswith(("STRATA_", "GGML_", "LLAMA_")) or key == "LD_PRELOAD":
            os.environ.pop(key)
    with tempfile.TemporaryDirectory(prefix="strata-synapse-bridge-") as temp:
        root = Path(temp)
        supplied, cfg = configuration(args, root)
        sys.path.insert(0, str(Path(supplied["source"]).resolve(strict=True)))
        from serve import server
        install_native_endpoints(server)
        original = server.start_engine_or_fallback

        def checked_start(*values, **kwargs):
            engine = verified_engine(original(*values, **kwargs), args.ctx_size,
                                     supplied.get("expected_slots", supplied["experts"]))
            print("VRAMBridge Strata verified: full routing, F16 KV, native MTP, no lookup or CPU-miss GPU tier",
                  flush=True)
            print(f"n_ctx_slot = {engine.info['context']}", flush=True)
            print("SYNAPSE_STRATA_VERIFIED " + json.dumps({
                "context": engine.info["context"], "kv": engine.info["kv"], "mtp": True,
                "full_routing": True, "cache_mib": engine.info["expert_cache_mib"],
                "cache_slots": engine.info["expert_slots"],
                "pid": engine.proc.pid}), flush=True)
            return engine

        server.start_engine_or_fallback = checked_start
        config = root / "server.json"
        config.write_text(json.dumps(cfg))
        return server.main(["--engine", "strata", "--config", str(config), "--host", args.host,
                            "--port", str(args.port), "--api-key", args.api_key, "--offline"])


if __name__ == "__main__":
    sys.exit(main())
