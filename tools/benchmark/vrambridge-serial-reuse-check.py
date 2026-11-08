#!/usr/bin/env python3
"""Compare full-vocabulary server probabilities with and without actual prefix reuse."""

import argparse
import http.client
import json
import math
import os
from pathlib import Path
import re
import secrets
import socket
import subprocess
import sys
import time


LOCAL_LIBRARY_NAMES = ("libllama-server-impl.so", "libllama-common.so.0", "libllama.so.0", "libmtmd.so.0",
                       "libggml.so.0", "libggml-base.so.0", "libggml-cpu.so.0", "libggml-cuda.so.0")


def vector(event, vocabulary):
    positions = event.get("completion_probabilities")
    if not isinstance(positions, list) or len(positions) != 1:
        raise ValueError("Expected one probability position.")
    entries = positions[0].get("top_logprobs")
    if not isinstance(entries, list) or len(entries) != vocabulary:
        raise ValueError("A full-vocabulary result is required.")
    result = {}
    for entry in entries:
        token, value = entry.get("id"), entry.get("logprob")
        if (type(token) is not int or token < 0 or token >= vocabulary or token in result or
                (value is not None and (type(value) not in (int, float) or not math.isfinite(value)))):
            raise ValueError("Invalid vocabulary probability.")
        result[token] = value
    return result


def compare(reference, candidate, vocabulary):
    a, b = vector(reference, vocabulary), vector(candidate, vocabulary)
    if any((a[token] is None) != (b[token] is None) for token in a):
        raise ValueError("Probability masks differ.")
    values = [(value, b[token]) for token, value in a.items() if value is not None]
    if not values:
        raise ValueError("No finite probabilities.")
    maximum = max(abs(x - y) for x, y in values)
    relative = math.sqrt(sum((x - y) ** 2 for x, y in values) / max(sum(x * x for x, _ in values), 1e-30))
    return {"max_absolute": maximum, "relative_l2": relative,
            "passes": maximum <= 0.01 and relative <= 0.001,
            "finite_entries": len(values), "kind": "pre-sampling log probabilities, not raw logits"}


def verify_local_libraries(maps, directory):
    directory = directory.resolve(strict=True)
    expected = {directory.joinpath(name).resolve(strict=True) for name in LOCAL_LIBRARY_NAMES}
    if len(expected) != len(LOCAL_LIBRARY_NAMES) or any(path.parent != directory for path in expected):
        raise ValueError("Backend library links escape their bound directory.")
    observed = set()
    pattern = re.compile(r"^lib(?:llama(?:-common|-server-impl)?|ggml(?:-base|-cpu|-cuda)?|mtmd)\.so(?:\.|$)")
    for line in maps.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith("/"):
            continue
        path = Path(fields[5])
        if pattern.match(path.name):
            observed.add(path.resolve(strict=True))
    if observed != expected:
        raise ValueError("The running backend did not bind every library to its staged directory.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vocabulary", type=int, required=True)
    parser.add_argument("--ubatch", type=int, choices=(1, 512), required=True)
    parser.add_argument("--overlay", type=Path)
    parser.add_argument("--experimental-single-token", action="store_true")
    parser.add_argument("--require-execution-marker", action="store_true")
    parser.add_argument("--require-local-libraries", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.vocabulary <= 1048576:
        parser.error("Use a new private report path and bounded vocabulary.")
    if args.experimental_single_token and (args.ubatch != 1 or
                                           (args.overlay is None and not args.require_local_libraries)):
        parser.error("The single-token experiment requires a bound library set and ubatch 1.")
    if args.require_execution_marker and not args.experimental_single_token:
        parser.error("The execution marker requires the single-token experiment.")
    if args.require_local_libraries and (sys.platform != "linux" or args.overlay is not None):
        parser.error("Local-library verification requires Linux and no library overlay.")
    os.umask(0o077)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.output.with_suffix(".log")
    if log_path.exists():
        parser.error("Refusing to overwrite the backend log.")
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("LLAMA_", "GGML_", "STRATA_", "VRAMBRIDGE_")) and
           key not in ("LD_PRELOAD", "LD_LIBRARY_PATH")}
    if args.overlay is not None:
        if not (args.overlay / "libllama.so.0").is_file():
            parser.error("The overlay must contain a usable libllama.so.0.")
        env["LD_LIBRARY_PATH"] = str(args.overlay.resolve()) + os.pathsep + str(args.backend.resolve().parent)
    if args.experimental_single_token:
        env["VRAMBRIDGE_EXPERIMENT_CAUSAL_SINGLE_TOKEN"] = "1"
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    key = secrets.token_hex(32)
    report = {"state": "starting", "ubatch": args.ubatch, "context": 16384, "cases": [], "cold_repeats": [],
              "experimental_single_token": args.experimental_single_token,
              "scope": "Single-slot causal prefix reuse at one fixed placement; probabilities, not raw logits."}
    process = None

    def request(path, body):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=180)
        try:
            conn.request("POST", path, json.dumps(body),
                         {"Content-Type": "application/json", "Authorization": "Bearer " + key})
            response = conn.getresponse()
            raw = response.read(64 * 1024 * 1024 + 1)
            if response.status != 200 or len(raw) > 64 * 1024 * 1024:
                raise ValueError(f"{path} failed with HTTP {response.status} or excessive output.")
            return json.loads(raw)
        finally:
            conn.close()

    try:
        with log_path.open("x") as log:
            process = subprocess.Popen([str(args.backend), "-m", str(args.model), "--host", "127.0.0.1",
                "--port", str(port), "--api-key", key, "--ctx-size", "16384", "--parallel", "1",
                "--threads", "8", "--threads-batch", "16", "--n-gpu-layers", "auto", "--fit", "on",
                "--fit-target", "1024", "--ubatch-size", str(args.ubatch), "--spec-type", "none",
                "--cache-type-k", "f16", "--cache-type-v", "f16", "--log-verbosity", "4",
                *(["--no-warmup"] if args.require_execution_marker else [])],
                stdout=log, stderr=subprocess.STDOUT, env=env)
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise ValueError("Server exited before readiness.")
                try:
                    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
                    conn.request("GET", "/health")
                    response = conn.getresponse()
                    response.read()
                    conn.close()
                    if response.status == 200:
                        break
                except OSError:
                    pass
                time.sleep(0.25)
            else:
                raise ValueError("Server readiness deadline exceeded.")
            startup = log_path.read_text()
            execution_marker = "EXPERIMENT_CAUSAL_SINGLE_TOKEN prefill executed"
            if args.require_execution_marker and execution_marker in startup:
                raise ValueError("The execution marker appeared before an inference request.")
            if (not re.search(r"n_ubatch\s*=\s*" + str(args.ubatch) + r"\b", startup) or
                    not re.search(r"n_ctx_slot\s*=\s*16384\b", startup) or
                    not re.search(r"K \(f16\):.*V \(f16\):", startup)):
                raise ValueError("Required context, microbatch and KV activation markers are missing.")
            devices = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=10, check=True)
            if str(process.pid) not in devices.stdout.split():
                raise ValueError("GPU execution was not observed.")
            if args.require_local_libraries:
                verify_local_libraries(Path(f"/proc/{process.pid}/maps").read_text(), args.backend.resolve().parent)
            phrase = "Agent: Design a safe, bounded queue for independent model requests."
            for repeat in (1, 40):
                prompt = "\n".join([phrase] * repeat)
                tokens = request("/tokenize", {"content": prompt, "add_special": True})["tokens"]
                if len(tokens) < 4:
                    raise ValueError("Unexpected token count.")
                common = {"n_predict": 1, "temperature": 0, "seed": 42, "ignore_eos": True,
                          "return_tokens": True, "post_sampling_probs": False}
                reference = request("/completion", {**common, "prompt": tokens,
                                                     "n_probs": args.vocabulary, "cache_prompt": False})
                if args.require_execution_marker and execution_marker not in log_path.read_text():
                    raise ValueError("The requested prefill execution marker was not observed.")
                if args.require_execution_marker:
                    report["execution_marker_verified_after_request"] = True
                repeated = request("/completion", {**common, "prompt": tokens,
                                                    "n_probs": args.vocabulary, "cache_prompt": False})
                if repeated.get("timings", {}).get("cache_n") != 0:
                    raise ValueError("Cold repeat unexpectedly reused prompt tokens.")
                repeat_result = compare(reference, repeated, args.vocabulary)
                report["cold_repeats"].append({"prompt_tokens": len(tokens),
                                               "same_token": reference.get("tokens") == repeated.get("tokens"),
                                               **repeat_result})
                print("prompt", len(tokens), "cold repeat passed", repeat_result["passes"], flush=True)
                for split in (2, len(tokens) // 2, len(tokens) - 1):
                    request("/completion", {**common, "prompt": tokens[:split], "cache_prompt": True})
                    candidate = request("/completion", {**common, "prompt": tokens,
                                                         "n_probs": args.vocabulary, "cache_prompt": True})
                    cached = candidate.get("timings", {}).get("cache_n")
                    if type(cached) is not int or cached < split - 1:
                        raise ValueError("Expected prefix reuse was not observed.")
                    result = compare(reference, candidate, args.vocabulary)
                    report["cases"].append({"prompt_tokens": len(tokens), "split": split, "cached_tokens": cached,
                                            "same_token": reference.get("tokens") == candidate.get("tokens"), **result})
                    print("prompt", len(tokens), "split", split, "cached", cached,
                          "passed", result["passes"], flush=True)
            report["state"] = ("passed" if all(case["passes"] for case in
                                              report["cases"] + report["cold_repeats"]) else "numerical_failure")
            if args.require_local_libraries:
                verify_local_libraries(Path(f"/proc/{process.pid}/maps").read_text(), args.backend.resolve().parent)
                report["local_libraries_verified"] = True
            if args.experimental_single_token and "EXPERIMENT_CAUSAL_SINGLE_TOKEN enabled" not in log_path.read_text():
                raise ValueError("The requested experimental branch was not confirmed.")
    except (OSError, ValueError, http.client.HTTPException, subprocess.SubprocessError) as exc:
        report.update(state="failed", error=str(exc))
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        report["owned_backend_stopped"] = process is None or process.poll() is not None
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("result:", report["state"], flush=True)
    return 0 if report["state"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
