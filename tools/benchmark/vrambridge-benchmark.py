#!/usr/bin/env python3
"""Run a reproducible llama-bench memory-placement experiment matrix."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import pathlib
import platform
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any


SCHEMA_VERSION = 1
DEFAULT_CONFIGS = ("cpu", "gpu-offload", "auto-fit", "unified-memory")


@dataclass
class Experiment:
    name: str
    arguments: list[str] = field(default_factory=list)
    environment: dict[str, str] = field(default_factory=dict)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=pathlib.Path, help="path to a local GGUF model")
    parser.add_argument("--llama-bench", type=pathlib.Path, default=os.environ.get("LLAMA_BENCH", "llama-bench"))
    parser.add_argument("--llama-source", type=pathlib.Path, help="llama.cpp checkout used to resolve the exact commit")
    parser.add_argument("--upstream-commit", help="explicit llama.cpp commit when no source checkout is available")
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("benchmarks/results"))
    parser.add_argument("--config", action="append", choices=(*DEFAULT_CONFIGS, "tensor-placement", "cpu-moe"))
    parser.add_argument("--gpu-layers", help="calibrated manual layer count for gpu-offload")
    parser.add_argument("--fit-target", type=int, default=1024, metavar="MIB")
    parser.add_argument("--fit-context", type=int, default=4096)
    parser.add_argument("--cpu-moe-layers", type=int, default=999)
    parser.add_argument("--tensor-override", action="append", default=[], metavar="PATTERN=BUFFER")
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--generation-tokens", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--no-op-offload", action="store_true")
    parser.add_argument("--no-kv-offload", action="store_true")
    parser.add_argument("--no-telemetry", action="store_true", help="disable nvidia-smi and process sampling")
    parser.add_argument("--extra-arg", action="append", default=[], help="additional llama-bench argument; repeat as needed")
    parser.add_argument("--dry-run", action="store_true", help="write commands without executing llama-bench")
    return parser.parse_args()


def git_commit(source: pathlib.Path | None) -> str | None:
    if source is None:
        return None
    completed = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def file_fingerprint(path: pathlib.Path) -> dict[str, Any]:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        head = model_file.read(1024 * 1024)
        digest.update(head)
        if stat.st_size > len(head):
            model_file.seek(max(0, stat.st_size - 1024 * 1024))
            digest.update(model_file.read(1024 * 1024))
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "sha256_first_last_mib": digest.hexdigest(),
    }


def build_experiments(args: argparse.Namespace) -> list[Experiment]:
    selected = args.config or list(DEFAULT_CONFIGS)
    experiments: list[Experiment] = []
    for name in selected:
        if name == "cpu":
            experiments.append(Experiment(name, ["-ngl", "0", "-dev", "none"]))
        elif name == "gpu-offload":
            if args.gpu_layers is None:
                raise ValueError("gpu-offload requires a calibrated --gpu-layers value")
            experiments.append(Experiment(name, ["-ngl", str(args.gpu_layers)]))
        elif name == "auto-fit":
            experiments.append(Experiment(name, ["-fitt", str(args.fit_target), "-fitc", str(args.fit_context)]))
        elif name == "unified-memory":
            experiments.append(Experiment(name, ["-ngl", "-1"], {"GGML_CUDA_ENABLE_UNIFIED_MEMORY": "1"}))
        elif name == "tensor-placement":
            if not args.tensor_override:
                raise ValueError("tensor-placement requires at least one --tensor-override")
            arguments = ["-ngl", "-1"]
            for override in args.tensor_override:
                arguments.extend(["-ot", override])
            experiments.append(Experiment(name, arguments))
        elif name == "cpu-moe":
            experiments.append(Experiment(name, ["-ngl", "-1", "-ncmoe", str(args.cpu_moe_layers)]))
    return experiments


def parse_json_output(stdout: str) -> tuple[Any | None, str | None]:
    try:
        return json.loads(stdout), None
    except json.JSONDecodeError as error:
        return None, f"llama-bench stdout was not valid JSON: {error}"


def read_process_sample(pid: int, started: float) -> dict[str, Any] | None:
    try:
        status: dict[str, str] = {}
        for line in pathlib.Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                status[key] = value.strip()
        stat_tail = pathlib.Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()
        meminfo: dict[str, int] = {}
        for line in pathlib.Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, value = line.split(":", 1)
            fields = value.split()
            if fields:
                meminfo[key] = int(fields[0]) * (1024 if len(fields) > 1 and fields[1] == "kB" else 1)
        clock_ticks = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
        return {
            "elapsed_seconds": time.monotonic() - started,
            "process_rss_bytes": int(status.get("VmRSS", "0 kB").split()[0]) * 1024,
            "process_cpu_seconds": (int(stat_tail[11]) + int(stat_tail[12])) / clock_ticks,
            "system_ram_available_bytes": meminfo.get("MemAvailable", 0),
        }
    except (FileNotFoundError, IndexError, ValueError):
        return None


def summarize_process_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {"sample_count": 0}
    def phase_summary(phase: list[dict[str, Any]]) -> dict[str, Any]:
        elapsed = phase[-1]["elapsed_seconds"] - phase[0]["elapsed_seconds"]
        cpu_seconds = phase[-1]["process_cpu_seconds"] - phase[0]["process_cpu_seconds"]
        return {
            "sample_count": len(phase),
            "duration_seconds": elapsed,
            "peak_process_rss_bytes": max(sample["process_rss_bytes"] for sample in phase),
            "minimum_system_ram_available_bytes": min(sample["system_ram_available_bytes"] for sample in phase),
            "process_cpu_seconds": cpu_seconds,
            "average_process_cpu_percent": cpu_seconds / elapsed * 100.0 if elapsed > 0 else None,
        }

    summary = {
        "sample_count": len(samples),
        "peak_process_rss_bytes": max(sample["process_rss_bytes"] for sample in samples),
        "minimum_system_ram_available_bytes": min(sample["system_ram_available_bytes"] for sample in samples),
        "whole_process": phase_summary(samples),
    }
    resident_threshold = summary["peak_process_rss_bytes"] * 0.9
    resident = [sample for sample in samples if sample["process_rss_bytes"] >= resident_threshold]
    if len(resident) >= 2:
        summary["resident_phase"] = phase_summary(resident)
    return summary


def parse_dmon(output: str) -> dict[str, Any]:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    header_index = next((index for index, line in enumerate(lines) if line.startswith("#Date")), None)
    if header_index is None:
        return {"sample_count": 0, "raw": output}
    header = [field.strip().lstrip("#") for field in next(csv.reader([lines[header_index]], skipinitialspace=True))]
    samples: list[dict[str, Any]] = []
    for line in lines[header_index + 1:]:
        if line.startswith("#"):
            continue
        fields = [field.strip() for field in next(csv.reader([line], skipinitialspace=True))]
        if len(fields) != len(header):
            continue
        sample: dict[str, Any] = {}
        for key, value in zip(header, fields):
            if value == "-":
                sample[key] = None
                continue
            try:
                sample[key] = float(value) if "." in value else int(value)
            except ValueError:
                sample[key] = value
        samples.append(sample)

    def phase_summary(phase: list[dict[str, Any]]) -> dict[str, Any]:
        phase_result: dict[str, Any] = {"sample_count": len(phase)}
        for field in ("pwr", "gtemp", "sm", "mem", "fb", "bar1", "rxpci", "txpci"):
            values = [sample[field] for sample in phase if isinstance(sample.get(field), (int, float))]
            if values:
                phase_result[field] = {"mean": sum(values) / len(values), "max": max(values)}
        return phase_result

    summary: dict[str, Any] = phase_summary(samples)
    framebuffer_values = [sample["fb"] for sample in samples if isinstance(sample.get("fb"), (int, float))]
    if framebuffer_values:
        resident_threshold = max(2048.0, max(framebuffer_values) * 0.5)
        resident = [sample for sample in samples if isinstance(sample.get("fb"), (int, float)) and sample["fb"] >= resident_threshold]
        if resident:
            summary["resident_phase"] = phase_summary(resident)
    return {
        "units": {
            "pwr": "W", "gtemp": "C", "sm": "percent", "mem": "percent",
            "fb": "MiB", "bar1": "MiB", "rxpci": "MiB/s", "txpci": "MiB/s",
        },
        "summary": summary,
        "samples": samples,
        "stderr": "",
    }


def run_experiment(args: argparse.Namespace, experiment: Experiment) -> dict[str, Any]:
    command = [
        str(args.llama_bench),
        "-m", str(args.model.resolve()),
        "-p", str(args.prompt_tokens),
        "-n", str(args.generation_tokens),
        "-r", str(args.repetitions),
        "-o", "json",
        *experiment.arguments,
    ]
    if args.threads is not None:
        command.extend(["-t", str(args.threads)])
    if args.no_op_offload:
        command.extend(["-nopo", "1"])
    if args.no_kv_offload:
        command.extend(["-nkvo", "1"])
    command.extend(args.extra_arg)

    result: dict[str, Any] = {
        "name": experiment.name,
        "command": command,
        "command_shell": shlex.join(command),
        "environment": experiment.environment,
        "status": "planned" if args.dry_run else "running",
    }
    if args.dry_run:
        return result

    environment = os.environ.copy()
    environment.update(experiment.environment)
    started_at = utc_now()
    started = time.monotonic()
    process_samples: list[dict[str, Any]] = []
    dmon_process: subprocess.Popen[str] | None = None
    dmon_output = ""
    dmon_error = ""
    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stdout_file, \
            tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stderr_file, \
            tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as dmon_stdout, \
            tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as dmon_stderr:
        if not args.no_telemetry and shutil.which("nvidia-smi"):
            dmon_process = subprocess.Popen(
                ["nvidia-smi", "dmon", "-i", "0", "-s", "pumt", "-d", "1", "-o", "DT", "--format", "csv,nounit"],
                stdout=dmon_stdout,
                stderr=dmon_stderr,
                text=True,
            )
        process = subprocess.Popen(command, stdout=stdout_file, stderr=stderr_file, text=True, env=environment)
        while process.poll() is None:
            if not args.no_telemetry:
                sample = read_process_sample(process.pid, started)
                if sample:
                    process_samples.append(sample)
            time.sleep(0.25)
        return_code = process.wait()
        if dmon_process is not None:
            dmon_process.terminate()
            try:
                dmon_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                dmon_process.kill()
                dmon_process.wait()
            dmon_stdout.seek(0)
            dmon_stderr.seek(0)
            dmon_output = dmon_stdout.read()
            dmon_error = dmon_stderr.read()
        stdout_file.seek(0)
        stderr_file.seek(0)
        stdout = stdout_file.read()
        stderr = stderr_file.read()
    elapsed = time.monotonic() - started
    parsed, parse_error = parse_json_output(stdout)
    load_and_warmup_seconds = None
    if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict) and parsed[0].get("test_time"):
        try:
            first_test = dt.datetime.fromisoformat(parsed[0]["test_time"].replace("Z", "+00:00"))
            run_start = dt.datetime.fromisoformat(started_at.replace("Z", "+00:00"))
            load_and_warmup_seconds = max(0.0, (first_test - run_start).total_seconds())
        except ValueError:
            pass
    gpu_telemetry = parse_dmon(dmon_output) if dmon_output else {"sample_count": 0}
    if isinstance(gpu_telemetry, dict):
        gpu_telemetry["stderr"] = dmon_error
    result.update({
        "started_at_utc": started_at,
        "duration_seconds": elapsed,
        "estimated_load_and_warmup_seconds": load_and_warmup_seconds,
        "return_code": return_code,
        "status": "ok" if return_code == 0 and parse_error is None else "failed",
        "measurements": parsed,
        "telemetry": {
            "process": {
                "summary": summarize_process_samples(process_samples),
                "samples": process_samples,
            },
            "gpu": gpu_telemetry,
        },
        "parse_error": parse_error,
        "stdout": stdout,
        "stderr": stderr,
    })
    return result


def output_path(output_dir: pathlib.Path, model: pathlib.Path) -> pathlib.Path:
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_model = "".join(character if character.isalnum() or character in "-_" else "_" for character in model.stem)
    return output_dir / f"{timestamp}-{safe_model}.json"


def main() -> int:
    args = parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(f"model does not exist: {args.model}")
    if min(args.prompt_tokens, args.generation_tokens, args.repetitions, args.fit_target, args.fit_context) <= 0:
        raise ValueError("token counts, repetitions, fit target, and fit context must be positive")

    experiments = build_experiments(args)
    upstream_commit = args.upstream_commit or git_commit(args.llama_source)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "model": file_fingerprint(args.model),
        "backend": {
            "llama_bench": str(args.llama_bench),
            "llama_cpp_commit": upstream_commit,
        },
        "parameters": {
            "prompt_tokens": args.prompt_tokens,
            "generation_tokens": args.generation_tokens,
            "repetitions": args.repetitions,
            "fit_target_mib": args.fit_target,
            "fit_context": args.fit_context,
        },
        "runs": [],
    }

    for experiment in experiments:
        print(f"[{experiment.name}]", file=sys.stderr, flush=True)
        report["runs"].append(run_experiment(args, experiment))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_path(args.output_dir, args.model)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    print(destination)

    failed = [run for run in report["runs"] if run["status"] == "failed"]
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"vrambridge-benchmark: {error}", file=sys.stderr)
        raise SystemExit(2)