#!/usr/bin/env python3
"""Partition observed CPU/CUDA intervals without crediting API duration as saved time."""

from __future__ import annotations

import argparse
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys


SPEC = importlib.util.spec_from_file_location("critical_path_windows",
    Path(__file__).with_name("vrambridge-cache-window.py"))
assert SPEC and SPEC.loader
windows = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(windows)


def partition(envelope, cpu, gpu, observer, api):
    merge, intersection, subtract, duration = windows.merge, windows.intersection, windows.subtract, windows.duration
    both = intersection(cpu, gpu)
    cpu_only = subtract(cpu, gpu)
    gpu_only = subtract(gpu, cpu)
    neither = subtract(envelope, cpu + gpu)
    parts = {"cpu_only": cpu_only, "gpu_only": gpu_only, "cpu_and_gpu": both, "neither_cpu_nor_gpu": neither}
    if sum(duration(value) for value in parts.values()) != duration(envelope):
        raise ValueError("CPU/GPU partition does not cover the measured envelope exactly.")
    return {**{key + "_ms": duration(value) / 1e6 for key, value in parts.items()},
            "envelope_ms": duration(envelope) / 1e6,
            "observer_inside_neither_ms": duration(intersection(neither, observer)) / 1e6,
            "cuda_api_inside_neither_ms": duration(intersection(neither, api)) / 1e6,
            "unattributed_neither_ms": duration(subtract(neither, observer + api)) / 1e6}


def analyze(cpu, timeline, request):
    report, frames, ranges, intervals = windows.read_inputs(cpu, timeline, request)
    lo, hi = intervals["envelope"][0]
    connection = sqlite3.connect(timeline.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        pids = windows.bounded_rows(connection, """
            SELECT DISTINCT e.globalTid FROM NVTX_EVENTS e LEFT JOIN StringIds s ON e.textId=s.id
            WHERE COALESCE(e.text,s.value) LIKE 'vrambridge.cpu_graph:%'
        """)
        global_pids = {tid & ~0xffffff for (tid,) in pids}
        if len(global_pids) != 1:
            raise ValueError("One traced backend process required.")
        global_pid = next(iter(global_pids))
        api_rows = windows.bounded_rows(connection, """
            SELECT a.start,a.end,s.value FROM CUPTI_ACTIVITY_KIND_RUNTIME a
            JOIN StringIds s ON a.nameId=s.id
            WHERE (a.globalTid & ~16777215)=? AND a.end>? AND a.start<?
        """, (global_pid, lo, hi))
        kernels = windows.bounded_rows(connection, """
            SELECT start,end,streamId FROM CUPTI_ACTIVITY_KIND_KERNEL
            WHERE globalPid=? AND end>? AND start<?
        """, (global_pid, lo, hi))
        has_memset = connection.execute(
            "SELECT count(*) FROM sqlite_master WHERE name='CUPTI_ACTIVITY_KIND_MEMSET'").fetchone()[0]
        memsets = windows.bounded_rows(connection, """
            SELECT start,end FROM CUPTI_ACTIVITY_KIND_MEMSET WHERE globalPid=? AND end>? AND start<?
        """, (global_pid, lo, hi)) if has_memset else []
    finally:
        connection.close()
    api = windows.intersection([(a, b) for a, b, _ in api_rows], intervals["envelope"])
    gpu = windows.merge(intervals["gpu"] + intervals["copies"] +
                        windows.intersection(memsets, intervals["envelope"]))
    decomposition = partition(intervals["envelope"], intervals["cpu"], gpu, intervals["observer"], api)
    names = defaultdict(list)
    for a, b, name in api_rows:
        names[name].append((max(a, lo), min(b, hi)))
    leaders = [{"name": name, "calls": len(calls),
                "sum_call_elapsed_ms": sum(b - a for a, b in calls) / 1e6,
                "union_elapsed_ms": windows.duration(windows.merge(calls)) / 1e6}
               for name, calls in names.items()]
    leaders.sort(key=lambda row: -row["sum_call_elapsed_ms"])
    cpu_groups = defaultdict(list)
    for frame in frames:
        key = ("expert" if frame["matmuls"] else "+".join(sorted(frame["ops"])), frame["threads"])
        cpu_groups[key].append(ranges["graph"][frame["sequence"]])
    grouped = [{"ops": key[0], "threads": key[1], "calls": len(value),
                "elapsed_ms": windows.duration(windows.merge(value)) / 1e6}
               for key, value in cpu_groups.items()]
    grouped.sort(key=lambda row: -row["elapsed_ms"])
    return {"schema_version": 1, "study": "observed_native_cpu_cuda_handoff_partition",
        "performance_measurement": False, "full_model_acceleration_claim": False,
        "output_tokens": report["output_tokens"], "cuda_runtime_environment": report["cuda_runtime_environment"],
        "cpu_groups": grouped, "partition": decomposition, "cuda_api_leaders": leaders[:12],
        "kernel_stream_ids": sorted({stream for _, _, stream in kernels}),
        "kernel_sum_elapsed_ms": sum(max(0, min(b, hi) - max(a, lo)) for a, b, _ in kernels) / 1e6,
        "kernel_union_elapsed_ms": windows.duration(intervals["gpu"]) / 1e6,
        "limitations": [
            "Profiler-perturbed CPU-enclosed subset of first-text generation interval, including partial boundary tokens.",
            "CUDA API elapsed time can include required GPU waits, driver work and profiling overhead; not removable latency.",
            "CPU/GPU intervals partition wall time; CUDA API and observer subsets can overlap and must not be summed.",
            "Kernel time-range overlap does not prove SM occupancy or successful QKV multi-stream activation.",
            "No projected tok/s or precision/placement changes are implied.",
        ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-profile", type=Path, required=True)
    parser.add_argument("--timeline", type=Path, required=True)
    parser.add_argument("--request-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists() or Path(str(args.output) + ".part").exists():
            raise ValueError("Refusing to overwrite report or partial.")
        result = analyze(args.cpu_profile, args.timeline, args.request_report)
        result["source_sha256"] = windows.file_hash(Path(__file__))
        result["input_sha256"] = {p.name: windows.file_hash(p)
                                 for p in (args.cpu_profile, args.timeline, args.request_report)}
        temporary = Path(str(args.output) + ".part")
        with temporary.open("x") as stream:
            stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
        args.output.hardlink_to(temporary)
        temporary.unlink()
        print(args.output)
        return 0
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as error:
        print(f"vrambridge-critical-path: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
