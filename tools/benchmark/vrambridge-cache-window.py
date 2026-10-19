#!/usr/bin/env python3
"""Bound CPU weight staging in observed GPU windows; never infer model tok/s."""

from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys


def merge(intervals):
    result = []
    for start, end in sorted(intervals):
        if end < start:
            raise ValueError("Reversed timeline interval.")
        if end == start:
            continue
        if result and start <= result[-1][1]:
            result[-1] = result[-1][0], max(end, result[-1][1])
        else:
            result.append((start, end))
    return result


def intersection(left, right):
    left, right = merge(left), merge(right)
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        start, end = max(left[i][0], right[j][0]), min(left[i][1], right[j][1])
        if start < end:
            result.append((start, end))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return result


def subtract(left, right):
    result = []
    right = merge(right)
    j = 0
    for start, end in merge(left):
        while j < len(right) and right[j][1] <= start:
            j += 1
        k = j
        while k < len(right) and right[k][0] < end:
            a, b = right[k]
            if start < a:
                result.append((start, min(a, end)))
            start = max(start, b)
            if start >= end:
                break
            k += 1
        if start < end:
            result.append((start, end))
    return result


def duration(intervals):
    return sum(end - start for start, end in intervals)


class WindowBudget:
    def __init__(self, windows):
        windows = merge(windows)
        self.starts = [a for a, _ in windows]
        self.ends = [b for _, b in windows]
        self.prefix = [0]
        for a, b in windows:
            self.prefix.append(self.prefix[-1] + b - a)

    def between(self, start, end):
        if end < start:
            raise ValueError("Staging deadline precedes release.")
        left, right = bisect_right(self.ends, start), bisect_left(self.starts, end)
        if left >= right:
            return 0
        return (self.prefix[right] - self.prefix[left] - max(0, start - self.starts[left])
                - max(0, self.ends[right - 1] - end))


class SpareBandwidth:
    def __init__(self, windows, host_copies, bandwidth_gbps):
        events = {}
        for a, b in merge(windows):
            events.setdefault(a, [0, 0.0])[0] += 1
            events.setdefault(b, [0, 0.0])[0] -= 1
        for a, b, count in host_copies:
            if b <= a or count < 0:
                raise ValueError("Invalid host DMA interval or byte count.")
            rate = count / (b - a)
            events.setdefault(a, [0, 0.0])[1] += rate
            events.setdefault(b, [0, 0.0])[1] -= rate
        self.starts, self.ends, self.rates, self.prefix = [], [], [], [0.0]
        active, debit, previous = 0, 0.0, None
        for point, (window_change, rate_change) in sorted(events.items()):
            if previous is not None and point > previous and active:
                rate = max(0.0, bandwidth_gbps - max(0.0, debit))
                self.starts.append(previous)
                self.ends.append(point)
                self.rates.append(rate)
                self.prefix.append(self.prefix[-1] + (point - previous) * rate)
            active += window_change
            debit += rate_change
            previous = point

    def between(self, start, end):
        if end < start:
            raise ValueError("Staging deadline precedes release.")
        left, right = bisect_right(self.ends, start), bisect_left(self.starts, end)
        if left >= right:
            return 0.0
        return max(0.0, self.prefix[right] - self.prefix[left]
                   - max(0, start - self.starts[left]) * self.rates[left]
                   - max(0, self.ends[right - 1] - end) * self.rates[right - 1])


def host_passes(name):
    kind = name.removeprefix("CUDA_MEMCPY_KIND_").removeprefix("UVM_")
    passes = {"HTOD": 1, "DTOH": 1, "HTOA": 1, "ATOH": 1, "HTOH": 2,
              "DTOD": 0, "ATOA": 0, "ATOD": 0, "DTOA": 0, "PTOP": 0}
    if kind not in passes:
        raise ValueError("Unknown CUDA copy kind; cannot safely debit shared DRAM.")
    return passes[kind]


def bounded_rows(connection, query, parameters=()):
    rows = connection.execute(query, parameters).fetchmany(2000001)
    if len(rows) > 2000000:
        raise ValueError("Timeline query exceeds two million rows.")
    return rows


def load(path):
    return json.loads(path.read_text())


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_inputs(cpu_path, timeline_path, request_path):
    request = load(request_path)
    if request["status"] != "complete" or len(request["samples"]) != 1 or not request["owned_backend_stopped"]:
        raise ValueError("A complete single-request owned-backend measurement is required.")
    sample = request["samples"][0]
    before = sample["first_text_event"]["monotonic_seconds"] * 1e9
    after = sample["response_complete"]["monotonic_seconds"] * 1e9
    if cpu_path.stat().st_size > 512 * 1024 * 1024:
        raise ValueError("CPU profile exceeds 512 MiB.")
    frames = {}
    with cpu_path.open() as stream:
        for line in stream:
            if len(line) > 2 * 1024 * 1024:
                raise ValueError("CPU record exceeds two MiB.")
            frame = json.loads(line)
            sequence = frame["sequence"]
            if (type(sequence) is not int or sequence < 0 or sequence in frames or len(frames) >= 100000
                    or frame["schema_version"] != 3 or frame["status"] != 0
                    or frame["contains_sensitive_routing"] is not True
                    or frame["end_ns"] <= frame["start_ns"]):
                raise ValueError("Invalid, duplicate or incomplete private CPU profile.")
            frames[sequence] = frame
    if not frames:
        raise ValueError("Empty CPU profile.")
    connection = sqlite3.connect(timeline_path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        rows = bounded_rows(connection, """
            SELECT e.start,e.end,e.globalTid,COALESCE(e.text,s.value)
            FROM NVTX_EVENTS e LEFT JOIN StringIds s ON e.textId=s.id
            WHERE COALESCE(e.text,s.value) LIKE 'vrambridge.cpu_%'
        """)
        ranges = {"graph": {}, "meta": {}, "record": {}}
        global_pids = set()
        for start, end, tid, text in rows:
            kind, separator, index = text.removeprefix("vrambridge.cpu_").partition(":")
            if kind not in ranges:
                continue
            if not separator or not index.isdecimal() or end is None or end <= start:
                raise ValueError("Malformed or incomplete CPU NVTX range.")
            sequence = int(index)
            if sequence in ranges[kind]:
                raise ValueError("Duplicate CPU NVTX range.")
            ranges[kind][sequence] = start, end
            global_pids.add(tid & ~0xffffff)
        if len(global_pids) != 1 or any(set(ranges[kind]) != set(frames) for kind in ranges):
            raise ValueError("CPU/NVTX coverage or process scope mismatch.")
        global_pid = next(iter(global_pids))
        if ((global_pid >> 24) & 0xffffff) != sample["before_request"]["pid"]:
            raise ValueError("Nsight process encoding does not match the owned backend.")
        selected = sorted((frame for frame in frames.values()
                           if frame["start_ns"] >= before and frame["end_ns"] <= after),
                          key=lambda frame: frame["sequence"])
        if not selected:
            raise ValueError("No fully contained generation CPU frames.")
        for previous, current in zip(selected, selected[1:]):
            if ranges["graph"][previous["sequence"]][1] > ranges["graph"][current["sequence"]][0]:
                raise ValueError("Concurrent CPU graphs require a different staging model.")
        lo = ranges["graph"][selected[0]["sequence"]][0]
        hi = ranges["graph"][selected[-1]["sequence"]][1]
        kernels = bounded_rows(connection, """
            SELECT start,end,deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL
            WHERE globalPid=? AND end>? AND start<?
        """, (global_pid, lo, hi))
        if not kernels or len({row[2] for row in kernels}) != 1:
            raise ValueError("One traced GPU with kernel activity is required.")
        copy_rows = bounded_rows(connection, """
            SELECT start,end,bytes,copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY
            WHERE globalPid=? AND end>? AND start<?
        """, (global_pid, lo, hi))
        kinds = dict(connection.execute("SELECT id,name FROM ENUM_CUDA_MEMCPY_OPER"))
        if any(b <= a or type(count) is not int or count < 0 for a, b, count, _ in copy_rows):
            raise ValueError("Invalid CUDA copy duration or payload.")
    finally:
        connection.close()
    envelope = [(lo, hi)]
    gpu = intersection([(a, b) for a, b, _ in kernels], envelope)
    cpu = intersection(list(ranges["graph"].values()), envelope)
    observer = intersection(list(ranges["meta"].values()) + list(ranges["record"].values()), envelope)
    generous = subtract(gpu, cpu + observer)
    copies = [(a, b) for a, b, _, _ in copy_rows]
    strict = subtract(generous, copies)
    gpu_work = subtract(intersection([(a, b) for a, b, _ in kernels] + copies, envelope), cpu + observer)
    expert_cpu = intersection([ranges["graph"][sequence] for sequence, frame in frames.items()
                               if frame["matmuls"]], envelope)
    control_overlap = subtract(intersection([(a, b) for a, b, _ in kernels] + copies, envelope),
                               expert_cpu + observer)
    host_copies = [(a, b, count * host_passes(kinds[kind])) for a, b, count, kind in copy_rows]
    return request, selected, ranges, {
        "envelope": envelope, "gpu": gpu, "cpu": cpu, "observer": observer,
        "copies": intersection(copies, envelope), "generous": generous, "strict": strict,
        "gpu_work_dma_budgeted": gpu_work, "host_dma_load": host_copies,
        "expert_cpu": expert_cpu, "gpu_work_control_overlap": control_overlap,
    }


def demands(frames, ranges):
    producers = {}
    history = {}
    result = []
    previous_end = ranges["graph"][frames[0]["sequence"]][0]
    for frame in frames:
        start, end = ranges["graph"][frame["sequence"]]
        matrices = frame["matmuls"]
        if matrices:
            required = 0
            ready = []
            early_starts, early_chunks = [], []
            previous_chunks = []
            unknown = False
            early_unknown = False
            for matrix in matrices:
                ids = matrix.get("cold_ids")
                if (matrix["rows"] != 1 or ids is None or matrix["cold_experts"] != len(ids)
                        or len(set(ids)) != len(ids) or matrix["expert_bytes"] <= 0
                        or any(type(i) is not int or not 0 <= i < matrix["expert_bank_size"] for i in ids)
                        or matrix["logical_weight_bytes"] != len(ids) * matrix["expert_bytes"]):
                    raise ValueError("Unobservable or inconsistent generation cold-weight accounting.")
                required += matrix["logical_weight_bytes"]
                projection = matrix["weight_name"]
                previous_chunks.extend((matrix["expert_bytes"], identifier in ids)
                                       for identifier in history.get(projection, ()))
                history[projection] = sorted(ids)
                keys = matrix.get("id_source_keys", [])
                if any(type(key) is not int or key <= 0 for key in keys):
                    raise ValueError("Invalid opaque tensor identity.")
                candidates = [producers[key] for key in keys
                              if key in producers and producers[key]["end"] >= previous_end]
                if candidates:
                    producer = max(candidates, key=lambda item: item["end"])
                    ready.append(producer["end"])
                    hints = producer["hints"]
                    if len(hints) == 1 and hints[0].get("observable") is True:
                        predicted = hints[0]["pre_admission_misses"]
                        selected = hints[0]["selected_ids"]
                        if (len(set(predicted)) != len(predicted) or not set(predicted) <= set(selected)
                                or not set(ids) <= set(predicted)):
                            raise ValueError("Early residency snapshot does not cover the actual cold experts.")
                        early_starts.append(producer["start"])
                        early_chunks.extend((matrix["expert_bytes"], identifier in ids)
                                            for identifier in sorted(predicted))
                    else:
                        early_unknown = True
                else:
                    unknown = True
                    early_unknown = True
            if required or early_chunks or previous_chunks:
                result.append({"start": start, "previous_end": previous_end, "bytes": required,
                               "route_ready": None if unknown else max(ready),
                               "early_ready": None if early_unknown else max(early_starts),
                               "early_chunks": early_chunks, "previous_chunks": previous_chunks})
            if required:
                previous_end = end
        hints = [node["early_route"] for node in frame.get("nodes", []) if "early_route" in node]
        for node in frame.get("nodes", []):
            if type(node["tensor_key"]) is not int or node["tensor_key"] <= 0:
                raise ValueError("Invalid producer tensor identity.")
            producers[node["tensor_key"]] = {"start": start, "end": end, "hints": hints}
    return result


def oracle(jobs, windows, capacity, bandwidth_gbps, tile_bytes, causal=False, host_copies=(), early=False,
           budget_index=None, previous=False):
    warmed = 0
    attempted = 0
    usable_ns = 0
    covered = 0
    known = 0
    budget_windows, spare = budget_index or (
        WindowBudget(windows), SpareBandwidth(windows, host_copies, bandwidth_gbps))
    for job in jobs:
        release = job.get("early_ready") if early else job["route_ready"] if causal else job["previous_end"]
        chunks = job["previous_chunks"] if previous else job["early_chunks"] if early else ()
        demand = sum(count for count, _ in chunks) if early or previous else job["bytes"]
        if demand == 0:
            continue
        if release is None:
            continue
        known += 1
        ns = budget_windows.between(release, job["start"])
        usable_ns += ns
        budget = int(spare.between(release, job["start"]))
        amount = min(demand, capacity, budget)
        amount = amount // tile_bytes * tile_bytes
        attempted += amount
        if early or previous:
            remaining = amount
            useful = 0
            for count, needed in chunks:
                used = min(remaining, count)
                if needed:
                    useful += used
                remaining -= used
                if remaining == 0:
                    break
            warmed += useful
        else:
            warmed += amount
        covered += amount > 0
    total = sum(job["bytes"] for job in jobs)
    return {"warmable_payload_bytes_upper_bound": warmed,
            "attempted_staging_bytes": attempted, "wasted_staging_bytes": attempted - warmed,
            "logical_payload_fraction": warmed / total if total else 0,
            "eligible_window_ms": usable_ns / 1e6,
            "demand_graphs": len(jobs), "known_release_graphs": known, "graphs_with_warmed_tiles": covered}


def analyze(request, frames, ranges, intervals, capacities, bandwidths, tile_bytes):
    jobs = demands(frames, ranges)
    if not jobs:
        raise ValueError("No generation cold-expert work.")
    if tile_bytes <= 0 or any(c < tile_bytes for c in capacities):
        raise ValueError("Cache capacities must hold at least one positive staging tile.")
    if any(not math.isfinite(b) or b <= 0 for b in bandwidths):
        raise ValueError("Warm bandwidths must be finite and positive.")
    families = ("generous", "strict", "gpu_work_dma_budgeted", "gpu_work_control_overlap")
    indexes = {}
    for name in families:
        for bandwidth in bandwidths:
            debits = intervals["host_dma_load"] if name.startswith("gpu_work") else ()
            indexes[name, bandwidth] = WindowBudget(intervals[name]), SpareBandwidth(
                intervals[name], debits, bandwidth)
    scenarios = []
    for capacity in capacities:
        for bandwidth in bandwidths:
            for name in families:
                debits = intervals["host_dma_load"] if name.startswith("gpu_work") else ()
                scenarios.append({"cache_bytes": capacity, "assumed_warm_gbps": bandwidth,
                    "window_policy": name,
                    "perfect_next_graph_oracle": oracle(jobs, intervals[name], capacity, bandwidth, tile_bytes,
                                                      host_copies=debits, budget_index=indexes[name, bandwidth]),
                    "fresh_observed_route_ready": oracle(jobs, intervals[name], capacity, bandwidth, tile_bytes,
                                                        True, debits, budget_index=indexes[name, bandwidth]),
                    "pre_admission_dispatch_entry": oracle(jobs, intervals[name], capacity, bandwidth, tile_bytes,
                                                          True, debits, True, indexes[name, bandwidth]),
                    "previous_same_projection_cold_set": oracle(jobs, intervals[name], capacity, bandwidth, tile_bytes,
                                                              host_copies=debits,
                                                              budget_index=indexes[name, bandwidth], previous=True)})
    return {"schema_version": 1, "study": "gpu_window_cpu_cache_staging_headroom",
        "full_model_acceleration_claim": False, "performance_measurement": False,
        "actual_prefetch_implemented": False, "raw_expert_ids_published": False,
        "output_tokens": request["output_tokens"], "generation_cpu_frames": len(frames),
        "cold_demand_graphs": sum(j["bytes"] > 0 for j in jobs), "staging_candidate_graphs": len(jobs),
        "logical_projection_weight_bytes": sum(j["bytes"] for j in jobs),
        "interval_ms": {name: duration(value) / 1e6 for name, value in intervals.items() if name != "host_dma_load"},
        "tile_bytes": tile_bytes, "scenarios": scenarios,
        "limitations": [
            "CPU-enclosed subset of the first-text generation interval; boundary GPU-only work is excluded.",
            "Only measured GPU kernel activity outside CPU graph and observer bookkeeping ranges is eligible.",
            "Strict windows exclude all overlapping CUDA memcpy; generous windows ignore its DRAM contention.",
            "GPU-work windows include copy-engine activity; host transfers debit one read/write pass at uniform measured copy rate.",
            "Shared-bandwidth model ignores read/write turnaround penalties; memset-only windows are not included.",
            "Control-overlap windows allow CPU bookkeeping graphs but block all CPU expert graphs; control RAM traffic is uncharged.",
            "Early dispatcher snapshot uses current CPU-visible router IDs and pre-admission residency; admitted false positives are charged.",
            "Previous-cold-set control uses only earlier observed cold IDs; it cannot filter predictions using future demand.",
            "Oracle knows the exact next CPU demand before routing; this is not an implementable predictor.",
            "Causal release is a fresh recorded CPU producer completion, not necessarily the earliest GPU router readiness.",
            "Cache is an ideal byte-capacity budget per next graph, not measured LLC hits or a cache replacement simulation.",
            "Assumes native-cache misses, perfect residency, zero helper/hint overhead and freely divisible supplied bandwidth.",
            "Tile budgets are optimistic bytes; actual execution order, cachelines, CCD locality and pollution need hardware tests.",
            "Profiler overhead can alter windows; no predicted tok/s or end-to-end speedup follows from this trace.",
        ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-profile", type=Path, required=True)
    parser.add_argument("--timeline", type=Path, required=True)
    parser.add_argument("--request-report", type=Path, required=True)
    parser.add_argument("--cache-mib", type=int, nargs="+", default=[1, 8, 16, 32, 64])
    parser.add_argument("--warm-gbps", type=float, nargs="+", required=True)
    parser.add_argument("--tile-kib", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists() or Path(str(args.output) + ".part").exists():
            raise ValueError("Refusing to overwrite an existing report or partial.")
        request, frames, ranges, intervals = read_inputs(args.cpu_profile, args.timeline, args.request_report)
        report = analyze(request, frames, ranges, intervals,
                         [c * 1024 * 1024 for c in args.cache_mib], args.warm_gbps, args.tile_kib * 1024)
        report["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        report["input_sha256"] = {p.name: file_hash(p)
                                 for p in (args.cpu_profile, args.timeline, args.request_report)}
        temporary = Path(str(args.output) + ".part")
        with temporary.open("x") as stream:
            stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
        args.output.hardlink_to(temporary)
        temporary.unlink()
        print(args.output)
        return 0
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as exc:
        print(f"vrambridge-cache-window: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
