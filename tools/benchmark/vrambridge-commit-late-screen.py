#!/usr/bin/env python3
"""Screen causal next-token admission guesses against an observed Classic hybrid trace."""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict, deque
import hashlib
import heapq
import importlib.util
import json
import os
from pathlib import Path
import re
import sqlite3
import statistics
import sys


spec = importlib.util.spec_from_file_location("commit_late_windows", Path(__file__).with_name("vrambridge-cache-window.py"))
assert spec is not None and spec.loader is not None
windows = importlib.util.module_from_spec(spec)
spec.loader.exec_module(windows)


def dispatches(frames, ranges):
    producers = {}
    events = []
    previous_end = -1
    for frame in frames:
        start, end = ranges["graph"][frame["sequence"]]
        matrices = frame["matmuls"]
        if matrices:
            links, layers, cold_sets = [], set(), []
            for matrix in matrices:
                match = re.fullmatch(r"blk\.(\d+)\.ffn_(?:gate|up|down)_exps\.weight", matrix["weight_name"])
                if match is None or matrix["rows"] != 1 or matrix["expert_bytes"] <= 0:
                    raise ValueError("Unsupported routed-expert projection")
                layers.add(int(match[1]))
                cold = matrix.get("cold_ids")
                if (not isinstance(cold, list) or len(cold) != matrix["cold_experts"]
                        or len(set(cold)) != len(cold) or type(matrix["expert_bank_size"]) is not int
                        or matrix["expert_bank_size"] < 1
                        or any(type(x) is not int or not 0 <= x < matrix["expert_bank_size"] for x in cold)
                        or matrix["logical_weight_bytes"] != len(cold) * matrix["expert_bytes"]):
                    raise ValueError("Incomplete CPU expert mask")
                cold_sets.append(set(cold))
                sources = [producers[key] for key in matrix.get("id_source_keys", [])
                           if key in producers and producers[key]["end"] >= previous_end]
                if not sources:
                    raise ValueError("CPU expert mask has no current dispatcher producer")
                links.append(max(sources, key=lambda x: x["end"]))
            if (len(layers) != 1 or len(matrices) != 3 or len({x["sequence"] for x in links}) != 1
                    or len({matrix["expert_bank_size"] for matrix in matrices}) != 1
                    or any(value != cold_sets[0] for value in cold_sets)):
                raise ValueError("Expert projections do not share one layer, dispatcher and mask")
            producer = links[0]
            hint = producer["hint"]
            if hint is None or hint.get("observable") is not True:
                raise ValueError("Dispatcher pre-admission map is not observable")
            selected = hint.get("selected_ids")
            misses = hint.get("pre_admission_misses")
            if (not isinstance(selected, list) or not isinstance(misses, list)
                    or len(set(selected)) != len(selected) or len(set(misses)) != len(misses)
                    or any(type(expert) is not int or not 0 <= expert < matrices[0]["expert_bank_size"]
                           for expert in selected + misses)
                    or not set(misses) <= set(selected) or not cold_sets[0] <= set(misses)):
                raise ValueError("Invalid pre- and post-dispatch expert masks")
            events.append({"layer": next(iter(layers)), "selected": tuple(selected), "cold": cold_sets[0],
                           "pre_misses": set(misses),
                           "admitted": set(misses) - cold_sets[0], "producer": producer["sequence"],
                           "start": producer["start"], "end": end,
                           "sizes": tuple(matrix["expert_bytes"] for matrix in matrices)})
            if any(matrix["logical_weight_bytes"] for matrix in matrices):
                previous_end = end
        hints = [node["early_route"] for node in frame.get("nodes", []) if "early_route" in node]
        for node in frame.get("nodes", []):
            key = node.get("tensor_key")
            if type(key) is not int or key <= 0:
                raise ValueError("Invalid private tensor identity")
            producers[key] = {"sequence": frame["sequence"], "start": start, "end": end,
                              "hint": hints[0] if len(hints) == 1 else None}
    if not events:
        raise ValueError("No reconstructable expert dispatches")
    return events


def complete_tokens(events):
    layer_count = max(event["layer"] for event in events) + 1
    if layer_count < 2:
        raise ValueError("At least two routed layers are required")
    for a, b in zip(events, events[1:]):
        if b["layer"] != (a["layer"] + 1) % layer_count:
            raise ValueError("Routed layers are missing or out of order")
    tokens, current = [], []
    for event in events:
        if event["layer"] == 0:
            current = [event]
        elif current:
            current.append(event)
        if current and len(current) == layer_count:
            tokens.append(current)
            current = []
    if len(tokens) < 2:
        raise ValueError("Insufficient complete decode tokens")
    return tokens, layer_count


def scratch_schedule(jobs, slots, service_ns):
    if slots < 1 or service_ns <= 0:
        raise ValueError("Scratch slots and service time must be positive")
    events = []
    for index, job in enumerate(jobs):
        if job["deadline"] < job["release"]:
            raise ValueError("Scratch deadline precedes release")
        events.append((job["release"], 1, index))
        events.append((job["deadline"], 2, index))
    heapq.heapify(events)
    states = {}
    queue = deque()
    active, peak, engine_busy = 0, 0, False
    result = Counter()

    def launch(time):
        nonlocal active, engine_busy
        while queue:
            queued = queue[0]
            if states.get(queued) != "queued":
                queue.popleft()
            elif jobs[queued]["deadline"] <= time:
                queue.popleft()
                states[queued] = "cancelled"
                active -= 1
                result["cancelled_before_copy"] += 1
            else:
                break
        if not engine_busy and queue:
            index = queue.popleft()
            states[index] = "copying"
            engine_busy = True
            result["weight_upload_bytes"] += jobs[index]["bytes"]
            heapq.heappush(events, (time + service_ns, 0, index))

    while events:
        time, action, index = heapq.heappop(events)
        job = jobs[index]
        if action == 0:
            engine_busy = False
            if states[index] == "late":
                active -= 1
                states[index] = "finished"
            else:
                states[index] = "ready"
            launch(time)
        elif action == 1:
            result["offered"] += 1
            if active >= slots:
                result["capacity_rejected"] += 1
                continue
            states[index] = "queued"
            queue.append(index)
            active += 1
            peak = max(peak, active)
            result["accepted"] += 1
            launch(time)
        elif index in states:
            if states[index] == "ready":
                active -= 1
                states[index] = "finished"
                result["completed_before_deadline"] += 1
                if job["matched"]:
                    result["useful_admissions"] += 1
                    result["useful_weight_bytes"] += job["bytes"]
            elif states[index] == "queued":
                states[index] = "cancelled"
                active -= 1
                result["cancelled_before_copy"] += 1
            elif states[index] == "copying":
                states[index] = "late"
                result["late"] += 1
    return {**dict(result), "slots": slots, "peak_slots": peak,
            "useful_admissions": result["useful_admissions"],
            "useful_weight_bytes": result["useful_weight_bytes"],
            "weight_upload_bytes": result["weight_upload_bytes"],
            "capacity_rejected": result["capacity_rejected"], "late": result["late"],
            "cancelled_before_copy": result["cancelled_before_copy"]}


def forecast(tokens, warmup, k, previous_cold, transfer_ns, history_min=0, mode="previous", scratch_slots=()):
    if warmup < 1 or warmup >= len(tokens) or k < 1:
        raise ValueError("Insufficient tokens or invalid prediction width")
    if mode not in ("previous", "known_cold_frequency", "known_cold_recency"):
        raise ValueError("Unknown causal predictor")
    if history_min and warmup < 63:
        raise ValueError("History-filtered prediction needs 63 complete prior tokens")
    result = Counter()
    windows_by_candidate = []
    scratch_jobs = []
    histories = [deque() for _ in tokens[0]]
    counts = [Counter() for _ in tokens[0]]
    known_cold = [{} for _ in tokens[0]]
    for token, layer_events in enumerate(tokens):
        for layer, current in enumerate(layer_events):
            if token < warmup:
                if len(histories[layer]) == 63:
                    counts[layer].subtract(histories[layer].popleft())
                histories[layer].append(current["selected"])
                counts[layer].update(current["selected"])
                for rank, expert in enumerate(current["selected"]):
                    if expert in current["cold"]:
                        known_cold[layer][expert] = (token, rank)
                    else:
                        known_cold[layer].pop(expert, None)
                continue
            previous = tokens[token - 1][layer]
            if previous["layer"] != current["layer"]:
                raise ValueError("Token layer layouts differ")
            if mode == "previous":
                order = [expert for expert in previous["selected"]
                         if (not previous_cold or expert in previous["cold"]) and counts[layer][expert] >= history_min]
            else:
                candidates = [expert for expert in known_cold[layer] if counts[layer][expert] >= history_min]
                if mode == "known_cold_frequency":
                    order = sorted(candidates, key=lambda expert: (-counts[layer][expert],
                                   -known_cold[layer][expert][0], known_cold[layer][expert][1], expert))
                else:
                    order = sorted(candidates, key=lambda expert: (-known_cold[layer][expert][0],
                                   -counts[layer][expert], known_cold[layer][expert][1], expert))
            predicted = order[:k]
            actual = current["admitted"]
            hits = set(predicted) & actual
            lead = current["start"] - previous["end"]
            if lead < 0:
                raise ValueError("Prediction source follows its target deadline")
            weight_bytes = sum(current["sizes"])
            result.update(layer_positions=1, actual_admissions=len(actual), predictions=len(predicted),
                          matched_admissions=len(hits), weight_prediction_bytes=len(predicted) * weight_bytes,
                          useful_weight_bytes=len(hits) * weight_bytes,
                          timely_matches=len(hits) if lead >= transfer_ns else 0,
                          timely_predictions=len(predicted) if lead >= transfer_ns else 0,
                          admissions_below_default_history_floor=sum(counts[layer][expert] < 3 for expert in actual),
                          predicted_selected_already_resident=len(
                              set(predicted) & (set(current["selected"]) - current.get("pre_misses", actual | current["cold"]))))
            if predicted:
                windows_by_candidate.append((previous["end"], current["start"], len(predicted)))
                for expert in predicted:
                    scratch_jobs.append({"release": previous["end"], "deadline": current["start"],
                                         "bytes": weight_bytes, "matched": expert in actual})
            if len(histories[layer]) == 63:
                counts[layer].subtract(histories[layer].popleft())
            histories[layer].append(current["selected"])
            counts[layer].update(current["selected"])
            for rank, expert in enumerate(current["selected"]):
                if expert in current["cold"]:
                    known_cold[layer][expert] = (token, rank)
                else:
                    known_cold[layer].pop(expert, None)
    peak, active = 0, 0
    points = []
    for start, end, count in windows_by_candidate:
        if end > start:
            points.extend(((start, count), (end, -count)))
    for _, delta in sorted(points, key=lambda item: (item[0], item[1])):
        active += delta
        peak = max(peak, active)
    attempted, matched, admitted = result["predictions"], result["matched_admissions"], result["actual_admissions"]
    output = {**dict(result), "precision": matched / attempted if attempted else None,
              "admission_recall": matched / admitted if admitted else None,
              "timely_admission_recall": result["timely_matches"] / admitted if admitted else None,
              "hypothetical_peak_unlimited_scratch_experts": peak}
    if scratch_slots:
        output["scratch_scenarios"] = {str(slots): scratch_schedule(scratch_jobs, slots, transfer_ns)
                                       for slots in scratch_slots}
    return output


def analyze(cpu_path, timeline_path, request_path, warmup):
    request, frames, ranges, spans = windows.read_inputs(cpu_path, timeline_path, request_path)
    markers = set(request.get("hybrid_markers", []))
    logs = "\n".join(request.get("backend_log_tail", []))
    if (request.get("execution_mode") != "hybrid" or not {"hybrid", "direct", "coverage"} <= markers
            or "direct asynchronous admissions" not in logs
            or "transient cold GPU offload enabled" in logs or "VRAMBridge hosted MoE enabled" in logs):
        raise ValueError("Trace does not establish a suitable Classic direct-admission profile")
    events = dispatches(frames, ranges)
    if len({event["producer"] for event in events}) != len(events):
        raise ValueError("Multiple expert dispatches share a producer graph")
    tokens, layer_count = complete_tokens(events)
    if warmup >= len(tokens):
        raise ValueError("Warmup consumes all complete decode tokens")
    lo, hi = spans["envelope"][0]
    connection = sqlite3.connect(timeline_path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        pids = windows.bounded_rows(connection, """
            SELECT DISTINCT e.globalTid FROM NVTX_EVENTS e LEFT JOIN StringIds s ON e.textId=s.id
            WHERE COALESCE(e.text,s.value) LIKE 'vrambridge.cpu_graph:%'
        """)
        global_pids = {tid & ~0xffffff for (tid,) in pids}
        if len(global_pids) != 1:
            raise ValueError("Expected one owned backend process")
        global_pid = next(iter(global_pids))
        kinds = dict(connection.execute("SELECT id,name FROM ENUM_CUDA_MEMCPY_OPER"))
        h2d = [key for key, name in kinds.items() if name == "CUDA_MEMCPY_KIND_HTOD"]
        if len(h2d) != 1:
            raise ValueError("Unrecognized CUDA H2D direction")
        copies = windows.bounded_rows(connection, """
            SELECT start,end,bytes,copyKind,streamId FROM CUPTI_ACTIVITY_KIND_MEMCPY
            WHERE globalPid=? AND end>? AND start<?
        """, (global_pid, lo, hi))
        kernel_streams = [stream for (stream,) in windows.bounded_rows(connection, """
            SELECT DISTINCT streamId FROM CUPTI_ACTIVITY_KIND_KERNEL
            WHERE globalPid=? AND end>? AND start<?
        """, (global_pid, lo, hi))]
        has_memset = connection.execute(
            "SELECT count(*) FROM sqlite_master WHERE name='CUPTI_ACTIVITY_KIND_MEMSET'").fetchone()[0]
        memsets = windows.bounded_rows(connection, """
            SELECT start,end FROM CUPTI_ACTIVITY_KIND_MEMSET WHERE globalPid=? AND end>? AND start<?
        """, (global_pid, lo, hi)) if has_memset else []
        joined = windows.bounded_rows(connection, """
            SELECT m.start,m.end,m.bytes,m.streamId,a.start,a.end,s.value
            FROM CUPTI_ACTIVITY_KIND_MEMCPY m
            JOIN CUPTI_ACTIVITY_KIND_RUNTIME a ON m.correlationId=a.correlationId
                AND (a.globalTid & ~16777215)=m.globalPid
            JOIN StringIds s ON a.nameId=s.id
            WHERE m.globalPid=? AND m.end>? AND m.start<? AND m.copyKind=?
        """, (global_pid, lo, hi, h2d[0]))
    finally:
        connection.close()
    sizes = {size for event in events for size in event["sizes"]}
    candidates = [(a, b, size, stream) for a, b, size, kind, stream in copies if kind == h2d[0] and size in sizes]
    linked = [(a, b, size, stream, api_start, api_end, name)
              for a, b, size, stream, api_start, api_end, name in joined if size in sizes]
    if Counter(candidates) != Counter((a, b, size, stream) for a, b, size, stream, _, _, _ in linked):
        raise ValueError("Expert-sized H2D copies lack unique runtime correlation")
    ordered = sorted(events, key=lambda event: event["start"])
    starts = [event["start"] for event in ordered]
    observed = defaultdict(Counter)
    durations = defaultdict(list)
    for a, b, size, _, api_start, api_end, name in linked:
        index = bisect_right(starts, api_start) - 1
        if (b <= a or api_end <= api_start or index < 0
                or api_end > ranges["graph"][ordered[index]["producer"]][1]
                or not name.startswith("cudaMemcpyAsync")):
            raise ValueError("Expert-sized transfer was not launched in an observed dispatcher")
        observed[ordered[index]["producer"]][size] += 1
        durations[size].append(b - a)
    for event in events:
        expected = Counter(event["sizes"])
        expected = Counter({size: count * len(event["admitted"]) for size, count in expected.items() if event["admitted"]})
        if observed[event["producer"]] != expected:
            raise ValueError("Inferred admissions do not match CUDA projection-copy launches")
    if not candidates or any(size not in durations for size in sizes):
        raise ValueError("No observed expert projection copies")
    predicted_service = sum(statistics.median(durations[size]) for size in tokens[0][0]["sizes"])
    if any(event["sizes"] != tokens[0][0]["sizes"] for event in events):
        raise ValueError("Variable projection sizes require a per-layer transfer model")
    candidate_intervals = windows.intersection([(a, b) for a, b, _, _ in candidates], spans["envelope"])
    other_intervals = windows.intersection([(a, b) for a, b, size, kind, _ in copies
                                             if not (kind == h2d[0] and size in sizes)], spans["envelope"])
    overlap = {
        "envelope_ms": windows.duration(spans["envelope"]) / 1e6,
        "candidate_h2d_union_ms": windows.duration(candidate_intervals) / 1e6,
        "overlap_cpu_graph_ms": windows.duration(windows.intersection(candidate_intervals, spans["cpu"])) / 1e6,
        "overlap_cpu_expert_ms": windows.duration(windows.intersection(candidate_intervals, spans["expert_cpu"])) / 1e6,
        "overlap_gpu_kernel_ms": windows.duration(windows.intersection(candidate_intervals, spans["gpu"])) / 1e6,
        "no_cpu_gpu_copy_or_observer_ms": windows.duration(windows.subtract(
            candidate_intervals, spans["cpu"] + spans["gpu"] + other_intervals + memsets + spans["observer"])) / 1e6,
    }
    return {
        "schema_version": 1, "study": "observed_classic_admission_and_causal_route_screen",
        "model": Path(request["model"]).name, "requested_output_tokens": request["output_tokens"],
        "actual_context": request["actual_context"], "warmup_complete_tokens": warmup,
        "complete_decode_tokens": len(tokens), "scored_decode_tokens": len(tokens) - warmup,
        "routed_layers": layer_count, "observed_dispatches": len(events),
        "inferred_admissions": sum(len(event["admitted"]) for event in events),
        "correlated_weight_copy_calls": len(candidates), "weight_copy_bytes": sum(x[2] for x in candidates),
        "weight_copy_streams": sorted({x[3] for x in candidates}), "kernel_streams": sorted(kernel_streams),
        "median_projection_copy_ns": {str(size): statistics.median(values) for size, values in durations.items()},
        "independent_expert_copy_service_ns": predicted_service, "observed_overlap": overlap,
        "forecasts": {**{f"{name}_top{k}": forecast(tokens, warmup, k, cold, predicted_service)
                         for name, cold in (("previous_selected", False), ("previous_cpu_cold", True)) for k in (1, 2, 4)},
                      **{f"previous_cpu_cold_recent{minimum}_top{k}":
                         forecast(tokens, warmup, k, True, predicted_service, minimum,
                                  scratch_slots=(1, 4, 8, 16) if minimum == 3 and k == 1 else ())
                         for minimum in (2, 3) for k in (1, 2, 4)},
                      **{f"{mode}_recent3_top{k}": forecast(tokens, warmup, k, True, predicted_service, 3, mode)
                         for mode in ("known_cold_frequency", "known_cold_recency") for k in (1, 2, 4)}},
        "scope": ["Classic pre-admission misses minus observed CPU mask infer promotions; same-dispatch CUDA API launches are cross-checked per projection",
                  "Scratch transfers, GPU-to-GPU commit, cache and CPU bandwidth contention are not modeled",
                  "Independent readiness uses the elapsed gap and observed median copies, without copy-engine or capacity sharing",
                  "Scratch scenarios start empty after warmup and use a first-free policy and one speculative copy queue with queued-job cancellation; they ignore baseline DMA, DRAM contention, biases and GPU-to-GPU commit",
                  "Isolated intervals exclude recorded CPU graphs, kernels, copies, memsets and observer ranges; unrecorded work may remain",
                  "Duration overlap and isolated activity are not causally exposed or removable latency",
                  "Single instrumented request; no full-logit, target-platform throughput or future model claim"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-profile", type=Path, required=True)
    parser.add_argument("--timeline", type=Path, required=True)
    parser.add_argument("--request-report", type=Path, required=True)
    parser.add_argument("--warmup-tokens", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.warmup_tokens < 63:
            raise ValueError("Warmup must contain at least 63 complete tokens for the history filter")
        if args.output.exists():
            raise ValueError("Refusing to overwrite an existing report")
        result = analyze(args.cpu_profile, args.timeline, args.request_report, args.warmup_tokens)
        result["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        result["input_sha256"] = {p.name: windows.file_hash(p) for p in
                                  (args.cpu_profile, args.timeline, args.request_report)}
        with args.output.open("x", encoding="utf-8") as out:
            json.dump(result, out, indent=2, sort_keys=True, allow_nan=False)
            out.write("\n")
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as error:
        parser.exit(2, f"error: {error}\n")
    print(args.output)


if __name__ == "__main__":
    os.umask(0o077)
    main()
