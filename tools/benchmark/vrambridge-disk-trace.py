#!/usr/bin/env python3
"""Attribute observed MoE I/O and replay byte-budget caches without predicting model speed."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys


spec = importlib.util.spec_from_file_location("disk_trace_routing", Path(__file__).with_name("vrambridge-routing.py"))
routing = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = routing
spec.loader.exec_module(routing)


def validate_interval(value):
    if not isinstance(value, dict):
        raise ValueError("Missing process I/O interval.")
    for key in ("elapsed_ns", "physical_read_bytes", "minor_faults", "major_faults"):
        routing.integer(value.get(key), key, 1 if key == "elapsed_ns" else 0)
    return value


def attribution(path, data):
    if data.metadata.get("io_observer") is not True:
        raise ValueError("Trace did not enable the I/O observer.")
    experts = {}
    tokens = {}
    with path.open("rb") as stream:
        for raw in stream:
            if not raw.strip():
                continue
            if len(raw) > 2 * 1024 ** 2:
                raise ValueError("I/O observation record exceeds 2 MiB.")
            record = json.loads(raw)
            if record["type"] == "route":
                key = record["request"], record["token"]
                observed = validate_interval(record.get("expert_observed_interval"))
                summary = experts.setdefault(key, {"read_bytes": 0, "elapsed_ns": 0, "major_faults": 0})
                summary["read_bytes"] += observed["physical_read_bytes"]
                summary["elapsed_ns"] += observed["elapsed_ns"]
                summary["major_faults"] += observed["major_faults"]
            elif record["type"] == "request_end":
                observations = record.get("observed_token_intervals")
                if not isinstance(observations, list) or len(observations) != record["decode_tokens"]:
                    raise ValueError("Missing or incomplete token I/O observations.")
                for number, observation in enumerate(observations):
                    if observation.get("token") != number:
                        raise ValueError("Missing, duplicate or out-of-order token I/O interval.")
                    key = record["id"], number
                    if key in tokens:
                        raise ValueError("Duplicate token I/O observation.")
                    whole = validate_interval(observation.get("interval"))
                    gathers = observation.get("engram_gathers")
                    if not isinstance(gathers, list):
                        raise ValueError("Missing Engram gather observation list.")
                    names = set()
                    engram_read = engram_time = 0
                    for gather in gathers:
                        name = routing.text(gather.get("name"), "Engram name")
                        if name in names:
                            raise ValueError("Duplicate Engram gather observation.")
                        names.add(name)
                        interval = validate_interval(gather.get("interval"))
                        engram_read += interval["physical_read_bytes"]
                        engram_time += interval["elapsed_ns"]
                    expert = experts.get(key)
                    if expert is None:
                        raise ValueError("Token has no expert intervals.")
                    covered = expert["read_bytes"] + engram_read
                    if covered > whole["physical_read_bytes"]:
                        raise ValueError("Observed intervals exceed whole-token read accounting.")
                    tokens[key] = {
                        "physical_read_bytes": whole["physical_read_bytes"],
                        "expert_interval_read_bytes": expert["read_bytes"],
                        "engram_gather_read_bytes": engram_read,
                        "unattributed_read_bytes": whole["physical_read_bytes"] - covered,
                        "observed_expert_ns": expert["elapsed_ns"], "observed_engram_ns": engram_time,
                        "observed_token_ns": whole["elapsed_ns"],
                        "expert_interval_major_faults": expert["major_faults"],
                    }
    expected = {(event.request, event.token) for event in data.events}
    if set(tokens) != expected:
        raise ValueError("Token I/O coverage differs from the validated routing trace.")
    result = {}
    for request in data.requests.values():
        values = [tokens[request.id, index] for index in range(request.tokens)]
        result[request.id] = {
            "split": request.split, "tokens": request.tokens,
            "totals": {key: sum(value[key] for value in values) for key in values[0]},
            "tokens_with_expert_interval_reads": sum(value["expert_interval_read_bytes"] > 0 for value in values),
        }
    return result


def replay(data, capacity, policy, bandwidth):
    routing.integer(capacity, "capacity")
    if policy not in ("train_static", "lru", "lfu"):
        raise ValueError("Unsupported causal disk-cache policy.")
    if not math.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("Disk bandwidth must be finite and positive.")
    train = data.events_for("train")
    heldout = data.events_for("test")
    if not train or not heldout:
        raise ValueError("Separate training and held-out routes are required.")
    cache = routing.Cache(policy, data.layers, capacity, 1)
    fixed = routing.hotset(routing.frequencies(train), data.layers, capacity) if policy == "train_static" else set()
    if policy != "train_static":
        for index, event in enumerate(train):
            cache.update(event.keys(), index)
    warm_used = sum(data.layers[key[0]].size for key in fixed) if fixed else cache.used
    initial_used = warm_used
    requests = {}
    demand_bytes = missed_bytes = admitted_bytes = evictions = 0
    peak = warm_used
    for index, event in enumerate(heldout, len(train)):
        resident = fixed if policy == "train_static" else cache.resident
        keys = event.keys()
        demand = sum(data.layers[key[0]].size for key in keys)
        misses = sum(data.layers[key[0]].size for key in keys if key not in resident)
        request = requests.setdefault(event.request, {"demand_bytes": 0, "logical_miss_bytes": 0})
        request["demand_bytes"] += demand
        request["logical_miss_bytes"] += misses
        demand_bytes += demand
        missed_bytes += misses
        if policy != "train_static":
            added, removed = cache.update(keys, index)
            admitted_bytes += added
            evictions += removed
            peak = max(peak, cache.used)
    count = data.tokens_for("test")
    for identifier, request in requests.items():
        request["tokens"] = data.requests[identifier].tokens
        request["byte_hit_fraction"] = 1 - request["logical_miss_bytes"] / request["demand_bytes"]
        request["logical_miss_bytes_per_token"] = request["logical_miss_bytes"] / request["tokens"]
    if peak > capacity:
        raise ValueError("Cache exceeded its byte budget.")
    return {
        "policy": policy, "capacity_bytes": capacity, "training_resident_bytes": initial_used,
        "peak_resident_bytes": peak, "heldout_tokens": count, "demand_bytes": demand_bytes,
        "logical_miss_bytes": missed_bytes, "byte_hit_fraction": 1 - missed_bytes / demand_bytes,
        "logical_miss_bytes_per_token": missed_bytes / count,
        "disk_bandwidth_ms_per_token_floor": missed_bytes / count / bandwidth * 1000,
        "admitted_bytes": admitted_bytes, "evicted_entries": evictions, "requests": requests,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--capacity-gib", type=float, nargs="+", default=[32, 64, 96, 108])
    parser.add_argument("--bandwidth-mbps", type=float, required=True, help="Measured decimal MB/s; sensitivity input only.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if (any(not math.isfinite(value) or not 0 <= value <= 256 for value in args.capacity_gib) or
                not math.isfinite(args.bandwidth_mbps) or args.bandwidth_mbps <= 0):
            raise ValueError("Invalid cache budget or measured bandwidth.")
        data = routing.read_trace(args.trace)
        if not all(request.verified for request in data.requests.values()):
            raise ValueError("Callback-inactive token sequence comparison did not pass.")
        result = {
            "schema_version": 1, "status": "complete", "model": data.metadata.get("model"),
            "backend_commit": data.metadata["backend_commit"], "trace_sha256": data.sha256,
            "analysis_source_sha256": routing.ANALYSIS_SOURCE_SHA256,
            "source_schema": "validated_decode_routing_with_io_observer",
            "measured_io": attribution(args.trace, data),
            "bandwidth_bytes_s_assumption": args.bandwidth_mbps * 1e6,
            "cache_replays": [replay(data, int(capacity * 1024 ** 3), policy, args.bandwidth_mbps * 1e6)
                              for capacity in args.capacity_gib for policy in ("train_static", "lru", "lfu")],
            "scope": [
                "Combined ideal RAM+VRAM expert-byte budget; no implemented disk cache or tier placement.",
                "Cache budget excludes active staging, dense weights, KV state and OS/interpreter memory.",
                "Cache is warmed only on training routes, retained through held-out requests; no future information.",
                "LFU is the existing unaged replay policy, not the production fill/replace admission bundle.",
                "Demand misses can be served without admission; admission/eviction counts are replay, not physical I/O.",
                "Bandwidth floors exclude Engram, prefetch waste, read amplification, seeks, transfers and computation.",
                "Observed expert intervals include routing-output scheduling and callback overhead, not kernel-only time.",
                "Engram intervals omit earlier prefetch; unattributed reads remain visible rather than reassigned.",
                "Reference replay warms OS pages between requests; attribution timing is not sustained model speed.",
                "Short calibration/held-out prompts do not establish general routing locality.",
            ],
        }
        with args.output.open("x", encoding="utf-8") as output:
            output.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(args.output)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Disk trace analysis failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
