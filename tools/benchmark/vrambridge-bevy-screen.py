#!/usr/bin/env python3
"""Screen aligned independent routing traces for resident/missing expert mixtures."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys


ROUTING_PATH = Path(__file__).with_name("vrambridge-routing.py")
spec = importlib.util.spec_from_file_location("vrambridge_routing", ROUTING_PATH)
assert spec is not None and spec.loader is not None
routing = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = routing
spec.loader.exec_module(routing)


def load_plan(path, layers):
    raw = path.read_bytes()
    plan = set()
    for line in raw.decode("utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 2:
            raise ValueError("Invalid plan entry")
        layer, expert = map(int, fields)
        if layer not in layers or expert < 0 or expert >= layers[layer].experts or (layer, expert) in plan:
            raise ValueError("Duplicate or out-of-range plan entry")
        plan.add((layer, expert))
    if not plan:
        raise ValueError("Plan is empty")
    return plan, hashlib.sha256(raw).hexdigest()


def screen(data, resident):
    requests = [r for r in data.requests.values() if r.split == "test"]
    if len(requests) < 2 or not all(r.verified for r in data.requests.values()):
        raise ValueError("At least two held-out requests and verified observer equivalence are required")
    positions = {r.id: {} for r in requests}
    for event in data.events_for("test"):
        positions[event.request][(event.token, event.layer)] = set(event.keys())
    levels = {}
    for width in range(2, min(len(requests), 3) + 1):
        total = Counter()
        groups = []
        for selected in itertools.combinations(requests, width):
            counters = Counter()
            for token in range(min(r.tokens for r in selected)):
                for layer in sorted(data.layers):
                    misses = [positions[r.id][(token, layer)] - resident for r in selected]
                    ready = sum(not miss for miss in misses)
                    unique = set().union(*misses)
                    counters.update(frames=1, ready_jobs=ready, jobs=width,
                                    mixed_frames=0 < ready < width,
                                    all_resident_frames=ready == width,
                                    all_missing_frames=ready == 0,
                                    selected_cpu_experts=sum(map(len, misses)),
                                    unique_cpu_experts=len(unique))
            frames = counters["frames"]
            if not frames:
                raise ValueError("Empty aligned request group")
            groups.append({"categories": [r.category for r in selected], **dict(counters),
                           "mixed_fraction": counters["mixed_frames"] / frames})
            total.update(counters)
        frames = total["frames"]
        levels[str(width)] = {**dict(total), "groups": groups,
                              "mixed_fraction": total["mixed_frames"] / frames,
                              "all_resident_fraction": total["all_resident_frames"] / frames,
                              "all_missing_fraction": total["all_missing_frames"] / frames,
                              "resident_job_fraction": total["ready_jobs"] / total["jobs"]}
    return levels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--cache-mib", type=int, required=True)
    parser.add_argument("--seed-plan", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.cache_mib < 0:
        parser.error("cache size must be nonnegative")
    try:
        data = routing.read_trace(args.trace)
        capacity = args.cache_mib * 1024**2
        training = routing.hotset(routing.frequencies(data.events_for("train")), data.layers, capacity)
        models = {"training_static": {"resident_experts": len(training),
                                      "resident_bytes": sum(data.layers[k[0]].size for k in training),
                                      "levels": screen(data, training)}}
        if args.seed_plan:
            plan, sha256 = load_plan(args.seed_plan, data.layers)
            plan_bytes = sum(data.layers[k[0]].size for k in plan)
            if plan_bytes > capacity:
                raise ValueError("Seed plan exceeds the supplied logical byte capacity")
            models["fixed_seed_plan"] = {"plan_sha256": sha256, "resident_experts": len(plan),
                                         "resident_bytes": plan_bytes, "levels": screen(data, plan)}
        report = {
            "schema_version": 1,
            "trace_sha256": data.sha256,
            "screen_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "routing_source_sha256": routing.ANALYSIS_SOURCE_SHA256,
            "model": Path(data.metadata.get("model", "unknown")).name,
            "architecture": data.metadata["architecture"],
            "cache_mib": args.cache_mib,
            "train_tokens": data.tokens_for("train"),
            "test_tokens": data.tokens_for("test"),
            "scope": "Fixed hypothetical residency; align independent serial traces by token and layer, not measured concurrency",
            "limitations": ["No Bevy scheduler, cache updates, timing, prefetch, prefill, KV or backend execution",
                            "A partly resident request can execute on both CPU and GPU; missing does not prove a stall",
                            "Aligned tokens are an artificial pairing and do not predict asynchronous progress",
                            "Fixed seed plan is not the dynamic runtime state; traces do not include model hashes"],
            "residencies": models,
        }
        with args.output.open("x", encoding="utf-8") as out:
            json.dump(report, out, indent=2, sort_keys=True)
            out.write("\n")
    except (OSError, UnicodeError, ValueError, KeyError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(args.output)


if __name__ == "__main__":
    main()
