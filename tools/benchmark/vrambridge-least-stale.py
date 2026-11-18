#!/usr/bin/env python3
"""Replay causal, decode-only expert cache policies on verified routing traces."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import heapq
import importlib.util
import json
from pathlib import Path
import sys


ROUTING_PATH = Path(__file__).with_name("vrambridge-routing.py")
spec = importlib.util.spec_from_file_location("vrambridge_routing", ROUTING_PATH)
assert spec is not None and spec.loader is not None
routing = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = routing
spec.loader.exec_module(routing)


class Cache:
    """Global, byte-bounded cache with end-of-layer admissions and CPU misses."""

    def __init__(self, policy, layers, capacity, admit_after):
        if policy not in ("lru", "lfu", "least_stale") or capacity < 0 or admit_after < 1:
            raise ValueError("Invalid policy, capacity or admission threshold")
        self.policy, self.layers = policy, layers
        self.capacity, self.admit_after = capacity, admit_after
        self.resident = {}
        self.counts = Counter()
        self.used = self.clock = self.serial = 0
        self.cycle = None
        self.current = set()
        self.heap = []
        self.stale = []
        self.current_heap = []

    def _push(self, key):
        self.serial += 1
        self.resident[key] = self.serial
        if self.policy == "least_stale":
            heap = self.current_heap if key in self.current else self.stale
            heapq.heappush(heap, (key[0], self.serial, key))
        else:
            priority = (self.counts[key], self.clock) if self.policy == "lfu" else (self.clock,)
            heapq.heappush(self.heap, (priority, self.serial, key))

    def _advance(self, cycle):
        if self.cycle == cycle:
            return
        if self.policy == "least_stale":
            previous = self.current
            self.current = set()
            self.current_heap = []
            for key in sorted(previous, key=lambda key: self.resident[key]):
                if key in self.resident:
                    self._push(key)
        self.cycle = cycle

    def _victims(self, size, protected, candidate):
        heaps = [self.stale, self.current_heap] if self.policy == "least_stale" else [self.heap]
        removed = []
        available = self.capacity - self.used
        for heap in heaps:
            while available < size and heap:
                item = heapq.heappop(heap)
                key, version = item[-1], item[-2]
                if self.resident.get(key) != version:
                    continue
                removed.append((heap, item))
                if key in protected:
                    continue
                if self.policy == "lfu" and self.counts[key] > self.counts[candidate]:
                    break
                available += self.layers[key[0]].size
            if available >= size:
                break
        victims = [item[-1] for _, item in removed if item[-1] not in protected]
        if available < size:
            victims = []
        victim_set = set(victims)
        for heap, item in removed:
            if item[-1] not in victim_set:
                heapq.heappush(heap, item)
        return victims

    def update(self, event):
        self._advance((event.request, event.token))
        self.clock += 1
        keys = event.keys()
        protected = set(keys)
        hits = sum(key in self.resident for key in keys)
        for key in keys:
            self.counts[key] += 1
            if key in self.resident:
                if self.policy == "least_stale":
                    self.current.add(key)
                self._push(key)
        uploads = evictions = 0
        for key in keys:
            size = self.layers[key[0]].size
            if key in self.resident or self.counts[key] < self.admit_after or size > self.capacity:
                continue
            victims = self._victims(size, protected, key)
            if self.capacity - self.used + sum(self.layers[v[0]].size for v in victims) < size:
                continue
            for victim in victims:
                del self.resident[victim]
                self.current.discard(victim)
                self.used -= self.layers[victim[0]].size
            evictions += len(victims)
            self.used += size
            if self.policy == "least_stale":
                self.current.add(key)
            self._push(key)
            uploads += size
        return hits, uploads, evictions


def run(data, capacity, admit_after):
    if not all(request.verified for request in data.requests.values()):
        raise ValueError("All requests must have verified observer token equivalence")
    fixed = routing.hotset(routing.frequencies(data.events_for("train")), data.layers, capacity)
    caches = {name: Cache(name, data.layers, capacity, admit_after)
              for name in ("lru", "lfu", "least_stale")}
    names = ("train_static", *caches)
    totals = {name: {request.id: Counter() for request in data.requests.values() if request.split == "test"}
              for name in names}
    warmup = {name: Counter() for name in caches}
    initial = {}
    started = False
    for event in data.events:
        request = data.requests[event.request]
        layer = data.layers[event.layer]
        keys = event.keys()
        if request.split == "test" and not started:
            initial = {name: cache.used for name, cache in caches.items()}
            initial["train_static"] = sum(data.layers[key[0]].size for key in fixed)
            started = True
        outcomes = {"train_static": (sum(key in fixed for key in keys), 0, 0)}
        outcomes.update({name: cache.update(event) for name, cache in caches.items()})
        for name, (hits, uploads, evictions) in outcomes.items():
            if request.split == "train":
                if name in warmup:
                    warmup[name].update(upload_bytes=uploads, evictions=evictions)
                continue
            total = totals[name][request.id]
            total.update(expert_calls=len(keys), hits=hits, total_bytes=len(keys) * layer.size,
                         hit_bytes=hits * layer.size, upload_bytes=uploads, evictions=evictions)
    results = {}
    for name in names:
        combined = sum(totals[name].values(), Counter())
        results[name] = {
            **dict(combined),
            "hit_fraction": combined["hits"] / combined["expert_calls"],
            "hit_byte_fraction": combined["hit_bytes"] / combined["total_bytes"],
            "initial_resident_bytes": initial[name],
            "warmup": dict(warmup.get(name, {})),
            "requests": {request: {"category": data.requests[request].category, **dict(values),
                                    "hit_fraction": values["hits"] / values["expert_calls"]}
                         for request, values in totals[name].items()},
        }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--cache-mib", type=int, action="append", required=True)
    parser.add_argument("--admit-after", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.admit_after < 1 or any(size < 0 for size in args.cache_mib):
        parser.error("cache sizes must be nonnegative and admission threshold positive")
    try:
        data = routing.read_trace(args.trace)
        report = {
            "schema_version": 1,
            "trace_sha256": data.sha256,
            "replay_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "routing_source_sha256": routing.ANALYSIS_SOURCE_SHA256,
            "model": Path(data.metadata.get("model", "unknown")).name,
            "architecture": data.metadata["architecture"],
            "train_tokens": data.tokens_for("train"),
            "test_tokens": data.tokens_for("test"),
            "admit_after": args.admit_after,
            "scope": "global hypothetical byte cache, CPU misses, end-of-layer admissions, no prefetch or prefill",
            "policy_note": "Least-stale prioritizes stale before current, then layer id and queue-entry order; tie semantics are an implementation choice",
            "results": {str(size): run(data, size * 1024**2, args.admit_after) for size in args.cache_mib},
        }
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
    except (OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(args.output)


if __name__ == "__main__":
    main()
