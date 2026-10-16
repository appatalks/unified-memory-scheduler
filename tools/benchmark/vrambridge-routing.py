#!/usr/bin/env python3
"""Validate decode routing traces and replay explicit expert-residency cost models."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
import heapq
import hashlib
import json
import math
import os
from pathlib import Path
import sys


Expert = tuple[int, int]
ANALYSIS_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def integer(value: object, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


@dataclass(frozen=True)
class Layer:
    experts: int
    active: int
    size: int
    host_size: int


@dataclass
class Request:
    id: str
    split: str
    category: str
    maximum: int
    prompt_tokens: int
    tokens: int = 0
    verified: bool = False


@dataclass(frozen=True)
class Event:
    request: str
    token: int
    layer: int
    experts: tuple[int, ...]

    def keys(self) -> list[Expert]:
        return [(self.layer, expert) for expert in self.experts]


@dataclass
class Dataset:
    metadata: dict
    layers: dict[int, Layer]
    requests: dict[str, Request]
    events: list[Event]
    sha256: str

    def events_for(self, split: str) -> list[Event]:
        return [event for event in self.events if self.requests[event.request].split == split]

    def tokens_for(self, split: str) -> int:
        return sum(request.tokens for request in self.requests.values() if request.split == split)


def read_trace(path: Path) -> Dataset:
    metadata = None
    layers = {}
    requests = {}
    events = []
    current = None
    token = layer_index = 0
    ended = testing = False
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for number, raw in enumerate(stream, 1):
            digest.update(raw)
            if not raw.strip():
                continue
            try:
                if len(raw) > 2 * 1024 * 1024:
                    raise ValueError("trace record exceeds 2 MiB")
                item = json.loads(raw)
                if not isinstance(item, dict):
                    raise ValueError("trace records must be objects")
                kind = item.get("type")
                if ended:
                    raise ValueError("records follow trace_end")
                if metadata is None:
                    if kind != "metadata" or type(item.get("schema_version")) is not int or item["schema_version"] != 1:
                        raise ValueError("first record must be schema-version-1 metadata")
                    if item.get("instrumented") is not True or item.get("capture_scope") != "decode_only":
                        raise ValueError("a decode-only instrumented trace is required")
                    integer(item.get("context_tokens"), "context_tokens", 1)
                    text(item.get("backend_commit"), "backend_commit")
                    text(item.get("architecture"), "architecture")
                    if not isinstance(item.get("layers"), list) or not item["layers"]:
                        raise ValueError("metadata.layers must be nonempty")
                    for entry in item["layers"]:
                        identifier = integer(entry["id"], "layer id")
                        count = integer(entry["experts"], "expert count", 1)
                        active = integer(entry["active"], "active experts", 1)
                        size = integer(entry["bytes_per_expert"], "expert bytes", 1)
                        host = integer(entry["static_host_bytes_per_expert"], "host expert bytes")
                        if identifier in layers or active > count or host > size:
                            raise ValueError("invalid or duplicate layer metadata")
                        layers[identifier] = Layer(count, active, size, host)
                    expected = sum((layer.size - layer.host_size) * layer.experts
                                   for layer in layers.values())
                    if integer(item.get("static_device_expert_bytes"), "static device bytes") != expected:
                        raise ValueError("static device byte accounting does not match tensors")
                    metadata = item
                elif kind == "request_start":
                    if current is not None:
                        raise ValueError("nested requests")
                    identifier = text(item.get("id"), "request id")
                    split = item.get("split")
                    if identifier in requests or split not in ("train", "test") or (testing and split == "train"):
                        raise ValueError("duplicate request or invalid train/test ordering")
                    testing = testing or split == "test"
                    maximum = integer(item.get("max_tokens"), "max_tokens", 1)
                    prompt = integer(item.get("prompt_tokens"), "prompt_tokens", 1)
                    if maximum + prompt > metadata["context_tokens"]:
                        raise ValueError("request exceeds declared actual context")
                    current = Request(identifier, split, text(item.get("category"), "category"), maximum, prompt)
                    requests[identifier] = current
                    token = layer_index = 0
                elif kind == "route":
                    if current is None or item.get("request") != current.id or item.get("phase") != "decode":
                        raise ValueError("route is outside its decode request")
                    identifier = integer(item.get("layer"), "route layer")
                    if identifier != sorted(layers)[layer_index] or integer(item.get("token"), "route token") != token:
                        raise ValueError("missing, duplicate, or out-of-order layer/token observation")
                    experts = item.get("experts")
                    layer = layers[identifier]
                    if not isinstance(experts, list) or len(experts) != layer.active:
                        raise ValueError("wrong number of selected experts")
                    for expert in experts:
                        integer(expert, "expert id")
                    if len(set(experts)) != len(experts) or any(expert >= layer.experts for expert in experts):
                        raise ValueError("duplicate or out-of-range expert id")
                    events.append(Event(current.id, token, identifier, tuple(experts)))
                    layer_index += 1
                    if layer_index == len(layers):
                        token += 1
                        layer_index = 0
                elif kind == "request_end":
                    if current is None or item.get("id") != current.id or layer_index:
                        raise ValueError("request ends with incomplete routing")
                    count = integer(item.get("decode_tokens"), "decode_tokens", 1)
                    if count != token or count > current.maximum:
                        raise ValueError("request token count does not match complete layer observations")
                    reason = item.get("stop_reason")
                    if (reason not in ("limit", "eos") or (reason == "limit" and count != current.maximum)
                            or (reason == "eos" and count == current.maximum)):
                        raise ValueError("invalid request stop reason or token limit")
                    if "token_sequence_equivalent" in item and item["token_sequence_equivalent"] is not True:
                        raise ValueError("routing observer token-equivalence check failed")
                    current.tokens = count
                    current.verified = item.get("token_sequence_equivalent") is True
                    current = None
                elif kind == "trace_end":
                    if current is not None or item.get("complete") is not True:
                        raise ValueError("trace is incomplete")
                    if integer(item.get("requests"), "request count", 1) != len(requests):
                        raise ValueError("trace request count mismatch")
                    ended = True
                else:
                    raise ValueError(f"unknown record type: {kind}")
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"Trace line {number}: {exc}") from exc
    if metadata is None or not ended or {request.split for request in requests.values()} != {"train", "test"}:
        raise ValueError("Trace must end completely and contain train and held-out test requests")
    return Dataset(metadata, layers, requests, events, digest.hexdigest())


@dataclass(frozen=True)
class Costs:
    ram_gbps: float
    pcie_gbps: float
    gpu_expert_us: float | None = None
    layer_sync_us: float | None = None
    staging_ram_passes: int = 2
    baseline_tok_s: float | None = None

    def validate(self) -> None:
        for name, value in (("RAM bandwidth", self.ram_gbps), ("PCIe bandwidth", self.pcie_gbps)):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name, value in (("GPU expert cost", self.gpu_expert_us), ("layer synchronization", self.layer_sync_us)):
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.baseline_tok_s is not None and (type(self.baseline_tok_s) not in (int, float)
                                              or not math.isfinite(self.baseline_tok_s) or self.baseline_tok_s <= 0):
            raise ValueError("Baseline throughput must be finite and positive")
        if type(self.staging_ram_passes) is not int or self.staging_ram_passes not in (0, 2):
            raise ValueError("Staging RAM passes must be 0 (already pinned) or 2 (one read+write copy)")

    def latency(self, cpu_bytes: float, gpu_experts: float, upload_bytes: int) -> tuple[float, float, float]:
        ram = self.ram_gbps * 1e9
        pcie = self.pcie_gbps * 1e9
        gpu = gpu_experts * (self.gpu_expert_us or 0) / 1e6
        sync = (self.layer_sync_us or 0) / 1e6
        staging = self.staging_ram_passes * upload_bytes / ram
        dma = max(upload_bytes / pcie, upload_bytes / ram)
        serial = cpu_bytes / ram + gpu + staging + dma + sync
        shared_ram = cpu_bytes + (self.staging_ram_passes + 1) * upload_bytes
        optimistic = max(shared_ram / ram, upload_bytes / pcie, gpu) + sync
        return serial, optimistic, shared_ram


def frequencies(events: list[Event]) -> Counter:
    result = Counter()
    for event in events:
        result.update(event.keys())
    return result


def hotset(counts: Counter, layers: dict[int, Layer], capacity: int) -> set[Expert]:
    selected = set()
    remaining = capacity
    for key in sorted(counts, key=lambda key: (-counts[key], layers[key[0]].size, key)):
        size = layers[key[0]].size
        if size <= remaining:
            selected.add(key)
            remaining -= size
    return selected


class Cache:
    def __init__(self, policy: str, layers: dict[int, Layer], capacity: int, admit_after: int,
                 future: dict[Expert, deque[int]] | None = None, costs: Costs | None = None,
                 payback_tokens: int = 64):
        if policy not in ("measured_static", "train_static", "lru", "lfu", "belady_hits", "cost_aware"):
            raise ValueError(f"Unknown residency policy: {policy}")
        integer(capacity, "cache capacity")
        integer(admit_after, "admission threshold", 1)
        integer(payback_tokens, "payback window", 1)
        if policy == "cost_aware" and costs is None:
            raise ValueError("Cost-aware admission requires explicit hardware costs")
        self.policy = policy
        self.layers = layers
        self.capacity = capacity
        self.admit_after = admit_after
        self.future = future
        self.resident: dict[Expert, int] = {}
        self.counts = Counter()
        self.used = 0
        self.clock = 0
        self.costs = costs
        self.window_events = payback_tokens * len(layers)
        self.recent: Counter = Counter()
        self.history: deque[tuple[int, list[Expert]]] = deque()
        self.rejected_admissions = 0

    def _benefit(self, key: Expert) -> float:
        if self.costs is None:
            raise ValueError("Missing admission costs")
        saved = self.layers[key[0]].size / (self.costs.ram_gbps * 1e9)
        extra_gpu = (self.costs.gpu_expert_us or 0) / 1e6
        return self.recent[key] * max(0.0, saved - extra_gpu)

    def _cost_admit(self, keys: list[Expert], index: int) -> tuple[int, int]:
        assert self.costs is not None
        self.history.append((index, keys))
        self.recent.update(keys)
        while self.history and self.history[0][0] <= index - self.window_events:
            _, expired = self.history.popleft()
            self.recent.subtract(expired)
            for key in expired:
                if self.recent[key] <= 0:
                    del self.recent[key]
        uploads = evictions = 0
        benefits = {key: self._benefit(key) for key in self.resident.keys() | set(keys)}
        def priority(key: Expert) -> tuple:
            return benefits[key] / self.layers[key[0]].size, key
        eviction_order = [priority(key) for key in self.resident]
        heapq.heapify(eviction_order)
        candidates = sorted(set(keys) - self.resident.keys(),
                            key=lambda key: (-priority(key)[0], key))
        for key in candidates:
            size = self.layers[key[0]].size
            if self.recent[key] < self.admit_after or size > self.capacity:
                continue
            available = self.capacity - self.used
            victims = []
            lost = 0.0
            while available < size and eviction_order:
                _, victim = heapq.heappop(eviction_order)
                available += self.layers[victim[0]].size
                lost += benefits[victim]
                victims.append(victim)
            upload_seconds = self.costs.latency(0, 0, size)[0] - (self.costs.layer_sync_us or 0) / 1e6
            if available < size or benefits[key] <= upload_seconds + lost:
                for victim in victims:
                    heapq.heappush(eviction_order, priority(victim))
                self.rejected_admissions += 1
                continue
            for victim in victims:
                del self.resident[victim]
                self.used -= self.layers[victim[0]].size
            self.resident[key] = self.clock
            heapq.heappush(eviction_order, priority(key))
            self.used += size
            uploads += size
            evictions += len(victims)
        return uploads, evictions

    def update(self, keys: list[Expert], index: int) -> tuple[int, int]:
        old = set(self.resident)
        self.clock += 1
        for key in keys:
            self.counts[key] += 1
            if key in self.resident:
                self.resident[key] = self.clock
        if self.policy == "cost_aware":
            uploads, evictions = self._cost_admit(keys, index)
        elif self.policy == "belady_hits":
            if self.future is None:
                raise ValueError("Belady replay requires future-use queues")
            for key in keys:
                if not self.future[key] or self.future[key].popleft() != index:
                    raise ValueError("Future-use queue is inconsistent")
            candidates = old | set(keys)
            live = [key for key in candidates if self.future[key]]
            size = next(iter(self.layers.values())).size
            keep = sorted(live, key=lambda key: (self.future[key][0], key))[:self.capacity // size]
            self.resident = {key: self.clock for key in keep}
            self.used = len(keep) * size
        else:
            for key in keys:
                size = self.layers[key[0]].size
                if key in self.resident or self.counts[key] < self.admit_after or size > self.capacity:
                    continue
                def order(candidate: Expert) -> tuple:
                    if self.policy == "lfu":
                        return self.counts[candidate], self.resident[candidate], candidate
                    return self.resident[candidate], candidate
                victims = []
                available = self.capacity - self.used
                for candidate in sorted(self.resident, key=order):
                    if available >= size:
                        break
                    if self.policy == "lfu" and self.counts[candidate] > self.counts[key]:
                        break
                    victims.append(candidate)
                    available += self.layers[candidate[0]].size
                if available < size:
                    continue
                for victim in victims:
                    del self.resident[victim]
                    self.used -= self.layers[victim[0]].size
                self.resident[key] = self.clock
                self.used += size
        if self.used > self.capacity or self.used != sum(self.layers[key[0]].size for key in self.resident):
            raise ValueError("Cache byte-capacity invariant failed")
        if self.policy == "cost_aware":
            # Every admitted copy is charged, even if replaced later in this layer.
            return uploads, evictions
        added = set(self.resident) - old
        return sum(self.layers[key[0]].size for key in added), len(old - set(self.resident))


@dataclass
class Totals:
    calls: int = 0
    full_hits: int = 0
    bytes: int = 0
    cpu_bytes: int = 0
    upload_bytes: int = 0
    evictions: int = 0
    serial: float = 0
    optimistic: float = 0
    shared_ram: float = 0

    def add(self, event: Event, layer: Layer, cpu_bytes: int, full_hits: int, uploads: int,
            evictions: int, costs: Costs) -> None:
        total = len(event.experts) * layer.size
        serial, optimistic, shared_ram = costs.latency(cpu_bytes, (total - cpu_bytes) / layer.size, uploads)
        self.calls += len(event.experts)
        self.full_hits += full_hits
        self.bytes += total
        self.cpu_bytes += cpu_bytes
        self.upload_bytes += uploads
        self.evictions += evictions
        self.serial += serial
        self.optimistic += optimistic
        self.shared_ram += shared_ram

    def result(self, tokens: int, overhead: float, costs: Costs, target: float) -> dict:
        seconds_serial = self.serial + tokens * overhead
        seconds_overlap = self.optimistic + tokens * overhead
        bandwidth_seconds = max(self.shared_ram / (costs.ram_gbps * 1e9),
                                self.upload_bytes / (costs.pcie_gbps * 1e9))
        bound = tokens / bandwidth_seconds if bandwidth_seconds else None
        return {
            "tokens": tokens, "expert_calls": self.calls, "full_expert_hits": self.full_hits,
            "full_expert_hit_fraction": self.full_hits / self.calls if self.calls else 0,
            "gpu_byte_fraction": 1 - self.cpu_bytes / self.bytes if self.bytes else 0,
            "cold_cpu_bytes": self.cpu_bytes, "weight_upload_bytes": self.upload_bytes,
            "evictions": self.evictions, "shared_ram_traffic_bytes": self.shared_ram,
            "serial_estimated_seconds": seconds_serial,
            "optimistic_overlap_estimated_seconds": seconds_overlap,
            "serial_estimated_tok_s": tokens / seconds_serial if seconds_serial else None,
            "optimistic_overlap_estimated_tok_s": tokens / seconds_overlap if seconds_overlap else None,
            "remaining_nonmemory_ms_per_token_at_target": (1 / target - self.optimistic / tokens) * 1000,
            "policy_bandwidth_upper_bound_tok_s": bound,
            "bandwidth_condition": "below_target_under_supplied_inputs" if bound is not None and bound < target
                                   else "not_ruled_out_requires_measurement",
        }


def static_overhead(data: Dataset, costs: Costs) -> float:
    if costs.baseline_tok_s is None:
        return 0
    total = 0.0
    for event in data.events_for("train"):
        layer = data.layers[event.layer]
        cold = len(event.experts) * layer.host_size
        gpu = len(event.experts) * (layer.size - layer.host_size) / layer.size
        total += costs.latency(cold, gpu, 0)[1]
    residual = 1 / costs.baseline_tok_s - total / data.tokens_for("train")
    if residual < -1e-12:
        raise ValueError("Supplied component costs exceed measured baseline time; calibration is inconsistent")
    return max(0.0, residual)


def replay(data: Dataset, policy: str, capacity: int, costs: Costs, overhead: float,
           admit_after: int, target: float, cold_cache: bool = False, payback_tokens: int = 64) -> dict:
    train = data.events_for("train")
    training_frequency = frequencies(train)
    fixed = hotset(training_frequency, data.layers, capacity) if policy == "train_static" else set()
    future = defaultdict(deque)
    if policy == "belady_hits":
        if len({layer.size for layer in data.layers.values()}) != 1:
            return {"supported": False, "reason": "Exact hit-oriented Belady replay requires equal-size experts"}
        for index, event in enumerate(data.events):
            for key in event.keys():
                future[key].append(index)
    cache = Cache(policy, data.layers, capacity, admit_after, future, costs, payback_tokens)
    total = Totals()
    per_request = {request.id: Totals() for request in data.requests.values() if request.split == "test"}
    warmup_uploads = 0
    testing_started = False
    initial = 0
    warmup_rejections = 0
    for index, event in enumerate(data.events):
        request = data.requests[event.request]
        layer = data.layers[event.layer]
        keys = event.keys()
        if request.split == "test" and not testing_started:
            if cold_cache:
                cache.resident.clear()
                cache.used = 0
            initial = (data.metadata["static_device_expert_bytes"] if policy == "measured_static"
                       else sum(data.layers[key[0]].size for key in fixed) if policy == "train_static" else cache.used)
            warmup_rejections = cache.rejected_admissions
            testing_started = True
        if policy == "measured_static":
            cold = len(keys) * layer.host_size
            hits = len(keys) if layer.host_size == 0 else 0
            uploads = evictions = 0
        elif policy == "train_static":
            hits = sum(key in fixed for key in keys)
            cold = (len(keys) - hits) * layer.size
            uploads = evictions = 0
        else:
            hits = sum(key in cache.resident for key in keys)
            cold = (len(keys) - hits) * layer.size
            uploads, evictions = cache.update(keys, index)
        if request.split == "train":
            warmup_uploads += uploads
        else:
            total.add(event, layer, cold, hits, uploads, evictions, costs)
            per_request[request.id].add(event, layer, cold, hits, uploads, evictions, costs)
    tokens = data.tokens_for("test")
    result = total.result(tokens, overhead, costs, target)
    startup_serial, _, _ = costs.latency(0, 0, initial)
    result.update({
        "supported": True, "policy": policy, "cache_capacity_bytes": capacity,
        "warmup_cache_upload_bytes": warmup_uploads, "initial_resident_upload_bytes": initial,
        "initial_upload_estimated_seconds": startup_serial,
        "decode_plus_initial_cache_upload_estimated_tok_s":
            tokens / (result["serial_estimated_seconds"] + startup_serial)
            if result["serial_estimated_seconds"] + startup_serial else None,
        "future_aware": policy == "belady_hits",
        "rejected_cost_admissions": cache.rejected_admissions - warmup_rejections if policy == "cost_aware" else None,
        "warmup_rejected_cost_admissions": warmup_rejections if policy == "cost_aware" else None,
        "payback_tokens": payback_tokens if policy == "cost_aware" else None,
        "oracle_scope": "hit-oriented, equal-size, end-of-layer demand admission; not a latency optimum"
                        if policy == "belady_hits" else None,
        "initialization": "training-derived counters; cache cleared before test" if cold_cache
                          else "training replay or preloaded static set",
        "per_request": {
            request.id: {"category": request.category, **per_request[request.id].result(request.tokens, overhead, costs, target)}
            for request in data.requests.values() if request.split == "test"
        },
    })
    return result


def fractional_fixed_bound(data: Dataset, capacity: int, costs: Costs) -> dict:
    counts = frequencies(data.events_for("test"))
    remaining = float(capacity)
    saved = 0.0
    for key in sorted(counts, key=lambda key: (-counts[key], key)):
        size = data.layers[key[0]].size
        used = min(remaining, size)
        saved += counts[key] * used
        remaining -= used
        if remaining <= 0:
            break
    all_bytes = sum(count * data.layers[key[0]].size for key, count in counts.items())
    cold = max(0.0, all_bytes - saved)
    tokens = data.tokens_for("test")
    return {
        "scope": "future-aware fractional fixed-residency upper bound only; not an adaptive-cache optimum",
        "fractional_experts_not_implementable": True,
        "gpu_byte_fraction": saved / all_bytes,
        "cold_cpu_bytes": cold,
        "bandwidth_only_upper_bound_tok_s": tokens * costs.ram_gbps * 1e9 / cold if cold else None,
        "unbounded_by_ram_only": cold == 0,
        "omits_gpu_compute_and_startup": True,
    }


def locality(data: Dataset, capacity: int) -> dict:
    total_refs = sum(len(event.experts) for event in data.events)
    tree = [0] * (total_refs + 1)
    last = {}
    last_token = {}
    histogram = Counter()
    token_gaps = Counter()
    rank = 0
    distinct = 0
    token_position = -1
    first_layer = min(data.layers)

    def add(index: int, value: int) -> None:
        while index < len(tree):
            tree[index] += value
            index += index & -index

    def prefix(index: int) -> int:
        result = 0
        while index:
            result += tree[index]
            index -= index & -index
        return result

    for event in data.events:
        if event.layer == first_layer:
            token_position += 1
        testing = data.requests[event.request].split == "test"
        for key in event.keys():
            rank += 1
            previous = last.get(key)
            if previous is None:
                distance = None
                distinct += 1
            else:
                distance = distinct - prefix(previous)
                add(previous, -1)
            add(rank, 1)
            last[key] = rank
            if testing:
                label = ("cold" if distance is None else "0-31" if distance < 32 else "32-127" if distance < 128
                         else "128-511" if distance < 512 else "512-1023" if distance < 1024 else "1024+")
                histogram[label] += 1
                gap = token_position - last_token[key] if key in last_token else None
                gap_label = ("cold" if gap is None else "1" if gap == 1 else "2-4" if gap <= 4
                             else "5-16" if gap <= 16 else "17-64" if gap <= 64 else "65+")
                token_gaps[gap_label] += 1
            last_token[key] = token_position
    train = frequencies(data.events_for("train"))
    test = frequencies(data.events_for("test"))
    return {
        "heldout_stack_distance_histogram": dict(histogram),
        "heldout_token_reuse_gap_histogram": dict(token_gaps),
        "token_gap_definition": "generated decode steps across requests; excludes prefill and wall-clock gaps",
        "distance_definition": "distinct expert keys since previous reference; router-rank order within a layer",
        "training_unique_experts": len(train), "test_unique_experts": len(test),
        "top_training_experts": [
            {"layer": key[0], "expert": key[1], "calls": count}
            for key, count in train.most_common(20)
        ],
        "trained_hotset_experts": len(hotset(train, data.layers, capacity)),
    }


def analyze(data: Dataset, costs: Costs, capacity: int | None = None,
            admit_after: int = 2, target: float = 60, payback_tokens: int = 64) -> dict:
    costs.validate()
    capacity = data.metadata["static_device_expert_bytes"] if capacity is None else integer(capacity, "capacity")
    integer(admit_after, "admit_after", 1)
    integer(payback_tokens, "payback_tokens", 1)
    if type(target) not in (int, float) or not math.isfinite(target) or target <= 0:
        raise ValueError("Target throughput must be finite and positive")
    if capacity < data.metadata["static_device_expert_bytes"]:
        raise ValueError("Cache budget is below the measured static expert allocation; collect a matching baseline")
    overhead = static_overhead(data, costs)
    policies = {
        policy: replay(data, policy, capacity, costs, overhead, admit_after, target,
                       payback_tokens=payback_tokens)
        for policy in ("measured_static", "train_static", "lru", "lfu", "belady_hits", "cost_aware")
    }
    categories = {
        split: sorted({request.category for request in data.requests.values() if request.split == split})
        for split in ("train", "test")
    }
    trained = hotset(frequencies(data.events_for("train")), data.layers, capacity)
    return {
        "schema_version": 1, "trace_sha256": data.sha256, "model": data.metadata.get("model"),
        "architecture": data.metadata["architecture"], "backend_commit": data.metadata["backend_commit"],
        "collector_source_sha256": data.metadata.get("collector_source_sha256"),
        "collector_initialization": data.metadata.get("initialization"),
        "context_tokens": data.metadata["context_tokens"],
        "actual_performance_claim": False, "screening_only": True, "target_tok_s": target,
        "production_target_met": False,
        "observation_equivalence": {
            "all_requests_verified": all(request.verified for request in data.requests.values()),
            "method": data.metadata.get("token_equivalence_check"),
            "does_not_establish_answer_quality": True,
        },
        "counts": {"train_tokens": data.tokens_for("train"), "test_tokens": data.tokens_for("test"),
                   "layers": len(data.layers), "requests": len(data.requests)},
        "coverage": {**categories, "required_categories_present":
                     all({"coding", "reasoning", "prose"} <= set(values) for values in categories.values())},
        "cost_inputs": {
            "ram_gbps": costs.ram_gbps, "pcie_gbps": costs.pcie_gbps,
            "gpu_expert_us": costs.gpu_expert_us, "layer_sync_us": costs.layer_sync_us,
            "staging_ram_passes": costs.staging_ram_passes, "baseline_tok_s": costs.baseline_tok_s,
            "calibrated_fixed_seconds_per_token": overhead,
            "capacity_matches_measured": capacity == data.metadata["static_device_expert_bytes"],
            "cache_capacity_bytes": capacity,
            "payback_tokens": payback_tokens,
        },
        "locality": locality(data, capacity),
        "trained_residency_plan": {
            "source": "training references only",
            "live_executor_implemented": False,
            "capacity_bytes": capacity,
            "used_bytes": sum(data.layers[key[0]].size for key in trained),
            "layers": [{"layer": layer, "experts": sorted(key[1] for key in trained if key[0] == layer)}
                       for layer in sorted(data.layers)],
        },
        "policies": policies,
        "cold_cache_replay": {
            policy: replay(data, policy, capacity, costs, overhead, admit_after, target,
                           cold_cache=True, payback_tokens=payback_tokens)
            for policy in ("lru", "lfu", "belady_hits", "cost_aware")
        },
        "fixed_residency_fractional_oracle": fractional_fixed_bound(data, capacity, costs),
        "limitations": [
            "Instrumented collection time is not a throughput measurement.",
            "GPU expert and layer-sync costs are zero only as an optimistic assumption when not supplied.",
            "Baseline residual is calibrated on training accesses and assumed unchanged across policies.",
            "CPU dot-product cost beyond bandwidth and new hybrid dispatch overhead are not separately measured.",
            "Staging passes model RAM traffic; they are not a measured memcpy kernel time.",
            "Optimistic overlap assumes pipelining but includes the shared RAM bus for CPU and DMA traffic.",
            "Prefill cache effects, output quality, and real execution equivalence are not evaluated.",
            "Training cache warm-up uploads and static initialization are reported separately from held-out decode.",
            "Belady optimizes hits under restricted equal-size admission, not total latency or proactive prefetch.",
            "The fractional oracle bounds fixed residency only; it cannot rule out all adaptive execution algorithms.",
            "Cost-aware admission extrapolates recent use over an equal future window; it is a causal heuristic, not an oracle.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ram-gbps", type=float, required=True)
    parser.add_argument("--pcie-gbps", type=float, required=True)
    parser.add_argument("--baseline-tok-s", type=float)
    parser.add_argument("--gpu-expert-us", type=float)
    parser.add_argument("--layer-sync-us", type=float)
    parser.add_argument("--cache-mib", type=float)
    parser.add_argument("--admit-after", type=int, default=2)
    parser.add_argument("--staging-ram-passes", type=int, choices=(0, 2), default=2)
    parser.add_argument("--target-tok-s", type=float, default=60)
    parser.add_argument("--payback-tokens", type=int, default=64,
                        help="past generated-token window extrapolated for cost-aware admission (default: 64)")
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(args.output.suffix + ".tmp").exists():
        parser.error("Refusing to overwrite an existing report or temporary report.")
    capacity = None
    if args.cache_mib is not None:
        if not math.isfinite(args.cache_mib) or args.cache_mib < 0:
            parser.error("--cache-mib must be finite and nonnegative")
        capacity = int(args.cache_mib * 1024**2)
    try:
        data = read_trace(args.trace)
        costs = Costs(args.ram_gbps, args.pcie_gbps, args.gpu_expert_us, args.layer_sync_us,
                      args.staging_ram_passes, args.baseline_tok_s)
        result = analyze(data, costs, capacity, args.admit_after, args.target_tok_s, args.payback_tokens)
        result["analysis_source_sha256"] = ANALYSIS_SOURCE_SHA256
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
        os.link(temporary, args.output)
        temporary.unlink()
    except (OSError, ValueError) as exc:
        print(f"vrambridge-routing: {exc}", file=sys.stderr)
        return 2
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
