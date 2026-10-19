#!/usr/bin/env python3
"""Compare causal prompt-conditioned seeds on fixed recorded routing, not runtime speed."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


SPEC = importlib.util.spec_from_file_location(
    "prefill_seed_routing", Path(__file__).with_name("vrambridge-routing.py"))
assert SPEC and SPEC.loader
routing = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = routing
SPEC.loader.exec_module(routing)


def read_prefill(path: Path, data) -> dict:
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("Prefill observations exceed 64 MiB.")
    source = json.loads(path.read_text())
    if not isinstance(source, dict):
        raise ValueError("Prefill observations must be an object.")
    if (type(source.get("schema_version")) is not int or source.get("schema_version") != 1
            or source.get("instrumented") is not True or source.get("complete") is not True
            or source.get("capture_scope") != "observed_prefill_expert_rows"
            or source.get("performance_measurement") is not False
            or source.get("context_tokens") != data.metadata["context_tokens"]
            or source.get("model") != data.metadata.get("model")
            or source.get("architecture") != data.metadata.get("architecture")
            or source.get("backend_commit") != data.metadata["backend_commit"]
            or source.get("collector_source_sha256") != data.metadata["collector_source_sha256"]):
        raise ValueError("Prefill/decode provenance or completeness mismatch.")
    requests = source.get("requests")
    if not isinstance(requests, list) or len(requests) != len(data.requests):
        raise ValueError("Prefill request count mismatch.")
    result = {}
    for request in requests:
        identifier = request["id"]
        if identifier not in data.requests or identifier in result:
            raise ValueError("Duplicate or unknown prefill request.")
        expected = data.requests[identifier]
        if (request["split"] != expected.split or request["category"] != expected.category
                or routing.integer(request["prompt_tokens"], "prefill prompt tokens", 1) != expected.prompt_tokens
                or request.get("token_sequence_equivalent") is not True):
            raise ValueError("Prefill request metadata or observer control mismatch.")
        counts, rows = Counter(), Counter()
        for batch in request["batches"]:
            layer = routing.integer(batch["layer"], "prefill layer")
            if layer not in data.layers:
                raise ValueError("Unknown prefill layer.")
            descriptor = data.layers[layer]
            count = routing.integer(batch["rows"], "prefill rows")
            histogram = batch["expert_row_counts"]
            if (routing.integer(batch["active_experts_per_row"], "prefill active experts", 1) != descriptor.active
                    or len(histogram) != descriptor.experts):
                raise ValueError("Prefill histogram shape mismatch.")
            for expert, value in enumerate(histogram):
                routing.integer(value, "prefill expert rows")
                if value > count:
                    raise ValueError("Expert count exceeds observed rows.")
                counts[layer, expert] += value
            if sum(histogram) != count * descriptor.active:
                raise ValueError("Prefill histogram does not account for every selection.")
            rows[layer] += count
        coverage = {}
        for item in request["coverage"]:
            layer = item["layer"]
            if (type(layer) is not int or layer not in data.layers or layer in coverage
                    or type(item["covers_all_prompt_rows"]) is not bool
                    or routing.integer(item["observed_rows"], "observed prefill rows", 1) != rows[layer]
                    or rows[layer] <= 0
                    or rows[layer] > expected.prompt_tokens
                    or item["covers_all_prompt_rows"] != (rows[layer] == expected.prompt_tokens)):
                raise ValueError("Prefill coverage inconsistent with actual rows.")
            coverage[layer] = item["covers_all_prompt_rows"]
        if set(coverage) != set(data.layers):
            raise ValueError("Missing prefill layer coverage.")
        result[identifier] = {"counts": counts, "rows": rows, "coverage": coverage}
    return result


def read_plan(path: Path, data) -> set:
    if path.stat().st_size > 1024 * 1024:
        raise ValueError("Baseline plan exceeds 1 MiB.")
    selected = set()
    for line in path.read_text().splitlines():
        values = line.split()
        if len(values) != 2:
            raise ValueError("Baseline plan needs exactly two integers per line.")
        layer, expert = map(int, values)
        key = layer, expert
        if (layer not in data.layers or expert < 0 or expert >= data.layers[layer].experts
                or key in selected):
            raise ValueError("Baseline plan has duplicate or out-of-domain entries.")
        selected.add(key)
    if not selected:
        raise ValueError("Baseline plan is empty.")
    return selected


def make_seeds(data, prefill: dict, baseline: set) -> dict:
    tokens = data.tokens_for("train")
    prior = routing.frequencies(data.events_for("train"))
    capacity = sum(data.layers[layer].size for layer, _ in baseline)
    prior = Counter({key: value / tokens for key, value in prior.items()})
    result = {}
    for identifier in data.requests:
        observation = prefill[identifier]
        probabilities = Counter()
        for layer, descriptor in data.layers.items():
            for expert in range(descriptor.experts):
                key = layer, expert
                probabilities[key] = (observation["counts"][key] / observation["rows"][layer]
                                      if observation["coverage"][layer] else prior[key])
        blended = Counter({key: 0.5 * (value + prior[key]) for key, value in probabilities.items()})
        result[identifier] = {
            "trained_external": set(baseline),
            "train_only": routing.hotset(prior, data.layers, capacity),
            "prompt_only": routing.hotset(probabilities, data.layers, capacity),
            "prompt_train_half": routing.hotset(blended, data.layers, capacity),
            "empty": set(),
        }
    return result


def replay(data, events: list, seed: set, baseline: set, capacity: int, costs, adaptive: bool) -> dict:
    cache = routing.Cache("cost_aware", data.layers, capacity, 3, costs=costs, payback_tokens=64)
    cache.resident = {key: 0 for key in seed}
    cache.used = sum(data.layers[layer].size for layer, _ in seed)
    initial_bytes = sum(data.layers[layer].size for layer, _ in seed - baseline)
    initial_serial, initial_overlap, _ = costs.latency(0, 0, initial_bytes)
    totals = routing.Totals()
    for index, event in enumerate(events):
        descriptor = data.layers[event.layer]
        misses = sum(key not in cache.resident for key in event.keys())
        uploads, evictions = cache.update(event.keys(), index) if adaptive else (0, 0)
        totals.add(event, descriptor, misses * descriptor.size,
                   descriptor.active - misses, uploads, evictions, costs)
    return {
        "seed_experts": len(seed), "seed_bytes": sum(data.layers[layer].size for layer, _ in seed),
        "retained_baseline_experts": len(seed & baseline), "seed_change_upload_bytes": initial_bytes,
        "expert_calls": totals.calls, "hit_fraction": totals.full_hits / totals.calls,
        "cold_cpu_bytes": totals.cpu_bytes, "decode_admission_bytes": totals.upload_bytes,
        "evictions": totals.evictions,
        "modeled_serial_expert_ms": (initial_serial + totals.serial) * 1000,
        "modeled_overlap_expert_ms": (initial_overlap + totals.optimistic) * 1000,
    }


def analyze(data, prefill: dict, baseline: set, costs) -> dict:
    costs.validate()
    capacity = sum(data.layers[layer].size for layer, _ in baseline)
    seeds = make_seeds(data, prefill, baseline)
    requests = []
    for identifier, request in data.requests.items():
        if request.split != "test":
            continue
        events = [event for event in data.events if event.request == identifier]
        policies = {}
        for adaptive in (False, True):
            policies["cost_aware_replay" if adaptive else "static"] = {
                name: replay(data, events, seed, baseline, capacity, costs, adaptive)
                for name, seed in seeds[identifier].items()}
        requests.append({"id": identifier, "category": request.category, "decode_tokens": request.tokens,
                         "partial_prefill_layers_using_training_prior": [
                             layer for layer, full in prefill[identifier]["coverage"].items() if not full],
                         "policies": policies})
    return {
        "schema_version": 1, "study": "prompt_conditioned_residency_fixed_route_replay",
        "full_model_acceleration_claim": False, "performance_measurement": False,
        "routing_modified": False, "heldout_decode_used_to_select_seeds": False,
        "blend_weight": 0.5, "capacity_bytes": capacity,
        "train_tokens": data.tokens_for("train"), "test_tokens": data.tokens_for("test"),
        "trace_sha256": data.sha256, "model": data.metadata.get("model"),
        "costs": {"ram_gbps": costs.ram_gbps, "pcie_gbps": costs.pcie_gbps,
                  "gpu_expert_us": costs.gpu_expert_us, "staging_ram_passes": costs.staging_ram_passes},
        "limitations": [
            "Recorded native CPU-reference routes are held fixed; placement-dependent numerical trajectories are not qualified.",
            "Cost-aware admission is the existing offline heuristic, not an emulation of the installed native adaptive policy.",
            "Initial trained-bank loading is common/excluded; replacement seed uploads are charged before decode.",
            "Prefill collection, policy computation, GPU synchronization and invariant model work are not measured.",
            "Expert-stage modeled milliseconds are not request time or generation throughput.",
            "Partial/pruned prefill layers use training-only priors, never fabricated full-row observations.",
        ],
        "requests": requests,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--prefill", type=Path, required=True)
    parser.add_argument("--baseline-plan", type=Path, required=True)
    parser.add_argument("--ram-gbps", type=float, required=True)
    parser.add_argument("--pcie-gbps", type=float, required=True)
    parser.add_argument("--gpu-expert-us", type=float, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists() or Path(str(args.output) + ".part").exists():
            raise ValueError("Refusing to overwrite report or partial.")
        data = routing.read_trace(args.trace)
        prefill = read_prefill(args.prefill, data)
        baseline = read_plan(args.baseline_plan, data)
        costs = routing.Costs(args.ram_gbps, args.pcie_gbps, args.gpu_expert_us, 0, 0)
        report = analyze(data, prefill, baseline, costs)
        report["prefill_sha256"] = hashlib.sha256(args.prefill.read_bytes()).hexdigest()
        report["baseline_plan_sha256"] = hashlib.sha256(args.baseline_plan.read_bytes()).hexdigest()
        report["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        temporary = Path(str(args.output) + ".part")
        with temporary.open("x") as stream:
            stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
        args.output.hardlink_to(temporary)
        temporary.unlink()
        print(args.output)
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"vrambridge-prefill-seed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
