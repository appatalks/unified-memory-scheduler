#!/usr/bin/env python3

import copy
from functools import lru_cache
import importlib.util
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from collections import defaultdict, deque


SCRIPT = Path(sys.argv.pop(1)).resolve()
spec = importlib.util.spec_from_file_location("routing_analysis", SCRIPT)
assert spec is not None and spec.loader is not None
routing = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = routing
spec.loader.exec_module(routing)


def records(layers=None, training=None, testing=None):
    layers = layers or [
        {"id": 0, "experts": 3, "active": 1, "bytes_per_expert": 10, "static_host_bytes_per_expert": 0},
        {"id": 1, "experts": 3, "active": 1, "bytes_per_expert": 10, "static_host_bytes_per_expert": 10},
    ]
    training = training or [(0, 0), (0, 1), (1, 0)]
    testing = testing or [(0, 2), (1, 0), (2, 1)]
    result = [{
        "type": "metadata", "schema_version": 1, "instrumented": True, "capture_scope": "decode_only",
        "context_tokens": 256, "backend_commit": "test", "architecture": "test-moe", "model": "test.gguf",
        "static_device_expert_bytes": sum((layer["bytes_per_expert"] - layer["static_host_bytes_per_expert"])
                                         * layer["experts"] for layer in layers),
        "layers": layers,
    }]
    for split, sequence in (("train", training), ("test", testing)):
        result.append({"type": "request_start", "id": split, "split": split, "category": "reasoning",
                       "prompt_tokens": 4, "max_tokens": len(sequence)})
        for token, selected in enumerate(sequence):
            for layer, expert in zip(layers, selected):
                result.append({"type": "route", "request": split, "phase": "decode", "token": token,
                               "layer": layer["id"], "experts": expert if isinstance(expert, list) else [expert]})
        result.append({"type": "request_end", "id": split, "decode_tokens": len(sequence), "stop_reason": "limit",
                       "token_sequence_equivalent": True})
    result.append({"type": "trace_end", "requests": 2, "complete": True})
    return result


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.trace = self.root / "trace.jsonl"

    def tearDown(self):
        self.temporary.cleanup()

    def read(self, values=None):
        values = records() if values is None else values
        self.trace.write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")
        return routing.read_trace(self.trace)

    def test_complete_trace_counts_and_sampling_scope(self):
        data = self.read()
        self.assertEqual(data.tokens_for("train"), 3)
        self.assertEqual(data.tokens_for("test"), 3)
        self.assertEqual(len(data.events), 12)
        self.assertEqual(len(data.sha256), 64)

    def test_rejects_incomplete_or_reordered_trace(self):
        original = records()
        bad = [original[:-1], original + [original[-1]], original[1:]]
        changed = copy.deepcopy(original)
        del changed[3]
        bad.append(changed)
        changed = copy.deepcopy(original)
        changed[2], changed[3] = changed[3], changed[2]
        bad.append(changed)
        for values in bad:
            with self.subTest(values=values[:2]), self.assertRaises(ValueError):
                self.read(values)

    def test_variable_layer_ids_expert_counts_and_partial_residency(self):
        layers = [
            {"id": 2, "experts": 4, "active": 2, "bytes_per_expert": 12, "static_host_bytes_per_expert": 6},
            {"id": 7, "experts": 3, "active": 1, "bytes_per_expert": 20, "static_host_bytes_per_expert": 20},
        ]
        data = self.read(records(layers, training=[([0, 1], 0), ([1, 2], 1)], testing=[([1, 3], 2)]))
        report = routing.analyze(data, routing.Costs(1, 1))
        measured = report["policies"]["measured_static"]
        self.assertEqual(measured["expert_calls"], 3)
        self.assertEqual(measured["cold_cpu_bytes"], 32)
        self.assertAlmostEqual(measured["gpu_byte_fraction"], 12 / 44)

    def test_rejects_invalid_shape_experts_and_boolean_numbers(self):
        for changes in ({"experts": [True]}, {"experts": [-1]}, {"experts": [3]}, {"experts": [0, 0]},
                        {"token": True}, {"phase": "prefill"}, {"request": "unknown"}, {"layer": 99}):
            values = records()
            values[2].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.read(values)
        values = records()
        values[0]["schema_version"] = True
        with self.assertRaises(ValueError):
            self.read(values)

    def test_rejects_inconsistent_metadata_and_stop_shape(self):
        for mutate in (
            lambda values: values[0].update(static_device_expert_bytes=29),
            lambda values: values[0]["layers"][0].update(static_host_bytes_per_expert=11),
            lambda values: values[0]["layers"].append(values[0]["layers"][0]),
            lambda values: values[8].update(decode_tokens=2),
            lambda values: values[8].update(stop_reason="eos"),
            lambda values: values[8].update(token_sequence_equivalent=False),
            lambda values: values[8].update(token_sequence_equivalent="true"),
            lambda values: values[-1].update(complete=False),
        ):
            values = records()
            mutate(values)
            with self.assertRaises(ValueError):
                self.read(values)

    def test_requires_training_before_heldout(self):
        values = records()
        for value in values:
            if value["type"] == "request_start":
                value["split"] = "test"
        with self.assertRaises(ValueError):
            self.read(values)

    def test_cost_model_counts_shared_ram_and_staging(self):
        costs = routing.Costs(2, 1, staging_ram_passes=2)
        serial, overlap, shared = costs.latency(100, 0, 200)
        self.assertEqual(shared, 700)
        self.assertAlmostEqual(serial, 450e-9)
        self.assertAlmostEqual(overlap, 350e-9)
        self.assertLessEqual(overlap, serial)

    def test_gpu_and_sync_costs_are_not_free_when_supplied(self):
        costs = routing.Costs(1, 1, gpu_expert_us=10, layer_sync_us=2)
        serial, overlap, _ = costs.latency(10000, 1, 0)
        self.assertAlmostEqual(serial, 22e-6)
        self.assertAlmostEqual(overlap, 12e-6)

    def test_invalid_costs_are_explicit(self):
        for costs in (routing.Costs(0, 1), routing.Costs(float("nan"), 1),
                      routing.Costs(1, float("inf")), routing.Costs(True, 1),
                      routing.Costs(1, 1, gpu_expert_us=-1), routing.Costs(1, 1, staging_ram_passes=1),
                      routing.Costs(1, 1, staging_ram_passes=False)):
            with self.assertRaises(ValueError):
                costs.validate()

    def test_calibration_reproduces_measured_static_and_accounts_partial_components(self):
        values = records()
        values[0]["layers"][0]["static_host_bytes_per_expert"] = 5
        values[0]["static_device_expert_bytes"] = 15
        data = self.read(values)
        costs = routing.Costs(1e-7, 1e-7, baseline_tok_s=5)
        report = routing.analyze(data, costs)
        result = report["policies"]["measured_static"]
        self.assertAlmostEqual(result["optimistic_overlap_estimated_tok_s"], 5)
        self.assertEqual(result["cold_cpu_bytes"], 45)
        self.assertEqual(result["full_expert_hits"], 0)
        self.assertAlmostEqual(result["gpu_byte_fraction"], 0.25)
        self.assertFalse(report["actual_performance_claim"])
        self.assertFalse(report["production_target_met"])
        self.assertTrue(report["observation_equivalence"]["all_requests_verified"])
        self.assertFalse(report["coverage"]["required_categories_present"])
        self.assertFalse(report["trained_residency_plan"]["live_executor_implemented"])
        self.assertLessEqual(report["trained_residency_plan"]["used_bytes"],
                             report["trained_residency_plan"]["capacity_bytes"])

    def test_inconsistent_calibration_is_rejected(self):
        data = self.read()
        with self.assertRaisesRegex(ValueError, "calibration"):
            routing.analyze(data, routing.Costs(1e-7, 1e-7, baseline_tok_s=1000))

    def test_train_hotset_does_not_peek_at_test_frequency(self):
        layers = {0: routing.Layer(3, 1, 10, 10)}
        selected = routing.hotset(routing.Counter({(0, 0): 10, (0, 1): 2}), layers, 10)
        self.assertEqual(selected, {(0, 0)})
        self.assertNotIn((0, 2), selected)

    def test_variable_size_caches_never_exceed_byte_budget(self):
        layers = {0: routing.Layer(3, 1, 10, 10), 1: routing.Layer(3, 1, 25, 25)}
        for policy in ("lru", "lfu"):
            cache = routing.Cache(policy, layers, 30, 1)
            for index, key in enumerate(((0, 0), (1, 0), (0, 1), (1, 1), (0, 2))):
                uploads, _ = cache.update([key], index)
                self.assertLessEqual(cache.used, 30)
                self.assertGreaterEqual(uploads, 0)
                self.assertEqual(cache.used, sum(layers[item[0]].size for item in cache.resident))

    def test_repeat_admission_and_layer_end_commit(self):
        layers = {0: routing.Layer(3, 2, 10, 10)}
        cache = routing.Cache("lru", layers, 10, 2)
        self.assertEqual(cache.update([(0, 0)], 0)[0], 0)
        self.assertEqual(cache.update([(0, 0)], 1)[0], 10)
        old = set(cache.resident)
        uploads, _ = cache.update([(0, 1), (0, 0)], 2)
        self.assertEqual(uploads, sum(layers[key[0]].size for key in set(cache.resident) - old))

    def test_cost_admission_bypasses_experts_that_cannot_repay_upload(self):
        layers = {0: routing.Layer(3, 1, 10, 10)}
        costs = routing.Costs(1, 0.001)
        cache = routing.Cache("cost_aware", layers, 10, 2, costs=costs, payback_tokens=4)
        for index in range(8):
            self.assertEqual(cache.update([(0, 0)], index), (0, 0))
        self.assertEqual(cache.used, 0)
        self.assertGreater(cache.rejected_admissions, 0)

    def test_cost_admission_charges_copies_and_prevents_unprofitable_eviction(self):
        layers = {0: routing.Layer(3, 1, 10, 10)}
        costs = routing.Costs(1, 1, staging_ram_passes=2)
        cache = routing.Cache("cost_aware", layers, 10, 2, costs=costs, payback_tokens=8)
        for index in range(3):
            self.assertEqual(cache.update([(0, 0)], index), (0, 0))
        self.assertEqual(cache.update([(0, 0)], 3), (10, 0))
        self.assertIn((0, 0), cache.resident)
        for index in range(4, 7):
            self.assertEqual(cache.update([(0, 1)], index), (0, 0))
        self.assertIn((0, 0), cache.resident)
        self.assertEqual(cache.used, 10)

    def test_cost_admission_is_causal_for_identical_prefixes(self):
        layers = {0: routing.Layer(3, 1, 10, 10)}
        costs = routing.Costs(1, 1)
        prefix = [0] * 5 + [1] * 5
        def replay_tail(tail):
            cache = routing.Cache("cost_aware", layers, 10, 2, costs=costs, payback_tokens=8)
            states = []
            for index, expert in enumerate(prefix + tail):
                copies = cache.update([(0, expert)], index)
                states.append((copies, set(cache.resident)))
            return states[:len(prefix)]
        self.assertEqual(replay_tail([0] * 8), replay_tail([2] * 8))

    def test_payback_window_and_costs_are_required(self):
        layers = {0: routing.Layer(2, 1, 10, 10)}
        with self.assertRaises(ValueError):
            routing.Cache("cost_aware", layers, 10, 2)
        with self.assertRaises(ValueError):
            routing.analyze(self.read(), routing.Costs(1, 1), payback_tokens=0)

    def test_unknown_policy_cannot_silently_become_lru(self):
        with self.assertRaisesRegex(ValueError, "Unknown residency"):
            routing.Cache("unknown", {0: routing.Layer(2, 1, 10, 10)}, 10, 1)

    def test_legacy_trace_never_claims_token_equivalence(self):
        values = records()
        for value in values:
            value.pop("token_sequence_equivalent", None)
        report = routing.analyze(self.read(values), routing.Costs(1, 1))
        self.assertFalse(report["observation_equivalence"]["all_requests_verified"])

    def test_belady_hits_match_small_exhaustive_oracle(self):
        groups = [{0, 1}, {1, 2}, {0, 2}, {1, 3}, {0, 3}, {2, 3}]
        slots = 2
        @lru_cache(None)
        def optimum(index, resident):
            if index == len(groups):
                return 0
            held = set(resident)
            candidates = held | groups[index]
            choices = (subset for size in range(min(slots, len(candidates)) + 1)
                       for subset in itertools.combinations(sorted(candidates), size))
            return len(held & groups[index]) + max(optimum(index + 1, tuple(choice)) for choice in choices)
        layers = {0: routing.Layer(4, 2, 10, 10)}
        future = defaultdict(deque)
        for index, group in enumerate(groups):
            for expert in group:
                future[(0, expert)].append(index)
        cache = routing.Cache("belady_hits", layers, slots * 10, 1, future)
        hits = 0
        for index, group in enumerate(groups):
            keys = [(0, expert) for expert in sorted(group)]
            hits += sum(key in cache.resident for key in keys)
            cache.update(keys, index)
        self.assertEqual(hits, optimum(0, ()))

    def test_fractional_oracle_is_labeled_and_variable_belady_is_not_claimed(self):
        values = records()
        values[0]["layers"][1]["bytes_per_expert"] = 20
        values[0]["layers"][1]["static_host_bytes_per_expert"] = 20
        data = self.read(values)
        report = routing.analyze(data, routing.Costs(1, 1))
        self.assertFalse(report["policies"]["belady_hits"]["supported"])
        self.assertTrue(report["fixed_residency_fractional_oracle"]["fractional_experts_not_implementable"])
        self.assertIn("fixed-residency", report["fixed_residency_fractional_oracle"]["scope"])

    def test_locality_histograms_account_for_every_test_reference(self):
        data = self.read()
        report = routing.analyze(data, routing.Costs(1, 1))
        expected = sum(len(event.experts) for event in data.events_for("test"))
        self.assertEqual(sum(report["locality"]["heldout_stack_distance_histogram"].values()), expected)
        self.assertEqual(sum(report["locality"]["heldout_token_reuse_gap_histogram"].values()), expected)

    def test_cold_cache_is_not_warmed_by_future_test_accesses(self):
        data = self.read()
        costs = routing.Costs(1, 1)
        cold = routing.replay(data, "lru", 30, costs, 0, 1, 60, cold_cache=True)
        self.assertEqual(cold["initial_resident_upload_bytes"], 0)
        self.assertGreater(cold["weight_upload_bytes"], 0)

    def test_cli_output_shape_and_no_overwrite(self):
        self.read()
        output = self.root / "report.json"
        command = [sys.executable, str(SCRIPT), str(self.trace), "--output", str(output),
                   "--ram-gbps", "42.16", "--pcie-gbps", "28.52"]
        completed = subprocess.run(command, capture_output=True, text=True, check=True)
        self.assertIn(str(output), completed.stdout)
        result = json.loads(output.read_text())
        self.assertEqual(set(result["policies"]),
                         {"measured_static", "train_static", "lru", "lfu", "belady_hits", "cost_aware"})
        self.assertFalse(result["production_target_met"])
        original = output.read_bytes()
        self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
        self.assertEqual(output.read_bytes(), original)

    def test_cli_invalid_trace_never_creates_success_report(self):
        self.trace.write_text("{}\n")
        output = self.root / "bad.json"
        result = subprocess.run([sys.executable, str(SCRIPT), str(self.trace), "--output", str(output),
                                 "--ram-gbps", "42.16", "--pcie-gbps", "28.52"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("Trace line", result.stderr)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
