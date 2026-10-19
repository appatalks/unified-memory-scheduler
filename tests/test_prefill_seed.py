#!/usr/bin/env python3

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location(
    "prefill_seed_under_test",
    Path(__file__).resolve().parents[1] / "tools/benchmark/vrambridge-prefill-seed.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)
routing = module.routing


class PrefillSeedTests(unittest.TestCase):
    def fixture(self):
        metadata = {"context_tokens": 16, "backend_commit": "620d222",
                    "collector_source_sha256": "a" * 64, "model": "toy"}
        layers = {0: routing.Layer(4, 1, 100, 100), 1: routing.Layer(4, 1, 200, 200)}
        requests = {
            "train": routing.Request("train", "train", "coding", 4, 3, 4, True),
            "test": routing.Request("test", "test", "coding", 4, 3, 4, True),
        }
        events = [routing.Event(identifier, token, layer, (0 if identifier == "train" else 2,))
                  for identifier in requests for token in range(4) for layer in layers]
        data = routing.Dataset(metadata, layers, requests, events, "b" * 64)
        source = {
            "schema_version": 1, "complete": True, "instrumented": True,
            "capture_scope": "observed_prefill_expert_rows", "performance_measurement": False,
            **metadata, "requests": [],
        }
        for identifier, request in requests.items():
            source["requests"].append({
                "id": identifier, "split": request.split, "category": request.category,
                "prompt_tokens": request.prompt_tokens, "token_sequence_equivalent": True,
                "coverage": [{"layer": layer, "observed_rows": 3, "covers_all_prompt_rows": True}
                             for layer in layers],
                "batches": [{"layer": layer, "rows": 3, "active_experts_per_row": 1,
                             "expert_row_counts": [0, 0, 3, 0]} for layer in layers],
            })
        return data, source, {(0, 0), (1, 0)}

    def read(self, source, data):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prefill.json"
            path.write_text(json.dumps(source))
            return module.read_prefill(path, data)

    def test_prefix_seed_reduces_fixed_route_misses_and_charges_replacements(self):
        data, source, baseline = self.fixture()
        report = module.analyze(data, self.read(source, data), baseline, routing.Costs(100, 100, 0, 0, 0))
        self.assertFalse(report["performance_measurement"])
        self.assertFalse(report["full_model_acceleration_claim"])
        self.assertFalse(report["heldout_decode_used_to_select_seeds"])
        self.assertEqual(report["capacity_bytes"], 300)
        self.assertEqual(len(report["requests"]), 1)
        static = report["requests"][0]["policies"]["static"]
        self.assertEqual(static["trained_external"]["cold_cpu_bytes"], 1200)
        self.assertEqual(static["prompt_only"]["cold_cpu_bytes"], 0)
        self.assertEqual(static["prompt_only"]["seed_change_upload_bytes"], 300)
        self.assertGreater(static["prompt_only"]["modeled_serial_expert_ms"], 0)
        for mode in report["requests"][0]["policies"].values():
            for result in mode.values():
                self.assertLessEqual(result["seed_bytes"], 300)
                self.assertNotIn("tok_s", result)

    def test_test_decode_cannot_change_selected_seeds(self):
        data, source, baseline = self.fixture()
        prefill = self.read(source, data)
        expected = module.make_seeds(data, prefill, baseline)
        data.events = [routing.Event(e.request, e.token, e.layer, (3,)) if e.request == "test" else e
                       for e in data.events]
        self.assertEqual(expected, module.make_seeds(data, prefill, baseline))

    def test_pruned_layer_uses_training_prior(self):
        data, source, baseline = self.fixture()
        for request in source["requests"]:
            request["coverage"][1] = {"layer": 1, "observed_rows": 1, "covers_all_prompt_rows": False}
            request["batches"][1]["rows"] = 1
            request["batches"][1]["expert_row_counts"] = [0, 0, 1, 0]
        seeds = module.make_seeds(data, self.read(source, data), baseline)
        self.assertIn((1, 0), seeds["test"]["prompt_only"])
        self.assertNotIn((1, 2), seeds["test"]["prompt_only"])

    def test_provenance_and_histogram_errors_are_rejected(self):
        data, source, _ = self.fixture()
        malformed = []
        for key, value in (("complete", False), ("schema_version", True),
                           ("collector_source_sha256", "c" * 64), ("context_tokens", 32),
                           ("model", "wrong-model")):
            invalid = copy.deepcopy(source)
            invalid[key] = value
            malformed.append(invalid)
        for field, value in (("rows", True), ("expert_row_counts", [0, 0, 4, 0]),
                             ("expert_row_counts", [0, 0, 2, 0]), ("layer", 3),
                             ("active_experts_per_row", True)):
            invalid = copy.deepcopy(source)
            invalid["requests"][1]["batches"][0][field] = value
            malformed.append(invalid)
        invalid = copy.deepcopy(source)
        invalid["requests"][1]["coverage"].pop()
        malformed.append(invalid)
        for invalid in malformed:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.read(invalid, data)

    def test_plan_rejects_duplicates_and_unknown_experts(self):
        data, _, baseline = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.txt"
            path.write_text("0 0\n1 0\n")
            self.assertEqual(module.read_plan(path, data), baseline)
            for text in ("0 0\n0 0\n", "0 4\n", "3 0\n", "", "0 0 1\n"):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    module.read_plan(path, data)


if __name__ == "__main__":
    unittest.main()
