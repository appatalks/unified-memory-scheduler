import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


path = Path(__file__).resolve().parents[1] / "tools/benchmark/vrambridge-disk-trace.py"
spec = importlib.util.spec_from_file_location("disk_trace", path)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


class DiskTraceTests(unittest.TestCase):
    def dataset(self):
        layers = {0: study.routing.Layer(4, 1, 16, 16), 1: study.routing.Layer(4, 1, 32, 32)}
        requests = {"train": study.routing.Request("train", "train", "coding", 1, 1, 1, True),
                    "test": study.routing.Request("test", "test", "coding", 1, 1, 1, True)}
        events = [study.routing.Event("train", 0, 0, (0,)), study.routing.Event("train", 0, 1, (0,)),
                  study.routing.Event("test", 0, 0, (0,)), study.routing.Event("test", 0, 1, (1,))]
        return study.routing.Dataset({"io_observer": True}, layers, requests, events, "hash")

    def test_byte_weighted_hits_not_expert_count(self):
        result = study.replay(self.dataset(), 48, "train_static", 1000)
        self.assertAlmostEqual(result["byte_hit_fraction"], 1 / 3)
        self.assertEqual(result["logical_miss_bytes_per_token"], 32)
        self.assertEqual(result["disk_bandwidth_ms_per_token_floor"], 32)
        self.assertLessEqual(result["peak_resident_bytes"], 48)

    def test_zero_budget_always_misses_and_never_admits(self):
        for policy in ("lru", "lfu", "train_static"):
            result = study.replay(self.dataset(), 0, policy, 1000)
            self.assertEqual(result["byte_hit_fraction"], 0)
            self.assertEqual(result["logical_miss_bytes"], 48)
            self.assertEqual(result["peak_resident_bytes"], 0)

    def test_cache_large_enough_for_training_is_not_future_aware(self):
        for policy in ("lru", "lfu", "train_static"):
            result = study.replay(self.dataset(), 1024, policy, 1000)
            self.assertEqual(result["logical_miss_bytes"], 32)
            self.assertEqual(result["training_resident_bytes"], 48)

    def records(self):
        interval = lambda size: {"elapsed_ns": 100, "physical_read_bytes": size,
                                 "minor_faults": 0, "major_faults": 0}
        values = []
        for identifier in ("train", "test"):
            for layer in (0, 1):
                values.append({"type": "route", "request": identifier, "token": 0, "layer": layer,
                               "expert_observed_interval": interval(20)})
            values.append({"type": "request_end", "id": identifier, "decode_tokens": 1,
                           "observed_token_intervals": [{"token": 0, "interval": interval(100),
                                                       "engram_gathers": [{"name": "engram_rows-1",
                                                                          "interval": interval(10)}]}]})
        return values

    def attribute(self, values):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.jsonl"
            trace.write_text("".join(json.dumps(value) + "\n" for value in values))
            return study.attribution(trace, self.dataset())

    def test_attribution_retains_unattributed_reads(self):
        result = self.attribute(self.records())
        totals = result["test"]["totals"]
        self.assertEqual(totals["expert_interval_read_bytes"], 40)
        self.assertEqual(totals["engram_gather_read_bytes"], 10)
        self.assertEqual(totals["unattributed_read_bytes"], 50)

    def test_invalid_and_missing_observations_fail(self):
        for change in ("missing", "order", "double_count"):
            values = self.records()
            observation = values[-1]["observed_token_intervals"][0]
            if change == "missing":
                del values[0]["expert_observed_interval"]
            elif change == "order":
                observation["token"] = 1
            else:
                observation["interval"]["physical_read_bytes"] = 0
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.attribute(values)

    def test_negative_or_invalid_intervals_fail(self):
        for value in ({}, {"elapsed_ns": 0, "physical_read_bytes": 0, "minor_faults": 0, "major_faults": 0},
                      {"elapsed_ns": 1, "physical_read_bytes": -1, "minor_faults": 0, "major_faults": 0}):
            with self.assertRaises(ValueError):
                study.validate_interval(value)


if __name__ == "__main__":
    unittest.main()
