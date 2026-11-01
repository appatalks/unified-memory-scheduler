import importlib.util
from pathlib import Path
import unittest


path = Path(__file__).resolve().parents[1] / "tools/benchmark/vrambridge-disk-roofline.py"
spec = importlib.util.spec_from_file_location("disk_roofline", path)
roofline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(roofline)


class DiskRooflineTests(unittest.TestCase):
    def analyze(self, **changes):
        arguments = dict(active_bytes=2_000_000_000, bandwidth=500_000_000,
                         iops=50_000, rows=48, row_bytes=144, target_tps=10,
                         hit_rate=0.9, amplification=1, prefetch_precision=1, compute_ms=0)
        return roofline.analyze(**{**arguments, **changes})

    def test_hit_threshold_and_page_accounting(self):
        result = self.analyze()
        self.assertEqual(result["engram"]["logical_bytes_per_token"], 6912)
        self.assertEqual(result["engram"]["page_bytes_min"], 196608)
        self.assertEqual(result["engram"]["page_bytes_max"], 393216)
        self.assertAlmostEqual(result["target"]["minimum_byte_hit_rate_bandwidth_only"], 0.975098304)
        self.assertEqual(result["engram"]["random_read_seconds_floor"], 48 / 50000)

    def test_compute_is_not_hidden_by_zero_misses(self):
        result = self.analyze(hit_rate=1, compute_ms=100)
        self.assertEqual(result["upper_bound_tps"], 10)
        self.assertLess(result["serial_reference_tps"], 10)

    def test_bad_prediction_debits_bandwidth(self):
        good = self.analyze()
        bad = self.analyze(prefetch_precision=0.5)
        self.assertEqual(bad["experts"]["physical_read_bytes_per_token_estimate"],
                         2 * good["experts"]["physical_read_bytes_per_token_estimate"])
        self.assertGreater(bad["target"]["minimum_byte_hit_rate_bandwidth_only"],
                           good["target"]["minimum_byte_hit_rate_bandwidth_only"])

    def test_hdd_random_latency_can_dominate(self):
        result = self.analyze(hit_rate=1, iops=150)
        self.assertAlmostEqual(result["upper_bound_tps"], 150 / 48)
        self.assertFalse(result["target"]["row_iops_condition_met"])

    def test_amplification_is_not_a_speedup(self):
        result = self.analyze(amplification=2)
        self.assertGreater(result["target"]["minimum_byte_hit_rate_bandwidth_only"],
                           self.analyze()["target"]["minimum_byte_hit_rate_bandwidth_only"])

    def test_reject_invalid_inputs(self):
        for changes in ({"active_bytes": 0}, {"bandwidth": -1}, {"iops": 0},
                        {"hit_rate": -0.1}, {"hit_rate": 1.1}, {"hit_rate": float("nan")},
                        {"amplification": 0.5}, {"prefetch_precision": 1.1},
                        {"compute_ms": -1}, {"target_tps": float("inf")}, {"rows": -1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.analyze(**changes)


if __name__ == "__main__":
    unittest.main()
