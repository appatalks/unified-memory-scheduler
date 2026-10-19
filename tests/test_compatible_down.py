#!/usr/bin/env python3

import json
from pathlib import Path
import subprocess
import sys
import unittest


BINARY = Path(sys.argv.pop(1)).resolve()


class CompatibleDownTests(unittest.TestCase):
    def test_packed_and_scalar_dots_preserve_native_cpu_bits(self):
        for variant in ([], ["--compatible-dp4a"]):
            for pattern in ("sine", "zero", "q8-ties"):
                with self.subTest(variant=variant, pattern=pattern):
                    result = subprocess.run(
                        [str(BINARY), "--self-test", "--compatible-down", "--rows", "7",
                         "--samples", "3", "--repetitions", "3", "--threads", "2",
                         "--activation-pattern", pattern, "--compatible-gpu-first", *variant],
                        capture_output=True, text=True, check=True, timeout=30)
                    report = json.loads(result.stdout)
                    self.assertEqual(report["kind"], "cpu_compatible_packed_down_probe")
                    self.assertTrue(report["all_checks_passed"])
                    self.assertFalse(report["full_model_qualification"])
                    self.assertFalse(report["full_model_acceleration_claim"])
                    self.assertFalse(report["raw_activations_recorded"])
                    self.assertFalse(report["raw_routing_recorded"])
                    self.assertEqual(report["timing_order"], "gpu_then_cpu")
                    self.assertEqual(report["integer_dot"], "byte_permute_dp4a" if variant else "scalar_quad")
                    self.assertEqual(report["packed_input_bytes"], 128 * 4 * 7 // 32 * 34)
                    self.assertEqual(report["f32_input_bytes"], 128 * 4 * 7 * 4)
                    self.assertEqual(report["raw_output_bytes"], 64 * 4 * 7 * 4)
                    self.assertEqual(report["down_weight_bytes"], 4 * 64 * (128 // 32) * 17)
                    self.assertEqual(len(report["samples"]), 3)
                    for sample in report["samples"]:
                        self.assertTrue(sample["cpu_all_iterations_identical"])
                        self.assertTrue(sample["gpu_all_iterations_identical"])
                        self.assertTrue(sample["timings_numerically_qualified"])
                        check = sample["gpu_raw_vs_cpu"]
                        self.assertTrue(check["bitwise_equal"])
                        self.assertEqual(check["different_bits_values"], 0)
                        self.assertEqual(check["max_abs_error"], 0)
                        self.assertEqual(check["relative_l2_error"], 0)
                        self.assertIsNone(sample["observer_control"])
                        self.assertIsNone(sample["candidate_vs_captured_boundary"])
                        for key in ("cpu_samples_ms", "gpu_inclusive_samples_ms", "gpu_cpu_quantize_samples_ms"):
                            self.assertEqual(len(sample[key]), 3)
                            self.assertTrue(all(value > 0 for value in sample[key]))
                        self.assertGreater(sample["raw_stage_speedup"], 0)


if __name__ == "__main__":
    unittest.main()
