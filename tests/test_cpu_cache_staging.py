#!/usr/bin/env python3

import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


BINARY = Path(sys.argv.pop(1)).resolve()


class CpuCacheStagingTests(unittest.TestCase):
    def test_owned_buffer_staging_preserves_bits_and_charges_warm_cost(self):
        cpu = min(os.sched_getaffinity(0))
        for budget in (0, 1):
            result = subprocess.run(
                [str(BINARY), "--self-test", "--cpu-cache-staging", "--cpu-experts", "2",
                 "--cache-warm-mib", str(budget), "--warm-cpu", str(cpu),
                 "--threads", "2", "--repetitions", "3"],
                capture_output=True, text=True, check=True, timeout=30)
            report = json.loads(result.stdout)
            self.assertTrue(report["all_checks_passed"])
            self.assertTrue(report["bitwise_equal_to_native_cpu"])
            self.assertFalse(report["full_model_acceleration_claim"])
            self.assertFalse(report["actual_gpu_window_execution"])
            self.assertFalse(report["kernel_math_changed"])
            self.assertEqual(report["projection_weight_bytes"], 64 * 96 * 4 * 3 * 2)
            self.assertEqual(report["warming_cpu"], cpu)
            self.assertEqual(report["warmed_prefix_bytes"], 0 if budget == 0 else report["projection_weight_bytes"])
            for key in ("cold_compute_ms", "warm_compute_ms", "warm_handoff_ms", "serial_warm_plus_compute_ms"):
                self.assertEqual(len(report[key]), 3)
                self.assertTrue(all(x > 0 for x in report[key]))
            for handoff, compute, serial in zip(report["warm_handoff_ms"], report["warm_compute_ms"],
                                                report["serial_warm_plus_compute_ms"]):
                self.assertAlmostEqual(handoff + compute, serial)

    def test_incompatible_options_fail_before_backend_loading(self):
        for extra in (["--cpu-experts", "0"], ["--cache-warm-mib", "65"], ["--warm-cpu", "-1"],
                      ["--gpu-reduce"], ["--operator-check"], ["--rows", "2"], ["--samples", "1"]):
            result = subprocess.run(
                [str(BINARY), "--self-test", "--cpu-cache-staging", *extra],
                capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 2)
            self.assertNotIn("ggml_cuda_init", result.stderr)


if __name__ == "__main__":
    unittest.main()
