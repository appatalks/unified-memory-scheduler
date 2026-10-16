#!/usr/bin/env python3

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


BINARY = Path(sys.argv.pop(1)).resolve()


class HybridLayerTests(unittest.TestCase):
    def test_cpu_partition_math_matches_unsplit_reference(self):
        for extra in ([], ["--gpu-reduce"]):
            result = subprocess.run([str(BINARY), "--self-test", "--samples", "5", "--repetitions", "5", *extra],
                                    capture_output=True, text=True, check=True, timeout=30)
            report = json.loads(result.stdout)
            self.assertTrue(report["all_checks_passed"])
            self.assertTrue(report["independent_scalar_reference"])
            self.assertFalse(report["full_model_acceleration_claim"])
            self.assertFalse(report["hardware_gpu_partition"])
            self.assertEqual(len(report["routing_weights"]), 4)
            for actual, expected in zip(report["routing_weights"], [0.05, 0.15, 0.30, 0.50]):
                self.assertAlmostEqual(actual, expected, places=6)
            self.assertEqual(len(report["partitions"]), 5)
            for split in report["partitions"]:
                self.assertEqual(len(split["correctness"]), 5)
                self.assertEqual(len(split["modes"]["serial"]["samples_ms"]), 5)
                self.assertGreater(split["modes"]["serial"]["median_ms"], 0)
                self.assertLess(split["modes"]["serial"]["worst_relative_l2_error"], 0.0001)
                self.assertTrue(all(check["passed"] for check in split["correctness"]))

    def test_invalid_cli_does_not_load_weights(self):
        for args in ([], ["--self-test", "--layer", "-1"], ["--self-test", "--threads", "0"],
                     ["--self-test", "--repetitions", "1", "--samples", "2"],
                     ["--self-test", "--layer", "1garbage"], ["--unknown", "value"]):
            result = subprocess.run([str(BINARY), *args], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 2)
            self.assertIn("vrambridge-hybrid-layer:", result.stderr)
            self.assertNotIn("ggml_cuda_init", result.stderr)

    def test_refuses_to_overwrite_final_or_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "probe.json"
            for path in (output, Path(str(output) + ".part")):
                path.write_text("preserve")
                result = subprocess.run([str(BINARY), "--self-test", "--output", str(output)],
                                        capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 2)
                self.assertIn("Refusing to overwrite", result.stderr)
                self.assertEqual(path.read_text(), "preserve")
                path.unlink()


if __name__ == "__main__":
    unittest.main()
