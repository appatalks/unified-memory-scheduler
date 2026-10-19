#!/usr/bin/env python3

import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import unittest


FIXTURE, LIBRARY, CPU = (Path(sys.argv.pop(1)).resolve() for _ in range(3))


def no_core():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


class ControlThreadTests(unittest.TestCase):
    def test_scalar_control_and_unchanged_other_graphs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            env = {**os.environ, "LD_PRELOAD": str(LIBRARY), "VRAMBRIDGE_CPU_CONTROL_THREADS": "one",
                   "VRAMBRIDGE_CPU_CONTROL_LIBRARY": str(CPU), "VRAMBRIDGE_CPU_CONTROL_OUT": str(output)}
            result = subprocess.run([str(FIXTURE)], env=env, capture_output=True, text=True,
                                    check=True, timeout=30, preexec_fn=no_core)
            self.assertTrue(json.loads(result.stdout)["all_checks_passed"])
            report = json.loads(output.read_text())
            self.assertTrue(report["complete"])
            self.assertEqual(report["eligible_calls"], 12)
            self.assertEqual(report["changed_calls"], 12)
            self.assertEqual(report["other_calls"], 12)
            self.assertEqual(report["maximum_original_threads"], 8)
            self.assertFalse(report["expert_plan_threads_changed"])
            self.assertFalse(report["full_model_acceleration_claim"])
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)

    def test_off_creates_no_report(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            result = subprocess.run([str(FIXTURE)],
                env={**os.environ, "LD_PRELOAD": str(LIBRARY), "VRAMBRIDGE_CPU_CONTROL_THREADS": "off",
                     "VRAMBRIDGE_CPU_CONTROL_LIBRARY": str(CPU), "VRAMBRIDGE_CPU_CONTROL_OUT": str(output)},
                capture_output=True, text=True, check=True, timeout=30, preexec_fn=no_core)
            self.assertTrue(json.loads(result.stdout)["all_checks_passed"])
            self.assertFalse(output.exists())

    def test_large_copies_and_batched_controls_keep_native_plans(self):
        for option in ("--large-copy", "--batch-two"):
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "report.json"
                result = subprocess.run([str(FIXTURE), option],
                    env={**os.environ, "LD_PRELOAD": str(LIBRARY), "VRAMBRIDGE_CPU_CONTROL_THREADS": "one",
                         "VRAMBRIDGE_CPU_CONTROL_LIBRARY": str(CPU), "VRAMBRIDGE_CPU_CONTROL_OUT": str(output)},
                    capture_output=True, text=True, check=True, timeout=30, preexec_fn=no_core)
                self.assertTrue(json.loads(result.stdout)["all_checks_passed"])
                report = json.loads(output.read_text())
                self.assertEqual(report["changed_calls"], 0)
                self.assertEqual(report["eligible_calls"], 0)
                self.assertEqual(report["other_calls"], 24)


if __name__ == "__main__":
    unittest.main()
