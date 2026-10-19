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


class LiveWarmerTests(unittest.TestCase):
    def run_case(self, mode, extra=(), kib="256", maximum="5000"):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            env = {**os.environ, "LD_PRELOAD": str(LIBRARY), "VRAMBRIDGE_CPU_WARM_MODE": mode,
                   "VRAMBRIDGE_CPU_WARM_LIBRARY": str(CPU), "VRAMBRIDGE_CPU_WARM_OUT": str(output),
                   "VRAMBRIDGE_CPU_WARM_CPU": str(min(os.sched_getaffinity(0))),
                   "VRAMBRIDGE_CPU_WARM_KIB": kib, "VRAMBRIDGE_CPU_WARM_MAX_US": maximum}
            result = subprocess.run([str(FIXTURE), *extra], env=env, capture_output=True, text=True,
                                    timeout=30, preexec_fn=no_core)
            self.assertEqual(result.returncode, 0, result.stderr)
            observed = json.loads(result.stdout)
            report = json.loads(output.read_text()) if mode != "off" else None
            if report is not None:
                self.assertEqual(output.stat().st_mode & 0o777, 0o600)
                self.assertFalse(report["weights_modified"])
                self.assertFalse(report["routing_modified"])
                self.assertFalse(report["full_model_acceleration_claim"])
                self.assertFalse(report["raw_ids_or_addresses_recorded"])
                self.assertTrue(report["complete"])
                self.assertGreater(report["pid"], 0)
                self.assertEqual(report["initial_compute_affinity"], sorted(os.sched_getaffinity(0)))
                self.assertEqual(report["read_payload_bytes"], report["bytes_completed_before_observed_deadline"] +
                                 report["cancellation_tail_payload_bytes"])
                self.assertLessEqual(report["read_payload_bytes"], report["planned_bytes"])
                self.assertLessEqual(report["read_snapshot_bytes_for_actual_cpu_experts"] +
                                     report["read_snapshot_bytes_for_admitted_gpu_experts"],
                                     report["bytes_completed_before_observed_deadline"])
                self.assertGreaterEqual(report["buffer_release_drains"], 3)
                self.assertEqual(report["source_invalidations"], 9)
            else:
                self.assertFalse(output.exists())
            return observed, report

    def test_off_noop_and_read_preserve_bits_across_unloads(self):
        expected, _ = self.run_case("off")
        for mode in ("noop", "read"):
            observed, report = self.run_case(mode)
            self.assertEqual(observed, expected)
            self.assertEqual(report["dispatches"], 60)
            self.assertEqual(report["jobs_submitted"], 60)
            self.assertEqual(report["dispatches_without_sources"], 0)
            self.assertGreater(report["jobs_started"], 0)
            if mode == "read":
                self.assertGreater(report["read_payload_bytes"], 0)
                self.assertGreater(report["bytes_completed_before_observed_deadline"], 0)
                self.assertGreater(report["read_snapshot_bytes_for_actual_cpu_experts"], 0)
                self.assertGreater(report["read_snapshot_bytes_for_admitted_gpu_experts"], 0)
            else:
                self.assertEqual(report["read_payload_bytes"], 0)

    def test_no_dispatch_zero_budget_and_expiry_shutdown(self):
        _, idle = self.run_case("read", ["--no-dispatch"])
        self.assertEqual(idle["jobs_submitted"], 0)
        self.assertEqual(idle["read_payload_bytes"], 0)
        _, zero = self.run_case("read", kib="0")
        self.assertEqual(zero["planned_bytes"], 0)
        self.assertEqual(zero["read_payload_bytes"], 0)
        _, expiry = self.run_case("read", maximum="1")
        self.assertGreater(expiry["jobs_started"], 0)
        self.assertGreater(expiry["jobs_expired"] + expiry["jobs_canceled"], 0)

    def test_queued_reads_are_drained_before_model_buffer_release(self):
        observed, report = self.run_case("read", ["--unload-only"])
        self.assertTrue(observed["all_checks_passed"])
        self.assertEqual(report["jobs_submitted"], 3)
        self.assertEqual(report["buffer_release_drains"], 3)
        self.assertEqual(report["source_invalidations"], 9)

    def test_invalid_configuration_fails_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            for key, value in (("VRAMBRIDGE_CPU_WARM_MODE", "unknown"), ("VRAMBRIDGE_CPU_WARM_KIB", "-1"),
                               ("VRAMBRIDGE_CPU_WARM_MAX_US", "0"), ("VRAMBRIDGE_CPU_WARM_CPU", "999999")):
                env = {**os.environ, "LD_PRELOAD": str(LIBRARY), "VRAMBRIDGE_CPU_WARM_MODE": "read",
                       "VRAMBRIDGE_CPU_WARM_LIBRARY": str(CPU), "VRAMBRIDGE_CPU_WARM_OUT": str(Path(directory) / "report.json"),
                       key: value}
                result = subprocess.run([str(FIXTURE)], env=env, capture_output=True, text=True,
                                        timeout=30, preexec_fn=no_core)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("CPU live warmer:", result.stderr)


if __name__ == "__main__":
    unittest.main()
