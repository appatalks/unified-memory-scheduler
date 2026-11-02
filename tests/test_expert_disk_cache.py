#!/usr/bin/env python3

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SELF_TEST, KERNEL_TEST, CACHE_LIBRARY, CPU_LIBRARY = [Path(sys.argv.pop(1)).resolve() for _ in range(4)]


class ExpertDiskCacheTests(unittest.TestCase):
    def test_cache_page_ownership_and_budget(self):
        result = subprocess.run([str(SELF_TEST)], capture_output=True, text=True, check=True, timeout=20)
        report = json.loads(result.stdout)
        for key in ("byte_parity", "hits_avoid_reads", "overlapping_pages", "eviction_readmission", "budget_rejection"):
            self.assertTrue(report[key])

    def test_native_kernel_bits_match_all_tested_formats(self):
        native = subprocess.run([str(KERNEL_TEST)], capture_output=True, text=True, check=True, timeout=30)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "cache.jsonl"
            environment = {**os.environ, "LD_PRELOAD": str(CACHE_LIBRARY),
                           "VRAMBRIDGE_EXPERT_DISK_LIBRARY": str(CPU_LIBRARY),
                           "VRAMBRIDGE_EXPERT_DISK_OUT": str(output),
                           "VRAMBRIDGE_EXPERT_DISK_MIB": "64",
                           "VRAMBRIDGE_EXPERT_DISK_WORKERS": "2"}
            cached = subprocess.run([str(KERNEL_TEST)], env=environment, capture_output=True, text=True,
                                    check=True, timeout=30)
            self.assertEqual(json.loads(native.stdout), json.loads(cached.stdout))
            self.assertEqual({case["type"] for case in json.loads(cached.stdout)["cases"]},
                             {"f32", "q4_0", "iq2_xxs", "iq3_s", "iq1_m"})
            self.assertIn("exact expert RAM cache active", cached.stderr)
            counters = json.loads(output.read_text().splitlines()[-1])
            self.assertEqual(counters["graphs"], 20)
            self.assertGreater(counters["hit_expert_bytes"], 0)
            self.assertGreater(counters["direct_read_requested_bytes"], 0)
            self.assertLessEqual(counters["peak_resident_bytes"], 64 * 1024 ** 2)

    def test_missing_explicit_configuration_fails_instead_of_native_fallback(self):
        environment = {key: value for key, value in os.environ.items()
                       if not key.startswith("VRAMBRIDGE_EXPERT_DISK_")}
        environment["LD_PRELOAD"] = str(CACHE_LIBRARY)
        result = subprocess.run([str(KERNEL_TEST)], env=environment, capture_output=True, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Explicit pinned library, output and RAM budget are required", result.stderr)

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "cache.jsonl"
            output.write_text("preserve")
            environment = {**os.environ, "LD_PRELOAD": str(CACHE_LIBRARY),
                           "VRAMBRIDGE_EXPERT_DISK_LIBRARY": str(CPU_LIBRARY),
                           "VRAMBRIDGE_EXPERT_DISK_OUT": str(output),
                           "VRAMBRIDGE_EXPERT_DISK_MIB": "64"}
            result = subprocess.run([str(KERNEL_TEST)], env=environment,
                                    capture_output=True, text=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(output.read_text(), "preserve")

    def test_source_aware_fills_preserve_native_kernel_bits(self):
        native = subprocess.run([str(KERNEL_TEST)], capture_output=True, text=True, check=True, timeout=30)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "aware.jsonl"
            environment = {**os.environ, "LD_PRELOAD": str(CACHE_LIBRARY),
                           "VRAMBRIDGE_EXPERT_DISK_LIBRARY": str(CPU_LIBRARY),
                           "VRAMBRIDGE_EXPERT_DISK_OUT": str(output),
                           "VRAMBRIDGE_EXPERT_DISK_MIB": "64",
                           "VRAMBRIDGE_EXPERT_DISK_SOURCE_AWARE": "1"}
            cached = subprocess.run([str(KERNEL_TEST)], env=environment, capture_output=True, text=True,
                                    check=True, timeout=30)
            self.assertEqual(json.loads(native.stdout), json.loads(cached.stdout))
            self.assertIn("source-residency-aware cache fills active", cached.stderr)
            counters = json.loads(output.read_text().splitlines()[-1])
            self.assertGreater(counters["source_ram_copy_bytes"], 0)

    def test_invalid_worker_count_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "workers.jsonl"
            for value in ("4294967298", "2junk", "0"):
                environment = {**os.environ, "LD_PRELOAD": str(CACHE_LIBRARY),
                               "VRAMBRIDGE_EXPERT_DISK_LIBRARY": str(CPU_LIBRARY),
                               "VRAMBRIDGE_EXPERT_DISK_OUT": str(output),
                               "VRAMBRIDGE_EXPERT_DISK_MIB": "64",
                               "VRAMBRIDGE_EXPERT_DISK_WORKERS": value}
                result = subprocess.run([str(KERNEL_TEST)], env=environment, capture_output=True, text=True, timeout=30)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Read worker count", result.stderr)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
