#!/usr/bin/env python3

import importlib.util
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location("critical_path_under_test",
    Path(__file__).resolve().parents[1] / "tools/benchmark/vrambridge-critical-path.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class CriticalPathTests(unittest.TestCase):
    def test_partition_conserves_wall_time_without_adding_api_waits(self):
        result = module.partition([(0, 1000000)], [(0, 500000)], [(300000, 700000)],
                                  [(700000, 800000)], [(400000, 600000), (750000, 900000)])
        self.assertEqual(result["envelope_ms"], 1)
        self.assertEqual(result["cpu_only_ms"], .3)
        self.assertEqual(result["gpu_only_ms"], .2)
        self.assertEqual(result["cpu_and_gpu_ms"], .2)
        self.assertEqual(result["neither_cpu_nor_gpu_ms"], .3)
        self.assertEqual(result["observer_inside_neither_ms"], .1)
        self.assertEqual(result["cuda_api_inside_neither_ms"], .15)
        self.assertEqual(result["unattributed_neither_ms"], .1)

    def test_overlapping_activity_is_not_double_counted(self):
        result = module.partition([(0, 1000000)], [(0, 600000), (200000, 700000)],
                                  [(0, 1000000)], [], [(0, 1000000), (0, 1000000)])
        self.assertEqual(result["cpu_and_gpu_ms"], .7)
        self.assertEqual(result["gpu_only_ms"], .3)
        self.assertEqual(result["neither_cpu_nor_gpu_ms"], 0)
        self.assertEqual(result["cuda_api_inside_neither_ms"], 0)

    def test_sqlite_api_counts_and_partition_use_aligned_fixture(self):
        spec = importlib.util.spec_from_file_location("window_fixtures",
            Path(__file__).with_name("test_cache_window.py"))
        fixtures = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixtures)
        with tempfile.TemporaryDirectory() as directory:
            cpu, timeline, request = fixtures.CacheWindowTests().fixture(directory)
            frames = [json.loads(line) for line in cpu.read_text().splitlines()]
            for frame in frames:
                frame.update(threads=8, ops={"MUL_MAT_ID": 1} if frame["matmuls"] else {"CUSTOM": 1})
            cpu.write_text("".join(json.dumps(frame) + "\n" for frame in frames))
            report = json.loads(request.read_text())
            report["cuda_runtime_environment"] = {"GGML_CUDA_GRAPH_OPT": "0"}
            request.write_text(json.dumps(report))
            connection = sqlite3.connect(timeline)
            try:
                connection.execute("ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL ADD COLUMN streamId INTEGER DEFAULT 14")
                connection.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,globalTid INTEGER,nameId INTEGER)")
                connection.execute("INSERT INTO StringIds VALUES(1,'cudaStreamSynchronize')")
                connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(1120,1500,?,1)",
                                   (((1 << 48) | (7 << 24) | 9),))
                connection.commit()
            finally:
                connection.close()
            result = module.analyze(cpu, timeline, request)
        self.assertEqual(result["cuda_api_leaders"][0]["calls"], 1)
        self.assertEqual(result["cuda_api_leaders"][0]["name"], "cudaStreamSynchronize")
        self.assertFalse(result["full_model_acceleration_claim"])
        self.assertEqual(len(result["cpu_groups"]), 2)


if __name__ == "__main__":
    unittest.main()
