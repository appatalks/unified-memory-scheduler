#!/usr/bin/env python3

import copy
from contextlib import closing
import importlib.util
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location(
    "cache_window_under_test",
    Path(__file__).resolve().parents[1] / "tools/benchmark/vrambridge-cache-window.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class CacheWindowTests(unittest.TestCase):
    def test_interval_unions_do_not_double_count(self):
        self.assertEqual(module.merge([(4, 7), (1, 5), (7, 9), (20, 20)]), [(1, 9)])
        self.assertEqual(module.intersection([(1, 8)], [(3, 5), (4, 7)]), [(3, 7)])
        self.assertEqual(module.subtract([(1, 10)], [(2, 4), (3, 7), (9, 12)]), [(1, 2), (7, 9)])
        self.assertEqual(module.subtract([(1, 2)], [(2, 3)]), [(1, 2)])
        with self.assertRaises(ValueError):
            module.merge([(2, 1)])

    def test_indexed_window_integral_matches_interval_reference(self):
        intervals = [(1, 4), (3, 8), (10, 11)]
        index = module.WindowBudget(intervals)
        for start in range(16):
            for end in range(start, 16):
                self.assertEqual(index.between(start, end),
                                 module.duration(module.intersection(intervals, [(start, end)])))

    def fixture(self, directory):
        directory = Path(directory)
        matrix = {"name": "hybrid_cpu_gate-0", "weight_name": "layer0.gate", "rows": 1,
                  "cold_experts": 1, "cold_ids": [2], "expert_bank_size": 4,
                  "expert_bytes": 128, "logical_weight_bytes": 128,
                  "id_sources": ["generic repeated name"], "id_source_keys": [1234]}
        frames = [
            {"schema_version": 3, "sequence": 0, "contains_sensitive_routing": True,
             "status": 0, "start_ns": 1000, "end_ns": 1100,
             "nodes": [{"name": "generic repeated name", "op": "CUSTOM", "tensor_key": 1234}], "matmuls": []},
            {"schema_version": 3, "sequence": 1, "contains_sensitive_routing": True,
             "status": 0, "start_ns": 2000, "end_ns": 2400, "nodes": [], "matmuls": [matrix]},
        ]
        cpu = directory / "cpu.jsonl"
        cpu.write_text("".join(json.dumps(f) + "\n" for f in frames))
        request = directory / "request.json"
        request.write_text(json.dumps({"status": "complete", "owned_backend_stopped": True, "output_tokens": 16,
            "samples": [{"first_text_event": {"monotonic_seconds": 0},
                         "response_complete": {"monotonic_seconds": 0.000003},
                         "before_request": {"pid": 7}}]}))
        timeline = directory / "timeline.sqlite"
        global_pid = (1 << 48) | (7 << 24)
        with closing(sqlite3.connect(timeline)) as connection:
            connection.executescript("""
                CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,globalTid INTEGER,text TEXT,textId INTEGER);
                CREATE TABLE StringIds(id INTEGER,value TEXT);
                CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,deviceId INTEGER,globalPid INTEGER);
                CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start INTEGER,end INTEGER,bytes INTEGER,copyKind INTEGER,globalPid INTEGER);
                CREATE TABLE ENUM_CUDA_MEMCPY_OPER(id INTEGER,name TEXT);
            """)
            for sequence, start, end in [(0, 1000, 1100), (1, 2000, 2400)]:
                for kind, a, b in [("meta", start - 200, start - 100),
                                   ("graph", start, end), ("record", end, end + 20)]:
                    connection.execute("INSERT INTO NVTX_EVENTS VALUES(?,?,?,?,NULL)",
                                       (a, b, global_pid | 9, f"vrambridge.cpu_{kind}:{sequence}"))
            connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(1150,1990,0,?)", (global_pid,))
            connection.execute("INSERT INTO ENUM_CUDA_MEMCPY_OPER VALUES(1,'CUDA_MEMCPY_KIND_HTOD')")
            connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES(1300,1400,50,1,?)", (global_pid,))
            connection.commit()
        return cpu, timeline, request

    def test_cuda_alignment_excludes_cpu_observer_and_dma(self):
        with tempfile.TemporaryDirectory() as directory:
            request, frames, ranges, intervals = module.read_inputs(*self.fixture(directory))
        self.assertEqual(module.duration(intervals["generous"]), 740)
        self.assertEqual(module.duration(intervals["strict"]), 640)
        report = module.analyze(request, frames, ranges, intervals, [64], [1], 32)
        self.assertFalse(report["full_model_acceleration_claim"])
        self.assertFalse(report["actual_prefetch_implemented"])
        scenario = report["scenarios"][0]
        self.assertEqual(scenario["perfect_next_graph_oracle"]["warmable_payload_bytes_upper_bound"], 64)
        self.assertEqual(scenario["fresh_observed_route_ready"]["known_release_graphs"], 1)

    def test_cache_capacity_bandwidth_tiles_and_unknown_routes(self):
        jobs = [{"start": 100, "previous_end": 0, "bytes": 1000, "route_ready": None}]
        result = module.oracle(jobs, [(0, 100)], 512, 1, 64)
        self.assertEqual(result["warmable_payload_bytes_upper_bound"], 64)
        unknown = module.oracle(jobs, [(0, 100)], 512, 1, 64, True)
        self.assertEqual(unknown["known_release_graphs"], 0)
        self.assertEqual(unknown["warmable_payload_bytes_upper_bound"], 0)

    def test_shared_dma_rate_is_debited_without_negative_capacity(self):
        budget = module.SpareBandwidth([(0, 100)], [(20, 40, 10), (60, 80, 100)], 1)
        self.assertEqual(budget.between(0, 100), 70)
        self.assertEqual(budget.between(25, 35), 5)
        self.assertEqual(budget.between(60, 80), 0)
        self.assertEqual(module.host_passes("CUDA_MEMCPY_KIND_UVM_HTOD"), 1)
        self.assertEqual(module.host_passes("CUDA_MEMCPY_KIND_DTOD"), 0)
        with self.assertRaises(ValueError):
            module.host_passes("unknown")

    def test_duplicate_labels_or_wrong_process_are_not_accepted(self):
        for bad_pid in (False, True):
            with tempfile.TemporaryDirectory() as directory:
                paths = self.fixture(directory)
                with closing(sqlite3.connect(paths[1])) as connection:
                    if bad_pid:
                        connection.execute("UPDATE NVTX_EVENTS SET globalTid=8 << 24")
                    else:
                        connection.execute("INSERT INTO NVTX_EVENTS SELECT * FROM NVTX_EVENTS LIMIT 1")
                    connection.commit()
                with self.assertRaises(ValueError):
                    module.read_inputs(*paths)

    def test_inactive_slots_and_unobservable_work_are_not_cold_payload(self):
        frame = {"sequence": 0, "nodes": [], "matmuls": [{
            "rows": 1, "cold_ids": [-1], "cold_experts": 1, "expert_bank_size": 4,
            "expert_bytes": 128, "logical_weight_bytes": 128, "id_sources": [], "weight_name": "toy"}]}
        with self.assertRaises(ValueError):
            module.demands([frame], {"graph": {0: (0, 1)}})
        invalid = copy.deepcopy(frame)
        invalid["matmuls"][0]["cold_ids"] = None
        with self.assertRaises(ValueError):
            module.demands([invalid], {"graph": {0: (0, 1)}})

    def test_repeated_names_do_not_substitute_for_tensor_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            request, frames, ranges, _ = module.read_inputs(*self.fixture(directory))
        frames[1]["matmuls"][0]["id_source_keys"] = [9999]
        job = module.demands(frames, ranges)[0]
        self.assertIsNone(job["route_ready"])

    def test_early_admission_false_positives_consume_staging_budget(self):
        job = {"start": 1000, "previous_end": 0, "route_ready": 900, "early_ready": 0,
               "bytes": 128, "early_chunks": [(128, False), (128, True)]}
        result = module.oracle([job], [(0, 1000)], 128, 1, 64, early=True)
        self.assertEqual(result["attempted_staging_bytes"], 128)
        self.assertEqual(result["warmable_payload_bytes_upper_bound"], 0)
        self.assertEqual(result["wasted_staging_bytes"], 128)
        result = module.oracle([job], [(0, 1000)], 256, 1, 64, early=True)
        self.assertEqual(result["warmable_payload_bytes_upper_bound"], 128)

    def test_previous_cold_set_spends_bandwidth_on_incorrect_predictions(self):
        jobs = [{"start": 1000, "previous_end": 0, "route_ready": 900, "bytes": 128,
                 "previous_chunks": [(128, False), (128, True)]}]
        result = module.oracle(jobs, [(0, 1000)], 128, 1, 64, previous=True)
        self.assertEqual(result["attempted_staging_bytes"], 128)
        self.assertEqual(result["warmable_payload_bytes_upper_bound"], 0)
        self.assertEqual(result["wasted_staging_bytes"], 128)


if __name__ == "__main__":
    unittest.main()
