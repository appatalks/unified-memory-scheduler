#!/usr/bin/env python3

import copy
from contextlib import closing
import importlib.util
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location(
    "cache_window_under_test",
    Path(__file__).resolve().parents[1] / "tools/benchmark/vrambridge-cache-window.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)

SCREEN_SPEC = importlib.util.spec_from_file_location(
    "commit_late_screen", Path(__file__).resolve().parents[1] / "tools/benchmark/vrambridge-commit-late-screen.py")
assert SCREEN_SPEC is not None and SCREEN_SPEC.loader is not None
screen = importlib.util.module_from_spec(SCREEN_SPEC)
SCREEN_SPEC.loader.exec_module(screen)

LEARNED_SPEC = importlib.util.spec_from_file_location(
    "commit_late_learned", Path(__file__).resolve().parents[1] / "tools/benchmark/vrambridge-commit-late-learned.py")
assert LEARNED_SPEC is not None and LEARNED_SPEC.loader is not None
learned = importlib.util.module_from_spec(LEARNED_SPEC)
LEARNED_SPEC.loader.exec_module(learned)


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

    def test_commit_late_infers_admissions_from_matching_masks(self):
        hint = {"observable": True, "selected_ids": [1, 2], "pre_admission_misses": [1, 2]}
        matrices = [{"weight_name": f"blk.0.ffn_{kind}_exps.weight", "rows": 1,
                     "cold_experts": 1, "cold_ids": [2], "expert_bank_size": 4, "expert_bytes": 10,
                     "logical_weight_bytes": 10, "id_source_keys": [7]}
                    for kind in ("gate", "up", "down")]
        frames = [{"sequence": 0, "nodes": [{"tensor_key": 7, "early_route": hint}], "matmuls": []},
                  {"sequence": 1, "nodes": [], "matmuls": matrices}]
        ranges = {"graph": {0: (10, 20), 1: (30, 40)}}
        event = screen.dispatches(frames, ranges)[0]
        self.assertEqual(event["admitted"], {1})
        self.assertEqual(event["cold"], {2})
        frames[1]["matmuls"][1]["cold_ids"] = [1]
        with self.assertRaises(ValueError):
            screen.dispatches(frames, ranges)

    def test_commit_late_predictions_use_only_previous_cpu_misses(self):
        def event(layer, selected, cold, admitted, start):
            return {"layer": layer, "selected": selected, "cold": set(cold), "admitted": set(admitted),
                    "start": start, "end": start + 5, "sizes": (10, 10, 10)}
        tokens = [
            [event(0, (0, 1), {1}, set(), 0), event(1, (0, 2), {2}, set(), 20)],
            [event(0, (1, 3), {3}, {1}, 100), event(1, (2, 3), {2}, {3}, 120)],
            [event(0, (3, 0), {0}, {3}, 200), event(1, (2, 0), {0}, {2}, 220)],
        ]
        result = screen.forecast(tokens, 1, 1, True, 10)
        self.assertEqual(result["predictions"], 4)
        self.assertEqual(result["matched_admissions"], 3)
        self.assertEqual(result["actual_admissions"], 4)
        self.assertEqual(result["timely_matches"], 3)
        altered = copy.deepcopy(tokens)
        altered[2][1]["admitted"] = {0}
        changed = screen.forecast(altered, 1, 1, True, 10)
        self.assertEqual(changed["predictions"], result["predictions"])
        self.assertEqual(changed["weight_prediction_bytes"], result["weight_prediction_bytes"])
        self.assertNotEqual(changed["matched_admissions"], result["matched_admissions"])

    def test_commit_late_rejects_missing_layers_and_insufficient_warmup(self):
        events = [{"layer": value} for value in (0, 1, 0, 1)]
        tokens, count = screen.complete_tokens(events)
        self.assertEqual((len(tokens), count), (2, 2))
        with self.assertRaises(ValueError):
            screen.complete_tokens([{"layer": value} for value in (0, 1, 1)])
        with self.assertRaises(ValueError):
            screen.forecast(tokens, 2, 1, True, 1)

    def test_commit_late_history_uses_only_last_63_observations(self):
        def event(layer, selected, cold, admitted, start):
            return {"layer": layer, "selected": selected, "cold": set(cold), "admitted": set(admitted),
                    "start": start, "end": start + 5, "sizes": (10, 10, 10)}
        tokens = [[event(0, (1, 2), {1}, {1}, t * 100),
                   event(1, (4, 5), {4}, {4}, t * 100 + 20)] for t in range(65)]
        result = screen.forecast(tokens, 64, 1, True, 10, history_min=3)
        self.assertEqual((result["predictions"], result["matched_admissions"]), (2, 2))
        self.assertEqual(result["admissions_below_default_history_floor"], 0)
        for token in tokens[2:63]:
            token[0]["selected"] = (2, 3)
        result = screen.forecast(tokens, 64, 1, True, 10, history_min=3)
        self.assertEqual(result["predictions"], 1)
        self.assertEqual(result["admissions_below_default_history_floor"], 1)
        with self.assertRaises(ValueError):
            screen.forecast(tokens, 32, 1, True, 10, history_min=3)

    def test_commit_late_known_cold_survives_unselected_tokens_causally(self):
        def event(layer, selected, cold, admitted, start):
            return {"layer": layer, "selected": selected, "cold": set(cold), "admitted": set(admitted),
                    "start": start, "end": start + 5, "sizes": (10, 10, 10)}
        tokens = [[event(0, (1, 2), {1}, set(), t * 100),
                   event(1, (4, 5), {4}, set(), t * 100 + 20)] for t in range(65)]
        for token in tokens[61:64]:
            token[0] = event(0, (2, 3), {3}, set(), token[0]["start"])
        tokens[64][0] = event(0, (1, 2), {2}, {1}, 6400)
        result = screen.forecast(tokens, 64, 1, True, 10, 3, "known_cold_frequency")
        self.assertEqual(result["matched_admissions"], 1)
        self.assertEqual(result["predictions"], 2)
        changed = copy.deepcopy(tokens)
        changed[64][0]["admitted"] = {2}
        changed[64][0]["cold"] = {1}
        other = screen.forecast(changed, 64, 1, True, 10, 3, "known_cold_frequency")
        self.assertEqual(other["predictions"], result["predictions"])
        self.assertEqual(other["matched_admissions"], 0)

    def test_learned_candidate_selection_does_not_read_holdout_labels(self):
        bucket = learned.feature(4, 2, 1)
        training = [{"candidates": [(7, bucket)], "admitted": {7}} for _ in range(40)]
        model = learned.fit(training)
        self.assertEqual(model[bucket], [40, 40])
        self.assertEqual(learned.choose([(3, bucket)], model, 0.5, 20), 3)
        self.assertIsNone(learned.choose([(3, bucket)], model, 0.8, 20))
        heldout = [{"candidates": [(3, bucket)], "admitted": {3}, "weight_bytes": 30,
                    "release": 0, "deadline": 100}]
        matched = learned.score(heldout, model, 0.5, 20, 10)
        heldout[0]["admitted"] = {4}
        missed = learned.score(heldout, model, 0.5, 20, 10)
        self.assertEqual(matched["predictions"], missed["predictions"])
        self.assertEqual(matched["matched_admissions"], 1)
        self.assertEqual(missed["matched_admissions"], 0)
        heldout[0]["deadline"] = -1
        with self.assertRaises(ValueError):
            learned.score(heldout, model, 0.5, 20, 10)

    def test_adjacent_route_is_available_before_the_target(self):
        def event(layer, selected, cold, start):
            return {"layer": layer, "selected": selected, "cold": set(cold), "admitted": set(),
                    "start": start, "end": start + 5, "sizes": (10, 10, 10)}
        tokens = [[event(0, (1, 2), {1}, t * 100), event(1, (1, 4), {1}, t * 100 + 20)]
                  for t in range(65)]
        first, second = list(learned.examples(tokens, 64, adjacent_route=True))
        self.assertEqual(first["release"], tokens[63][1]["start"])
        self.assertEqual(second["release"], tokens[64][0]["start"])
        self.assertLess(second["release"], second["deadline"])
        self.assertEqual(second["candidates"][0][1][-1], 0)

    def test_scratch_scheduler_charges_capacity_and_late_copies(self):
        jobs = [{"release": 0, "deadline": 10, "bytes": 30, "matched": True},
                {"release": 1, "deadline": 2, "bytes": 30, "matched": False},
                {"release": 11, "deadline": 20, "bytes": 30, "matched": False}]
        result = screen.scratch_schedule(jobs, 1, 5)
        self.assertEqual(result["accepted"], 2)
        self.assertEqual(result["capacity_rejected"], 1)
        self.assertEqual(result["useful_admissions"], 1)
        self.assertEqual(result["weight_upload_bytes"], 60)
        self.assertEqual(result["useful_weight_bytes"], 30)
        changed = copy.deepcopy(jobs)
        for job in changed:
            job["matched"] = not job["matched"]
        other = screen.scratch_schedule(changed, 1, 5)
        self.assertEqual((other["accepted"], other["capacity_rejected"], other["weight_upload_bytes"]),
                         (result["accepted"], result["capacity_rejected"], result["weight_upload_bytes"]))
        late = screen.scratch_schedule([{"release": 0, "deadline": 2, "bytes": 30, "matched": True},
                                       {"release": 3, "deadline": 12, "bytes": 30, "matched": True}], 1, 5)
        self.assertEqual(late["late"], 1)
        self.assertEqual(late["capacity_rejected"], 1)
        self.assertEqual(late["useful_admissions"], 0)
        self.assertLessEqual(late["peak_slots"], 1)
        cancellable = screen.scratch_schedule(
            [{"release": 0, "deadline": 10, "bytes": 30, "matched": True},
             {"release": 1, "deadline": 3, "bytes": 30, "matched": False},
             {"release": 4, "deadline": 20, "bytes": 30, "matched": True}], 2, 5)
        self.assertEqual(cancellable["accepted"], 3)
        self.assertEqual(cancellable["cancelled_before_copy"], 1)
        self.assertEqual(cancellable["weight_upload_bytes"], 60)
        self.assertEqual(cancellable["useful_admissions"], 2)

    def test_commit_late_clis_reject_invalid_settings_without_output(self):
        root = Path(__file__).resolve().parents[1] / "tools/benchmark"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            commands = [
                [sys.executable, str(root / "vrambridge-commit-late-screen.py"),
                 "--cpu-profile", "missing", "--timeline", "missing", "--request-report", "missing",
                 "--warmup-tokens", "62", "--output", str(output)],
                [sys.executable, str(root / "vrambridge-commit-late-learned.py"),
                 "--train", "missing", "missing", "missing", "--test", "x", "missing", "missing", "missing",
                 "--minimum-support", "0", "--output", str(output)],
            ]
            for command in commands:
                self.assertEqual(subprocess.run(command, capture_output=True).returncode, 2)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
