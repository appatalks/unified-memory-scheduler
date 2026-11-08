import importlib.util
from pathlib import Path
import threading
import time
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "tools/benchmark/vrambridge-agent-concurrency.py"
spec = importlib.util.spec_from_file_location("agent_concurrency", SCRIPT)
assert spec and spec.loader
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class AgentConcurrencyTests(unittest.TestCase):
    def test_startup_requires_exact_slots_context_and_gpu_placement(self):
        log = (
            "I load_tensors: CPU_Mapped model buffer size = 144.00 MiB\n"
            "I load_tensors: CPU_Mapped model buffer size = 210.00 MiB\n"
            "I load_tensors: CUDA0 model buffer size = 12825.75 MiB\n"
            "I llama_kv_cache: K (f16): 1280.00 MiB, V (f16): 1280.00 MiB\n"
            "I srv load_model: initializing, n_slots = 2, n_ctx_slot = 16384, kv_unified = 'false'\n"
        )
        startup = bench.parse_startup(log, 2, 16384)
        self.assertEqual(startup["model_buffer_mib"], {"CPU_Mapped": 354.0, "CUDA0": 12825.75})
        for slots, context, text in ((1, 16384, log), (2, 8192, log), (2, 16384, log.replace("CUDA0", "CPU"))):
            with self.subTest(slots=slots, context=context), self.assertRaises(RuntimeError):
                bench.parse_startup(text, slots, context)
        with self.assertRaises(RuntimeError):
            bench.parse_startup(log.replace("K (f16)", "K (q8_0)"), 2, 16384)

    def test_gpu_guard_rejects_other_processes_or_missing_owned_server(self):
        self.assertEqual(bench.check_gpu_processes({}, None), 0)
        self.assertEqual(bench.check_gpu_processes({42: 1234}, 42), 1234)
        for processes, owner in (({42: 1}, None), ({42: 1, 43: 2}, 42), ({}, 42)):
            with self.assertRaises(RuntimeError):
                bench.check_gpu_processes(processes, owner)

    def test_backend_slot_activity_tracks_overlap_and_rejects_inconsistency(self):
        log = (
            "I slot launch_slot_: id 0 | task 10 | processing task, is_child = 0\n"
            "I slot launch_slot_: id 1 | task 11 | processing task, is_child = 0\n"
            "I slot release: id 0 | task 10 | stop processing: n_tokens = 20\n"
            "I slot release: id 1 | task 11 | stop processing: n_tokens = 20\n"
        )
        self.assertEqual(bench.parse_peak_slots(log), 2)
        with self.assertRaises(RuntimeError):
            bench.parse_peak_slots(log.replace("task 11 | stop", "task 12 | stop"))

    def test_gpu_monitor_propagates_contamination(self):
        monitor = bench.GpuMonitor(42)
        with mock.patch.object(monitor.stop_event, "wait", return_value=False), \
             mock.patch.object(bench, "gpu_processes", return_value={42: 100, 43: 200}):
            monitor.start()
            monitor.thread.join()
            with self.assertRaisesRegex(RuntimeError, "Another GPU compute process"):
                monitor.stop()

    def test_cohort_limits_active_requests_and_includes_queue_delay(self):
        lock = threading.Lock()
        active = 0
        peak = 0

        def fake_sample(base, tokens, headers, *, prompt):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.03)
            with lock:
                active -= 1
            return {"first_token_seconds": 0.01, "elapsed_seconds": 0.03,
                    "wall_decode_tok_s": 50.0, "timings": {"cache_n": 0}}

        corpus = [{"prompt": "synthetic", "category": "coding"}]
        with mock.patch.object(bench.sweep, "sample", side_effect=fake_sample):
            result = bench.run_cohort("http://127.0.0.1", "key", corpus, 4, 2, 8)
        self.assertEqual(peak, 2)
        self.assertEqual(len(result["samples"]), 4)
        self.assertGreater(result["samples"][-1]["queue_seconds"], 0.015)
        self.assertGreater(result["max_agent_ttft_seconds"], 0.025)
        self.assertGreater(result["aggregate_end_to_end_output_tok_s"], 0)

    def test_failed_or_reused_samples_cannot_become_successful_cohorts(self):
        corpus = [{"prompt": "synthetic", "category": "coding"}]
        sample = {"first_token_seconds": 0.1, "elapsed_seconds": 0.2,
                  "wall_decode_tok_s": 10, "timings": {"cache_n": 1}}
        with mock.patch.object(bench.sweep, "sample", return_value=sample), self.assertRaises(RuntimeError):
            bench.run_cohort("http://127.0.0.1", "key", corpus, 1, 1, 8)
        with self.assertRaises(ValueError):
            bench.run_cohort("http://127.0.0.1", "key", corpus, 1, 2, 8)

    def test_partial_or_failed_trials_have_no_performance_median(self):
        cohort = {key: 1.0 for key in ("aggregate_end_to_end_output_tok_s", "median_agent_ttft_seconds",
                  "max_agent_ttft_seconds", "median_agent_latency_seconds", "max_agent_latency_seconds",
                  "median_active_agent_decode_tok_s")}
        runs = [{"concurrency": 2, "status": "completed", "cohort": cohort},
                {"concurrency": 2, "status": "failed", "cohort": cohort}]
        summary = bench.summarize(runs, [2], 2)[0]
        self.assertEqual(summary["completed_runs"], 1)
        self.assertNotIn("aggregate_end_to_end_output_tok_s", summary)


if __name__ == "__main__":
    unittest.main()
