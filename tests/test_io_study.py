#!/usr/bin/env python3

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock


SCRIPT = Path(sys.argv.pop(1)).resolve()
spec = importlib.util.spec_from_file_location("io_study", SCRIPT)
assert spec and spec.loader
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


@unittest.skipUnless(sys.platform == "linux", "Linux /proc research probe")
class IOStudyTests(unittest.TestCase):
    def test_real_self_snapshot_has_physical_and_logical_counters(self):
        value = study.snapshot(os.getpid())
        self.assertEqual(value["pid"], os.getpid())
        self.assertGreaterEqual(value["physical_read_bytes"], 0)
        self.assertGreater(value["read_syscall_bytes"], 0)
        self.assertGreater(value["rss_bytes"], 0)
        self.assertIn("Cached", value["system_memory"])

    def test_interval_distinguishes_disk_reads_from_cached_read_calls(self):
        before = {"pid": 42, "monotonic_seconds": 1, "physical_read_bytes": 0,
                  "read_syscall_bytes": 100, "minor_faults": 0, "major_faults": 0, "cpu_seconds": 0}
        after = {**before, "monotonic_seconds": 3, "read_syscall_bytes": 1048676, "minor_faults": 100}
        result = study.interval(before, after)
        self.assertEqual(result["physical_read_bytes_s"], 0)
        self.assertEqual(result["read_syscall_bytes"], 1048576)
        for invalid in ({**after, "pid": 43}, {**after, "monotonic_seconds": 1},
                        {**after, "major_faults": -1}):
            with self.assertRaises(ValueError):
                study.interval(before, invalid)

    def test_report_publication_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            study.write_report(output, {"status": "failed"})
            self.assertEqual(json.loads(output.read_text())["status"], "failed")
            with self.assertRaises(FileExistsError):
                study.write_report(output, {"status": "complete"})
            self.assertEqual(json.loads(output.read_text())["status"], "failed")

    def test_cpu_runtime_environment_is_attested_without_exposing_other_values(self):
        with mock.patch.dict(os.environ, {"GOMP_SPINCOUNT": "1000"}, clear=True), \
                mock.patch.object(Path, "read_bytes", return_value=b"GOMP_SPINCOUNT=1000\0API_KEY=private\0"):
            self.assertEqual(study.cpu_runtime_environment(42), {"GOMP_SPINCOUNT": "1000"})
        with mock.patch.dict(os.environ, {"GOMP_SPINCOUNT": "1000"}, clear=True), \
                mock.patch.object(Path, "read_bytes", return_value=b"GOMP_SPINCOUNT=0\0"):
            with self.assertRaisesRegex(RuntimeError, "differs"):
                study.cpu_runtime_environment(42)

    def test_cpu_backend_mapping_keeps_paths_with_spaces_and_deduplicates(self):
        maps = ("0-1 r-xp 0 00:00 1 /test path/libggml-cpu-zen4.so\n"
                "1-2 r--p 0 00:00 1 /test path/libggml-cpu-zen4.so\n"
                "2-3 r-xp 0 00:00 2 /test/libggml-cuda.so\n")
        with mock.patch.object(Path, "read_text", return_value=maps):
            self.assertEqual(study.cpu_backend_modules(42), ["/test path/libggml-cpu-zen4.so"])
            self.assertEqual(study.backend_library_modules(42),
                             ["/test path/libggml-cpu-zen4.so", "/test/libggml-cuda.so"])

    def test_cuda_environment_attests_graph_option_without_other_values(self):
        with mock.patch.dict(os.environ, {"GGML_CUDA_GRAPH_OPT": "1"}, clear=True), \
                mock.patch.object(Path, "read_bytes", return_value=b"GGML_CUDA_GRAPH_OPT=1\0API_KEY=private\0"):
            self.assertEqual(study.cuda_runtime_environment(42), {"GGML_CUDA_GRAPH_OPT": "1"})
        with mock.patch.dict(os.environ, {"GGML_CUDA_GRAPH_OPT": "1"}, clear=True), \
                mock.patch.object(Path, "read_bytes", return_value=b"GGML_CUDA_GRAPH_OPT=0\0"):
            with self.assertRaisesRegex(RuntimeError, "CUDA runtime environment differs"):
                study.cuda_runtime_environment(42)

    def test_help_and_invalid_options_need_no_model_or_gpu(self):
        result = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            for extra in (["--tokens", "0"], ["--threads", "0"], ["--context", "1"],
                          ["--cpu-poll", "-1"], ["--cpu-poll", "101"],
                          ["--repetitions", "11"], ["--load-timeout", "nan"],
                          ["--batch-size", "4096", "--ubatch-size", "512"],
                          ["--batch-size", "512", "--ubatch-size", "1024"],
                          ["--context", "512", "--batch-size", "2048", "--ubatch-size", "512"]):
                result = subprocess.run(
                    [sys.executable, str(SCRIPT), "--llama-server", "/missing/server",
                     "--model", "/missing/model.gguf", "--output", str(output), *extra],
                    capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 2)
                self.assertIn("Invalid bounded", result.stderr)
                self.assertFalse(output.exists())
            for option in ("--batch-size", "--ubatch-size"):
                result = subprocess.run(
                    [sys.executable, str(SCRIPT), "--llama-server", "/missing/server",
                     "--model", "/missing/model.gguf", "--output", str(output), option, "512"],
                    capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 2)
                self.assertIn("Supply both", result.stderr)
                self.assertFalse(output.exists())
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--llama-server", "/missing/server",
                 "--model", "/missing/model.gguf", "--output", str(output), "--verbatim-prompt"],
                capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 2)
            self.assertIn("requires --prompt-file", result.stderr)

    def test_unconfirmed_batches_fail_before_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "target.gguf"
            model.write_bytes(b"GGUF")
            args = SimpleNamespace(model=model, hybrid_profile=None, context=16384, threads=8,
                                   batch_threads=16, tokens=256, repetitions=1,
                                   llama_server="/test/server", load_timeout=600, prompt_file=None,
                                   batch_size=2048, ubatch_size=1024)
            for observed in ({}, {"n_batch": 2048, "n_ubatch": 512}):
                with self.subTest(observed=observed):
                    runtime = SimpleNamespace(
                        lock=threading.RLock(), process=SimpleNamespace(pid=42), state="ready", error=None,
                        context=16384, hybrid_active=False, hybrid_markers=set(), key="", logs=[],
                        load=mock.Mock(), execution_mode="standard", placement="auto", load_mode="auto",
                        cpu_threads=8, batch_threads=16, expert_cache_mib=0, hybrid_plan_sha256=None,
                        observed_batch_sizes=observed)
                    runtime.stop = mock.Mock(side_effect=lambda **kwargs: setattr(runtime, "process", None))
                    base = {"pid": 42, "physical_read_bytes": 0, "read_syscall_bytes": 0,
                            "minor_faults": 0, "major_faults": 0, "cpu_seconds": 0}
                    with mock.patch.object(study, "Runtime", return_value=runtime), \
                            mock.patch.object(study, "snapshot", side_effect=[
                                {**base, "monotonic_seconds": value} for value in (1, 2)]), \
                            mock.patch.object(study.sweep, "sample") as sampler:
                        status, report = study.measure(args)
                    self.assertEqual(status, 1)
                    self.assertEqual(report["status"], "failed")
                    self.assertIn("did not confirm", report["error"])
                    sampler.assert_not_called()
                    runtime.stop.assert_called_once_with(closing=True)
                    self.assertTrue(report["owned_backend_stopped"])

    def test_generation_failure_is_recorded_and_owned_backend_is_stopped(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "target.gguf"
            model.write_bytes(b"GGUF")
            args = SimpleNamespace(model=model, hybrid_profile=None, context=16384, threads=8,
                                   batch_threads=16, tokens=256, repetitions=3,
                                   llama_server="/test/server", load_timeout=600, prompt_file=None,
                                   batch_size=None, ubatch_size=None)
            runtime = SimpleNamespace(
                lock=threading.RLock(), process=SimpleNamespace(pid=42), state="ready", error=None,
                context=16384, hybrid_active=False, hybrid_markers=set(), port=12345,
                key="unit-test-marker", logs=[], load=mock.Mock(), execution_mode="standard",
                placement="auto", load_mode="auto", cpu_threads=8, batch_threads=16,
                expert_cache_mib=0, hybrid_plan_sha256=None, observed_batch_sizes={})
            runtime.stop = mock.Mock(side_effect=lambda **kwargs: setattr(runtime, "process", None))
            base = {"pid": 42, "physical_read_bytes": 0, "read_syscall_bytes": 0,
                    "minor_faults": 0, "major_faults": 0, "cpu_seconds": 0}
            snapshots = [{**base, "monotonic_seconds": value} for value in (1, 2, 3)]
            with mock.patch.object(study, "Runtime", return_value=runtime), \
                    mock.patch.object(study, "snapshot", side_effect=snapshots), \
                    mock.patch.object(study, "cpu_runtime_environment", return_value={}), \
                    mock.patch.object(study, "cuda_runtime_environment", return_value={}), \
                    mock.patch.object(study, "cpu_backend_modules", return_value=[]), \
                    mock.patch.object(study, "backend_library_modules", return_value=[]), \
                    mock.patch.object(study.sweep, "sample",
                                      side_effect=RuntimeError("generation failed: unit-test-marker")):
                status, report = study.measure(args)
            self.assertEqual(status, 1)
            self.assertEqual(report["status"], "failed")
            self.assertTrue(report["owned_backend_stopped"])
            self.assertNotIn("unit-test-marker", json.dumps(report))
            self.assertIn("[redacted]", report["error"])
            runtime.stop.assert_called_once_with(closing=True)

    def test_empty_or_unbounded_context_is_rejected_before_launcher(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "target.gguf"
            model.write_bytes(b"GGUF")
            prompt = Path(directory) / "context.txt"
            args = SimpleNamespace(model=model, hybrid_profile=None, prompt_file=prompt)
            for data in ("", "x" * (512 * 1024 + 1)):
                prompt.write_text(data)
                with mock.patch.object(study, "Runtime") as runtime, self.assertRaises(ValueError):
                    study.measure(args)
                runtime.assert_not_called()


if __name__ == "__main__":
    unittest.main()
