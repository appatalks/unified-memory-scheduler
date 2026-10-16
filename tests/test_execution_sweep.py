#!/usr/bin/env python3

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(sys.argv.pop(1)).resolve()
spec = importlib.util.spec_from_file_location("execution_sweep", SCRIPT)
assert spec and spec.loader
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)


def initial_status():
    return {
        "state": "ready", "chat_busy": False, "backend_kind": "llama.cpp",
        "draft_model": None, "model": "/private/models/target.gguf", "context": 16384,
        "placement": "auto", "cpu_threads": 4, "batch_threads": 16, "load_mode": "auto",
        "expert_cache_mib": 0, "fit_target_mib": 1024, "host_op_offload": True, "cpu_poll": 50,
        "op_offload_min_batch": 32,
        "logs": ["initializing, n_ctx_slot=16384"], "resources": {},
    }


def sample():
    return {"wall_decode_tok_s": 25.0, "end_to_end_tok_s": 20.0,
            "generated_text_sha256": "a" * 64}


class SweepTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.output = Path(self.temp.name) / "report.json"

    def tearDown(self):
        self.temp.cleanup()

    def test_sample_checks_exact_count_and_done(self):
        timings = {"predicted_n": 512, "prompt_ms": 1000, "predicted_ms": 20000,
                   "predicted_per_second": 25.55}
        usage = {"completion_tokens": 512}
        result = sweep.validate_sample(timings, usage, 512, 21, True)
        self.assertEqual(result["wall_decode_tok_s"], 511 / 20)
        self.assertEqual(result["end_to_end_tok_s"], 512 / 21)
        for updates in ({"predicted_n": 511}, {"prompt_ms": float("nan")},
                        {"predicted_ms": -1}, {"predicted_per_second": None},
                        {"predicted_ms": 0}, {"predicted_per_second": 0}):
            with self.assertRaises(ValueError):
                sweep.validate_sample({**timings, **updates}, usage, 512, 21, True)
        for count, elapsed, done in ((511, 21, True), (512, 0, True), (512, 21, False)):
            with self.assertRaises(ValueError):
                sweep.validate_sample(timings, {"completion_tokens": count}, 512, elapsed, done)

    def test_original_settings_restored_after_success(self):
        original = initial_status()
        with mock.patch.object(sweep, "request_json", return_value=original), \
                mock.patch.object(sweep, "load_profile", return_value=original) as load, \
                mock.patch.object(sweep, "sample", side_effect=lambda *args: sample()):
            self.assertEqual(sweep.execute("http://127.0.0.1:8080", ["control", "host-cpu"], 512, 2, self.output), 0)
        self.assertEqual(load.call_count, 3)
        self.assertTrue(load.call_args.args[1]["host_op_offload"])
        report = json.loads(self.output.read_text())
        self.assertTrue(report["restore"]["original_settings_restored"])
        self.assertTrue(report["runs"][1]["matches_control_text"])
        self.assertFalse(report["performance_target_met"])
        self.assertNotIn("/private/", self.output.read_text())

    def test_restores_original_after_candidate_failure(self):
        original = initial_status()
        with mock.patch.object(sweep, "request_json", return_value=original), \
                mock.patch.object(sweep, "load_profile", side_effect=[original, RuntimeError("candidate failed"), original]) as load, \
                mock.patch.object(sweep, "sample", return_value=sample()):
            self.assertEqual(sweep.execute("http://127.0.0.1:8080", ["control", "poll0"], 512, 1, self.output), 1)
        self.assertEqual(load.call_count, 3)
        report = json.loads(self.output.read_text())
        self.assertEqual(report["runs"][1]["status"], "failed")
        self.assertEqual(report["restore"]["status"], "ready")

    def test_restore_failure_is_reported_not_hidden(self):
        original = initial_status()
        with mock.patch.object(sweep, "request_json", return_value=original), \
                mock.patch.object(sweep, "load_profile", side_effect=[original, RuntimeError("restore failed")]), \
                mock.patch.object(sweep, "sample", return_value=sample()):
            self.assertEqual(sweep.execute("http://127.0.0.1:8080", ["control"], 512, 1, self.output), 1)
        self.assertEqual(json.loads(self.output.read_text())["restore"]["status"], "failed")

    def test_keyboard_interrupt_still_restores(self):
        original = initial_status()
        with mock.patch.object(sweep, "request_json", return_value=original), \
                mock.patch.object(sweep, "load_profile", return_value=original) as load, \
                mock.patch.object(sweep, "sample", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                sweep.execute("http://127.0.0.1:8080", ["control"], 512, 1, self.output)
        self.assertEqual(load.call_count, 2)
        self.assertTrue(json.loads(self.output.read_text())["restore"]["original_settings_restored"])

    def test_busy_app_is_never_unloaded(self):
        original = {**initial_status(), "chat_busy": True}
        with mock.patch.object(sweep, "request_json", return_value=original), \
                mock.patch.object(sweep, "load_profile") as load:
            with self.assertRaises(ValueError):
                sweep.execute("http://127.0.0.1:8080", ["control"], 512, 1, self.output)
        load.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_requires_backend_context_confirmation(self):
        original = initial_status()
        settings = {key: original[key] for key in sweep.SETTING_KEYS}
        for logs in ([], ["n_ctx_slot=4096"]):
            status = {**original, "logs": logs}
            with mock.patch.object(sweep, "request_json", return_value=status):
                with self.assertRaisesRegex(RuntimeError, "actual slot context"):
                    sweep.load_profile("http://127.0.0.1:8080", settings)

    def test_output_changes_are_visible(self):
        summary = sweep.summarize([sample(), {**sample(), "generated_text_sha256": "b" * 64}], "a" * 64)
        self.assertFalse(summary["repeated_text_hashes_match"])
        self.assertFalse(summary["matches_control_text"])

    def test_cli_requires_local_explicit_reload_consent(self):
        for extra in ([], ["--allow-reloads", "--base-url", "https://example.com"],
                      ["--allow-reloads", "--profiles", "poll0"],
                      ["--allow-reloads", "--tokens", "1"]):
            result = subprocess.run([sys.executable, str(SCRIPT), "--output", str(self.output), *extra],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
