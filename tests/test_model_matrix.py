#!/usr/bin/env python3

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock


SCRIPT = Path(sys.argv.pop(1)).resolve()
spec = importlib.util.spec_from_file_location("model_matrix", SCRIPT)
matrix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(matrix)


class ModelMatrixTests(unittest.TestCase):
    def test_cli_requires_consent_loopback_and_valid_lengths(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            for extra in ([], ["--allow-reloads", "--base-url", "https://example.com"],
                          ["--allow-reloads", "--tokens", "511"],
                          ["--allow-reloads", "--repetitions", "0"],
                          ["--allow-reloads", "--base-url", "http://user:pass@localhost:8080"]):
                result = subprocess.run([sys.executable, str(SCRIPT), "missing.gguf",
                                         "--output", str(output), *extra], capture_output=True, timeout=10)
                self.assertEqual(result.returncode, 2)
                self.assertFalse(output.exists())

    def test_exact_arithmetic_and_ast_checks_never_execute_model_output(self):
        self.assertTrue(matrix.answer_check("arithmetic", "323\n"))
        self.assertFalse(matrix.answer_check("arithmetic", "The answer is 323."))
        self.assertTrue(matrix.answer_check("coding", "def clamp(value, low, high):\n return min(high, max(low, value))"))
        self.assertFalse(matrix.answer_check("coding", "import os\nos.remove('do-not-execute')"))
        self.assertFalse(matrix.answer_check("coding", "```python\ninvalid\n```"))

    def test_restores_model_mode_and_settings_after_success_failure_and_interrupt(self):
        for outcome in ("ok", "bad-answer", "failed", "interrupt"):
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                model = root / "test.gguf"
                model.write_bytes(b"GGUF")
                args = argparse.Namespace(models=[model], base_url="http://127.0.0.1:8080",
                                          output=root / "report.json", context=16384, threads=8,
                                          batch_threads=16, tokens=512, repetitions=3)
                original = {"model": "/private/original.gguf", "context": 16384, "placement": "auto",
                            "cpu_threads": 8, "batch_threads": 16, "load_mode": "auto", "expert_cache_mib": 10240,
                            "fit_target_mib": 1024, "host_op_offload": False, "cpu_poll": 50,
                            "op_offload_min_batch": 32, "state": "ready", "chat_busy": False,
                            "hybrid_available": True, "execution_mode": "auto", "hybrid_plan_sha256": "b" * 64}
                settings = tuple(key for key in original if key not in
                                 ("state", "chat_busy", "hybrid_available", "execution_mode", "hybrid_plan_sha256"))
                sweep = types.SimpleNamespace(SETTING_KEYS=settings, PROMPT="public prompt")
                sweep.request_json = mock.Mock(return_value=original)
                sweep.save_report = lambda path, report: path.write_text(json.dumps(report))
                sweep.load_profile = mock.Mock(side_effect=lambda base, values: {
                    "execution_profile": "standard" if values["execution_mode"] == "standard" else "experimental-hybrid-direct",
                    "hybrid_verified": values["execution_mode"] == "auto",
                    "selection_reason": "test selection", "backend": "/opt/llama-server"})
                sweep.sample = mock.Mock(return_value={"wall_decode_tok_s": 40.0,
                                                      "generated_text_sha256": "a" * 64})
                if outcome == "failed":
                    sweep.sample.side_effect = RuntimeError("test inference failed")
                elif outcome == "interrupt":
                    sweep.sample.side_effect = KeyboardInterrupt
                sweep.summarize = lambda samples, control: {"median_wall_decode_tok_s": 40.0}
                app = types.SimpleNamespace(PINNED_LLAMA_COMMIT="test",
                                            model_metadata=lambda path: {"general.architecture": "gpt-oss"},
                                            supports_hybrid=lambda metadata: True)
                with mock.patch.object(matrix, "load_module", side_effect=[sweep, app]), \
                        mock.patch.object(matrix, "quality_sample", return_value={"check_passed": outcome != "bad-answer"}):
                    if outcome == "interrupt":
                        with self.assertRaises(KeyboardInterrupt):
                            matrix.execute(args)
                    else:
                        self.assertEqual(matrix.execute(args), 0 if outcome == "ok" else 1)
                restored = sweep.load_profile.call_args.args[1]
                self.assertEqual(restored["model"], "/private/original.gguf")
                self.assertEqual(restored["execution_mode"], "auto")
                report = json.loads(args.output.read_text())
                self.assertEqual(report["restore"]["status"], "ready")
                self.assertFalse(report["performance_target_met"])
                if outcome == "bad-answer":
                    self.assertTrue(all(run["status"] == "answer_check_failed" for run in report["runs"]))


if __name__ == "__main__":
    unittest.main()
