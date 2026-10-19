#!/usr/bin/env python3

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "tools/benchmark/vrambridge-live-warmer-study.py"
SPEC = importlib.util.spec_from_file_location("live_study_under_test", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class LiveWarmerStudyTests(unittest.TestCase):
    def fixture(self, root):
        root = Path(root)
        for name in ("model.gguf", "warmer.so", "cpu.so"):
            (root / name).write_bytes(b"fixture")
        profile = root / "profile.json"
        profile.write_text(json.dumps({"context": 16384, "cache_mib": 12000, "executable": "/test/server"}))
        corpus = root / "corpus.jsonl"
        corpus.write_text(json.dumps({"split": "test", "category": "coding", "prompt": "exact prompt"}) + "\n")
        return SimpleNamespace(output_dir=root / "results", profile=profile, model=root / "model.gguf",
            warmer=root / "warmer.so", cpu_library=root / "cpu.so", corpus=corpus, category="coding",
            pair=1, label="test", helper_cpu=min(os.sched_getaffinity(0)), warm_kib=256, maximum_us=200, tokens=512)

    def fake_process(self, command, *, env, **kwargs):
        output = Path(command[command.index("--output") + 1])
        self.assertTrue(kwargs["start_new_session"])
        prompt = Path(command[command.index("--prompt-file") + 1]).read_text()
        report = {"status": "complete", "owned_backend_stopped": True, "actual_context": 16384,
            "mapped_backend_libraries": [str(Path(env.get("VRAMBRIDGE_CPU_WARM_LIBRARY", self.cpu_path)).resolve()),
                                         "/native/libggml-base.so.0"],
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "applied_settings": {"expert_cache_mib": 12000},
            "samples": [{"before_request": {"pid": 42}, "timing": {"generated_text_sha256": "same",
                "timings": {"predicted_per_second": 50}, "elapsed_seconds": 12},
                "generation_interval": {"cpu_seconds": 80}}]}
        output.write_text(json.dumps(report))
        mode = env.get("VRAMBRIDGE_CPU_WARM_MODE", "native")
        if getattr(self, "bad_base", False) and mode == "native":
            report["mapped_backend_libraries"][1] = "/different/libggml-base.so.0"
            output.write_text(json.dumps(report))
        self.modes.append(mode)
        self.assertNotIn("GOMP_SPINCOUNT", env)
        if mode == "native":
            self.assertNotIn("LD_PRELOAD", env)
        else:
            warm = {"complete": True, "mode": mode, "pid": 99 if self.bad_pid else 42,
                "helper_cpu": int(env["VRAMBRIDGE_CPU_WARM_CPU"]),
                "initial_compute_affinity": sorted(os.sched_getaffinity(0)),
                "warm_budget_bytes": int(env["VRAMBRIDGE_CPU_WARM_KIB"]) * 1024}
            Path(env["VRAMBRIDGE_CPU_WARM_OUT"]).write_text(json.dumps(warm))
        return SimpleNamespace(wait=mock.Mock(return_value=0), pid=123)

    def test_reversed_order_child_configuration_and_prompt_cleanup(self):
        self.modes, self.bad_pid = [], False
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(directory)
            self.cpu_path = str(args.cpu_library)
            with mock.patch.object(module.subprocess, "Popen", side_effect=self.fake_process), \
                    mock.patch.dict(os.environ, {"GOMP_SPINCOUNT": "1", "VRAMBRIDGE_CPU_WARM_MODE": "wrong"}):
                module.run(args)
            self.assertEqual(self.modes, ["read", "noop", "native"])
            self.assertFalse(list(args.output_dir.glob("user-prompt-*")))
            with self.assertRaisesRegex(ValueError, "overwrite"):
                module.run(args)

    def test_helper_pid_mismatch_is_not_a_success(self):
        self.modes, self.bad_pid = [], True
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(directory)
            self.cpu_path = str(args.cpu_library)
            with mock.patch.object(module.subprocess, "Popen", side_effect=self.fake_process):
                with self.assertRaisesRegex(RuntimeError, "acknowledgement"):
                    module.run(args)

    def test_timeout_terminates_only_the_owned_process_group(self):
        with tempfile.TemporaryDirectory() as directory:
            process = SimpleNamespace(pid=123, wait=mock.Mock(side_effect=[subprocess.TimeoutExpired("owned", 180), 0]))
            with mock.patch.object(module.subprocess, "Popen", return_value=process), \
                    mock.patch.object(module.os, "killpg") as stop:
                with self.assertRaisesRegex(RuntimeError, "stopped the owned"):
                    module.run(self.fixture(directory))
            stop.assert_called_once_with(123, module.signal.SIGTERM)

    def test_mismatched_backend_library_sets_are_rejected(self):
        self.modes, self.bad_pid, self.bad_base = [], False, True
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(directory)
            self.cpu_path = str(args.cpu_library)
            with mock.patch.object(module.subprocess, "Popen", side_effect=self.fake_process):
                with self.assertRaisesRegex(RuntimeError, "library mappings differ"):
                    module.run(args)

    def test_help_and_invalid_parameters_do_not_start_model(self):
        result = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0)
        result = subprocess.run([sys.executable, str(SCRIPT), "--model", "/missing/model", "--profile", "/missing/profile",
            "--warmer", "/missing/warmer", "--cpu-library", "/missing/cpu", "--category", "coding", "--pair", "0",
            "--helper-cpu", "0", "--output-dir", "/missing/results", "--warm-kib", "-1"],
            capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 2)
        self.assertIn("Invalid bounded", result.stderr)


if __name__ == "__main__":
    unittest.main()
