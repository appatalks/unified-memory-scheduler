#!/usr/bin/env python3

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock
import tempfile


SPEC = importlib.util.spec_from_file_location("handoff_under_test",
    Path(__file__).resolve().parents[1] / "tools/benchmark/vrambridge-handoff-study.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class HandoffStudyTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        for name in ("model", "library", "cpu"):
            (root / name).write_text("fixture")
        (root / "profile").write_text(json.dumps({"context": 16384, "cache_mib": 12000,
                                                "executable": "/test/server"}))
        (root / "corpus").write_text(json.dumps({"category": "coding", "split": "test", "prompt": "test"}) + "\n")
        return SimpleNamespace(output_dir=root / "results", profile=root / "profile", corpus=root / "corpus",
            model=root / "model", library=root / "library", cpu_library=root / "cpu",
            variant="cpu-one", categories=["coding"], pairs=2, tokens=512, label="test")

    def measure(self, _):
        mode = "cpu-one" if "VRAMBRIDGE_CPU_CONTROL_THREADS" in os.environ else "native"
        self.modes.append(mode)
        if mode == "cpu-one":
            Path(os.environ["VRAMBRIDGE_CPU_CONTROL_OUT"]).write_text(json.dumps({
                "complete": True, "pid": 42, "changed_calls": 12, "expert_plan_threads_changed": False}))
        report = {"status": "complete", "owned_backend_stopped": True, "actual_context": 16384,
                  "applied_settings": {"expert_cache_mib": 12000},
                  "cuda_runtime_environment": {"GGML_CUDA_GRAPH_OPT": os.environ["GGML_CUDA_GRAPH_OPT"]},
                  "mapped_backend_libraries": ["/native/libggml-base.so"],
                  "samples": [{"before_request": {"pid": 42},
                               "timing": {"generated_text_sha256": "changed" if self.bad_hash and mode == "cpu-one" else "same",
                                          "timings": {"predicted_per_second": 50}, "elapsed_seconds": 12}}]}
        return 0, report

    def test_pair_order_environment_restore_and_prompt_cleanup(self):
        self.modes, self.bad_hash = [], False
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(directory)
            with mock.patch.object(module.io, "measure", side_effect=self.measure), \
                    mock.patch.dict(os.environ, {"GOMP_SPINCOUNT": "1000"}):
                before = dict(os.environ)
                module.run(args)
                self.assertEqual(dict(os.environ), before)
            self.assertEqual(self.modes, ["native", "cpu-one", "cpu-one", "native"])
            self.assertFalse(list(args.output_dir.glob("prompts-*")))
            with self.assertRaisesRegex(ValueError, "overwrite"):
                module.run(args)

    def test_hash_mismatch_is_not_a_success_and_environment_is_restored(self):
        self.modes, self.bad_hash = [], True
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(directory)
            before = dict(os.environ)
            with mock.patch.object(module.io, "measure", side_effect=self.measure):
                with self.assertRaisesRegex(RuntimeError, "Output hashes differ"):
                    module.run(args)
            self.assertEqual(dict(os.environ), before)

    def test_bad_profile_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(directory)
            args.profile.write_text(json.dumps({"context": 8192, "cache_mib": 12000}))
            with mock.patch.object(module.io, "measure") as measure:
                with self.assertRaisesRegex(ValueError, "16K/12000"):
                    module.run(args)
            measure.assert_not_called()


if __name__ == "__main__":
    unittest.main()
