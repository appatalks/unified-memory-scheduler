#!/usr/bin/env python3

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


BINARY = Path(sys.argv.pop(1)).resolve()


class CollectorInputTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.corpus = self.root / "corpus.jsonl"
        self.output = self.root / "trace.jsonl"

    def tearDown(self):
        self.temporary.cleanup()

    def invoke(self, values):
        self.corpus.write_text("".join(json.dumps(value) + "\n" for value in values))
        return subprocess.run([str(BINARY), "--trace-corpus", str(self.corpus),
                               "--trace-output", str(self.output)], capture_output=True, text=True)

    def test_help_and_missing_options(self):
        self.assertEqual(subprocess.run([str(BINARY), "--help"], capture_output=True).returncode, 0)
        result = subprocess.run([str(BINARY)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("required", result.stderr)

    def test_prefill_histogram_self_test(self):
        result = subprocess.run([str(BINARY), "--prefill-self-test"], capture_output=True, text=True,
                                check=True, timeout=20)
        report = json.loads(result.stdout)
        self.assertEqual(report["strided_histogram"]["expert_row_counts"], [1, 2, 2, 1])
        self.assertEqual(report["strided_histogram"]["rows"], 3)
        self.assertTrue(report["invalid_routes_rejected"])
        self.assertTrue(report["invalid_shape_rejected"])
        self.assertTrue(report["inactive_callback_skips_capture"])
        self.assertTrue(report["zero_rows_recorded"])

    def test_prefill_output_paths_are_checked_before_backend_loading(self):
        values = [
            {"id": "train", "split": "train", "category": "coding", "prompt": "First", "max_tokens": 1},
            {"id": "test", "split": "test", "category": "coding", "prompt": "Second", "max_tokens": 1},
        ]
        self.corpus.write_text("".join(json.dumps(value) + "\n" for value in values))
        for path in (self.corpus, self.output, Path(str(self.output) + ".part")):
            with self.subTest(alias=path):
                result = subprocess.run(
                    [str(BINARY), "--trace-corpus", str(self.corpus), "--trace-output", str(self.output),
                     "--prefill-output", str(path)], capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 2)
                self.assertIn("must not alias", result.stderr)
                self.assertNotIn("ggml_cuda_init", result.stderr)
        summary = self.root / "prefill.json"
        for path in (summary, Path(str(summary) + ".part")):
            path.write_text("preserve")
            result = subprocess.run(
                [str(BINARY), "--trace-corpus", str(self.corpus), "--trace-output", str(self.output),
                 "--prefill-output", str(summary)], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 2)
            self.assertIn("Refusing to overwrite", result.stderr)
            self.assertEqual(path.read_text(), "preserve")
            path.unlink()

    def test_invalid_corpus_never_loads_a_model_or_writes_a_trace(self):
        base = {"id": "train", "split": "train", "category": "coding", "prompt": "A prompt", "max_tokens": 2}
        cases = [
            [],
            [base],
            [{**base, "max_tokens": True}],
            [{**base, "max_tokens": 1.5}],
            [{**base, "max_tokens": 2**32 + 2}],
            [{**base, "max_tokens": -1}],
            [{**base, "unsupported": 5}],
            [base, {**base, "id": "test", "split": "test"}],
            [base, {**base, "prompt": "Different", "split": "test"}],
            [{**base, "split": "test"}, {**base, "prompt": "Different"}],
        ]
        for values in cases:
            with self.subTest(values=values):
                result = self.invoke(values)
                self.assertEqual(result.returncode, 2)
                self.assertIn("vrambridge-trace:", result.stderr)
                self.assertNotIn("loading model", result.stderr)
                self.assertFalse(self.output.exists())
                self.assertFalse(Path(str(self.output) + ".part").exists())

    def test_existing_final_or_partial_is_preserved(self):
        values = [
            {"id": "train", "split": "train", "category": "coding", "prompt": "First", "max_tokens": 2},
            {"id": "test", "split": "test", "category": "coding", "prompt": "Second", "max_tokens": 2},
        ]
        for path in (self.output, Path(str(self.output) + ".part")):
            path.write_text("preserve")
            result = self.invoke(values)
            self.assertEqual(result.returncode, 2)
            self.assertIn("Refusing to overwrite", result.stderr)
            self.assertEqual(path.read_text(), "preserve")
            path.unlink()

    def test_gpu_request_requires_an_initialized_device_before_model_load(self):
        values = [
            {"id": "train", "split": "train", "category": "coding", "prompt": "First", "max_tokens": 1},
            {"id": "test", "split": "test", "category": "coding", "prompt": "Second", "max_tokens": 1},
        ]
        self.corpus.write_text("".join(json.dumps(value) + "\n" for value in values))
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = "-1"
        environment.pop("GGML_BACKEND_PATH", None)
        result = subprocess.run(
            [str(BINARY), "--trace-corpus", str(self.corpus), "--trace-output", str(self.output),
             "--model", "/nonexistent-test-model.gguf", "--n-gpu-layers", "auto"],
            env=environment, capture_output=True, text=True, timeout=20,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("no usable GPU device initialized", result.stderr)
        self.assertNotIn("failed to load model", result.stderr)
        self.assertFalse(self.output.exists())
        self.assertFalse(Path(str(self.output) + ".part").exists())


if __name__ == "__main__":
    unittest.main()
