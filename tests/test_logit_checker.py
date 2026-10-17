#!/usr/bin/env python3

import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest


BINARY = Path(sys.argv.pop(1)).resolve()


class LogitCheckerInputs(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.corpus = self.root / "corpus.jsonl"
        self.output = self.root / "report.json"
        self.reference = self.root / "reference.bin"
        self.record = {"id": "held-out", "split": "test", "category": "coding", "prompt": "A public prompt."}
        self.corpus.write_text(json.dumps(self.record) + "\n")

    def tearDown(self):
        self.temporary.cleanup()

    def invoke(self, *extra):
        return subprocess.run([str(BINARY), "--corpus", str(self.corpus), "--output", str(self.output),
                               "--reference-in", str(self.reference), *extra],
                              capture_output=True, text=True, timeout=10)

    def rejected(self, result):
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("vrambridge-logit-check:", result.stderr)
        self.assertNotIn("ggml_cuda_init", result.stderr)
        self.assertNotIn("loading model", result.stderr)
        self.assertFalse(self.output.exists())

    def test_help_and_required_options(self):
        self.assertEqual(subprocess.run([str(BINARY), "--help"], capture_output=True).returncode, 0)
        self.rejected(subprocess.run([str(BINARY)], capture_output=True, text=True))

    def test_invalid_step_counts(self):
        for value in ("0", "-1", "129", "1.5", "12tail", "99999999999999999"):
            with self.subTest(value=value):
                self.rejected(self.invoke("--steps", value))
        for value in ("0", "-1", "33", "1.5", "4tail"):
            with self.subTest(block=value):
                self.rejected(self.invoke("--batch-check", value))

    def test_colliding_output_paths_are_rejected(self):
        result = self.invoke("--debug-output", str(self.output))
        self.rejected(result)
        self.assertIn("paths overlap", result.stderr)

    def test_all_layer_diagnostics_requires_output_before_backend_initialization(self):
        result = self.invoke("--debug-all-layers")
        self.rejected(result)
        self.assertIn("--debug-all-layers requires --debug-output", result.stderr)

    def test_cpu_reference_does_not_bypass_reference_validation(self):
        self.reference.write_bytes(b"BADMAGIC")
        result = self.invoke("--cpu-reference")
        self.rejected(result)
        self.assertIn("Reference magic mismatch", result.stderr)

    def test_layer_limit_is_bounded_and_requires_diagnostic_output(self):
        for value in ("0", "-1", "1025", "1.5", "4tail"):
            with self.subTest(value=value):
                self.rejected(self.invoke("--debug-layer-limit", value))
        result = self.invoke("--debug-layer-limit", "4")
        self.rejected(result)
        self.assertIn("--debug-layer-limit requires --debug-output", result.stderr)
        result = self.invoke("--debug-layer-limit", "4", "--debug-all-layers",
                             "--debug-output", str(self.root / "debug.json"))
        self.rejected(result)
        self.assertIn("not both", result.stderr)

    def test_attention_diagnostics_requires_output(self):
        result = self.invoke("--debug-attention")
        self.rejected(result)
        self.assertIn("--debug-attention requires --debug-output", result.stderr)

    def test_template_epoch_is_bounded_before_backend_initialization(self):
        for value in ("-1", "1.5", "12tail", "9223372037", "99999999999999999999"):
            with self.subTest(value=value):
                result = self.invoke("--template-epoch", value)
                self.rejected(result)
                if value == "99999999999999999999":
                    self.assertIn("stoll", result.stderr)
                else:
                    self.assertIn("Template epoch must be", result.stderr)
        self.reference.write_bytes(b"BADMAGIC")
        result = self.invoke("--template-epoch", "0")
        self.rejected(result)
        self.assertIn("Reference magic mismatch", result.stderr)

    def test_invalid_or_duplicate_corpus_records(self):
        for records in ([], [{**self.record, "category": "unknown"}],
                        [{**self.record, "prompt": ""}], [{**self.record, "split": "unknown"}],
                        [{**self.record, "category": None}], [self.record, self.record]):
            with self.subTest(records=records):
                self.corpus.write_text("".join(json.dumps(record) + "\n" for record in records))
                self.rejected(self.invoke())

    def test_existing_outputs_are_preserved(self):
        for path in (self.output, Path(str(self.output) + ".part"),
                     self.root / "debug.json", self.root / "debug.json.part"):
            path.write_text("preserve")
            result = self.invoke("--debug-output", str(self.root / "debug.json"))
            self.assertEqual(result.returncode, 2)
            self.assertIn("already exists", result.stderr)
            self.assertEqual(path.read_text(), "preserve")
            path.unlink()

    def test_bad_reference_format_is_rejected_before_backend_initialization(self):
        header = {"schema_version": 1, "steps": 32, "requests": 1, "vocab": 2, "context": 128,
                  "model": "synthetic.gguf", "backend_commit": "620d222", "contains_token_ids": True}
        packed = json.dumps(header).encode()
        prefix = b"VRLOG001" + struct.pack("=Q", len(packed)) + packed
        body = struct.pack("=Qi", 1, 1) + struct.pack("=iff", 1, 0.25, 0.5) * 32
        cases = [b"", b"BADMAGIC", b"VRLOG001", b"VRLOG001" + struct.pack("=Q", 1048577),
                 prefix, prefix + body, prefix + body + b"BADTRAIL",
                 prefix + body + b"ENDLOG01extra", prefix + struct.pack("=Q", 129)]
        for changes in ({"vocab": 0}, {"vocab": 1000001}, {"vocab": 1.5},
                        {"context": 32}, {"requests": 2}, {"steps": 31}, {"contains_token_ids": False}):
            invalid = json.dumps({**header, **changes}).encode()
            cases.append(b"VRLOG001" + struct.pack("=Q", len(invalid)) + invalid)
        for index, value in enumerate(cases):
            with self.subTest(case=index):
                self.reference.write_bytes(value)
                self.rejected(self.invoke())
                self.assertEqual(self.reference.read_bytes(), value)

    def test_missing_gpu_is_rejected_before_model_loading(self):
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = "-1"
        environment.pop("GGML_BACKEND_PATH", None)
        result = subprocess.run(
            [str(BINARY), "--corpus", str(self.corpus), "--output", str(self.output),
             "--reference-out", str(self.reference), "--model", str(self.root / "missing.gguf"),
             "--n-gpu-layers", "all"], env=environment, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("no usable GPU device initialized", result.stderr)
        self.assertNotIn("loading model", result.stderr)
        self.assertFalse(self.reference.exists())
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
