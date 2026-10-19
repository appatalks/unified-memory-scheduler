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

    def invoke(self, *extra, env=None):
        return subprocess.run([str(BINARY), "--corpus", str(self.corpus), "--output", str(self.output),
                               "--reference-in", str(self.reference), *extra],
                              capture_output=True, text=True, timeout=10, env=env)

    def rejected(self, result):
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("vrambridge-logit-check:", result.stderr)
        self.assertNotIn("ggml_cuda_init", result.stderr)
        self.assertNotIn("loading model", result.stderr)
        self.assertFalse(self.output.exists())

    def test_help_and_required_options(self):
        self.assertEqual(subprocess.run([str(BINARY), "--help"], capture_output=True).returncode, 0)
        self.rejected(subprocess.run([str(BINARY)], capture_output=True, text=True))

    def test_rollback_requires_a_multi_position_reference_comparison(self):
        result = self.invoke("--rollback-check", "--batch-check", "1")
        self.rejected(result)
        self.assertIn("Rollback checking requires a reference", result.stderr)

    def test_invalid_prefill_research_flag_fails_before_backend_loading(self):
        for value in ("", "yes", "2", "-1"):
            result = self.invoke(env={**os.environ, "VRAMBRIDGE_HYBRID_PREFILL_DOWN": value})
            self.rejected(result)
            self.assertIn("VRAMBRIDGE_HYBRID_PREFILL_DOWN must be 0 or 1", result.stderr)

    def test_prefill_diagnostic_controls_cannot_run_without_enabled_mode(self):
        for name in ("VRAMBRIDGE_HYBRID_PREFILL_DOWN_CPU_CONTROL", "VRAMBRIDGE_HYBRID_PREFILL_DOWN_AUDIT",
                     "VRAMBRIDGE_HYBRID_PREFILL_DOWN_LAYER", "VRAMBRIDGE_HYBRID_PREFILL_DOWN_LOCALIZE"):
            for value in ("1", "yes", ""):
                self.rejected(self.invoke(env={**os.environ, "VRAMBRIDGE_HYBRID_PREFILL_DOWN": "0", name: value}))

    def test_localization_reference_is_bounded_before_backend_initialization(self):
        source = self.root / "localization.bin"
        header = json.dumps({"schema_version": 1, "records": 180, "contains_sensitive_activations": True}).encode()
        cases = [b"", b"BADMAGIC", b"VRLOC001" + struct.pack("=Q", 16385),
                 b"VRLOC001" + struct.pack("=Q", len(header)) + header]
        prefix = b"VRLOC001" + struct.pack("=Q", len(header)) + header
        meta = {"request": "held-out", "layer": 0, "component": "input", "rows": 1, "width": 2880, "type": "f32"}
        for change in ({"rows": 513}, {"layer": 36}, {"width": 64}, {"type": "i32"}, {"request": []}):
            serialized = json.dumps({**meta, **change}).encode()
            cases.append(prefix + struct.pack("=Q", len(serialized)) + serialized)
        for data in cases:
            source.write_bytes(data)
            result = self.invoke("--localize-reference", str(source), env={
                **os.environ, "VRAMBRIDGE_HYBRID_PREFILL_DOWN": "0", "GGML_CUDA_DISABLE_FUSION": "1"})
            self.rejected(result)

    def test_localization_focus_and_fusion_guards_are_explicit(self):
        for value in ("ffn", "attention1", "rounding1", "unknown"):
            self.rejected(self.invoke("--localize-focus", value))
        result = self.invoke("--localize-reference", str(self.root / "missing-trace.bin"),
                             env={**os.environ, "GGML_CUDA_DISABLE_FUSION": "0"})
        self.rejected(result)
        self.assertIn("CUDA fusion disabled", result.stderr)

    def test_teacher_forced_export_requires_comparison_and_private_storage(self):
        destination = self.root / "teacher.bin"
        result = subprocess.run(
            [str(BINARY), "--corpus", str(self.corpus), "--output", str(self.output),
             "--reference-out", str(self.reference), "--teacher-forced-reference-out", str(destination)],
            capture_output=True, text=True, timeout=10)
        self.rejected(result)
        self.assertIn("Teacher-forced export requires --reference-in", result.stderr)
        public = self.root / "shared"
        public.mkdir(mode=0o755)
        public.chmod(0o755)
        result = self.invoke("--teacher-forced-reference-out", str(public / "teacher.bin"))
        self.rejected(result)
        self.assertIn("private parent directory", result.stderr)

    def test_localization_distinguishes_route_order_from_expert_membership(self):
        result = subprocess.run([str(BINARY), "--localization-self-test"],
                                capture_output=True, text=True, check=True, timeout=10)
        self.assertEqual(json.loads(result.stdout), {
            "different_ordered_rows": 2, "different_expert_set_rows": 1, "first_different_expert_set_row": 1})
        self.assertNotIn("ggml_cuda_init", result.stderr)

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
        result = self.invoke("--prefill-capture-output", str(self.output))
        self.rejected(result)
        self.assertIn("paths overlap", result.stderr)

    def test_private_prefill_capture_paths_are_checked_before_backend_initialization(self):
        destination = self.root / "capture.json"
        destination.write_text("preserve")
        result = self.invoke("--prefill-capture-output", str(destination))
        self.rejected(result)
        self.assertIn("already exists", result.stderr)
        self.assertEqual(destination.read_text(), "preserve")
        public = self.root / "public"
        public.mkdir(mode=0o755)
        public.chmod(0o755)
        result = self.invoke("--prefill-capture-output", str(public / "capture.json"))
        self.rejected(result)
        self.assertIn("private parent directory", result.stderr)

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

    def test_expert_diagnostics_requires_output(self):
        result = self.invoke("--debug-experts")
        self.rejected(result)
        self.assertIn("--debug-experts requires --debug-output", result.stderr)

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

    def test_reasoning_effort_is_validated_before_backend_initialization(self):
        for value in ("", "invalid", "LOW", "null"):
            with self.subTest(value=value):
                result = self.invoke("--reasoning-effort", value)
                self.rejected(result)
                self.assertIn("Reasoning effort must be", result.stderr)
        self.reference.write_bytes(b"BADMAGIC")
        for value in ("default", "none", "low", "medium", "high"):
            with self.subTest(value=value):
                result = self.invoke("--reasoning-effort", value)
                self.rejected(result)
                self.assertIn("Reference magic mismatch", result.stderr)

    def test_reference_reasoning_effort_must_match(self):
        header = {"schema_version": 1, "steps": 32, "requests": 1, "vocab": 2, "context": 128,
                  "model": "synthetic.gguf", "backend_commit": "620d222", "contains_token_ids": True}
        body = struct.pack("=Qi", 1, 1) + struct.pack("=iff", 1, 0.25, 0.5) * 32
        for recorded, requested in ((None, "none"), ("none", "low"), ("default", "none"), (8, "low")):
            with self.subTest(recorded=recorded, requested=requested):
                metadata = header if recorded is None else {**header, "reasoning_effort": recorded}
                packed = json.dumps(metadata).encode()
                self.reference.write_bytes(b"VRLOG001" + struct.pack("=Q", len(packed)) + packed + body + b"ENDLOG01")
                result = self.invoke("--reasoning-effort", requested)
                self.rejected(result)
                self.assertIn("Reference reasoning effort differs", result.stderr)

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
