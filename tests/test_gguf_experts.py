#!/usr/bin/env python3

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

INSPECT, LLAMA_SOURCE, GGML_BASE = [Path(value).resolve() for value in sys.argv[1:4]]
del sys.argv[1:4]
sys.path.insert(0, str(LLAMA_SOURCE / "gguf-py"))
import gguf

FIXTURE = Path(__file__).resolve().parents[1] / "tools/trace/make_moe_fixture.py"


class GGUFExpertTests(unittest.TestCase):
    def test_selected_expert_slices_match_independent_python_dequantizer(self):
        with tempfile.TemporaryDirectory() as directory:
            for arch, quant in (("llama", "F32"), ("qwen3moe", "Q8_0"), ("gpt-oss", "MXFP4")):
                with self.subTest(architecture=arch, quant=quant):
                    model = Path(directory) / f"{arch}.gguf"
                    created = subprocess.run([sys.executable, str(FIXTURE), "--llama-source", str(LLAMA_SOURCE),
                                              "--ggml-base", str(GGML_BASE), "--architecture", arch,
                                              "--expert-type", quant, "--output", str(model)],
                                             capture_output=True, text=True, timeout=30)
                    self.assertEqual(created.returncode, 0, created.stderr)
                    result = subprocess.run([str(INSPECT), str(model)], capture_output=True, text=True, timeout=30)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    actual = json.loads(result.stdout)
                    tensors = {tensor.name: tensor for tensor in gguf.GGUFReader(model).tensors}
                    for part in ("gate", "up", "down"):
                        for kind in ("weight", "bias"):
                            key = part if kind == "weight" else f"{part}_bias"
                            tensor = tensors.get(f"blk.0.ffn_{part}_exps.{kind}")
                            if tensor is None:
                                self.assertEqual(actual[key], [])
                                continue
                            values = gguf.quants.dequantize(tensor.data, tensor.tensor_type)
                            expected = []
                            for expert in (2, 0, 1):
                                row = values[expert].reshape(-1)
                                expected.extend(float(row[i]) for i in (0, len(row) // 3, len(row) // 2, len(row) - 1))
                            self.assertEqual(len(actual[key]), len(expected))
                            for left, right in zip(actual[key], expected):
                                self.assertAlmostEqual(left, right, delta=1e-6 * max(1.0, abs(right)))

    def test_missing_model_fails(self):
        result = subprocess.run([str(INSPECT), "/missing/model.gguf"], capture_output=True, text=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing", result.stderr)


if __name__ == "__main__":
    unittest.main()
