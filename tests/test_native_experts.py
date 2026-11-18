#!/usr/bin/env python3

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

TEST, LLAMA_SOURCE, GGML_BASE = [Path(value).resolve() for value in sys.argv[1:4]]
del sys.argv[1:4]
FIXTURE = Path(__file__).resolve().parents[1] / "tools/trace/make_moe_fixture.py"


class NativeExpertTests(unittest.TestCase):
    def test_native_f32_and_validation(self):
        plugins = sorted(GGML_BASE.parent.glob("libggml-cpu*.so"))
        self.assertTrue(plugins, "CPU plugin required")
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "untrained.gguf"
            created = subprocess.run([sys.executable, str(FIXTURE), "--llama-source", str(LLAMA_SOURCE),
                                      "--ggml-base", str(GGML_BASE), "--architecture", "qwen3moe",
                                      "--expert-type", "F32", "--output", str(model)],
                                     capture_output=True, errors="replace", text=True, timeout=30)
            self.assertEqual(created.returncode, 0, created.stderr)
            result = subprocess.run([str(TEST), str(model), str(plugins[0])], capture_output=True,
                                    errors="replace", text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
