#!/usr/bin/env python3

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

TEST, CUDA_TEST, LLAMA_SOURCE, GGML_BASE = [Path(value).resolve() for value in sys.argv[1:5]]
del sys.argv[1:5]
FIXTURE = Path(__file__).resolve().parents[1] / "tools/trace/make_moe_fixture.py"


class QuantizedCudaTests(unittest.TestCase):
    def test_packed_gpu_dot_matches_native_cpu_bits(self):
        plugins = sorted(GGML_BASE.parent.glob("libggml-cpu*.so"))
        self.assertTrue(plugins, "CPU plugin required")
        with tempfile.TemporaryDirectory() as directory:
            for architecture, quant, width in (("qwen3moe", "IQ4_NL", 64), ("gpt-oss", "IQ4_NL", 64),
                                                ("qwen3moe", "Q6_K", 256), ("gpt-oss", "Q6_K", 256),
                                                ("qwen3moe", "MXFP4", 64), ("gpt-oss", "MXFP4", 64)):
                with self.subTest(architecture=architecture, quant=quant):
                    model = Path(directory) / f"{architecture}-{quant}.gguf"
                    created = subprocess.run([sys.executable, str(FIXTURE), "--llama-source", str(LLAMA_SOURCE),
                                              "--ggml-base", str(GGML_BASE), "--architecture", architecture,
                                              "--expert-type", quant, "--embedding", str(width),
                                              "--feed-forward", str(max(width, 128)), "--output", str(model)],
                                             capture_output=True, errors="replace", text=True, timeout=30)
                    self.assertEqual(created.returncode, 0, created.stderr)
                    result = subprocess.run([str(TEST), str(model), str(plugins[0])], capture_output=True,
                                            errors="replace", text=True, timeout=60)
                    self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    gpu = subprocess.run([str(CUDA_TEST)], capture_output=True, text=True, timeout=30)
    if gpu.returncode == 77:
        print("CUDA unavailable or insufficient headroom")
        sys.exit(77)
    if gpu.returncode:
        print(gpu.stderr, file=sys.stderr)
        sys.exit(gpu.returncode)
    unittest.main()
