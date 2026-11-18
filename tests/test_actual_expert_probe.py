#!/usr/bin/env python3

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

PROBE, CUDA_TEST, LLAMA_SOURCE, GGML_BASE = [Path(value).resolve() for value in sys.argv[1:5]]
del sys.argv[1:5]
FIXTURE = Path(__file__).resolve().parents[1] / "tools/trace/make_moe_fixture.py"


class ActualExpertProbeTests(unittest.TestCase):
    def test_native_cpu_and_gpu_down_on_observed_model_routes(self):
        cpu = sorted(GGML_BASE.parent.glob("libggml-cpu*.so"))
        cuda = GGML_BASE.parent / "libggml-cuda.so"
        self.assertTrue(cpu and cuda.is_file(), "explicit CPU and CUDA plugins required")
        with tempfile.TemporaryDirectory() as directory:
            for arch, quant, activation in (("qwen3moe", "Q8_0", "silu"), ("qwen3moe", "IQ4_NL", "silu"),
                                             ("qwen3moe", "Q6_K", "silu"), ("gpt-oss", "MXFP4", "openai")):
                with self.subTest(architecture=arch, quant=quant):
                    model = Path(directory) / f"{arch}-{quant}.gguf"
                    width = 256 if quant == "Q6_K" else 64
                    created = subprocess.run([sys.executable, str(FIXTURE), "--llama-source", str(LLAMA_SOURCE),
                                              "--ggml-base", str(GGML_BASE), "--architecture", arch,
                                              "--expert-type", quant, "--embedding", str(width),
                                              "--feed-forward", str(max(width, 128)), "--output", str(model)],
                                             capture_output=True, errors="replace", text=True, timeout=30)
                    self.assertEqual(created.returncode, 0, created.stderr)
                    extra = (["--iq4-exact"] if quant == "IQ4_NL" else ["--q6-exact"] if quant == "Q6_K" else
                             ["--mxfp4-exact"] if quant == "MXFP4" else [])
                    env = os.environ.copy()
                    if quant == "Q6_K":
                        strict = subprocess.run([str(PROBE), str(model), str(cpu[0]), str(cuda), activation,
                                                 "--native-activation", "--native-kernel", "--gpu-layers", "0",
                                                 "--case", "coding", "--inject", "--steps", "2", *extra],
                                                capture_output=True, errors="replace", text=True, timeout=60, env=env)
                        if strict.returncode == 2:
                            self.assertIn("quantized layer bits differ", strict.stderr)
                        else:
                            self.assertEqual(strict.returncode, 0, strict.stdout + strict.stderr[-3000:])
                            self.assertTrue(json.loads(strict.stdout)["injection"]["layer_bitwise"])
                        decode = subprocess.run([str(PROBE), str(model), str(cpu[0]), str(cuda), activation,
                                                 "--native-activation", "--native-kernel", "--gpu-layers", "0",
                                                 "--case", "coding", "--inject", "--steps", "2", "--decode-only", *extra],
                                                capture_output=True, errors="replace", text=True, timeout=60, env=env)
                        self.assertEqual(decode.returncode, 0, decode.stdout + decode.stderr[-3000:])
                        self.assertTrue(json.loads(decode.stdout)["injection"]["layer_bitwise"])
                        self.assertEqual(json.loads(decode.stdout)["injection"]["interventions"], 2)
                        env["GGML_CPU_TILED_MM"] = "0"
                    result = subprocess.run([str(PROBE), str(model), str(cpu[0]), str(cuda), activation,
                                             "--native-activation", "--native-kernel", "--gpu-layers", "0",
                                             "--case", "coding", "--benchmark", "3", "--inject", "--steps", "2", *extra],
                                            capture_output=True, errors="replace", text=True, timeout=60, env=env)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr[-3000:])
                    metrics = json.loads(result.stdout)
                    for key in ("baseline_repeat", "observer", "cpu_native_kernel", "native_cpu_gpu_down",
                                "native_cpu_gpu_down_vs_cpu"):
                        self.assertTrue(metrics[key]["pass"], (key, metrics[key]))
                    self.assertTrue(metrics["benchmark"]["all_correct"])
                    self.assertTrue(metrics["injection"]["pass"], metrics["injection"])
                    self.assertEqual(metrics["injection"]["positions"], 3)
                    self.assertEqual(metrics["injection"]["interventions"], 3)
                    self.assertGreater(metrics["benchmark"]["cpu_median_ms"], 0)
                    self.assertGreater(metrics["benchmark"]["cpu_two_workers_median_ms"], 0)
                    self.assertGreater(metrics["benchmark"]["mixed_median_ms"], 0)
                    for key in ("cpu_samples_ms", "cpu_two_workers_samples_ms", "mixed_samples_ms"):
                        self.assertEqual(len(metrics["benchmark"][key]), 3)
                    if quant == "IQ4_NL":
                        self.assertTrue(metrics["benchmark"]["iq4_bitwise"])
                        self.assertEqual(len(metrics["benchmark"]["iq4_exact_samples_ms"]), 3)
                    if quant == "Q6_K":
                        self.assertTrue(metrics["benchmark"]["q6_bitwise"])
                        self.assertEqual(len(metrics["benchmark"]["q6_exact_samples_ms"]), 3)
                    if quant == "MXFP4":
                        self.assertTrue(metrics["benchmark"]["mxfp4_bitwise"])
                        self.assertEqual(len(metrics["benchmark"]["mxfp4_exact_samples_ms"]), 3)
                    for mode in ("original", "compute-only", "native-cpu"):
                        control = subprocess.run([str(PROBE), str(model), str(cpu[0]), str(cuda), activation,
                                                  "--native-activation", "--native-kernel", "--gpu-layers", "0",
                                                  "--case", "coding", "--inject", "--steps", "2", "--injection-mode", mode, *extra],
                                                 capture_output=True, errors="replace", text=True, timeout=60, env=env)
                        self.assertEqual(control.returncode, 0, control.stdout + control.stderr[-3000:])
                        self.assertTrue(json.loads(control.stdout)["injection"]["pass"], mode)


if __name__ == "__main__":
    gpu = subprocess.run([str(CUDA_TEST)], capture_output=True, text=True, timeout=30)
    if gpu.returncode == 77:
        print("CUDA unavailable or insufficient headroom")
        sys.exit(77)
    if gpu.returncode:
        print(gpu.stderr, file=sys.stderr)
        sys.exit(gpu.returncode)
    unittest.main()
