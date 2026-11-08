import importlib.util
import os
from pathlib import Path
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "tools/benchmark/vrambridge-tandem-screen.py"
spec = importlib.util.spec_from_file_location("tandem_screen", SCRIPT)
assert spec and spec.loader
screen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(screen)


class TandemScreenTests(unittest.TestCase):
    def test_environment_drops_inherited_experiments_and_forces_no_skipping(self):
        with mock.patch.dict(os.environ, {"LLAMA_MOE_CPU": "auto", "LLAMA_MOE_TAIL_SKIP": "7",
                                         "GGML_CUDA_Q8_CACHE": "1", "LD_LIBRARY_PATH": "/other"}):
            reference = screen.environment("reference", 8, Path("private-stats"))
            fixed = screen.environment("tandem-fixed", 8, Path("private-stats"))
            defaults = screen.environment("tandem-defaults", 8, Path("private-stats"))
        self.assertNotIn("LLAMA_MOE_CPU", reference)
        self.assertNotIn("GGML_CUDA_Q8_CACHE", reference)
        self.assertNotIn("LD_LIBRARY_PATH", reference)
        self.assertEqual(fixed["LLAMA_MOE_CPU"], "0.2")
        self.assertEqual(defaults["LLAMA_MOE_NO_DEFAULTS"], "0")
        for env in (reference, fixed, defaults):
            self.assertEqual(env["LLAMA_MOE_TAIL_SKIP"], "0")
            self.assertEqual(env["LLAMA_MOE_PIN_EXPS"], "0")

    def test_activation_requires_cache_and_hybrid_without_fallback(self):
        log = "CUDA0 MoE cache size = 8000.00 MiB\nhybrid mode: 20% of the misses are uploaded\n"
        self.assertEqual(screen.activation(log, "tandem-fixed")["expert_cache_mib"], 8000)
        for invalid in ("", log + "not supported by the CPU path",
                        log + "the cache is disabled", log.replace("8000.00", "0.00"),
                        log.replace("misses are uploaded", "unconfirmed")):
            with self.subTest(log=invalid), self.assertRaises(RuntimeError):
                screen.activation(invalid, "tandem-fixed")
        with self.assertRaises(RuntimeError):
            screen.activation(log, "tandem-defaults")
        screen.activation(log + "tandem.cpp defaults", "tandem-defaults")
        gpu_only = "CUDA0 MoE cache size = 4096.00 MiB\n"
        screen.activation(gpu_only, "reference-cache")
        screen.activation(gpu_only, "tandem-cache")
        with self.assertRaises(RuntimeError):
            screen.activation(gpu_only, "tandem-fixed")

    def test_incomplete_runs_have_no_speed_medians(self):
        run = {"arm": "reference", "status": "completed", "samples": [
            {"category": "coding", "wall_decode_tok_s": 20, "end_to_end_tok_s": 10,
             "first_token_seconds": 1}]}
        partial = screen.summarize([run], ["reference"], 2, ["coding"])[0]
        self.assertNotIn("workloads", partial)
        full = screen.summarize([run], ["reference"], 1, ["coding"])[0]
        self.assertEqual(full["workloads"][0]["wall_decode_tok_s"], 20)
        failed = {**run, "status": "failed"}
        self.assertNotIn("workloads", screen.summarize([failed], ["reference"], 1, ["coding"])[0])

    def test_components_do_not_enable_cpu_sharing_and_require_actual_markers(self):
        env = screen.environment("tandem-launch", 8, Path("private-stats"))
        self.assertNotIn("LLAMA_MOE_CPU", env)
        self.assertEqual(env["GGML_SCHED_BATCH_INPUTS"], "3")
        with self.assertRaises(RuntimeError):
            screen.component_activation("requested flags only", "tandem-launch")
        markers = "\n".join(screen.COMPONENTS["tandem-launch"][1])
        screen.component_activation(markers, "tandem-launch")
        with self.assertRaises(RuntimeError):
            screen.component_activation(markers, "tandem-q8")


if __name__ == "__main__":
    unittest.main()
