#!/usr/bin/env python3

import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ENGINE, REFERENCE, LLAMA_SOURCE, GGML_BASE = [Path(value).resolve() for value in sys.argv[1:5]]
del sys.argv[1:5]
FIXTURE = Path(__file__).resolve().parents[1] / "tools/trace/make_moe_fixture.py"


def run(command, timeout=90):
    return subprocess.run(command, capture_output=True, text=True, timeout=timeout)


class ExperimentalEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        candidates = sorted(GGML_BASE.parent.glob("libggml-cpu*.so"))
        if not candidates:
            raise RuntimeError("no CPU backend plugin next to the selected GGML library")
        cls.cpu = candidates[0]
        cls.directory = tempfile.TemporaryDirectory()
        cls.models = {}
        for architecture in ("llama", "qwen3moe", "gpt-oss"):
            path = Path(cls.directory.name) / f"{architecture}.gguf"
            result = run([sys.executable, str(FIXTURE), "--llama-source", str(LLAMA_SOURCE),
                          "--ggml-base", str(GGML_BASE), "--architecture", architecture,
                          "--output", str(path)])
            if result.returncode:
                raise RuntimeError(result.stderr)
            cls.models[architecture] = path

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_usage_and_invalid_options_fail_before_loading(self):
        help_result = run([str(ENGINE), "--help"])
        self.assertEqual(help_result.returncode, 0)
        self.assertIn("Experimental", help_result.stdout)
        for args in (["--model", "missing"], ["--threads", "-1"], ["--max-tokens", "1x"],
                     ["--model", "x", "--cpu-backend", "y", "--prompt", "hi", "--gpu-layers", "1"]):
            with self.subTest(args=args):
                self.assertNotEqual(run([str(ENGINE), *args]).returncode, 0)

    def test_greedy_full_logits_match_independent_reference(self):
        prompt = "Review a bounded queue."
        for architecture, model in self.models.items():
            for chunk in (1, 8, 512):
                with self.subTest(architecture=architecture, batch=chunk):
                    reference = run([str(REFERENCE), str(model), str(self.cpu), prompt, "6", str(chunk)])
                    self.assertEqual(reference.returncode, 0, reference.stderr)
                    expected = [json.loads(line) for line in reference.stdout.splitlines()]
                    self.assertTrue(expected)
                    result = run([str(ENGINE), "--model", str(model), "--cpu-backend", str(self.cpu),
                                  "--prompt", prompt, "--threads", "2", "--batch", str(chunk),
                                  "--max-tokens", "6", "--trace-logits"])
                    self.assertEqual(result.returncode, 0, result.stderr)
                    actual = [json.loads(line) for line in result.stdout.splitlines()]
                    self.assertEqual(len(actual), len(expected))
                    for left, right in zip(expected, actual):
                        self.assertEqual(left["token"], right["token"])
                        self.assertEqual(len(left["logits"]), len(right["logits"]))
                        diffs = [abs(a - b) for a, b in zip(left["logits"], right["logits"])]
                        self.assertTrue(all(math.isfinite(x) for x in right["logits"]))
                        self.assertLessEqual(max(diffs), 1e-5)

    def test_context_limit_is_explicit(self):
        result = run([str(ENGINE), "--model", str(self.models["llama"]), "--cpu-backend", str(self.cpu),
                      "--prompt", "a" * 80, "--context", "32", "--max-tokens", "1"])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("exceed the requested context", result.stderr)

    def test_token_ids_and_backend_errors(self):
        common = ["--model", str(self.models["llama"]), "--cpu-backend", str(self.cpu),
                  "--prompt", "Test token output.", "--threads", "2", "--max-tokens", "3"]
        tokens = run([str(ENGINE), *common, "--token-ids"])
        logits = run([str(ENGINE), *common, "--trace-logits"])
        self.assertEqual(tokens.returncode, 0, tokens.stderr)
        self.assertEqual(logits.returncode, 0, logits.stderr)
        self.assertEqual([json.loads(line)["token"] for line in tokens.stdout.splitlines()],
                         [json.loads(line)["token"] for line in logits.stdout.splitlines()])
        failure = run([str(ENGINE), "--model", str(self.models["llama"]), "--cpu-backend", "/missing/plugin",
                       "--prompt", "hi"])
        self.assertNotEqual(failure.returncode, 0)
        self.assertIn("CPU backend could not be loaded", failure.stderr)
        failure = run([str(ENGINE), "--model", "/missing/model.gguf", "--cpu-backend", str(self.cpu),
                       "--prompt", "hi"])
        self.assertNotEqual(failure.returncode, 0)
        self.assertIn("model load failed", failure.stderr)


if __name__ == "__main__":
    unittest.main()
