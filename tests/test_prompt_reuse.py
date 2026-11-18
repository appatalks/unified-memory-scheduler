import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "tools/benchmark/vrambridge-prompt-reuse.py"
spec = importlib.util.spec_from_file_location("prompt_reuse", SCRIPT)
assert spec and spec.loader
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


class PromptReuseTests(unittest.TestCase):
    def test_exact_replay_cases_preserve_prefix_and_only_edit_one_token(self):
        tokens = [4] * 8192
        cases = dict(study.replay_cases(tokens, [6, 7], 9))
        self.assertEqual(cases["cold"], tokens)
        self.assertEqual(cases["repeat"], tokens)
        self.assertEqual(cases["append"], tokens + [6, 7])
        self.assertEqual(cases["return-original"], tokens)
        self.assertEqual(sum(a != b for a, b in zip(tokens, cases["middle-edit"])), 1)
        self.assertEqual(tokens, [4] * 8192)
        with self.assertRaises(ValueError):
            study.replay_cases(tokens, [6], 4)

    def test_incomplete_or_invalid_native_samples_are_not_successes(self):
        event = {"tokens": [4] * 16, "timings": {
            "cache_n": 4096, "prompt_n": 4096, "prompt_ms": 1000,
            "predicted_n": 16, "predicted_ms": 500, "predicted_per_second": 30}}
        summary = study.summarize_response(event, 1.5, 16)
        self.assertEqual(summary["processed_tokens"], 4096)
        self.assertEqual(summary["prompt_cache_tokens"], 4096)
        self.assertNotIn("tokens", summary)
        for key, value in (("predicted_n", 15), ("prompt_ms", float("nan")),
                           ("cache_n", -1), ("predicted_ms", 0)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                study.summarize_response({**event, "timings": {**event["timings"], key: value}}, 1.5, 16)
        with self.assertRaises(ValueError):
            study.summarize_response({**event, "tokens": [4] * 15}, 1.5, 16)
        one = {**event, "tokens": [4], "timings": {
            **event["timings"], "predicted_n": 1, "predicted_ms": 0, "predicted_per_second": 0}}
        self.assertFalse(study.summarize_response(one, 1, 1)["performance_measurement"])

    def test_probability_gate_requires_complete_unique_vocabulary_and_masks(self):
        def event(values):
            return {"completion_probabilities": [{"top_logprobs": values}]}
        values = [{"id": 0, "logprob": -1.0}, {"id": 1, "logprob": -2.0},
                  {"id": 2, "logprob": None}]
        self.assertTrue(study.compare_probabilities(event(values), event(list(reversed(values))), 3)["passes"])
        changed = [{"id": 0, "logprob": -1.1}, *values[1:]]
        self.assertFalse(study.compare_probabilities(event(values), event(changed), 3)["passes"])
        for invalid in (values[:2], [values[0], values[0], values[2]],
                        [*values[:2], {"id": 2, "logprob": -3.0}]):
            with self.assertRaises(ValueError):
                study.compare_probabilities(event(values), event(invalid), 3)


if __name__ == "__main__":
    unittest.main()
