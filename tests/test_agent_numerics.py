import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "tools/benchmark/vrambridge-agent-numerics.py"
spec = importlib.util.spec_from_file_location("agent_numerics", SCRIPT)
assert spec and spec.loader
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def event(values, token=0):
    return {"tokens": [token], "completion_probabilities": [
        {"top_logprobs": [{"id": i, "logprob": value} for i, value in enumerate(values)]}
    ]}


class AgentNumericsTests(unittest.TestCase):
    def test_equal_probabilities_are_matched_by_token_id(self):
        first = event([-1.0, -2.0])
        second = event([-1.0, -2.0])
        second["completion_probabilities"][0]["top_logprobs"].reverse()
        result = probe.compare_probabilities(first, second, 2)
        self.assertTrue(result["passes"])
        self.assertTrue(result["first_token_matches"])
        self.assertEqual(result["max_absolute_logprob_error"], 0)

    def test_numerical_and_token_differences_remain_explicit(self):
        first = event([-1.0, -2.0], token=0)
        changed = event([-1.1, -2.0], token=1)
        result = probe.compare_probabilities(first, changed, 2)
        self.assertFalse(result["passes"])
        self.assertFalse(result["first_token_matches"])

    def test_rejects_partial_duplicate_nonfinite_or_mismatched_masks(self):
        first = event([-1.0, -2.0])
        partial = event([-1.0])
        duplicate = event([-1.0, -2.0])
        duplicate["completion_probabilities"][0]["top_logprobs"][1]["id"] = 0
        nonfinite = event([-1.0, float("nan")])
        masked = event([-1.0, None])
        for invalid in (partial, duplicate, nonfinite, masked):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                probe.compare_probabilities(first, invalid, 2)

    def test_rejects_missing_or_invalid_output_token(self):
        first = event([-1.0, -2.0])
        for invalid in ({"completion_probabilities": first["completion_probabilities"]},
                        event([-1.0, -2.0], token=True),
                        event([-1.0, -2.0], token=2)):
            with self.assertRaises(ValueError):
                probe.compare_probabilities(first, invalid, 2)

    def test_cpu_placement_and_f16_context_are_required(self):
        log = (
            "I load_tensors: CPU_Mapped model buffer size = 1234.00 MiB\n"
            "I llama_kv_cache: K (f16): 1.00 MiB, V (f16): 1.00 MiB\n"
            "I srv load_model: initializing, n_slots = 8, n_ctx_slot = 16384, kv_unified = 'false'\n"
        )
        self.assertEqual(probe.cpu_startup(log, 8, 16384)["placement"], "CPU")
        for invalid in (log.replace("n_slots = 8", "n_slots = 4"),
                        log.replace("V (f16)", "V (q8_0)"),
                        log + "I load_tensors: CUDA0 model buffer size = 2.00 MiB\n"):
            with self.assertRaises(RuntimeError):
                probe.cpu_startup(invalid, 8, 16384)


if __name__ == "__main__":
    unittest.main()
