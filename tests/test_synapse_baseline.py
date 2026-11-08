import importlib.util
import io
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "tools/benchmark/vrambridge-synapse-baseline.py"
spec = importlib.util.spec_from_file_location("synapse_baseline", SCRIPT)
assert spec and spec.loader
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


class BaselineTests(unittest.TestCase):
    def settings(self, variant="classic"):
        return baseline.trial_settings(Path("/models/target.gguf"), 16384, variant, 8192)

    def state(self, variant="classic"):
        return {**self.settings(variant), "hybrid_active": variant != "standard",
                "hybrid_verified": variant != "standard", "hybrid_hosted": variant == "hosted",
                "hybrid_hosted_verified": variant == "hosted", "draft_model": None}

    def test_exact_capacity_and_scalar_precision_are_preserved(self):
        for variant in baseline.VARIANTS:
            settings = self.settings(variant)
            self.assertEqual(settings["context"], 16384)
            self.assertEqual(settings["kv_cache_type"], "f16")
            self.assertEqual(settings["draft_type"], "none")
            self.assertEqual(settings["expert_cache_mib"], 0 if variant == "standard" else 8192)
            baseline.verify_activation(self.state(variant), settings, variant)

    def test_context_rounding_is_not_hidden(self):
        for context in (True, 511, 142128, 16384.0):
            with self.assertRaises(ValueError):
                baseline.trial_settings(Path("/models/target.gguf"), context, "classic", 8192)
        self.assertEqual(baseline.trial_settings(Path("/models/target.gguf"), 142336, "classic", 8192)["context"], 142336)

    def test_missing_markers_or_wrong_engine_fail(self):
        for changes in ({"hybrid_verified": False}, {"hybrid_active": False},
                        {"hybrid_hosted": True}, {"context": 8192},
                        {"kv_cache_type": "q8_0"}, {"draft_model": "/models/draft.gguf"}):
            with self.subTest(changes=changes), self.assertRaises(RuntimeError):
                baseline.verify_activation({**self.state(), **changes}, self.settings(), "classic")
        with self.assertRaises(RuntimeError):
            baseline.verify_activation({**self.state("hosted"), "hybrid_hosted_verified": False},
                                       self.settings("hosted"), "hosted")

    def test_alternating_order_does_not_mutate_input(self):
        variants = ["standard", "classic", "hosted"]
        self.assertEqual(baseline.trial_order(variants, 0), variants)
        self.assertEqual(baseline.trial_order(variants, 1), ["hosted", "classic", "standard"])
        self.assertEqual(variants, ["standard", "classic", "hosted"])

    def test_failed_or_partial_trials_do_not_become_successful_medians(self):
        sample = {"category": "coding", "wall_decode_tok_s": 30, "end_to_end_tok_s": 20,
                  "generated_text_sha256": "a"}
        runs = [
            {"status": "completed", "context": 16384, "variant": "standard", "samples": [sample]},
            {"status": "failed", "context": 16384, "variant": "classic", "samples": [sample]},
            {"status": "running", "context": 16384, "variant": "hosted", "samples": [sample]},
        ]
        summary = baseline.summarize(runs)
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["variant"], "standard")
        self.assertEqual(summary[0]["median_wall_decode_tok_s"], 30)
        self.assertTrue(summary[0]["matches_standard_text"])

    def test_hash_difference_is_not_reported_as_equivalence(self):
        sample = {"category": "coding", "wall_decode_tok_s": 30, "end_to_end_tok_s": 20,
                  "generated_text_sha256": "a"}
        runs = [
            {"status": "completed", "context": 16384, "variant": "standard", "samples": [sample]},
            {"status": "completed", "context": 16384, "variant": "classic",
             "samples": [{**sample, "generated_text_sha256": "b"}]},
        ]
        self.assertFalse(baseline.summarize(runs)[1]["matches_standard_text"])

    def test_capacity_deadline_cancels_the_probe_without_claiming_success(self):
        server = mock.Mock()
        server.tuner.cancelled = threading.Event()

        def probe(runtime, context):
            self.assertTrue(server.tuner.cancelled.wait(1))
            raise RuntimeError("Auto Tune was cancelled.")

        server.tuner._probe.side_effect = probe
        with self.assertRaisesRegex(RuntimeError, "capacity remains unverified"):
            baseline.capacity_probe(server, object(), 16384, 0.01)

    def test_capacity_success_and_unrelated_failures_are_preserved(self):
        server = mock.Mock()
        server.tuner.cancelled = threading.Event()
        server.tuner._probe.return_value = {"capacity_passed": True}
        self.assertEqual(baseline.capacity_probe(server, object(), 16384, 1),
                         {"capacity_passed": True})
        self.assertFalse(server.tuner.cancelled.is_set())
        server.tuner._probe.side_effect = RuntimeError("anchor retrieval failed")
        with self.assertRaisesRegex(RuntimeError, "anchor retrieval failed"):
            baseline.capacity_probe(server, object(), 16384, 1)

    def test_explicit_runtime_marker_is_required(self):
        runtime = mock.Mock()
        runtime.status.return_value = {"state": "ready", "logs": ["EXPERIMENT enabled"]}
        baseline.wait_runtime_marker(runtime, "EXPERIMENT enabled", timeout=0.1)
        with self.assertRaisesRegex(RuntimeError, "activation marker"):
            baseline.wait_runtime_marker(runtime, "missing", timeout=0.01)

    def test_marker_tracker_preserves_stream_and_captures_early_lines(self):
        seen = []
        runtime = SimpleNamespace(_read_logs=lambda process, key: seen.extend(process.stdout))
        observed = baseline.track_runtime_marker(runtime, "EXPERIMENT enabled")
        runtime._read_logs(SimpleNamespace(stdout=io.StringIO("first\nEXPERIMENT enabled\nlast\n")), "key")
        self.assertTrue(observed.is_set())
        self.assertEqual(seen, ["first\n", "EXPERIMENT enabled\n", "last\n"])

    def test_causal_singleton_contract_rejects_mismatched_runtime_facts(self):
        log = ("n_ubatch = 1\ncausal_attn = 1\n"
               "n_slots = 1, n_ctx_slot = 131072\n"
               "K (f16): 100 MiB, V (f16): 100 MiB\n"
               "CUDA0 model buffer size = 100.00 MiB\n")
        baseline.verify_causal_singleton_log(log, 131072)
        for before, after in (("n_ubatch = 1", "n_ubatch = 512"),
                              ("causal_attn = 1", "causal_attn = 0"),
                              ("n_slots = 1", "n_slots = 2"),
                              ("n_ctx_slot = 131072", "n_ctx_slot = 16384"),
                              ("K (f16)", "K (q8_0)"),
                              ("CUDA0 model buffer size = 100.00", "CUDA0 model buffer size = 0.00")):
            with self.subTest(before=before), self.assertRaises(RuntimeError):
                baseline.verify_causal_singleton_log(log.replace(before, after), 131072)
        with self.assertRaises(RuntimeError):
            baseline.verify_causal_singleton_log(log + "\nn_ubatch = 512\n", 131072)


if __name__ == "__main__":
    unittest.main()
