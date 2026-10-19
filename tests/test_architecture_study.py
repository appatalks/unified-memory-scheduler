#!/usr/bin/env python3

import copy
import hashlib
import importlib.util
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools/benchmark" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


assets = load("vrambridge-architecture-assets")
summary = load("vrambridge-architecture-summary")
restore = load("vrambridge-restore-mtp")


class ArchitectureStudyTests(unittest.TestCase):
    def test_partial_profile_never_fills_unranked_experts(self):
        raw = assets.strata_profile_bytes(2, 3, [(1, 2), (0, 1)])
        self.assertEqual(raw[:4], b"STRP")
        self.assertEqual(struct.unpack_from("<5I", raw, 4), (1, 2, 3, 2, 2))
        self.assertEqual(struct.unpack_from("<4H", raw, 24), (1, 2, 0, 1))
        self.assertEqual(struct.unpack_from("<6i", raw, 32), (-1, 1, -1, -1, -1, 0))
        self.assertEqual(len(raw), 56)

    def test_invalid_profiles_are_explicit_errors(self):
        for ranking in ([], [(0, 1), (0, 1)], [(2, 0)], [(0, 3)]):
            with self.subTest(ranking=ranking), self.assertRaises(ValueError):
                assets.strata_profile_bytes(2, 3, ranking)
        with self.assertRaises(ValueError):
            assets.strata_profile_bytes(65536, 3, [(0, 0)])

    def test_native_regions_and_digests(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "first", Path(directory) / "second"
            first.write_bytes(b"xxxpayload"); second.write_bytes(b"ypayload")
            assets.compare_region(first, 3, second, 1, 7)
            self.assertEqual(restore.digest(first), hashlib.sha256(b"xxxpayload").hexdigest())
            with self.assertRaisesRegex(ValueError, "differ"):
                assets.compare_region(first, 3, second, 1, 8)

    def test_derivative_rejects_unapproved_storage_before_reading_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.gguf"
            with self.assertRaisesRegex(ValueError, "approved model root"):
                restore.restore(SimpleNamespace(output=path))
            self.assertFalse(list(Path(directory).iterdir()))

    def pairs(self):
        common = {"status": "complete", "owned_backend_stopped": True, "engine": "strata", "window_capacity": 2,
            "observed_control_environment": {"STRATA_PREFILL_CPU_SHARE": "0", "STRATA_ROUTE_TAIL_SKIP": "0"},
            "requests": []}
        before, after = copy.deepcopy(common), copy.deepcopy(common)
        before["mode"], after["mode"] = "scalar", "mtp"
        for category in ("coding", "reasoning", "prose"):
            record = {"id": category, "category": category, "prompt_tokens": 10, "output_tokens": 512,
                "streamed_token_ids": 512, "output_token_sha256": "same-" + category,
                "accepted_output_after_first_tok_s": 51.1, "request_seconds": 12,
                "engine_native_times_not_cross_engine_normalized": {"drafts_offered": 0, "lookups": 1024},
                "process_interval": {"cpu_seconds": 80},
                "generation_interval": {"elapsed_seconds": 10, "physical_read_bytes": 0},
                "process_swap_bytes_at_end": 0}
            before["requests"].append(record)
            candidate = copy.deepcopy(record)
            candidate.update(accepted_output_after_first_tok_s=63.875, request_seconds=10,
                engine_native_times_not_cross_engine_normalized={"drafts_offered": 256, "lookups": 1026},
                process_interval={"cpu_seconds": 64},
                generation_interval={"elapsed_seconds": 8, "physical_read_bytes": 0})
            after["requests"].append(candidate)
        return [copy.deepcopy(before) for _ in range(3)], [copy.deepcopy(after) for _ in range(3)]

    def test_aggregate_reports_real_paired_gain_without_promoting_quality(self):
        before, after = self.pairs()
        rows = summary.paired_summary(before, after, 2, 1)
        self.assertEqual(len(rows), 3)
        self.assertAlmostEqual(rows[0]["ratio_of_medians_gain_percent"], 25)
        self.assertEqual(rows[0]["paired_gain_percent"], [25, 25, 25])
        self.assertTrue(rows[0]["all_greedy_hashes_equal"])

    def test_mismatched_outputs_cannot_be_promoted(self):
        before, after = self.pairs()
        after[1]["requests"][0]["output_token_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "outputs differ"):
            summary.paired_summary(before, after, 2, 1)

    def test_dynamic_prefill_cannot_be_a_confirmation(self):
        before, after = self.pairs()
        after[1]["observed_control_environment"]["STRATA_PREFILL_CPU_SHARE"] = "1"
        with self.assertRaisesRegex(ValueError, "Dynamic prefill"):
            summary.paired_summary(before, after, 2, 1)

    def test_wrong_target_row_accounting_cannot_be_promoted(self):
        before, after = self.pairs()
        after[0]["requests"][0]["engine_native_times_not_cross_engine_normalized"]["lookups"] = 1024
        with self.assertRaisesRegex(ValueError, "accounting"):
            summary.paired_summary(before, after, 2, 1)


if __name__ == "__main__":
    unittest.main()
