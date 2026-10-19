#!/usr/bin/env python3

import json
import copy
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


BINARY = Path(sys.argv.pop(1)).resolve()


class HybridLayerTests(unittest.TestCase):
    def test_cpu_partition_math_matches_unsplit_reference(self):
        for extra in ([], ["--gpu-reduce"]):
            result = subprocess.run([str(BINARY), "--self-test", "--samples", "5", "--repetitions", "5", *extra],
                                    capture_output=True, text=True, check=True, timeout=30)
            report = json.loads(result.stdout)
            self.assertTrue(report["all_checks_passed"])
            self.assertTrue(report["independent_scalar_reference"])
            self.assertFalse(report["full_model_acceleration_claim"])
            self.assertFalse(report["hardware_gpu_partition"])
            self.assertEqual(len(report["routing_weights"]), 4)
            for actual, expected in zip(report["routing_weights"], [0.05, 0.15, 0.30, 0.50]):
                self.assertAlmostEqual(actual, expected, places=6)
            self.assertEqual(len(report["partitions"]), 5)
            self.assertIn("cuda_mmq_precision_override", report)
            for split in report["partitions"]:
                self.assertTrue(split["all_checks_passed"])
                self.assertTrue(split["modes"]["serial"]["all_checks_passed"])
                self.assertEqual(len(split["correctness"]), 5)
                self.assertEqual(len(split["modes"]["serial"]["samples_ms"]), 5)
                self.assertGreater(split["modes"]["serial"]["median_ms"], 0)
                self.assertLess(split["modes"]["serial"]["worst_relative_l2_error"], 0.0001)
                self.assertTrue(all(check["passed"] for check in split["correctness"]))

    def test_invalid_cli_does_not_load_weights(self):
        for args in ([], ["--self-test", "--layer", "-1"], ["--self-test", "--threads", "0"],
                     ["--self-test", "--repetitions", "1", "--samples", "2"],
                     ["--self-test", "--layer", "1garbage"], ["--unknown", "value"],
                     ["--self-test", "--rows", "0"], ["--self-test", "--rows", "257"],
                     ["--self-test", "--rows", "2garbage"], ["--self-test", "--gpu-row-limit", "-1"],
                     ["--self-test", "--gpu-row-limit", "257"], ["--self-test", "--cpu-row-limit", "-1"],
                     ["--self-test", "--cpu-row-limit", "257"],
                     ["--self-test", "--operator-check", "--rows", "65"],
                     ["--self-test", "--activation-pattern", "unknown"],
                     ["--self-test", "--expert-ids", "1,2,3,4"],
                     ["--expert-ids", "1,1,2,3"], ["--expert-ids", "0,1,2,128"],
                     ["--expert-ids", "0,1,2"], ["--expert-ids", "0,1,2,3,"]):
            result = subprocess.run([str(BINARY), *args], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 2)
            self.assertIn("vrambridge-hybrid-layer:", result.stderr)
            self.assertNotIn("ggml_cuda_init", result.stderr)

    def test_compatible_down_rejects_incompatible_modes_before_backend_loading(self):
        for extra in (["--gpu-reduce"], ["--operator-check"], ["--rows", "65"],
                      ["--gpu-row-limit", "2"], ["--cpu-row-limit", "2"], ["--reverse-splits"]):
            result = subprocess.run(
                [str(BINARY), "--self-test", "--compatible-down", *extra],
                capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 2)
            self.assertIn("vrambridge-hybrid-layer:", result.stderr)
            self.assertNotIn("ggml_cuda_init", result.stderr)
        for flag in ("--compatible-gpu-first", "--compatible-dp4a"):
            result = subprocess.run(
                [str(BINARY), "--self-test", flag],
                capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 2)
            self.assertIn("requires --compatible-down", result.stderr)
            self.assertNotIn("ggml_cuda_init", result.stderr)

    def test_operator_checks_include_observer_controls_and_no_raw_values(self):
        result = subprocess.run(
            [str(BINARY), "--self-test", "--operator-check", "--rows", "7", "--gpu-row-limit", "2",
             "--samples", "2", "--repetitions", "2"],
            capture_output=True, text=True, check=True, timeout=30)
        report = json.loads(result.stdout)
        self.assertTrue(report["all_checks_passed"])
        self.assertFalse(report["timings_qualified"])
        diagnostics = report["operator_diagnostics"]
        self.assertFalse(diagnostics["raw_activations_recorded"])
        self.assertFalse(diagnostics["performance_measurement"])
        self.assertEqual(diagnostics["candidate_backend"], "cpu_self_test")
        self.assertEqual(len(diagnostics["samples"]), 2)
        def check_no_raw_payload(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    self.assertNotIn(key, ("values", "packed", "effective"))
                    check_no_raw_payload(child)
            elif isinstance(value, list):
                for child in value:
                    check_no_raw_payload(child)
        for sample in diagnostics["samples"]:
            self.assertTrue(sample["cpu_output_fnv1a64"].isdigit())
            self.assertTrue(sample["gpu_output_fnv1a64"].isdigit())
            self.assertTrue(sample["cpu_observer_control"]["passed"])
            self.assertTrue(sample["gpu_observer_control"]["passed"])
            self.assertIsNone(sample["first_failing_operator"])
            self.assertEqual(len(sample["chained_operator_checks"]), 9)
            self.assertEqual(len(sample["identical_input_replays"]), 9)
            self.assertTrue(all(check["passed"] for check in sample["identical_input_replays"]))
            self.assertFalse(sample["cpu_effective_q8_replays_applicable"])
            self.assertEqual(sample["cpu_q8_roundtrip_self_check"]["different_scale_blocks"], 0)
            self.assertEqual(sample["cpu_q8_roundtrip_self_check"]["different_code_values"], 0)
            self.assertIsNone(sample["effective_q8_propagation"])
            check_no_raw_payload(sample)

    def test_zero_and_rounding_tie_inputs_keep_scalar_reference(self):
        for pattern in ("zero", "q8-ties"):
            result = subprocess.run(
                [str(BINARY), "--self-test", "--operator-check", "--rows", "3",
                 "--samples", "2", "--repetitions", "2", "--activation-pattern", pattern],
                capture_output=True, text=True, check=True, timeout=30)
            report = json.loads(result.stdout)
            self.assertEqual(report["activation_pattern"], pattern)
            self.assertTrue(report["all_checks_passed"])
            self.assertTrue(report["independent_scalar_reference"])

    def test_batched_rows_and_timed_uploads_preserve_scalar_math(self):
        for rows in (2, 7):
            for extra in ([], ["--gpu-reduce"]):
                with self.subTest(rows=rows, extra=extra):
                    result = subprocess.run(
                        [str(BINARY), "--self-test", "--rows", str(rows), "--upload-weights",
                         "--samples", "3", "--repetitions", "3", "--gpu-row-limit", "2",
                         "--cpu-row-limit", "2", *extra],
                        capture_output=True, text=True, check=True, timeout=30)
                    report = json.loads(result.stdout)
                    self.assertTrue(report["all_checks_passed"])
                    self.assertTrue(report["independent_scalar_reference"])
                    self.assertFalse(report["warm_cache_microbenchmark"])
                    self.assertTrue(report["weight_uploads_in_timing"])
                    self.assertEqual(report["weight_upload_destination"], "cpu_self_test")
                    self.assertEqual(report["rows_per_expert"], rows)
                    self.assertEqual(report["gpu_row_limit"], 2)
                    self.assertEqual(report["cpu_row_limit"], 2)
                    self.assertEqual(report["activation_bytes"], 64 * rows * 4)
                    for split in report["partitions"]:
                        self.assertEqual(split["weight_upload_bytes_per_iteration"],
                                         report["selected_weight_bytes"] * split["split_index"] // 4)
                        self.assertTrue(all(check["passed"] for check in split["correctness"]))
                        self.assertGreaterEqual(split["modes"]["serial"]["median_weight_upload_ms"], 0)

    def test_maximum_row_count_preserves_all_output_rows(self):
        result = subprocess.run(
            [str(BINARY), "--self-test", "--rows", "256", "--gpu-reduce", "--reverse-splits",
             "--gpu-row-limit", "1", "--cpu-row-limit", "1",
             "--samples", "1", "--repetitions", "1"],
            capture_output=True, text=True, check=True, timeout=30)
        report = json.loads(result.stdout)
        self.assertTrue(report["all_checks_passed"])
        self.assertEqual(report["rows_per_expert"], 256)
        self.assertEqual(report["activation_bytes"], 65536)
        self.assertTrue(report["warm_cache_microbenchmark"])
        self.assertFalse(report["weight_uploads_in_timing"])
        self.assertEqual([split["split_index"] for split in report["partitions"]], [4, 3, 2, 1, 0])
        self.assertEqual(report["split_order"], "descending_gpu_experts")
        self.assertTrue(all(split["weight_upload_bytes_per_iteration"] == 0 for split in report["partitions"]))

    def test_cpu_activation_split_preserves_scalar_math_and_accounts_for_copies(self):
        for extra in ([], ["--gpu-reduce", "--reverse-splits"]):
            result = subprocess.run(
                [str(BINARY), "--self-test", "--cpu-activation", "--rows", "7",
                 "--samples", "3", "--repetitions", "3", "--upload-weights",
                 "--gpu-row-limit", "2", "--cpu-row-limit", "3", *extra],
                capture_output=True, text=True, check=True, timeout=30)
            report = json.loads(result.stdout)
            self.assertTrue(report["all_checks_passed"])
            self.assertTrue(report["independent_scalar_reference"])
            self.assertEqual(report["execution_split"], "cpu_activation_gpu_down")
            self.assertEqual(report["cpu_activation_quantization"], "none")
            self.assertEqual(report["activation_stage_copy_destination"], "cpu_self_test")
            self.assertFalse(report["full_model_acceleration_claim"])
            for split in report["partitions"]:
                count = split["split_index"]
                self.assertEqual(split["expert_partition_scope"], "down_projection_only")
                self.assertEqual(split["cpu_gate_up_experts"], 4)
                self.assertEqual(split["gpu_gate_up_experts"], 0)
                self.assertEqual(split["cpu_down_experts"], 4)
                self.assertEqual(split["gpu_down_experts"], 0)
                self.assertEqual(split["activation_stage_copy_bytes_per_iteration"], 96 * count * 7 * 4)
                self.assertEqual(split["weight_upload_bytes_per_iteration"], (64 * 96 + 64) * 4 * count)
                self.assertTrue(split["all_checks_passed"])
                self.assertGreaterEqual(split["modes"]["serial"]["median_cpu_activation_prepare_ms"], 0)
                self.assertGreaterEqual(split["modes"]["serial"]["median_activation_stage_copy_ms"], 0)

    def test_cpu_activation_split_handles_maximum_rows_and_single_row_chunks(self):
        result = subprocess.run(
            [str(BINARY), "--self-test", "--cpu-activation", "--rows", "256", "--gpu-reduce",
             "--gpu-row-limit", "1", "--cpu-row-limit", "1", "--samples", "1", "--repetitions", "1"],
            capture_output=True, text=True, check=True, timeout=30)
        report = json.loads(result.stdout)
        self.assertTrue(report["all_checks_passed"])
        self.assertEqual(report["partitions"][-1]["activation_stage_copy_bytes_per_iteration"], 96 * 4 * 256 * 4)

    def test_cpu_activation_split_operator_checks_cover_both_stages(self):
        result = subprocess.run(
            [str(BINARY), "--self-test", "--cpu-activation", "--operator-check", "--rows", "7",
             "--gpu-row-limit", "2", "--cpu-row-limit", "3", "--samples", "3", "--repetitions", "3"],
            capture_output=True, text=True, check=True, timeout=30)
        report = json.loads(result.stdout)
        self.assertTrue(report["all_checks_passed"])
        self.assertFalse(report["timings_qualified"])
        for sample in report["operator_diagnostics"]["samples"]:
            self.assertTrue(sample["cpu_activation_observer_control"]["passed"])
            self.assertTrue(sample["cpu_observer_control"]["passed"])
            self.assertTrue(sample["gpu_observer_control"]["passed"])
            self.assertTrue(sample["uninstrumented_cpu_gpu"]["passed"])
            self.assertEqual(len(sample["chained_operator_checks"]), 9)
            self.assertEqual([check["backend"] for check in sample["chained_operator_checks"]],
                             ["cpu"] * 5 + ["gpu_down_stage"] * 4)
            self.assertTrue(all(check["passed"] for check in sample["chained_operator_checks"]))
            self.assertIsNone(sample["first_failing_operator"])

    def routed_fixture(self):
        return {"schema_version": 1, "kind": "synthetic_routed_fixture", "records": [{
            "request": "routed-scalar", "category": "coding", "layer": 17, "rows": 3,
            "native_rows": 8, "row_offset": 0, "hidden": 64,
            "input": [0.2 * math.sin(i * 0.17) for i in range(64 * 3)],
            "expert_ids": [2, 31, 78, 127, 4, 19, 55, 102, 127, 55, 2, 102],
            "probabilities": [0.05, 0.15, 0.3, 0.5, 0.6, 0.1, 0.2, 0.1, 0.25, 0.25, 0.25, 0.25],
        }]}

    def test_actual_route_layout_uses_each_rows_original_ids_and_probabilities(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "fixture.json"
            source.write_text(json.dumps(self.routed_fixture()))
            result = subprocess.run(
                [str(BINARY), "--self-test", "--cpu-activation", "--activation-capture", str(source)],
                capture_output=True, text=True, check=True, timeout=30)
            report = json.loads(result.stdout)
            self.assertTrue(report["all_checks_passed"])
            self.assertTrue(report["independent_scalar_reference"])
            self.assertEqual(report["activation_source"], "synthetic_routed_fixture")
            self.assertEqual(report["expert_bank_size"], 8)
            self.assertEqual(report["active_experts_per_row"], 4)
            self.assertEqual(report["distinct_ordered_routes"], 3)
            self.assertEqual(sum(report["expert_row_counts"]), 12)
            self.assertFalse(report["routing_probabilities_modified"])
            self.assertFalse(report["routing_experts_modified"])
            self.assertFalse(report["performance_measurement"])
            for key in ("cpu_vs_reference", "candidate_vs_reference", "candidate_vs_cpu"):
                self.assertTrue(report[key]["passed"])
            self.assertTrue(report["operator_diagnostics"]["all_checks_passed"])

    def test_malformed_actual_route_payloads_fail_before_backend_loading(self):
        changes = [
            ("rows", 0), ("rows", 65), ("rows", True), ("rows", 2 ** 32 + 3), ("hidden", 2880),
            ("request", {"input": [1, 2, 3]}), ("category", "unknown"),
            ("row_offset", -1), ("row_offset", 6), ("native_rows", 2 ** 32 + 8),
            ("expert_ids", [2, 2, 78, 127] * 3), ("expert_ids", [2, 31, 78, 128] * 3),
            ("expert_ids", [2.5, 31, 78, 127] * 3), ("probabilities", [-1] * 12),
            ("expert_ids", [2 ** 32 + 2, 31, 78, 127] * 3),
            ("probabilities", [0] * 12), ("input", [0] * 63),
        ]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "fixture.json"
            for key, value in changes:
                fixture = copy.deepcopy(self.routed_fixture())
                fixture["records"][0][key] = value
                source.write_text(json.dumps(fixture))
                result = subprocess.run(
                    [str(BINARY), "--self-test", "--cpu-activation", "--activation-capture", str(source)],
                    capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("vrambridge-hybrid-layer:", result.stderr)
                self.assertNotIn("ggml_cuda_init", result.stderr)
            source.write_text(json.dumps(self.routed_fixture()))
            for extra in (["--capture-record", "9"], ["--rows", "3"], ["--layer", "17"]):
                result = subprocess.run(
                    [str(BINARY), "--self-test", "--cpu-activation", "--activation-capture", str(source), *extra],
                    capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 2)
                self.assertNotIn("ggml_cuda_init", result.stderr)

    def test_unqualified_observed_capture_is_rejected_before_backend_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "unqualified.json"
            source.write_text(json.dumps({
                "schema_version": 1, "kind": "actual_prefill_ffn_capture",
                "observer_full_logit_checks_passed": False, "records": [{}],
            }))
            result = subprocess.run(
                [str(BINARY), "--model", "/missing/target.gguf", "--output", str(Path(directory) / "result.json"),
                 "--cpu-activation", "--activation-capture", str(source)],
                capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 2)
            self.assertIn("passing full-logit observer controls", result.stderr)
            self.assertNotIn("ggml_cuda_init", result.stderr)

    def test_refuses_to_overwrite_final_or_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "probe.json"
            for path in (output, Path(str(output) + ".part")):
                path.write_text("preserve")
                result = subprocess.run([str(BINARY), "--self-test", "--output", str(output)],
                                        capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 2)
                self.assertIn("Refusing to overwrite", result.stderr)
                self.assertEqual(path.read_text(), "preserve")
                path.unlink()


if __name__ == "__main__":
    unittest.main()
