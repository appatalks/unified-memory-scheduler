#!/usr/bin/env python3
"""Publish allowlisted aggregates from the private architecture study."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def paired_summary(baseline, candidate, layers, top_k):
    require(len(baseline) == len(candidate) == 3, "Three pairs required.")
    rows = []
    for category in ("coding", "reasoning", "prose"):
        before, after, paired, request_changes, window_ms, accepted_per_window = [], [], [], [], [], []
        hashes = set()
        reads, swaps, cpu_changes = [], [], []
        for control, experiment in zip(baseline, candidate):
            for run, mode in ((control, "scalar"), (experiment, "mtp")):
                require(run["status"] == "complete" and run["owned_backend_stopped"], "Incomplete/unstopped run.")
                require(run["engine"] == "strata" and run["mode"] == mode and run["window_capacity"] == 2,
                        "Wrong paired engine/mode/capacity.")
                require(run["observed_control_environment"]["STRATA_PREFILL_CPU_SHARE"] == "0",
                        "Dynamic prefill is not a confirmation control.")
                require(run["observed_control_environment"]["STRATA_ROUTE_TAIL_SKIP"] == "0",
                        "Approximate tail skipping invalidates the control.")
            x = next(r for r in control["requests"] if r["category"] == category)
            y = next(r for r in experiment["requests"] if r["category"] == category)
            require(x["id"] == y["id"] and x["prompt_tokens"] == y["prompt_tokens"], "Unaligned paired requests.")
            require(x["output_tokens"] == y["output_tokens"] == x["streamed_token_ids"] == y["streamed_token_ids"] == 512,
                    "Incomplete output counts.")
            require(x["output_token_sha256"] == y["output_token_sha256"], "Paired greedy outputs differ.")
            hashes.update((x["output_token_sha256"], y["output_token_sha256"]))
            sx, sy = x["engine_native_times_not_cross_engine_normalized"], y["engine_native_times_not_cross_engine_normalized"]
            require(sx["drafts_offered"] == 0 and sy["drafts_offered"] > 0, "Unacknowledged speculation control.")
            require(sx["lookups"] == 512 * layers * top_k, "Scalar target-row accounting changed.")
            require(sy["lookups"] == (1 + 2 * sy["drafts_offered"]) * layers * top_k,
                    "Two-row verification accounting changed.")
            before.append(x["accepted_output_after_first_tok_s"])
            after.append(y["accepted_output_after_first_tok_s"])
            paired.append(100 * (after[-1] / before[-1] - 1))
            request_changes.append(100 * (x["request_seconds"] / y["request_seconds"] - 1))
            cpu_changes.append(100 * (y["process_interval"]["cpu_seconds"] / x["process_interval"]["cpu_seconds"] - 1))
            window_ms.append(1000 * 511 / y["accepted_output_after_first_tok_s"] / sy["drafts_offered"])
            accepted_per_window.append(511 / sy["drafts_offered"])
            reads.append({"baseline": x["generation_interval"]["physical_read_bytes"],
                          "candidate": y["generation_interval"]["physical_read_bytes"]})
            swaps.extend((x["process_swap_bytes_at_end"], y["process_swap_bytes_at_end"]))
        require(len(hashes) == 1, "Confirmation greedy outputs vary across repetitions.")
        median_before, median_after = statistics.median(before), statistics.median(after)
        rows.append({"category": category, "pairs": 3, "output_tokens_per_request": 512,
            "baseline_accepted_wall_tok_s": before, "candidate_accepted_wall_tok_s": after,
            "baseline_median_tok_s": median_before, "candidate_median_tok_s": median_after,
            "ratio_of_medians_gain_percent": 100 * (median_after / median_before - 1),
            "paired_gain_percent": paired, "median_paired_gain_percent": statistics.median(paired),
            "request_throughput_gain_percent": request_changes, "request_cpu_time_change_percent": cpu_changes,
            "all_greedy_hashes_equal": True, "output_token_sha256": next(iter(hashes)),
            "median_post_first_verify_window_ms": statistics.median(window_ms),
            "accepted_outputs_per_post_first_window": accepted_per_window,
            "generation_physical_read_bytes": reads, "all_owned_process_swap_zero": all(s == 0 for s in swaps)})
    return rows


def publish(root, output):
    require(not output.exists(), "Refusing to overwrite a reviewed summary.")
    assets = read(root / "private-assets.json")
    baseline = [read(root / f"strata-scalar-fixed-prefill-r{i}.json") for i in range(3)]
    candidate = [read(root / f"strata-mtp-fixed-prefill-r{i}.json") for i in range(3)]
    paired = paired_summary(baseline, candidate, assets["geometry"]["block_count"], assets["geometry"]["expert_used_count"])
    sanity = read(root / "sanity-validation.json")
    require(sanity["all_passed"] and len(sanity["cases"]) == 15, "Natural-EOS sanity gate incomplete.")
    gates = {}
    for name in ("canonical-b2", "canonical-b4", "optimized-v-canonical"):
        report = read(root / (name + ".json"))
        gates[name] = {"passed": report["all_checks_passed"],
            "positions": report["steps_per_request"] * len(report["requests"]),
            "max_abs": max(r["max_abs_logit_error"] for r in report["requests"]),
            "max_relative_l2": max(r["max_relative_l2"] for r in report["requests"])}
    require(gates["canonical-b2"]["passed"] and gates["canonical-b4"]["passed"]
            and not gates["optimized-v-canonical"]["passed"], "Numerical gate history changed.")
    native = {}
    for category in ("coding", "reasoning", "prose"):
        rates = []
        for i in range(3):
            run = read(root / f"native-scalar-audited-r{i}.json")
            require(run["status"] == "complete" and run["owned_backend_stopped"], "Native scalar cohort incomplete.")
            record = next(r for r in run["requests"] if r["category"] == category)
            require(record["audit_all_streamed_positions_match"] and record["output_tokens"] == 512,
                    "Native output audit incomplete.")
            rates.append(record["accepted_output_after_first_tok_s"])
        native[category] = {"median_accepted_wall_tok_s": statistics.median(rates), "samples": rates}
    screens = {}
    for name in ("native-canonical-screen", "native-mtp-screen-v2", "native-mtp-window2-screen",
                 "strata-mtp-screen", "strata-mtp-window2-screen"):
        report = read(root / (name + ".json"))
        require(report["status"] == "complete", "An exploratory screen is incomplete.")
        screens[name] = {r["category"]: r["accepted_output_after_first_tok_s"] for r in report["requests"]}
    report = {"schema_version": 1, "kind": "reviewed_model_agnostic_architecture_study",
        "date": "2026-10-19", "research_repository": "https://github.com/appatalks/unified-memory-scheduler",
        "downstream_repository": "https://github.com/appatalks/Synapse",
        "downstream_read_only_revision": "545a7ebf4a6dea3dd4a1ac04760057eed2de1b15",
        "target": "Qwen3.8-Flash-Next GSQ-RCO IQ3_S", "context": 16384, "kv_type": "F16",
        "target_weights_unchanged": True, "all_selected_target_experts_preserved": True,
        "hardware": {"cpu": "Ryzen 9 9950X", "gpu": "RTX 5070 Ti 16GB", "host_ram_gib": 123.48},
        "target_shards": assets["shards"], "dense_tensors_byte_verified": assets["dense_tensors_byte_identical"],
        "native_expert_layers_verified": len(assets["native_expert_rows"]),
        "shared_resident_experts": assets["common_seed_experts"],
        "shared_expert_payload_bytes": assets["common_seed_payload_bytes"],
        "frozen_profile_sha256": digest(root / "common-frozen-strata-profile.bin"),
        "executed_strata_version": "0.1.43", "strata_source_revision": "7695859aa13afb248bdb4d36f9b2e95707db079d",
        "strata_binary_sha256": "73b66e36e16a8336c09fd834ae00fe009edc1d3b535f9f65fd0f6fc1c29ee47e",
        "strata_binary_source_binding_proven": False,
        "native_backend_pin": "620d2225cc436ab0b001e3200e590e3b286ef9ab",
        "metric": "(accepted_output_tokens - 1) / (completion_wall_time - first_output_wall_time)",
        "confirmation": paired, "native_optimized_scalar": native, "single_run_exploratory_screens": screens,
        "natural_eos_sanity": sanity, "native_numerical_gates": gates,
        "strata_full_vocabulary_gate_established": False,
        "strata_greedy_consistency_only": True, "gpt_60_tok_s_goal_met": False,
        "synapse_changes_made": False, "installed_runtime_or_defaults_changed": False,
        "generic_policy_prototype_promoted": False,
        "policy": {"capability_based_not_checkpoint_named": True,
            "criterion": "(draft_cost + verify_cost + commit_cost) / expected_accepted_outputs < scalar_target_cost",
            "unsupported_or_numerically_incompatible_modes": "remain disabled",
            "do_not_hardcode_two_rows_for_every_model": True},
        "blockers": [
            "Native canonical and optimized scalar execution fail the frozen interchange gate.",
            "Strata matching greedy output hashes and bounded sanity do not establish the full-vocabulary gate.",
            "The executed native-pack serve verifier has no valid no-token-graph CLI ablation.",
            "The verified downstream hybrid path explicitly rejects speculation; this is not a drop-in setting.",
            "Qwen has a trained MTP capability; GPT-OSS-120B transfer has not been established."],
        "controls_and_retained_failures": [
            "Original complete Strata profile admitted3991 experts; excluded from matched timing.",
            "Default dynamic CPU prefill-share cohorts are exploratory; final confirmation disables it.",
            "Native SSE can omit UTF-8-buffered raw IDs; every native timing is separately output-audited.",
            "Initial collector stop-reason mismatch and missing native MTP opt-in are retained failed runs.",
            "Strata/native draft quantization, vocabulary and hidden-state conventions differ."],
        "source_sha256": digest(Path(__file__))}
    with output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = publish(args.private_results, args.output)
    for row in report["confirmation"]:
        print(row["category"], round(row["baseline_median_tok_s"], 2), "->",
              round(row["candidate_median_tok_s"], 2), f"({row['ratio_of_medians_gain_percent']:+.2f}%)")


if __name__ == "__main__":
    main()
