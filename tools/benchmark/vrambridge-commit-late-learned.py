#!/usr/bin/env python3
"""Train a tiny causal admission predictor on one trace and score distinct traces."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sqlite3


spec = importlib.util.spec_from_file_location("commit_late_screen", Path(__file__).with_name("vrambridge-commit-late-screen.py"))
assert spec is not None and spec.loader is not None
screen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(screen)


def feature(count, age, rank, adjacent_rank=None):
    count_bin = min(count, 8)
    age_bin = 1 if age == 1 else 2 if age <= 3 else 4 if age <= 7 else 8 if age <= 15 else 16 if age <= 31 else 32
    result = (count_bin, age_bin, rank)
    return result if adjacent_rank is None else (*result, adjacent_rank)


def examples(tokens, warmup, adjacent_route=False):
    if warmup < 63 or warmup >= len(tokens):
        raise ValueError("Training and holdout require 63 complete prior tokens")
    history = [deque() for _ in tokens[0]]
    counts = [Counter() for _ in tokens[0]]
    known_cold = [{} for _ in tokens[0]]
    for token_index, layers in enumerate(tokens):
        for layer, current in enumerate(layers):
            if token_index >= warmup:
                previous = tokens[token_index - 1][layer]
                adjacent = (layers[layer - 1] if layer else tokens[token_index - 1][-1]) if adjacent_route else None
                candidates = []
                for expert, (last_token, rank) in known_cold[layer].items():
                    count = counts[layer][expert]
                    if count >= 3:
                        adjacent_rank = (adjacent["selected"].index(expert) if expert in adjacent["selected"]
                                         else len(adjacent["selected"])) if adjacent else None
                        candidates.append((expert, feature(count, token_index - last_token, rank, adjacent_rank)))
                yield {"candidates": candidates, "admitted": current["admitted"],
                       "release": adjacent["start"] if adjacent else previous["end"], "deadline": current["start"],
                       "weight_bytes": sum(current["sizes"])}
            if len(history[layer]) == 63:
                counts[layer].subtract(history[layer].popleft())
            history[layer].append(current["selected"])
            counts[layer].update(current["selected"])
            for rank, expert in enumerate(current["selected"]):
                if expert in current["cold"]:
                    known_cold[layer][expert] = (token_index, rank)
                else:
                    known_cold[layer].pop(expert, None)


def fit(training):
    counts = defaultdict(lambda: [0, 0])
    for row in training:
        for expert, bucket in row["candidates"]:
            counts[bucket][0] += 1
            counts[bucket][1] += expert in row["admitted"]
    return dict(counts)


def choose(candidates, model, threshold, minimum_support):
    eligible = []
    for expert, bucket in candidates:
        support, hits = model.get(bucket, (0, 0))
        probability = (hits + 1) / (support + 20)
        if support >= minimum_support and probability >= threshold:
            eligible.append((-probability, -support, bucket, expert))
    return min(eligible)[-1] if eligible else None


def score(rows, model, threshold, minimum_support, service_ns):
    total = Counter()
    for row in rows:
        if row["deadline"] < row["release"]:
            raise ValueError("Prediction source follows its target deadline")
        total["layer_positions"] += 1
        total["actual_admissions"] += len(row["admitted"])
        candidate = choose(row["candidates"], model, threshold, minimum_support)
        if candidate is None:
            continue
        total["predictions"] += 1
        total["weight_prediction_bytes"] += row["weight_bytes"]
        if candidate in row["admitted"]:
            total["matched_admissions"] += 1
            total["useful_weight_bytes"] += row["weight_bytes"]
            if row["deadline"] - row["release"] >= service_ns:
                total["independently_timely_matches"] += 1
    predictions, matched, admitted = total["predictions"], total["matched_admissions"], total["actual_admissions"]
    return {**dict(total), "predictions": predictions, "matched_admissions": matched,
            "actual_admissions": admitted, "useful_weight_bytes": total["useful_weight_bytes"],
            "weight_prediction_bytes": total["weight_prediction_bytes"],
            "independently_timely_matches": total["independently_timely_matches"],
            "precision": matched / predictions if predictions else None,
            "admission_recall": matched / admitted if admitted else None,
            "independently_timely_recall": total["independently_timely_matches"] / admitted if admitted else None}


def load_profile(paths, warmup):
    cpu, timeline, request = (Path(value) for value in paths)
    verified = screen.analyze(cpu, timeline, request, warmup)
    report, frames, ranges, _ = screen.windows.read_inputs(cpu, timeline, request)
    tokens, _ = screen.complete_tokens(screen.dispatches(frames, ranges))
    prompt_hash = report.get("prompt_sha256")
    if not isinstance(prompt_hash, str) or not re.fullmatch(r"[a-f0-9]{64}", prompt_hash):
        raise ValueError("A private prompt identity hash is required")
    settings = report.get("applied_settings", {})
    profile = {"expert_cache_mib": settings.get("expert_cache_mib"),
               "hybrid_plan_sha256": settings.get("hybrid_plan_sha256"),
               "hybrid_transfer": settings.get("hybrid_transfer"),
               "decode_pipeline": settings.get("decode_pipeline"), "async_ids": settings.get("async_ids"),
               "cpu_threads": report.get("cpu_threads"), "batch_threads": report.get("batch_threads")}
    if (type(profile["expert_cache_mib"]) is not int or profile["expert_cache_mib"] < 1
            or not isinstance(profile["hybrid_plan_sha256"], str)
            or not re.fullmatch(r"[a-f0-9]{64}", profile["hybrid_plan_sha256"])
            or profile["hybrid_transfer"] != "direct" or profile["decode_pipeline"] is not True
            or profile["async_ids"] is not True or type(profile["cpu_threads"]) is not int
            or type(profile["batch_threads"]) is not int):
        raise ValueError("A verified direct-admission profile and plan identity are required")
    return tokens, verified, prompt_hash, {p.name: screen.windows.file_hash(p) for p in (cpu, timeline, request)}, profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", nargs=3, required=True, metavar=("CPU", "TIMELINE", "REQUEST"))
    parser.add_argument("--test", nargs=4, action="append", required=True, metavar=("LABEL", "CPU", "TIMELINE", "REQUEST"))
    parser.add_argument("--warmup-tokens", type=int, default=64)
    parser.add_argument("--minimum-support", type=int, default=20)
    parser.add_argument("--adjacent-route", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.warmup_tokens < 63 or args.minimum_support < 1:
            raise ValueError("Invalid warmup or support threshold")
        if args.output.exists():
            raise ValueError("Refusing to overwrite an existing report")
        train, train_info, train_hash, train_inputs, profile = load_profile(args.train, args.warmup_tokens)
        model = fit(examples(train, args.warmup_tokens, args.adjacent_route))
        tests = {}
        prompt_hashes = {train_hash}
        for label, *paths in args.test:
            if not re.fullmatch(r"[A-Za-z0-9_-]+", label) or label in tests:
                raise ValueError("Duplicate or unsafe holdout label")
            tokens, info, prompt_hash, inputs, current_profile = load_profile(paths, args.warmup_tokens)
            if (prompt_hash in prompt_hashes or info["model"] != train_info["model"]
                    or info["actual_context"] != train_info["actual_context"] or current_profile != profile):
                raise ValueError("Holdout must use a distinct prompt and matching model, context and recorded profile")
            prompt_hashes.add(prompt_hash)
            tests[label] = {"input_sha256": inputs, "prompt_sha256": prompt_hash,
                            "complete_decode_tokens": info["complete_decode_tokens"],
                            "service_ns": info["independent_expert_copy_service_ns"],
                            "thresholds": {str(t): score(examples(tokens, args.warmup_tokens, args.adjacent_route), model, t,
                                                          args.minimum_support, info["independent_expert_copy_service_ns"])
                                           for t in (0.1, 0.2, 0.3, 0.5)}}
        result = {
            "schema_version": 1, "study": "technical_trace_trained_causal_admission_predictor",
            "model": train_info["model"], "actual_context": train_info["actual_context"],
            "recorded_profile": profile,
            "warmup_complete_tokens": args.warmup_tokens, "minimum_bucket_support": args.minimum_support,
            "training": {"input_sha256": train_inputs, "prompt_sha256": train_hash,
                         "complete_decode_tokens": train_info["complete_decode_tokens"],
                         "bucket_count": len(model), "candidate_examples": sum(v[0] for v in model.values()),
                         "positive_examples": sum(v[1] for v in model.values())},
            "features": ("past 63-token selection count, age and previous router rank of an expert last observed CPU-cold"
                         + (", plus its rank in the already-observed preceding-layer route" if args.adjacent_route else "")),
            "adjacent_route": args.adjacent_route,
            "release": ("preceding-layer pre-dispatch route snapshot" if args.adjacent_route
                        else "previous same-layer CPU graph completion"),
            "score": "(training hits + 1) / (training observations + 20); one expert per layer at fixed thresholds",
            "holdouts": tests,
            "limitations": ["One training request and three distinct holdouts on one model; no model-generalization claim",
                            "No heldout labels influence training or selection thresholds",
                            "No live scratch capacity, copy-engine queue, GPU-to-GPU commit or CPU/PCIe contention is modeled",
                            "Independent timeliness is not evidence that concurrent transfers fit or reduce latency",
                            "Inferred decisions are corroborated against CUDA projection-copy launches per dispatcher",
                            "Trace filenames and plan hashes match; model-shard and all backend library hashes are not recorded"],
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "screen_sha256": hashlib.sha256(Path(screen.__file__).read_bytes()).hexdigest(),
        }
        with args.output.open("x", encoding="utf-8") as out:
            json.dump(result, out, indent=2, sort_keys=True, allow_nan=False)
            out.write("\n")
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as error:
        parser.exit(2, f"error: {error}\n")
    print(args.output)


if __name__ == "__main__":
    os.umask(0o077)
    main()
