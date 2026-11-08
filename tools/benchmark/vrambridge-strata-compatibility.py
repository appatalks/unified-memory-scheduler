#!/usr/bin/env python3
"""Bounded same-engine Strata verification; reuse upstream speed evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import threading

MAGIC = 0x315041434C525453


def collect_tokens(stream):
    result = []
    for token in stream:
        if token is None:
            continue
        if type(token) is not int or token < 0:
            raise ValueError("Invalid engine token.")
        result.append(token)
    return result


def read_logits(path, expected_rows=None):
    import numpy as np
    with Path(path).open("rb") as stream:
        header = stream.read(16)
        if len(header) != 16:
            raise ValueError("Missing logits capture header.")
        magic, vocab = struct.unpack("<QQ", header)
        if magic != MAGIC or not 2 <= vocab <= 1048576:
            raise ValueError("Invalid logits capture header.")
        blocks = []
        total = 0
        while raw := stream.read(8):
            if len(raw) != 8:
                raise ValueError("Truncated logits block header.")
            rows, = struct.unpack("<Q", raw)
            if not 1 <= rows <= 8 or total + rows > (expected_rows if expected_rows is not None else 1024):
                raise ValueError("Unexpected logits copy: row count does not match scoring.")
            data = stream.read(rows * vocab * 4)
            if len(data) != rows * vocab * 4:
                raise ValueError("Truncated logits block.")
            block = np.frombuffer(data, dtype="<f4").reshape(rows, vocab)
            if not np.isfinite(block).all():
                raise ValueError("Non-finite captured logits.")
            blocks.append(block)
            total += rows
    if total == 0 or (expected_rows is not None and total != expected_rows):
        raise ValueError("Captured logits do not cover every scored position.")
    return np.concatenate(blocks), vocab


def scored_logits(path, positions):
    import numpy as np
    rows, vocab = read_logits(path)
    lines = [line.split("\t") for line in Path(positions).read_text().splitlines()]
    chosen, cursor, previous, skipped = [], 0, None, 0
    for fields in lines:
        position, target, top = int(fields[0]), int(fields[1]), int(fields[3])
        if not 0 <= target < vocab or not 0 <= top < vocab:
            raise ValueError("Scored token lies outside captured vocabulary.")
        boundary = previous is not None and position <= previous
        limit = 7 if boundary else 0
        for padding in range(limit + 1):
            if cursor >= len(rows):
                raise ValueError("Scored position has no captured row.")
            row = rows[cursor].astype("float64")
            argmax = int(np.argmax(row))
            lse = row[argmax] + np.log(np.exp(row - row[argmax]).sum())
            if (argmax == top and abs(row[target] - lse - float(fields[2])) <= 2e-8 and
                    abs(row[top] - lse - float(fields[4])) <= 2e-8):
                chosen.append(cursor)
                cursor += 1
                break
            if padding == limit:
                raise ValueError("Captured head does not reproduce the engine's scored position.")
            cursor += 1
            skipped += 1
        previous = position
    if len(rows) - cursor > 7:
        raise ValueError("Unexpected extra captured rows after the scored continuation.")
    skipped += len(rows) - cursor
    return rows[chosen], {"captured_rows": len(rows), "scored_rows": len(chosen),
                         "unscored_window_suffix_rows": skipped, "vocab": vocab}


def compare(reference, candidate):
    import numpy as np
    if reference.shape != candidate.shape:
        raise ValueError("Logit shapes differ.")
    error = candidate.astype("float64") - reference
    absolute = np.max(np.abs(error), axis=1)
    relative = np.linalg.norm(error, axis=1) / np.maximum(np.linalg.norm(reference, axis=1), 1e-30)
    passed = (absolute <= 0.01) & (relative <= 0.001)
    return {"positions": len(passed), "passed": int(passed.sum()),
            "max_absolute": float(absolute.max()), "max_relative_l2": float(relative.max()),
            "all_passed": bool(passed.all())}


def run(args):
    os.umask(0o077)
    root = args.output.resolve()
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.source.resolve()))
    from serve.server import StrataEngine
    records = json.loads(args.corpus.read_text())["records"]
    records = [row for row in records if row.get("split") in ("held", "test")][:args.requests]
    if not records:
        raise ValueError("No held-out token corpus records.")
    report = {"schema_version": 1, "scope": "same_strata_engine_bounded_compatibility",
              "upstream_speed_cohort_repeated": False, "arms": [], "status": "incomplete"}
    outputs = None
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("STRATA_", "VRAMBRIDGE_", "GGML_", "LLAMA_")) and k != "LD_PRELOAD"}
    env.update(STRATA_ROUTE_TAIL_SKIP="0", STRATA_ROUTE_PRIOR="0", STRATA_ROUTE_RESIDENT="0",
               STRATA_ROUTE_RESIDENT_MTP="0", STRATA_DROP_CACHE_ON_EXIT="0", STRATA_PREFILL_CPU_SHARE="0",
               STRATA_IQ_MT_MIN="1", STRATA_SPEC_PROB="1")
    env["LD_LIBRARY_PATH"] = str(args.binary.resolve().parent)
    base = ["--pack", str(args.pack.resolve()), "--native", str(args.model.resolve()),
            "--ple-gguf", str(args.ple.resolve()), "--ple-io", "mmap",
            "--expert-cache", str(args.experts), "--prefill", "512", "--spec", "2",
            "--max-context", "16384", "--kv", "fp16", "--suffix-draft", "0", "--lookup-chain", "0",
            "--prompt-cache", "0", "--adapt-every", "0", "--adapt-swaps", "0", "--pcie-frac", "0",
            "--pool-workers", "7", "--gpu", "0"]
    captures = {}
    cache_identity = None
    try:
        for mode in ("generate-control", "scalar", "mtp"):
            profile = root / (mode + ".profile.bin")
            shutil.copyfile(args.profile, profile)
            settings = base + ["--expert-profile", str(profile)]
            arm_env = dict(env)
            if mode == "mtp":
                settings += ["--mtp", str(args.mtp.resolve()), "--spec-min-p", "0"]
            if mode != "generate-control":
                follow = root / (mode + ".follow.txt")
                follow.write_text("".join(",".join(map(str, ids)) + "\n" for ids in outputs))
                arm_env.update(STRATA_FOLLOW=str(follow),
                               STRATA_FOLLOW_LOGPOS=str(root / (mode + ".positions.tsv")),
                               VRAMBRIDGE_STRATA_LOGITS=str(root / (mode + ".logits.bin")),
                               VRAMBRIDGE_STRATA_VOCAB=str(args.vocab), LD_PRELOAD=str(args.capture.resolve()))
            engine = None
            arm = {"mode": mode, "owned_backend_stopped": False, "outputs": []}
            report["arms"].append(arm)
            try:
                engine = StrataEngine(str(args.binary.resolve()), settings, cwd=str(root),
                                      log=str(root / (mode + ".engine.log")), env=arm_env)
                arm["info"] = engine.info
                if ((not args.byte_budget and engine.info.get("expert_slots") != args.experts) or
                        type(engine.info.get("expert_slots")) is not int or engine.info["expert_slots"] <= 0 or
                        engine.info.get("kv") != "fp16"):
                    raise ValueError("Actual cache residency/KV differs from the frozen control.")
                current_cache = (engine.info["expert_slots"], engine.info["expert_cache_mib"])
                if cache_identity is None:
                    cache_identity = current_cache
                elif current_cache != cache_identity:
                    raise ValueError("Scalar and MTP cache placements have different capacity.")
                if mode == "mtp" and engine.info.get("drafting") != "mtp":
                    raise ValueError("Requested MTP did not activate.")
                for row in records:
                    ids = collect_tokens(engine.generate(row["tokens"], args.tokens, {}, threading.Event()))
                    if len(ids) != args.tokens:
                        raise ValueError("Incomplete bounded output.")
                    arm["outputs"].append(ids)
                if outputs is None:
                    outputs = arm["outputs"]
                elif arm["outputs"] != outputs:
                    raise ValueError("Token forcing did not preserve the control continuation.")
            finally:
                if engine is not None:
                    process = engine.proc
                    arm["last"] = engine.last
                    engine.close()
                    arm["owned_backend_stopped"] = process is None or process.poll() is not None
            if mode != "generate-control":
                positions = (root / (mode + ".positions.tsv")).read_text().splitlines()
                if len(positions) != len(records) * args.tokens:
                    raise ValueError("Teacher-forced score does not cover exactly the requested positions.")
                expected = [(len(record["tokens"]) - 1 + offset, token)
                            for record, continuation in zip(records, outputs)
                            for offset, token in enumerate(continuation)]
                actual = [(int(line.split("\t")[0]), int(line.split("\t")[1])) for line in positions]
                if actual != expected:
                    raise ValueError("Scored positions/targets differ from the fixed continuation.")
                captures[mode], arm["capture"] = scored_logits(
                    root / (mode + ".logits.bin"), root / (mode + ".positions.tsv"))
                if arm["capture"]["vocab"] != args.vocab:
                    raise ValueError("Captured vocabulary differs from declared model vocabulary.")
        report["numerics"] = compare(captures["scalar"], captures["mtp"])
        report["status"] = "passed" if report["numerics"]["all_passed"] else "numerical_gate_failed"
    except Exception as exc:
        report["status"], report["error"] = "operation_failed", str(exc)
    finally:
        report["binary_sha256"] = hashlib.file_digest(args.binary.open("rb"), "sha256").hexdigest()
        (root / "private-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "numerics", "error") if key in report}))
    return 0 if report["status"] == "passed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "binary", "capture", "model", "ple", "pack", "mtp", "profile", "corpus", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--vocab", type=int, required=True)
    parser.add_argument("--experts", type=int, default=3067)
    parser.add_argument("--byte-budget", action="store_true",
                        help="Interpret the upstream expert argument as its byte-budget proxy; require equal actual cache capacity.")
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--requests", type=int, default=3)
    args = parser.parse_args()
    if not 2 <= args.vocab <= 1048576 or not 1 <= args.tokens <= 128 or not 1 <= args.requests <= 6:
        parser.error("Bounded vocabulary/token/request limits exceeded.")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
