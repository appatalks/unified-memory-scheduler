#!/usr/bin/env python3
"""Compare exact bounded row reads on an owned SSD fixture; never report model tok/s."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import mmap
import os
from pathlib import Path
import random
import resource
import shutil
import statistics
import sys
import tempfile
import time


def load_helper(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rows = load_helper("row_stream", "vrambridge-row-stream.py")
io = load_helper("row_study_io", "vrambridge-io-study.py")
PAGE = mmap.PAGESIZE
ROW_BYTES = 144
TENSOR_OFFSET = 416
RESERVE_BYTES = 16 * 1024 ** 3
METHODS = ("mmap", "upstream-madvise", "direct-serial", "direct-concurrent", "direct-cache")


def fixture_page(index, seed):
    return hashlib.shake_256(f"vrambridge-row-fixture-v1:{seed}:{index}".encode()).digest(PAGE)


FIXTURE_SEED = 20261101


def expected_row(row, seed):
    begin = TENSOR_OFFSET + row * ROW_BYTES
    end = begin + ROW_BYTES
    first = begin // PAGE
    data = b"".join(fixture_page(index, seed) for index in range(first, (end - 1) // PAGE + 1))
    return data[begin % PAGE:begin % PAGE + ROW_BYTES]


def write_fixture(path, size, seed):
    if size <= TENSOR_OFFSET or size % PAGE:
        raise ValueError("Fixture size must contain a tensor and whole pages.")
    if shutil.disk_usage(path.parent).free < size + RESERVE_BYTES:
        raise RuntimeError("Insufficient SSD space with the 16 GiB reserve.")
    with path.open("xb") as target:
        chunk_pages = max(1, 1024 * 1024 // PAGE)
        for first in range(0, size // PAGE, chunk_pages):
            if shutil.disk_usage(path.parent).free < RESERVE_BYTES + min(1024 * 1024, size - first * PAGE):
                raise RuntimeError("SSD free-space reserve reached while creating the fixture.")
            target.write(b"".join(fixture_page(index, seed)
                                  for index in range(first, min(first + chunk_pages, size // PAGE))))
        target.flush()
        os.fsync(target.fileno())


def request_stream(tensor_rows, requests, seed, warmup=4):
    if tensor_rows < 48 or requests < 1:
        raise ValueError("Fixture needs at least 48 rows and one measured request.")
    generator = random.Random(seed)
    result = []
    for index in range(requests + warmup):
        ids = generator.sample(range(tensor_rows), 48)
        result.append((ids, index < warmup, False))
        result.append((ids, index < warmup, True))
    return result


def percentile(values, fraction):
    return sorted(values)[max(0, min(len(values) - 1, int(len(values) * fraction + .999999) - 1))]


def summarize(samples, reused):
    selected = [sample for sample in samples if sample["immediate_reuse"] == reused]
    pages = sum(sample["pages"] for sample in selected)
    return {
        "requests": len(selected),
        "median_ms": statistics.median(sample["total_ms"] for sample in selected),
        "p95_ms": percentile([sample["total_ms"] for sample in selected], .95),
        "median_submit_ms": statistics.median(sample["submit_ms"] for sample in selected),
        "median_gather_ms": statistics.median(sample["gather_ms"] for sample in selected),
        "explicit_cache_page_hit_fraction": sum(sample["cache_hit_pages"] for sample in selected) / pages,
        "submitted_page_bytes": sum(sample["submitted_bytes"] for sample in selected),
        "logical_row_bytes": sum(sample["logical_bytes"] for sample in selected),
    }


def run_case(path, method, requests, seed, workers, cache_pages):
    if method not in METHODS:
        raise ValueError("Unknown row benchmark method.")
    tensor_rows = (path.stat().st_size - TENSOR_OFFSET) // ROW_BYTES
    direct = method.startswith("direct-")
    store = None
    mapping = None
    samples = []
    physical_reads = 0
    phases = {False: {"physical_read_bytes": 0, "minor_faults": 0, "major_faults": 0},
              True: {"physical_read_bytes": 0, "minor_faults": 0, "major_faults": 0}}
    with path.open("rb") as source:
        try:
            if direct:
                store = rows.RowStore(
                    path, row_bytes=ROW_BYTES, tensor_offset=TENSOR_OFFSET, tensor_rows=tensor_rows,
                    cache_pages=cache_pages if method == "direct-cache" else 0,
                    workers=1 if method == "direct-serial" else workers)
            else:
                mapping = mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ)
                mapping.madvise(mmap.MADV_RANDOM)
            started = time.monotonic()
            for ids, warmup, reused in request_stream(tensor_rows, requests, seed):
                spans, pages = rows.row_plan(ids, ROW_BYTES, TENSOR_OFFSET, tensor_rows, PAGE)
                before = io.snapshot(os.getpid())
                then = time.perf_counter_ns()
                hits = submitted_bytes = 0
                if store:
                    old_submitted = store.submitted_bytes
                    pending = store.submit(ids)
                    submitted = time.perf_counter_ns()
                    result = pending.gather()
                    hits = pending.hit_pages
                    submitted_bytes = store.submitted_bytes - old_submitted
                else:
                    if method == "upstream-madvise":
                        for begin, end in rows.merge_pages(pages, PAGE):
                            mapping.madvise(mmap.MADV_WILLNEED, begin, end - begin)
                    submitted = time.perf_counter_ns()
                    result = [mapping[begin:end] for begin, end in spans]
                completed = time.perf_counter_ns()
                after = io.snapshot(os.getpid())
                observed = io.interval(before, after)
                physical_reads += observed["physical_read_bytes"]
                # The independent deterministic reference never reads the backing file or warms its pages.
                if result != [expected_row(row, seed=FIXTURE_SEED) for row in ids]:
                    raise RuntimeError("Gathered bytes differ from the deterministic fixture.")
                if not warmup:
                    phase = phases[reused]
                    for counter in phase:
                        phase[counter] += observed[counter]
                    samples.append({
                        "immediate_reuse": reused, "pages": len(pages), "cache_hit_pages": hits,
                        "logical_bytes": len(ids) * ROW_BYTES, "submitted_bytes": submitted_bytes,
                        "physical_read_bytes": observed["physical_read_bytes"],
                        "total_ms": (completed - then) / 1e6,
                        "submit_ms": (submitted - then) / 1e6,
                        "gather_ms": (completed - submitted) / 1e6,
                    })
                if time.monotonic() - started > 60:
                    raise RuntimeError("One row benchmark case exceeded its 60-second limit.")
            report = {
                "method": method, "request_seed": seed, "all_rows_byte_equal": True,
                "samples": samples, "physical_reads_including_warmup": physical_reads,
                "novel_ids": {**summarize(samples, False), "observed_io": phases[False]},
                "immediate_reuse": {**summarize(samples, True), "observed_io": phases[True]},
                "payload_memory_bound_bytes": store.payload_memory_bound if store else None,
                "explicit_cache_pages": store.capacity if store else 0,
                "peak_pending_pages": store.peak_requested_pages if store else None,
                "peak_cached_pages": len(store.cache) if store else None,
            }
            for name in ("novel_ids", "immediate_reuse"):
                phase = report[name]
                phase["observed_physical_read_amplification"] = (
                    phase["observed_io"]["physical_read_bytes"] / phase["logical_row_bytes"])
            if direct and physical_reads <= 0:
                raise RuntimeError("No physical I/O observed for the direct-read control; result not qualified.")
            return report
        finally:
            if store is not None:
                store.close()
            if mapping is not None:
                mapping.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", type=Path, required=True, help="Existing directory on a block-backed SSD.")
    parser.add_argument("--fixture-mib", type=int, default=128)
    parser.add_argument("--requests", type=int, default=64, help="Novel row sets per case, each immediately repeated.")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cache-pages", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != "linux" or not hasattr(os, "O_DIRECT"):
        parser.error("Linux direct I/O and /proc observations are required.")
    if not (4 <= args.fixture_mib <= 256 and 4 <= args.requests <= 256 and 1 <= args.trials <= 5 and
            1 <= args.workers <= 16 and 64 <= args.cache_pages <= 4096):
        parser.error("Invalid bounded study parameters.")
    if args.output.exists() or Path(str(args.output) + ".part").exists():
        parser.error("Refusing to overwrite an existing report or partial.")
    report = {
        "schema_version": 1, "status": "preparing", "kind": "synthetic_row_fixture_not_model_inference",
        "settings": {"fixture_bytes": args.fixture_mib * 1024 ** 2, "rows_per_request": 48,
                     "row_bytes": ROW_BYTES, "tensor_offset": TENSOR_OFFSET, "page_bytes": PAGE,
                     "requests_per_case": args.requests * 2, "trials": args.trials,
                     "workers": args.workers, "cache_pages": args.cache_pages,
                     "free_space_reserve_bytes": RESERVE_BYTES},
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "row_stream_sha256": hashlib.sha256(Path(rows.__file__).read_bytes()).hexdigest(),
        "cases": [], "fixture_removed": False, "failed_wd_volume_accessed": False,
        "scope": [
            "Recently written small SSD fixture; controller caches and filesystem page residency are not forced cold.",
            "mmap and upstream-madvise are warm-cache references, not cold-disk comparators to O_DIRECT.",
            "Direct-read physical traffic is attested with /proc counters, not inferred from requested bytes.",
            "Matched row IDs are used across methods within a trial; method order rotates.",
            "Immediate reuse is an artificial positive cache control, not a measured model hit rate.",
            "No model, tokenizer, generated tokens, prefetch prediction or compute/I/O overlap is measured.",
            "Payload bounds exclude Python objects, OS cache and interpreter RSS; process peak RSS includes fixture setup.",
            "No global cache eviction, filesystem repair, device reset or existing file modification.",
        ],
    }
    try:
        parent = args.fixture_dir.resolve(strict=True)
        device = parent.stat().st_dev
        device_path = Path("/sys/dev/block") / f"{os.major(device)}:{os.minor(device)}"
        if not parent.is_dir() or not device_path.exists():
            raise ValueError("Fixture directory must be block-backed, not tmpfs or an absent mount.")
        whole_device = device_path.resolve()
        if (whole_device / "partition").exists():
            whole_device = whole_device.parent
        if (whole_device / "queue/rotational").read_text().strip() != "0":
            raise ValueError("Use a non-rotational SSD for this fixture study.")
        if (parent / ".vrambridge-staging.json").exists():
            raise ValueError("Use a session-owned directory, not the managed model-staging root.")
        report["fixture_device"] = device_path.resolve().name
        with tempfile.TemporaryDirectory(prefix="vrambridge-row-fixture-", dir=parent) as directory:
            fixture = Path(directory) / "rows.bin"
            write_fixture(fixture, report["settings"]["fixture_bytes"], FIXTURE_SEED)
            report["status"] = "running"
            for trial in range(args.trials):
                order = METHODS[trial:] + METHODS[:trial]
                for method in order:
                    result = run_case(fixture, method, args.requests, 20261102 + trial,
                                      args.workers, args.cache_pages)
                    result["trial"] = trial
                    report["cases"].append(result)
                    print(f"{trial} {method}: novel {result['novel_ids']['median_ms']:.3f} ms, "
                          f"reuse {result['immediate_reuse']['median_ms']:.3f} ms, "
                          f"reads {result['physical_reads_including_warmup'] / 1e6:.2f} MB", flush=True)
        report["fixture_removed"] = True
        report["status"] = "complete"
    except (OSError, ValueError, RuntimeError) as exc:
        report["status"], report["error"] = "failed", str(exc)
        if "fixture" in locals():
            report["fixture_removed"] = not fixture.exists()
        print(f"Row-stream study failed: {exc}", file=sys.stderr)
    report["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    try:
        io.write_report(args.output, report)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Could not save row-stream study: {exc}\n")
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
