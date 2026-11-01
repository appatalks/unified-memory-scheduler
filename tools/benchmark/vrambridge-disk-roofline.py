#!/usr/bin/env python3
"""Cost a disk-backed model from verified tensor bytes and measured storage rates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def positive(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive.")


def analyze(active_bytes: int, bandwidth: float, iops: float, rows: int,
            row_bytes: int, target_tps: float, hit_rate: float, amplification: float,
            prefetch_precision: float, compute_ms: float) -> dict:
    for value, name in ((active_bytes, "Active routed bytes"), (bandwidth, "Bandwidth"),
                        (iops, "IOPS"), (row_bytes, "Row bytes"), (target_tps, "Target tok/s"),
                        (amplification, "Read amplification"), (prefetch_precision, "Prefetch precision")):
        positive(value, name)
    if (type(rows) is not int or rows < 0 or not math.isfinite(hit_rate) or
            not 0 <= hit_rate <= 1 or prefetch_precision > 1 or amplification < 1 or
            not math.isfinite(compute_ms) or compute_ms < 0):
        raise ValueError("Invalid row count, hit rate, precision, amplification or compute time.")
    # 4 KiB aligned reads; a row can straddle two pages. They are bounds, not observed physical I/O.
    row_pages_max = math.ceil((row_bytes + 4095) / 4096)
    row_logical = rows * row_bytes
    row_read_min = rows * math.ceil(row_bytes / 4096) * 4096
    row_read_max = rows * row_pages_max * 4096
    expert_logical_miss = active_bytes * (1 - hit_rate)
    expert_read = expert_logical_miss * amplification / prefetch_precision
    io_seconds = (expert_read + row_read_min) / bandwidth
    row_latency_floor = rows / iops
    compute_seconds = compute_ms / 1000
    # Unrealistically favorable complete overlap is an upper bound. Serial is a separate reference.
    full_overlap_seconds = max(io_seconds, row_latency_floor, compute_seconds)
    serial_seconds = max(io_seconds, row_latency_floor) + compute_seconds
    io_budget = max(0.0, bandwidth / target_tps - row_read_min)
    max_logical_miss = io_budget * prefetch_precision / amplification
    bandwidth_hit = max(0.0, 1 - max_logical_miss / active_bytes)
    return {
        "scope": "Optimistic throughput bounds, not model generation measurements or quality qualification.",
        "inputs": {"active_routed_bytes_per_token": active_bytes, "bandwidth_bytes_s": bandwidth,
                   "random_4k_iops": iops, "engram_rows_per_token": rows, "engram_row_bytes": row_bytes,
                   "target_tps": target_tps, "combined_byte_hit_rate": hit_rate,
                   "read_amplification": amplification, "prefetch_precision": prefetch_precision,
                   "resident_compute_ms": compute_ms},
        "engram": {"logical_bytes_per_token": row_logical, "page_bytes_min": row_read_min,
                   "page_bytes_max": row_read_max, "random_read_seconds_floor": row_latency_floor},
        "experts": {"logical_miss_bytes_per_token": expert_logical_miss,
                    "physical_read_bytes_per_token_estimate": expert_read},
        "target": {"minimum_byte_hit_rate_bandwidth_only": bandwidth_hit,
                   "maximum_logical_expert_miss_bytes_per_token": max_logical_miss,
                   "row_iops_condition_met": rows * target_tps <= iops,
                   "compute_condition_met": compute_seconds <= 1 / target_tps},
        "upper_bound_tps": 1 / full_overlap_seconds if full_overlap_seconds else None,
        "serial_reference_tps": 1 / serial_seconds if serial_seconds else None,
        "limitations": [
            "Storage rate and IOPS are independently measured; their resources are shared, not additive.",
            "Layer routing dependencies and per-miss tail latency reduce actual overlap.",
            "Engram page coalescing, caching and alignment are workload-dependent.",
            "No cache hit rate, compute time or prefetch precision is inferred from model parameter count.",
            "Wrong predictions fetch extra bytes but must never replace actual router-selected experts.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active-bytes", type=int, required=True)
    parser.add_argument("--bandwidth-mbps", type=float, required=True, help="Measured decimal MB/s.")
    parser.add_argument("--iops", type=float, required=True)
    parser.add_argument("--rows", type=int, default=0)
    parser.add_argument("--row-bytes", type=int, default=144)
    parser.add_argument("--target-tps", type=float, default=10)
    parser.add_argument("--hit-rate", type=float, required=True)
    parser.add_argument("--amplification", type=float, default=1)
    parser.add_argument("--prefetch-precision", type=float, default=1)
    parser.add_argument("--compute-ms", type=float, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = analyze(args.active_bytes, args.bandwidth_mbps * 1e6, args.iops, args.rows,
                         args.row_bytes, args.target_tps, args.hit_rate, args.amplification,
                         args.prefetch_precision, args.compute_ms)
        text = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8") as output:
                output.write(text)
        else:
            print(text, end="")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Disk roofline failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
