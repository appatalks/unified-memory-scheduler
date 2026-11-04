# SC117 core decode at 96K and 128K

Date: 2026-11-04 UTC. **Existing-engine evidence and a selection-path fix,
not a new kernel or broadly qualified acceleration default.**

The [reviewed aggregate](../../benchmarks/results/sc117-core-decode-summary.json)
contains exact-target measurements, retained memory failures, and aligned
CPU/CUDA diagnostics. It follows the
[16K deployment comparison](sc117-hosted-porting-baseline.md).
The user's installed backend, saved settings and original 262K configuration
were restored after each experiment phase.

## What was measured

Existing SC117 abliterated Qwen3.8-Flash-Next IQ3_S weights on the Ryzen 9 9950X
and RTX 5070 Ti 16GB, eight generation/sixteen prompt threads, F16 K/V.
No target conversion, expert skipping, draft model, driver change, model
download or archive remount. Expert scheduling remains descriptor-driven.

Initial 256-output training screens compared installed Standard/Classic
at 96K/128K, banks of 8/6/4 GiB, and same-build candidate Classic/Hosted
at 4 GiB. Successful loads confirmed actual context, full routed-layer
coverage, direct registration, pipeline and asynchronous-ID activation.

The selected existing Classic configurations then received three fresh,
alternating Standard/Classic load pairs per context, each with held-out
coding/reasoning/prose and 512 forced outputs. Warmup excludes 64 tokens;
adaptive residency continues across the subsequent workload requests.
Rate is `(outputs - 1) / (request wall seconds - reported prompt seconds)`.
Do not substitute profiler rates or multiply earlier studies' gains.

## Repeated results

| Allocated context | Execution | Expert bank | Coding | Reasoning | Prose |
|---|---|---:|---:|---:|---:|
| 96K (98,304) | Standard | None | 29.39 | 29.35 | 29.38 |
| 96K (98,304) | Existing Classic hybrid | 6 GiB | **45.13** | **44.76** | **40.44** |
| 128K (131,072) | Standard | None | 26.29 | 26.49 | 27.12 |
| 128K (131,072) | Existing Classic hybrid | 4 GiB | **36.86** | **36.41** | **33.77** |

Values are three-run median decode tok/s, not full-input or natural-answer
quality guarantees. All 18 paired workload gains are positive. The 96K paired
gains range from 37.59% to 54.67%; 128K gains range from 17.45% to 43.00%.
Each comparison holds its allocated context fixed. Comparing the 96K row
against 128K also changes the fitting bank budget and must not be attributed
to context alone.

The original short-prompt screens found 8 GiB banks failed at both capacities.
A 6 GiB bank loaded at 96K but failed a 534.26 MiB CUDA compute-buffer
allocation at 128K. A 4 GiB bank loaded at both capacities. These are
machine/load-specific fit observations, not universal budgets.

The candidate in-graph/Hosted screens did not improve on same-build Classic
here. At 128K/4 GiB, candidate Classic screened at 40.69/39.64/37.09 tok/s;
Hosted screened at 39.43/37.04/34.38. These single screens do not establish
universal regressions, but do not justify promoting the newer method.

## Longer-input check

A separate held-out coding screen supplied **8,233 backend-reported input
tokens** at the same allocated 128K capacity, unchanged F16 K/V and 512 forced
outputs. Standard reached **24.73**, Classic **36.76 decode tok/s**.
Whole-request throughput was **8.35 versus 8.95 tok/s**, respectively:
the decode gain does not erase the substantial prefill cost.

This is one matched longer-input pair, not a fully occupied 96K/128K test
or a three-pair long-input confirmation. Prompt token counts are measured;
allocated capacity is not presented as occupied context.

## Where a token's time goes

The [owned Synapse diagnostic adapter](../../tools/benchmark/vrambridge-synapse-critical-path.py)
reuses the existing CPU graph profiler, request counter helpers and
critical-path analyzer. Unlike the older request runner, it can preserve
the current Synapse runtime's 96K/128K capacities. Nsight traces include
actual CUDA kernels/copies and separately marked CPU graph/observer ranges.
The CPU profiler's library must match the actual mapped CPU backend.

One 128K coding diagnostic per engine, 64 outputs after excluded warmup,
produced identical complete output hashes. Both generation intervals had
zero process physical disk reads, zero read syscalls and zero major faults.
These are warm-generation observations, not proof of RAM bandwidth
saturation or an argument for RAM-disk staging.

| Observed envelope, milliseconds | Standard | Classic, 4 GiB |
|---|---:|---:|
| CPU only | 1,326.99 | 472.98 |
| GPU only, including copies/memsets | 694.76 | 726.40 |
| CPU and GPU overlap | 0.00 | 104.11 |
| Neither recorded CPU graph nor GPU activity | 502.65 | 562.00 |
| Total envelope | 2,524.40 | 1,865.49 |

CPU expert-graph elapsed time separately falls from **1,326.03 to 542.75 ms**.
The existing pipeline therefore demonstrably reduces CPU expert work and
overlaps some work; it is not merely a different checkbox label.

These are profiler-perturbed, CPU-enclosed partial-boundary-token envelopes.
They are not uninstrumented throughput measurements. The Classic gap
contains 435.80 ms of CUDA API intervals and 60.67 ms of observer intervals;
these subsets can overlap. API time includes required waits/driver work and
cannot be credited as removable latency. Do not remove fences or claim a
projected speedup from this table.

## Concrete production issue

Synapse's `AutoTuner._wait_ready` previously retained only the last three
failure diagnostics. An early CUDA allocation error followed by generic
`failed to create context` / `exiting` lines could disappear from the trial
error. The existing smaller-bank retry classifier then stopped after the
first failing budget.

The minimal downstream fix preserves the first relevant memory failure
alongside the last three diagnostics. A regression fixture reproduces the
real ordering and verifies 8 -> 6 -> 4 GiB trial progression without
changing context, precision, bound plans, engine math or the measured-winner
rule. This makes an existing internal policy reliable; it adds no user
controls and does not automatically benchmark on first load.

## Remaining gates and decision

All bounded arithmetic/AST-only checks pass, but generated hashes differ
between Standard and Hybrid; 128K Standard also varies across fresh loads.
These facts do not identify an incorrect answer or establish full-vocabulary
equivalence. The existing acceleration quality warnings remain applicable.
GPT-OSS-120B originals remain unavailable, so no new cross-family engine
technique or global context/cache default is promoted.

The supported immediate conclusion is to use the existing measured
expert-sharing design where it fits and wins, with robust memory budgeting
and retained diagnostics. Further kernel work should target a specifically
measured residual cost, preserve scalar numerical contracts and buffer
lifetimes, then pass repeated request and numerical gates. Earlier failed
warmer/repack/verification experiments are not reset by these results.

The private directory retains raw CPU/routing/tensor identities, full
Nsight traces/databases, synthetic input, failed attempts and original-state
restoration manifests. Only the sanitized aggregate is allowlisted.
