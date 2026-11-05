# Periodic prompt checkpoints: latency gain, numerical qualification blocked

Date: 2026-11-05 UTC. **Research-only, default off. Do not install or promote.**

The [reviewed aggregate](../../benchmarks/results/prompt-reuse-checkpoint-summary.json)
and [owned replay runner](../../tools/benchmark/vrambridge-prompt-reuse.py)
investigate repeated long-prefill pauses while retaining the existing
96K Hybrid decode configuration.

## Reproduced mechanism

The pinned server normally checkpoints at selected message boundaries and
near the prompt end. Lowering `--checkpoint-min-step` alone does not permit
ordinary mid-message checkpoints.

A synthetic 8,192-token raw-completion prompt followed by an exact repeat,
append, one-token middle edit at zero-based position 4,353, and return to the
original prompt reproduces the problem. The original server, rebuilt
default-off candidate, and denser-boundary-only control all process 8,192
tokens again after the edit. Private numeric logs explicitly report full
reprocessing due to missing usable cache data.

The [experimental server-only patch](../../third_party/patches/llama-b11669-prompt-checkpoint-grid-experimental.patch)
adds `VRAMBRIDGE_PROMPT_CHECKPOINT_GRID=1`, accepting only 0/1 and requiring
positive checkpoint count/spacing. It permits checkpoint creation at
existing prefill batch boundaries. Existing recurrent/SWA eligibility,
checkpoint capacity, invalidation and native state serialization remain.
It does not split batches, alter token selection, change weights, move
expert-bank allocations, or change decode operations.

The patch is isolated from the canonical Hybrid patch and is applied only
to a copied pinned backend tree. With the opt-in absent, the initial
baseline/candidate complete output hashes match in all five replay cases.
The installed source, binaries, profile and saved state were not edited.

## Matched confirmation

Ryzen 9 9950X, RTX 5070 Ti 16GB, existing abliterated SC117 IQ3_S target,
98,304 allocated context, F16 K/V, 6,144 MiB causal expert bank,
8 generation/16 prompt threads. Prompt cache remains 8,192 MiB and the
checkpoint limit remains 32. The candidate uses a 1,024-token minimum
spacing, producing actual periodic checkpoints at existing 2,048-token
batch boundaries in this workload.

Three fresh control/candidate pairs alternate order. Each load excludes a
64-output warmup and returns exactly 256 forced output tokens per timed
request. Confirmation uses reduced verbosity, not the initial detailed
trace logging. Values below are medians, not model-load time or direct
time-to-first-token measurements.

| Replay case | Control prompt time | Candidate prompt time | Control request wall | Candidate request wall |
|---|---:|---:|---:|---:|
| Cold prompt | 42.59 s | 42.88 s | 49.46 s | 49.63 s |
| Exact repeat | 0.160 s | 0.159 s | 6.58 s | 6.54 s |
| Append | 0.392 s | 0.384 s | 7.21 s | 7.37 s |
| Middle edit | 42.78 s | **22.28 s** | 49.72 s | **29.32 s** |
| Return to original | 42.53 s | **21.77 s** | 49.03 s | **28.40 s** |

Every paired middle-edit prompt change is negative: -49.60%, -48.21%,
-46.46%. Every return-to-original change is approximately -49%.
The candidate restores the checkpoint at 4,096 and evaluates 4,096
remaining tokens, versus zero reused and 8,192 evaluated in the control.
Exact-repeat and append reuse were already effective.

Decode medians remain about 36-40 tok/s. All paired decode changes are
within approximately 5%; this is not proof of no regression on every
workload or machine. These repeated sequences are synthetic calibration
evidence, not a held-out real CLI workload.

Each checkpoint in this target is approximately 112.571 MiB. Backend RSS
after replay is about 53.02 GiB with the candidate versus 52.68 GiB in the
control, approximately 340 MiB more. The count cap is not a universal byte
budget. A production candidate still needs a measured host-memory reserve
and snapshot-budget policy; do not assume RAM availability is free memory.

## Numerical qualification failed

All six timed middle-edit output hashes match their own uncached edited
controls, but append outputs vary across fresh loads. Hash equality alone
is insufficient.

Separate single-output diagnostics compare the complete 248,320-entry
pre-sampling log-probability distribution, checking unique complete token
coverage. They are explicitly not throughput measurements and not a
substitute for the project's raw-logit gate. The diagnostic thresholds
remain maximum absolute error <= 0.01 and relative L2 <= 0.001.

| Contract / case | Control max absolute / relative L2 | Candidate max absolute / relative L2 |
|---|---:|---:|
| Adaptive, middle edit | 0.69334 / 0.00807 | 2.57193 / 0.03296 |
| Adaptive, append | 2.52239 / 0.02462 | 2.51996 / 0.02461 |
| Frozen empty bank, middle edit | 1.14078 / 0.01087 | 1.14078 / 0.01087 |
| Frozen empty bank, append | 1.80874 / 0.01972 | 1.80874 / 0.01972 |

**All listed diagnostics fail.** The frozen empty-bank experiment is a
separate attribution contract with no admissions or direct transfers,
not the measured production-speed configuration. Its mismatch persists
in the unmodified control, so adaptive expert placement alone does not
explain the discrepancy. Matching error magnitudes do not establish that
the entire distributions match each other.

No correctness threshold was relaxed, no unchanged control failure was
silently treated as success, and no experimental checkpoint behavior is
promoted. The result supports a latency mechanism and identifies further
numerical work; it does not establish which answer, if any, is incorrect.

An intermediate diagnostic attempt incorrectly applied multi-output decode
duration requirements to a one-output probability probe. It was retained
as a failed harness attempt and the user's model was restored. The runner
now explicitly marks one-output probes as non-performance measurements.

## Reproduction and safeguards

Unload the ordinary model with consent. Use a private 0700 output directory,
the existing local target, the installed matching CUDA plugin, and isolated
baseline/candidate builds:

```sh
python3 tools/benchmark/vrambridge-prompt-reuse.py \
  --baseline /private/baseline/bin/llama-server \
  --candidate /private/candidate/bin/llama-server \
  --profile /private/hybrid-profile.json \
  --model /models/target-00001-of-00002.gguf \
  --context 98304 --cache-mib 6144 --prompt-tokens 8192 \
  --variants off grid --repetitions 3 --tokens 256 \
  --verbosity 3 --probability-check \
  --output-dir /private/new-run
```

For numerical attribution add `--freeze-adaptation --numerical-only`.
That disables adaptive admissions and keeps the bank empty; do not compare
its decode timings to the adaptive cohort.

Activation must confirm full routed-layer coverage, exact context, pipeline
and ID controls, plus the experimental marker when selected. The native
server executable is a small loader: **the server implementation shared
library identifies this change**, not just the executable or `libllama.so`.
The reviewed aggregate records both implementation hashes.

All owned servers close on normal and failed trials. Original 96K Hybrid
settings and protected saved files were restored after every experiment
phase. Raw private logs remain ignored because verbose server output can
contain prompt tokens/text. The reviewed aggregate contains only numeric
measurements, hashes, controls, failures and limitations.

## Next gate

First isolate the unchanged-control log-probability discrepancy using exact
raw-logit capture, fixed token/batch shapes and audited context reset/restore
state. Then qualify checkpoint replay against that execution contract.
Only after numerical, bounded memory, multiple-workload and cross-family
checks pass should this become a capability-detected, measured internal
engine behavior. GPT-OSS-120B originals remain unavailable in this run.

The experimental patch, runner and evidence remain in this research
repository. No prompt-reuse diagnostic or checkpoint behavior is currently
being promoted to Synapse, and no backend pin or installation is changed.
