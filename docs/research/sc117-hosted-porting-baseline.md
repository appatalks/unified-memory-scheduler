# SC117 deployment baseline and hosted-dispatch follow-up

Date: 2026-11-04 UTC. **Measured research evidence, not rollout approval.**

The [reviewed aggregate](../../benchmarks/results/sc117-hosted-porting-summary.json)
records fresh measurements on the existing Ryzen 9 9950X / RTX 5070 Ti 16GB
machine. The target is the user's existing **SC117 abliterated Qwen3.8
Flash-Next IQ3_S**, not the original checkpoint in the older Strata study.
Target files were reused in place; no weights were converted, no selected
experts were skipped, and no drafting was enabled.

## What was already implemented

The canonical hybrid patch already contains metadata-driven hosted expert
execution, LFU admission, and the CUDA host-call mailbox. Synapse already
detects this capability and can benchmark it through Auto Tune. Its installed
GPU plugin did not expose hosted support, so this study used an existing
separate candidate build without replacing the installed backend.

This is evaluation of an existing port, not a new Strata runtime integration.
Classic and hosted both retain GPU-hot/CPU-cold expert sharing. Hosted moves
the CPU expert call inside the GPU graph; it does not stack a second expert
cache beneath the first engine.

## Protocol

- F16 K/V, 16,384 allocated context, eight generation/sixteen prompt threads.
- 8,192 MiB causal bank for every hybrid variant; no checkpoint-bound GPT
  training plan is reused for these Qwen targets.
- Three fresh loads per variant, alternating forward/reverse/forward order.
- One excluded 64-token warmup, then coding/reasoning/prose from the held-out
  CPU-decode corpus. All **63 measured requests** finish the requested 512
  generated tokens.
- Greedy sampling, seed 42, prompt reuse off, reasoning effort low.
- Forced-length throughput ignores EOS; this is not 512 natural answer tokens.
- Rate: `(output_tokens - 1) / (request wall seconds - reported prompt seconds)`.
  Draft proposals are absent. Do not mix this estimator with an older study's
  post-first-output timing or multiply independent studies' gains.
- Actual prompts contain 132-155 tokens for SC117 and 100-123 for Coder.
  Allocated context is not occupied context.

The installed Classic arm uses the installed hybrid executable/plugin. The
candidate Classic and Hosted arms share the same candidate executable/plugin.
Only the latter comparison isolates the hosted/LFU variant bundle from a
backend-build change. Neither comparison isolates graph orchestration from
admission policy.

### SC117 IQ3_S: three-run medians, tok/s

| Execution | Coding | Reasoning | Prose |
|---|---:|---:|---:|
| Installed Standard | 27.01 | 26.84 | 27.14 |
| Installed Classic hybrid | 39.48 | 37.43 | 34.20 |
| Candidate Classic hybrid | 40.36 | 38.77 | 34.76 |
| Candidate Hosted + LFU | 43.16 | 39.13 | 34.83 |

Hosted versus same-build Classic has paired coding gains of 8.17%, 5.81%,
and 7.01%. Reasoning gains are only 0.47-1.19%; prose ranges from -2.05%
to +0.72%. The latter two do not establish improvement over the approximately
5% noise band. Most of the gain over Standard is already available in the
classic expert-sharing design, not a new speculative-decoding multiplier.

### Qwen3-Coder-30B-A3B Q4_K_M: three-run medians, tok/s

| Execution | Coding | Reasoning | Prose |
|---|---:|---:|---:|
| Installed Standard | 69.65 | 69.41 | 69.85 |
| Candidate Classic hybrid | 86.86 | 92.84 | 106.05 |
| Candidate Hosted + LFU | 116.23 | 107.48 | 136.16 |

Every paired Hosted gain is positive: coding 31.11-36.23%, reasoning
15.27-18.47%, prose 26.07-30.84%. This supports measuring the variant per
model and workload, not promising these rates for the many-expert SC117 model.

## Original context capacity: fit is not a full-input check

The original requested capacity was 142,128 tokens; the actual backend slot
capacity was 142,336. The private runtime retained that actual capacity.

An 8 GiB hybrid bank **failed explicitly** during CUDA allocation of its
8,187.07 MiB payload. There was no retry as Standard and no silent context
reduction. A separate 4 GiB bank loaded and completed a single 256-output,
short-prompt screen:

| Execution at 142,336 capacity | Coding | Reasoning | Prose |
|---|---:|---:|---:|
| Standard | 25.63 | 25.23 | 25.23 |
| Classic hybrid, 4 GiB bank | 34.09 | 32.77 | 29.53 |

These are **single screens**, not three-run long-context medians. Changing the
private runtime's profile context was an explicit research override, not a
change to Synapse's installed 16K hybrid profile.

A separate near-capacity anchor-retrieval probe was interrupted during lengthy
Standard prefill after the confirmation cohort. Its private report is
unfinished; no capacity or retrieval success is inferred from it. Interruption
is not an incorrect-answer finding. The benchmark runner now has a bounded
capacity deadline with explicit failure reporting, tested cancellation, and
owned-process cleanup. A genuinely filled 142K comparison remains unverified.

## Accuracy and release gates

All 14 bounded arithmetic/AST-only checks in the two confirmation cohorts
pass. Generated model code is parsed, never executed. These checks are not a
general quality benchmark.

Accelerated output hashes differ from Standard. SC117 Standard also differs
across fresh loads; its hybrid arms repeat their respective hashes. Coder
repeats within each arm but differs across engines. This does not identify
which answer, if any, is incorrect, and does not establish full-vocabulary
numerical equivalence.

No performance default, backend pin, installed profile, or saved engine choice
is promoted. Remaining gates include:

1. Establish reproducible source-to-binary binding for the candidate and run
   the unchanged full-vocabulary numerical gates on the exact targets and
   placements. Existing binary hashes and activation markers identify what
   ran, not a reproducible build.
2. Restore access to the original GPT-OSS-120B shards for the required third
   MoE family. The approved SSD root contains only a GPT proposal draft;
   original shard symlinks point into the unavailable archive root. No drive
   was mounted and no model was downloaded.
3. Qualify natural answers and actual long-input behavior separately from
   short-prompt rates and allocation success.
4. Keep selection metadata-driven and measured through Synapse's existing
   detect -> Auto Tune -> remember flow. Mixed results do not justify a
   universal Hosted default.

For further Strata-inspired work, first profile residual CPU expert,
GPU-launch, transfer, and synchronization costs after enabling the existing
expert-sharing path in an isolated runtime. The earlier exact CPU-packing and
selective-KV/speculation studies lost real request throughput; do not repeat
them or port a batched verifier merely because an isolated operation is fast.
Exact multirow amortization remains a research lead, not a qualified next
default or a demonstrated 100 tok/s path for SC117.

## Reproduction and isolation

The new [owned-runtime runner](../../tools/benchmark/vrambridge-synapse-baseline.py)
imports a specified Synapse source tree, creates private loopback app/backend
instances, uses a temporary `XDG_CONFIG_HOME`, disables update checks, and
loads with `remember=False`. Required engine/context/precision markers must
match; failed and unfinished trials cannot become successful medians.

Unload the ordinary model with consent before benchmarking and restore it
afterward. The runner owns only its private processes; it does not stop or
restore an unrelated app itself. This study's outer orchestration restored
the original SC117 model and 142,336 capacity after every experiment phase.
Model settings, engine choices, tuning state, and the installed profile were
verified byte-for-byte unchanged.

For a separately prepared matching candidate profile:

```sh
python3 tools/benchmark/vrambridge-synapse-baseline.py \
  --synapse-source /path/to/Synapse \
  --backend /path/to/installed/llama-server \
  --profile /path/to/installed/hybrid-profile.json \
  --candidate-profile /path/to/private/candidate-profile.json \
  --model /models/target-00001-of-00002.gguf \
  --contexts 16384 \
  --variants standard classic candidate-classic hosted \
  --tokens 512 --repetitions 3 --split test --answer-checks \
  --output benchmarks/results/private-run.json
```

`--capacity-check --capacity-timeout 300` adds the near-capacity input probe;
a deadline is a reported failure, never a successful capacity estimate.
Raw output, private logs, and unfinished trials remain ignored. Only the
sanitized reviewed aggregate is allowlisted.
