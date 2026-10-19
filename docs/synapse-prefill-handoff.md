# Synapse handoff: real-model prefill investigation

Date: 2026-10-19. **Research review packet; no new optimization is qualified
for rollout.** Real before/after measurements are available, but larger native
batching fails numerical qualification. A subsequent CPU-activation/GPU-down
prototype passes isolated layer checks and has repeated layer-level gains;
its separate full-model mask prototype now builds but fails the target-logit
gate. A newer packed CPU-compatible raw-down kernel is bit-exact in isolated
replays and faster in resident raw-stage tests, but is not integrated or
full-model qualified. Do not apply either
an unqualified configuration or the experimental CUDA patch as a product default.
The newer CPU decode follow-up below identifies an optional measured CPU-use
tradeoff with preserved logits, not a faster default or a 60 tok/s result.
The subsequent live CPU-cache warmer is now implemented and tested, but its
matched request matrix also shows no throughput improvement.

The subsequent architecture study does find a repeated Strata/Qwen two-row
MTP throughput lead, with identical paired outputs and passing bounded sanity
cases. It remains a separate, not-yet-full-logit-qualified engine contract;
it is not a demonstrated speedup for Synapse's current native execution.

The [optimized-verification follow-up](research/optimized-verification-study.md)
now removes the native numerical blocker on bounded tested windows without
changing optimized scalar attention. GPT rollback and selective-KV checks
pass, but the live DFlash and exact CPU-packing candidates remain slower.
This advances qualification, not deployment readiness or the 60 tok/s goal.

## Workspace and current downstream contract

Research and artifacts belong in https://github.com/appatalks/unified-memory-scheduler.
Implementation belongs in https://github.com/appatalks/Synapse through its own
agent/coding sessions. No Synapse code was edited or pushed.

Synapse `main` was rechecked read-only at
`545a7ebf4a6dea3dd4a1ac04760057eed2de1b15`:

- Installer: https://github.com/appatalks/Synapse/blob/545a7ebf4a6dea3dd4a1ac04760057eed2de1b15/scripts/setup_acceleration.py
- AutoTune: https://github.com/appatalks/Synapse/blob/545a7ebf4a6dea3dd4a1ac04760057eed2de1b15/synapse_app/tuning.py
- llama.cpp pin: `620d2225cc436ab0b001e3200e590e3b286ef9ab`.
- Research artifact base:
  https://raw.githubusercontent.com/appatalks/unified-memory-scheduler/189db7f1ee3e087beccef790cf97cfcc9c0d335a
- Existing hybrid patch SHA256:
  `961544562fa0f9c45c1b5ee71e91196d693b01285572c40197c5c260df07319b`.
- Trained plan SHA256:
  `6a679bec76f1a410ed65fa79c186d164a388fc76419d96dcad5978782343949c`.
- AutoTune uses up to eight generation and sixteen batch threads. It does not
  establish that either count is universally optimal.

Recheck those contracts before downstream work. Publishing these research
artifacts does not update the downstream installer pin. Do not change installer
URLs or checksums to an unverified revision; promotion requires its own gates.

## Architecture packet: a real throughput lead, with explicit transfer gates

See the [reviewed architecture evidence](../benchmarks/results/model-agnostic-architecture-summary.json)
and [protocol/results](research/model-agnostic-architecture-study.md).
The latest downstream source was rechecked read-only at the revision above.
Its [hybrid constructor](https://github.com/appatalks/Synapse/blob/545a7ebf4a6dea3dd4a1ac04760057eed2de1b15/synapse_app/app.py#L404)
explicitly rejects hybrid execution with speculation. Do not bypass that guard.

Final Strata 0.1.43, unchanged Qwen IQ3_S target, 16K/F16 KV, exactly 3,067
frozen resident experts, no skipped selected experts, and fixed GPU prefill:

| 512-token workload | Scalar median | Two-row MTP median | Ratio-of-medians gain |
| --- | ---: | ---: | ---: |
| Coding | 47.67 tok/s | 57.20 tok/s | +20.01% |
| Reasoning | 34.37 tok/s | 43.85 tok/s | +27.57% |
| Prose | 38.43 tok/s | 45.26 tok/s | +17.77% |

Three fresh alternating pairs per workload all improve. All nine paired full
output hashes match, with deterministic repeats once dynamic CPU prefill sharing
is disabled. Separate natural-EOS tests pass all 15 bounded cases across native
scalar/canonical/MTP and Strata scalar/MTP. Neither fact substitutes for
full-vocabulary or broad quality qualification.

Two-row verification produces about 1.73-1.89 accepted outputs per window at
about 1.48 scalar-step cost. The transferable policy criterion is:

`(draft + verify + commit cost) / expected accepted outputs < scalar target cost`

It must be evaluated from loaded capabilities, shapes, numerical compatibility,
acceptance and observed costs. Do not hardcode a model name, require a trained
MTP head for every architecture, or assume that two rows always win. Four-row
screens are mixed, and native two-row screens do not reproduce the Strata gain.

### Implementation sequence for Synapse's own session

1. Retain the current hybrid/speculation guard and default-off behavior. Expose
   accepted-output and draft/verification counters before adding a controller.
2. Describe trained-draft availability, hidden width, proposal format/vocabulary
   and target rollback/batched-reduction compatibility from model/runtime
   metadata. Unsupported combinations must remain off with a diagnostic.
3. Preserve the distinction between execution contracts. Canonical block sizes two and
   four are exact at 96 checked positions against canonical scalar, but
   canonical versus the optimized baseline fails the original thresholds.
   Current canonical MTP is not a transparent replacement for optimized scalar.
   The newer optimized-preserving window prototype passes the tested full-logit
   gates; do not substitute global canonical flags for that separate result.
4. Qualify actual speculative acceptance/rejection and committed target state,
   not just contiguous batched target evaluation. Add full-vocabulary capture
   for the applicable execution path; Strata's matching 512-token hashes are
   not that gate. The newer GPT rejected-suffix tests pass, while Qwen's tested
   tail-removal-only rollback remains unsupported; snapshot/copy is unqualified.
5. Only after those gates pass, select windows from measured accepted-output
   cost using training/calibration data. Confirm on fresh held-out cases with
   scalar fallback, fixed placement, alternating real loads, natural EOS and
   unchanged thresholds. Do not choose the fastest workload result after the
   fact and call it a live controller.

No generic controller, native default or Synapse code was changed. This packet
supports a measured research lead and a concrete compatibility worklist, not
deployment approval, novelty claims or reaching 60 tok/s on GPT-OSS-120B.

### Optimized verification and KV follow-up: reference material, not rollout

The [new aggregate](../benchmarks/results/optimized-verification-summary.json)
and [detailed study](research/optimized-verification-study.md) supersede only
the claim that optimized-preserving native windows are numerically blocked.
The old canonical failure, performance limits and current downstream guard
remain. Synapse `main` was rechecked read-only at
`545a7ebf4a6dea3dd4a1ac04760057eed2de1b15`; its hybrid/speculation rejection
and backend pin are unchanged.

The installed package and managed-launcher profile were subsequently inspected
read-only as well. They already configure 16K context, 12,000 MiB cache, direct
transfers, the decode pipeline and asynchronous expert IDs. The installed
runtime source matches the local runtime file at
`f7b1668f83140f1b284eb73beb29fbd1dd40ea59`; this does not bind the entire
package to that revision. No deployed-model timing was collected during that
inspection. The historical asynchronous-ID gain is not a new optimization
waiting to be enabled, and neither the frozen/staged GPT research arm nor the
Strata/Qwen MTP gain establishes a faster replacement for this installation.

Potential implementation references for Synapse's own future session:

- Preserve single-query attention and scalar product reductions inside
  verification without altering normal prefill. Both batch-limit and slot-map
  placement must be guarded; fixing only one caused a retained CUDA failure.
- Use explicit inference roles rather than treating all-output small batches
  as a universal decode classifier. Preserve actual accepted/rejected state;
  recurrent models need separately qualified snapshot/copy support.
- Tiny verification batches may benefit from generation rather than prefill
  threads. This reduced CPU use and improved a slower speculative screen,
  not the original scalar baseline.
- Account for physical GPU capacity using actual device/NVML counters.
  The pinned backend's global-managed free-memory log can report host RAM.
  Selective KV staging passed correctness but demonstrated neither a
  throughput gain nor physical VRAM/PCIe savings.
- Keep exact expert-column splitting in the qualified path. Its explicit
  native-batched ablation fails full logits. A future exact multirow kernel
  needs new numerical and live-cost evidence, not relaxed thresholds.

The fresh 256-output training comparison is scalar 41.03/25.10/35.65 versus
selective-KV DFlash 37.49/24.46/30.51 tok/s for coding/reasoning/prose, with
identical complete outputs. Do not deploy the known-losing candidates, promote
the approximately 2.1x CPU microbenchmark as a request gain, or bypass the
downstream guard. No Synapse implementation or installer URL was changed.

**Recommended next gate:** benchmark the installed-profile behavior in an
owned research process against matched candidates, preserving target bytes,
prompts, context/KV, output caps and sampling. Record actual transfer,
adaptation and cache-history controls. Only then profile the verified
multirow path and implement an exact shared-work operator where measured
headroom justifies it. Do not spend another confirmation cohort on the
already-losing staging/repack configurations.

## Actual before/after measurements

See the [machine-readable record](../benchmarks/results/gpt120-prefill-real-workload-summary.json)
for individual samples, hashes, exact configurations and provenance.

Conditions: Ryzen 9 9950X, RTX 5070 Ti, GPT-OSS-120B MXFP4, actual 16,384 context,
12,000 MiB bank, direct DMA, decode pipeline and asynchronous IDs. Weights are
on the existing root Samsung 870 EVO SATA SSD and are cache-warm. The installed
backend and weight files are unchanged; the separate activation-parity
prototype is **not** used in these request benchmarks.

The frozen technical-review input renders to 3,577 tokens. Each run starts a
fresh owned backend and produces exactly 256 tokens with temperature zero,
seed 42, low reasoning, ignored EOS and prompt caching disabled. Three pairs
alternate A/B, B/A, A/B. Both requested native batch sizes are confirmed from
context initialization logs; context, bank and thread counts stay fixed.

| Median | Baseline: logical 2048 / physical 512 | Candidate: logical 2048 / physical 1024 |
| --- | ---: | ---: |
| Cache-warm launch to ready | 4.228 s | 4.225 s |
| Backend prompt processing | 29.321 s | 28.283 s |
| First received output text | 29.461 s | 28.433 s |
| Full request, excluding model launch | 35.300 s | 34.097 s |
| Backend decode, unqualified output | 42.736 tok/s | 43.700 tok/s |

The raw prompt-latency reduction is about 3.5%, but **it is not a qualified
improvement**:

- All three candidate outputs have a different hash from all three controls.
- A separate held-out render of the same context is 3,590 tokens: the checker
  adds its fixed helpful-assistant system message and pins the template to epoch
  zero. It is not represented as the identical HTTP render.
- At 32 teacher-forced full-vocabulary positions, the unchanged 512-row control
  is exact. The 1024-row candidate already fails at position zero and reaches
  max absolute logit error 12.253, relative L2 0.3343, and 29/32 top-one matches.
- Gates remain max absolute <= 0.01 and relative L2 <= 0.001. These establish
  matched-baseline preservation, not general answer quality or CPU-reference
  equivalence.

All six paired request windows record zero physical reads, zero major faults
and zero endpoint process swap. RSS snapshots overlap page cache and are not
true peak memory. No RAM disk, cache drop, colder storage or memory-pressure
qualification is implied. A faster decode rate with a different answer is
not accepted-output goodput.

Single-run batch-thread screens at 8/12/16/20/24/32 do not improve on sixteen.
Logical-batch screens at 512/1024 with physical 512 also do not improve prompt
latency; logical 512 changes the output hash. Do not promote these screens.

## Separate CUDA numerical investigation

The [original parity record](../benchmarks/results/mxfp4-activation-parity-summary.json)
and [effective-Q8 follow-up](../benchmarks/results/mxfp4-effective-q8-propagation-summary.json)
concern a separate default-off research build, not the installed backend.

The prototype aligns native x86 MXFP4 activation rounding, FP16 scales and Q8
matrix inputs without changing expert weights, selections or router
probabilities. It passes 57/60 broader operator configurations, not all sixty.
The original three failures are retained.

New aggregate diagnostics show that a few upstream code/scale differences
are sufficient to reproduce a severe down-projection error through CPU replay.
GPU down versus CPU down on the same effective input is accurate near 1e-7
relative L2. In the severe sample, the complete-chain CPU-down lesion has
max-absolute error 1.73584, versus 0.12219 for the activation-only lesion.
Projection propagation therefore cannot be ignored or assumed fixed by
changing SwiGLU alone.

Isolated, correctness-passing fixed-set transfer-inclusive timings favor the
GPU at 32/64 rows per expert. They do not establish full-model prefill or decode
improvements, and CPU preparation outside timing must not be counted as free.
Full racecheck remains inconclusive after its timeout; scoped sanitizer passes
are not a complete race qualification.

## CPU activation / GPU down follow-up

The next bounded experiment has now been implemented in the isolated probe:
see the [reviewed measurements](../benchmarks/results/cpu-activation-gpu-down-summary.json)
and [reproduction guide](../benchmarks/README.md#cpu-activation-with-gpu-down-projection).

Gate/up projections, their biases and SwiGLU remain CPU-native. A timed native
Q8_0 roundtrip reconstructs the CPU down projection's effective F32 input; only
the down projection moves to GPU with explicit Q8 input precision. This needs
**no experimental CUDA parity patch or flag** on the tested stock backend.
Weights, selected experts and router probabilities remain unchanged.

- All sixty selected layer/expert/row configurations pass nine operator gates,
  observer controls and all output partitions/schedules.
- Zero/tie, maximum-row, chunked and ordered-reconciliation controls also pass.
- In 108 three-repeat resident/upload reports, fresh down-weight uploads give
  about 1.15-1.19x at 32 rows and 1.32-1.34x at 64 versus matched CPU execution.
  Fresh one/eight-row uploads lose; sixteen rows are not a stable win.
- Resident 32/64-row down weights give about 1.45-1.49x in the fixed-set probe.
- The tested 64-row split completes memcheck with zero errors/leaks and
  racecheck with zero reported hazards/errors/warnings. This does not change
  the older GPU-first racecheck's inconclusive status.

The total includes CPU quantization/reconstruction, F32 activation transfer,
GPU requantization, optional actual down-only weight uploads and synchronization.
Graph allocation, initial file loading and validation readback are excluded.
No raw activations/codes are published.

These are synthetic, concentrated-reuse layer batches, **not prefill or
generation tokens/s**. They justify actual-activation/full-target investigation,
not immediate product integration or a universal 32-row cutoff. A useful first
integration hypothesis is prefill-only reuse of already resident down weights,
leaving existing single-token decode unchanged and avoiding extra evictions.
Actual routing, workspace needs, cache interference and request latency must
still be measured.
An integrated study must preserve every token's routing and legitimate
zero/one-row pruning; the probe's four-experts-per-row reuse must not become
a shortcut that substitutes the last token's experts for a whole prompt.

## Research tooling improvements already validated here

- Optional native batch overrides are internal to the research launcher
  constructor and I/O CLI. Normal application launches retain their defaults;
  there is no new UI or public load-API tuning setting.
- Requested sizes are bounded, paired and checked against actual context logs.
  Missing/mismatched confirmation fails before generation. Verbosity 3 did
  not expose the required logs; verbosity 4 does.
- The logit checker now caps total and per-sequence output reservations to the
  decode group, while retaining full vocabulary/position coverage. The first
  uncapped run exhausted CUDA memory; its failed log is retained. Context and
  expert bank were not reduced to make the check fit.
- Effective-Q8 output is aggregate-only and schema-typed. Raw activation/code
  payloads are not published.

These are research harness improvements, not a request to replace Synapse's
runtime wholesale.

## Actual hidden-state / per-token routing gate

The [actual-input record](../benchmarks/results/actual-prefill-activation-replay-summary.json)
now passes all nine bounded frames: coding, prose and reasoning at layers
0/17/35, with 387 rows and 1,548 original expert selections. The replay bank
contains the exact original slices needed by each frame (4-94 experts), not
four replacement experts reused across every row. Probabilities are neither
substituted nor renormalized.

Native CPU replay matches actual captured FFN boundaries within the original
gate. CPU-activation/GPU-down versus CPU has worst relative L2 about 1.52e-7;
all nine intermediate gates and observer controls pass. The final layer keeps
its zero/one-row pruning. Scoped actual-input memcheck and racecheck pass.

There is an important capture qualification limit:

- Initial active captures under normal optimized CUDA execution perturb full
  logits beyond the gate; unchanged and inactive-observer controls are exact.
- Disabling CUDA fusion in **both numerical-reference and capture processes**
  gives exact agreement at all 96 held-out full-vocabulary positions.
- That reference deliberately uses native CPU experts and no MoE bank, not the
  12,000 MiB hybrid serving profile. It is not a capacity workaround or a speed
  comparison. The diagnostic CPU reference-kernel flag remains off.
- Accepted replay uses normal GPU fusion. The 96 logit positions qualify the
  capture observer, **not an integrated GPU-down model**.

The precise observer/fusion interaction is not yet isolated. Keep the failed
optimized-profile attempt and do not enable fusion-off globally to manufacture
a deployment result. Sensitive captures/references remain mode 0600 under a
0700 ignored directory; public reports contain only metadata/errors/hashes.
See the [capture/replay guide](../benchmarks/README.md#actual-prefill-activation-and-routing-gate).

### Bounded full-model prototype contract

The next research prototype should:

1. Be separate and default-off. Require an explicit prefill phase and genuinely
   multi-row FFNs; preserve zero/one-row, decode and verification paths. Row
   count alone must not classify speculative or multi-request decoding as prefill.
2. Reuse original native CPU gate/up/bias/SwiGLU and CPU Q8 semantics. Select
   GPU down work using live per-token routing and original probabilities.
3. Start with already resident down weights to avoid fresh admissions or
   evictions of the decoder's useful residents. Keep its existing bank budget.
4. For mixed residency, use checked gather/scatter of actual `(token, rank)`
   pairs and lossless original-ID-to-slot mapping. Do not feed unsupported
   sentinel IDs, reuse the last token's routes or fabricate zeros for required
   computation.
5. Bound all CPU/GPU quantization, activation, ID, staging and scatter storage;
   retain view ownership and synchronize before buffer reuse. Account for
   these buffers at the same context and bank, rather than shrinking either.
6. Compare capture-free baseline/candidate target logits with the unchanged
   production fusion policy, at the original gates. Operator accuracy alone
   does not establish whole-model equivalence through later quantization.
7. Only after that gate passes, alternate actual end-to-end requests and
   measure loading, prefill, TTFT, accepted-output decode, memory and residency.

This contract now has a separate research mask-control implementation, described
below. It is not a qualified product patch or a request to change Synapse's runtime.

## Full-model mask control: implemented, numerical gate blocked

The user selected a mask-based correctness control first, followed by compact
resident-aware dispatch once validated. The
[experimental patch](../third_party/patches/llama-b11669-resident-down-prefill-experimental.patch)
stacks on the existing published hybrid patch; its
[reviewed full-target results](../benchmarks/results/resident-down-full-model-prototype-summary.json)
explicitly block rollout.

Implementation uses explicit checker/server prompt phase, a phase-aware graph
cache key, CPU gate/up/SwiGLU and native-Q8 preparation, cached GPU down weights,
live per-token routes/probabilities and CPU output reduction. Bulk prefill
neither admits nor evicts experts; zero/one-row and decode use their old paths.
Protected zero GPU slots and supported CPU masks preserve all required work,
but run inactive GPU rows. That overhead is not free; this is the temporary
control, not the intended compact design.

All tests retain 16K context, a 12,000 MiB bank, 2048/512 batches, 8/16 threads
and ordinary CUDA fusion for the primary optimized comparison:

- Final default-off build matches the installed hybrid exactly at all 96
  checked positions.
- Enabled mode fails all three held-out workloads, reaching max absolute logit
  error 3.861 and relative L2 0.1813.
- A CPU-only control on the same new graph also fails the optimized-profile
  comparison, but is exact against a matched fusion-off baseline.
- Real GPU down still fails the fusion-off target check; limiting it to
  layer 0 is sufficient to fail. This is not merely cumulative offloading
  of many layers.
- All 735 live local weighted-FFN comparisons against a CPU oracle pass the
  original primitive gate, worst relative L2 about 1.17e-7. This supports
  investigating downstream numerical amplification, not blaming a large
  standalone projection error. The attention-state localization below was
  established after this initial experiment.

The functional authenticated server smoke succeeds and stops its owned
process; eight focused CTest targets pass. Earlier operator sanitizer passes
do not qualify this complete prototype. No request-latency benchmark or
compaction promotion was performed after the failed target gate.

Next isolate optimized-graph fusion behavior and follow the small GPU/CPU
arithmetic differences through subsequent activation quantization/router
decisions. If CPU-reference fidelity requires different GPU accumulation,
measure that implementation before accepting its performance. Do not snap
near-boundary values, widen tolerances or select only passing layers to
manufacture readiness. Compact dispatch alone has not been shown to solve
this numerical blocker.

## Numerical localization: attention-state rounding confirmed

The [localization follow-up](../benchmarks/results/resident-down-numerical-localization-summary.json)
now identifies a concrete amplification path for the layer-0-only experiment.
It covers one held-out coding request and the first 512-row prefill batch,
with 16K context, the original 12,000 MiB bank and matched CUDA fusion-off
controls. All three trace focuses preserve their respective capture-free
reference logits exactly at all 32 checked positions. The candidate still
fails the original model gate; observer preservation is not candidate approval.

1. Layer 0 inputs, routes and probabilities are identical. Its FFN output
   first differs by at most 5.72e-6.
2. Layer 1 Q/K/V projection outputs remain close, with relative L2 around
   2-4e-8. Their weights are original Q8_0.
3. Native F16 attention-cache storage has **19 changed keys and 30 changed
   values**, reaching differences of 0.001953125 and 0.0009765625.
   Actual stores match their original F32 sources' IEEE-F16 rounding
   cell-for-cell at verified cache indices. Nothing is overwritten or snapped.
4. Attention context/output and post-normalization differences grow. Native
   CPU-Q8 FFN-input replay changes 1,307 codes and 448 scales; layer-1 FFN
   output differs by as much as 0.046009.
5. Layer 2 first changes routing: five rows change rank order, but only one
   changes the expert set, at first-batch row 138 (zero-based).

Query-F16 rounding replay predicts 195 code changes, but actual internal
query conversion is not captured. The separate effects of Q, K and V have not
been ablated. This is the first directly observed large/discrete amplification
boundary, not proof that every source of the final logit error is solved.
The optimized-fusion graph discrepancy remains separate.

This cache is the model's attention working state, not the Linux file cache.
More RAM, a RAM disk or faster model storage does not remove this rounding
mechanism. Merely selecting F32 KV storage is also not an established fix:
the pinned native Flash Attention implementation can convert F32 K/V back
to F16 internally.

### Next advancements to test

1. **Precision-compatible GPU down projection.** The isolated implementation
   below now matches raw dot and CPU post-op bits on this host. Extend that
   contract into the real graph, including native quantized-state parity.
   The pinned
   [native x86 dot implementation](https://github.com/ggml-org/llama.cpp/blob/620d2225cc436ab0b001e3200e590e3b286ef9ab/ggml/src/ggml-cpu/arch/x86/quants.c#L982)
   uses paired eight-lane FMA accumulators and a fixed horizontal sum;
   simply requesting more precision does not guarantee identical rounding.
   Passing actual-input and adversarial probe checks is not universal CPU/GPU
   support or a full-model qualification.
2. **Packed native-Q8 input and fusion-preserving integration.** Direct Q8_0
   input could avoid F32 reconstruction and GPU requantization. Its format is
   34 bytes per 32 values versus 128 bytes for F32, about 3.76x fewer input
   bytes before other costs. That input path is now implemented and timed in
   the isolated probe; all raw outputs return to CPU for original post-ops.
   Separately, make the CPU-only optimized graph control exact without intrusive tensor
   observations or globally disabling fusion.
3. **Compact resident-aware prefill dispatch.** After numerical/state and
   normal-fusion full-target gates pass, compact live token/rank work, reuse
   bounded buffers and test overlap without changing decode or evicting its
   residents. Then compare real loading, prefill, TTFT, accepted-output decode,
   memory and whole-request time before asking Synapse to integrate anything.

The next compatibility tests should include native KV/Q8 state parity, not
only a small local floating-point L2 error. These are proposed research
advancements, not demonstrated full-model performance improvements.

## Packed compatible-down follow-up: local bits and performance now pass

The [new study](../benchmarks/results/cpu-compatible-packed-down-summary.json)
implements a standalone CUDA raw MXFP4/Q8 dot with native CPU accumulation
order, plus a faster byte-permute/DP4A integer variant. Original CPU
quantization, bias, probabilities and ordered reduction are retained.

- **384/384 checks pass across both variants.** Raw projections and final
  post-processed FFNs match local native CPU controls bit-for-bit.
- Actual routed frames cover coding/prose/reasoning at layers 0/17/35,
  387 unique rows and 1,548 original selections, repeated three times.
  The historical captured-boundary discrepancy remains at most 5.504e-8
  relative L2 and is not declared bit-exact.
- With all required replay down weights resident, the DP4A raw stage takes
  about 3.25/3.33/3.33 ms versus CPU 10.32/10.41/10.04 ms at layer 0.
  Layer-17 comparisons are about 2.99/2.98/2.89 ms versus
  6.83/6.68/6.90 ms. These include Q8 preparation, input/ID transfers,
  raw-output return and synchronization, but exclude shared CPU
  gate/up/SwiGLU and post-ops.
- Fresh uploads remain unfavorable at 16/32 rows and mixed at 64.
  This supports testing already-resident work, not uploading every expert.
- Memcheck, racecheck and synccheck pass on an actual 94-expert frame for
  both implementations. Ordinary and opt-in probe tests pass.

This is a meaningful arithmetic/performance feasibility result, **not a
deployable Synapse improvement**. The probe loads all required replay experts
on GPU rather than honoring the model's mixed residency plan. It has no
model attention cache or target-logit trajectory. The existing enabled mask
prototype still fails, and the optimized CPU-only graph discrepancy is open.

Next make that optimized control exact and integrate the new dot with original
CPU post-ops in a separate research backend, at the same context/bank and
unchanged zero/one-row/decode behavior. Qualify actual native KV/Q8 state and
all 96 held-out full-vocabulary positions before compact dispatch. Real
loading, prefill, TTFT, accepted-output generation and total-request timings
remain mandatory before asking Synapse to adopt anything.

## Native handoff follow-up: exact scalar controls, mixed performance

The [native handoff study](../benchmarks/results/handoff-critical-path-summary.json)
leaves the warmer off and traces exact installed backend libraries. It finds
that tiny scalar dispatch/copy graphs still request eight CPU workers.
A separate default-off research interposer tests those bounded graphs with
one worker, retaining every native operation and the original expert plans.

The candidate is exact at all 96 original long-prompt full-vocabulary positions.
Three matched 512-output pairs per coding/reasoning/prose workload preserve
all observed hashes/counts and original 16K/12,000 MiB/8+16/2048+512 settings,
normal fusion, ordinary affinity and identical backend mappings.

| Workload | Native tok/s | One-worker scalar controls | Change |
| --- | ---: | ---: | ---: |
| Coding | 51.46 | 51.90 | +0.85% |
| Reasoning | 51.74 | 51.81 | +0.13% |
| Prose | 57.52 | 57.29 | -0.41% |

**Do not promote this as a general performance optimization.** The small
cross-workload result is mixed. Scalar controls account for only about 49 ms
of a 3.068-second diagnostic envelope; this is not enough budget for the
overall throughput goal. CUDA APIs occupy a larger observed part of the
CPU/GPU gaps, but their durations include necessary waits/driver work and
cannot simply be credited as removable latency.

The native path already uses pinned intermediates and CUDA graph replay.
Native Q/K/V graph optimization passes short numerical controls but does not
win a three-pair coding screen; reported metadata does not prove active
concurrency. A CUDA12 pinned-copy/sync microbenchmark also finds no explicit
spin advantage over auto, while blocking is slower.

Keep these as research findings. Further implementation should demonstrate a
different, costed handoff protocol with proven buffer lifetime/order before
request-speed qualification, not merely rename existing overlap/packet work
or skip required fences. No Synapse code/default was changed.

## Live CPU-cache warmer: tested, not a throughput recommendation

At the user's request, the [live study](../benchmarks/results/live-cpu-cache-warmer-summary.json)
implements the warmer despite the offline early-information concern. This is
a separate opt-in research interposer; no installed plugin, target weight,
Synapse file or application default changed.

The helper reads original native CPU expert-weight prefixes using the
current real router IDs and pre-admission residency map. There is no
prediction, expert substitution, model copy or changed quantization. Work is
bounded by a one-job pending queue, byte cap, checked wall budget and
cancellation at CPU expert entry. Expert execution never waits for warming;
model-buffer release does drain the reader before memory is freed.

Final matched tests use the original 16K context/12,000 MiB bank/8+16 threads/
2048+512 batches and normal fusion, with the same CCD CPU mask for every
native/noop/read variant. The helper uses logical CPU 16 in that CCD, sharing
physical core 0. CPU affinity is an explicit experimental condition, not an
uncredited optimization.

| 512-output workload, three triplets each | Native tok/s | No-op helper tok/s | Live-read tok/s |
| --- | ---: | ---: | ---: |
| Coding | 51.74 | 51.47 | 51.48 |
| Reasoning | 51.92 | 51.82 | 51.80 |
| Prose | 57.60 | 57.20 | 56.97 |

All 27 observed output hashes/counts match within workload. Live-read
throughput is 0.2-1.1% below native medians: no demonstrated gain, and too
small a difference to treat as a precise universal regression. Single-sample
budget and cross-CCD screens do not establish a winner. An unrestricted
coding triplet also shows no gain.
The clean matrix requires identical installed backend-library mappings across
all variants. An initial preload pulled a different research GGML base into
the candidate; those earlier timings are retained as exploratory, not silently
reused. The original long numerical gate and full request matrix were rerun
after rebuilding against the installed library set.

The final read path passes the **original long-prompt 96-position**
coding/prose/reasoning full-vocabulary gate exactly. Off/noop/read earlier
short-corpus controls also pass exactly. Four fixture methods cover lifecycle,
configurations, zero/noop/expiry and repeated unload/reload. Three scoped
address/undefined-behavior sanitizer cases pass with native plugins
uninstrumented and leak detection off.

Actual helper-job NVTX ranges overlap GPU kernels by about 0.012% and any
GPU activity by about 4.6%, mainly transfers; about 21% overlaps CPU control
graph calls. Whole-process job ranges include checks/cancellation and any
scalar initialization/prefill. They are not individual cacheline timestamps
or request-speed evidence.

This identifies the integration problem: the current exact route trigger
mostly starts warming **before GPU expert kernels launch**, not inside a
substantial independent GPU-compute window. Reading GB of logical cacheline
payload is not proof of LLC hits or removed critical-path time. The aggregate
also separates CPU-needed versus GPU-admitted read snapshots and cancellation
tails instead of crediting all warmed bytes as useful.

**Do not integrate or enable this warmer in Synapse as a performance feature.**
The research implementation is retained for future experiments. A genuinely
different early-hint or execution-order design would need a new gate; simply
enlarging the warm budget has not demonstrated a solution.

## GPU-window CPU-cache staging: conditional research candidate

The [new feasibility study](../benchmarks/results/gpu-window-cpu-cache-summary.json)
tests staging original CPU expert weights into CPU cache during measured GPU
work. This is neither more GPU residency nor a RAM disk. It leaves target
arithmetic, routing, context/bank, installed plugins and application defaults
unchanged.

### Positive primitive result

Across three real-weight layers and three repeated invocations, warming the
full 13.22 MB projection payload of one expert into the **same CCD** makes its
isolated native CPU FFN about 3.0-3.13x faster, bit-for-bit. A two-expert,
16 MiB-prefix calibration improves compute about 1.65-1.78x. The helper's
touch/handoff is charged separately and uses owned copies, not original model
mappings. These tests use synthetic one-row activations.

This is not free: fully warming one expert costs about 0.28 ms, and serial
warming plus compute loses. Cross-CCD warming is actively harmful, around
0.61x compute speed. The two 32 MiB CCD caches are not one uniform 64 MiB
working tier. Small helper wake/ack jobs alone cost 7-9 us.

### Integration gate: early information, not the cache primitive

Four native 128-output requests have aligned CUDA-kernel/copy, CPU-graph and
observer-bookkeeping timelines. At an ideal 32 MiB / 40 GB/s model:

- Perfect knowledge of the next CPU demand could stage about **27-33%** of
  logical projection payload before its use.
- Current CPU-visible cold IDs and pre-admission dispatcher hints stage
  **less than 0.3%**.
- A previous-same-projection cold-set control covers only about **2-3%**,
  with just 6-9% of staged bytes useful. Wrong predictions consume the
  modeled bandwidth/cache budget; they are not filtered retrospectively.

The real selected IDs/slot map are inspected read-only before admission.
Experts subsequently admitted to GPU are charged as wasted staging work.
Opaque tensor identity, not repeated generic names, binds the producer.
GPU-copy windows debit shared RAM traffic. Ideal retention, zero helper
overhead and profiler-perturbed scheduling still make the bounds optimistic;
no tok/s estimate is derived.

### Recommendation

Do not add a default CPU warmer or launch a predictor implementation yet.
First demonstrate earlier hints with useful byte-weighted predictions **under
the real residency/admission policy**. Any next prototype must:

1. Place helper and consuming workers in a compatible cache domain, measuring
   affinity costs rather than assuming fixed-CCD calibration is a faster app.
2. Warm bounded native-layout tiles in actual consumption order; preserve
   original weights, CPU math, routing and GPU residents.
3. Charge helper dispatch, cache pollution, wrong predictions, DRAM/DMA
   contention and per-token service costs.
4. Pass original full-vocabulary gates and repeated real request timings
   before promotion. A 3x isolated warm-cache kernel ratio is not a model gain.

Traced output hashes match all four controls. A separate short-corpus
coding/reasoning/prose profiler preservation check is exact at 96 positions.
The historical long-prompt preservation attempt timed out after two exact
32-position requests; it remains incomplete and is not presented as passing
all 96. Raw expert IDs, addresses, CUDA traces and private references are
excluded from the handoff packet. Synapse was inspected read-only only.

## CPU decode follow-up: optional lower-CPU-use profile

The [reviewed CPU study](../benchmarks/results/cpu-decode-efficiency-summary.json)
investigates decode instead of continuing GPU-prefill integration. The checked
native CPU backend uses GNU OpenMP. Its idle-spin behavior keeps process CPU
use near eight cores, even though synchronous CPU graph intervals occupy only
about 60% of an instrumented generation window. Those intervals can overlap
GPU work; they are not marginal critical-path savings.

Setting **`GOMP_SPINCOUNT=1000` only in the backend child environment** gives
this repeated result at unchanged 16K context, 12,000 MiB bank, 8/16 threads,
2048/512 batches, original weights and the existing direct-DMA pipeline:

| Workload | Native tok/s | Low-spin tok/s | Generation-window CPU reduction | Whole-request CPU reduction |
| --- | ---: | ---: | ---: | ---: |
| Coding, 512 outputs | 50.08 | 47.77 | 28.5% | 20.8% |
| Reasoning, 512 outputs | 49.50 | 47.54 | 28.7% | 19.9% |
| Prose, 512 outputs | 55.50 | 53.21 | 31.9% | 22.3% |
| Long 3,577-token prompt, 256 outputs | 45.32 | 43.68 | 27.0% | 3.1% |

Each row uses three fresh-process pairs in A/B, B/A, A/B order. Short request
wall time rises about 4.2-4.5%; the long-request median rises 1.7%. Individual
runs can vary more than these median effects. All 24 requests retain matched
output hashes and exact counts. Both native and low-spin modes are exact at
all 96 held-out full-vocabulary positions under the original gates.

Paired diagnostic profiles retain the same 55,728 ordered CPU matmul/count/
byte records and 254.18 GB of logical projection weights per run. They support
the idle-waiting explanation, not a reduction in required expert work.
Diagnostic timings are not included in the throughput comparisons.

### Appropriate downstream scope

- Keep the fastest native behavior as the default.
- If useful to users, expose an explicit **Lower CPU use (may reduce generation
  speed)** option for the checked GNU OpenMP backend. Recheck the current
  Synapse launch surface before implementation.
- Apply the variable only to a freshly launched owned backend process, never
  the global shell/system environment. Remove it for rollback.
- Attest the selected child environment and supported CPU runtime. Native
  pthread `--poll` is a different control; this setting is not portable to
  arbitrary OpenMP implementations or already-running processes.
- Preserve original model/context/bank/routing and normal execution fallback.
  Do not ship the profiler/VNNI shims or replace installed CPU libraries.
- Do not label this energy saving: package-energy reads are permission-denied.
  The measured benefit is fewer process CPU-seconds, with a speed cost.

This is the first new request-level **resource** result in this follow-up,
not a new throughput optimization. The small corpus and one machine do not
qualify all models, hardware, concurrent-user behavior or general answer quality.

### Negative directions retained

The VNNI replacement is bit-exact at raw and full-model levels but loses about
1.3% decode throughput in three pairs. Close OpenMP placement is flat; spread
is slower in bounded pilots. A separate non-OpenMP CPU backend and native-pool
polling 50/0/1 show no robust win and are not promoted or full-logit qualified.

Six native CPU-reference requests supply 768 training and 768 held-out routing
tokens, with inactive-observer continuations matching. Prompt-only and
50/50 prompt/training full-cache rebuilds lose modeled net cost against
trained initialization plus causal admission on all three held-out requests,
even with zero marginal GPU cost. Static hit-rate improvements are not a live
adaptive-cache performance result. Partial/pruned prefill layers use training
priors; held-out decode never selects the seed.

## Reproduction and private evidence

Use the [batch study commands](../benchmarks/README.md#real-model-prefill-configuration-qualification)
and [full-logit checker guide](../benchmarks/README.md).
Set the installed hybrid profile, freeze the context file, use a new report
for every load, and alternate baseline/candidate order. Preserve output counts,
actual context, bank, threads, plan hash and backend activation markers.

The machine-readable summaries name and hash the retained local raw reports
under the ignored results directories. Native binary references contain token
IDs and logits and remain private with restrictive permissions. Do not include
them, raw prompts/activations or authentication keys in an external packet.
The published summaries contain hashes, aggregate metrics and sanitized paths.

## Next experiment and rollout gates

The isolated split and bounded actual-input replays pass their stated gates,
but the implemented full-model mask control fails target qualification.
Resolve that numerical blocker before compact dispatch or rollout, retaining
the capture-profile limitation above. Do not replace Synapse's
runtime or decoder with the probe. Retain CPU Q8 canonicalization, preparation,
transfers, synchronization and residency effects in the measured cost.

Before asking Synapse to integrate any new optimization:

1. Preserve the new split's passing operator/stress/default-off controls and
   expand them on actual inputs at the original primitive gate. The old
   GPU-first prototype remains a separate 57/60 experiment.
2. Extend bounded actual-activation coverage to mixed residency and a complete
   integrated target at fixed context and bank, not just favorable expert sets.
3. Pass unchanged full-vocabulary gates across held-out coding, reasoning and
   prose, with long prompts that exercise the changed batching and at least
   96 teacher-forced positions. A failing check blocks promotion; do not widen
   tolerances, discard difficult experts or substitute zero outputs.
4. Repeat fresh-process, alternating real requests across those workloads.
   Report loading, prompt latency, TTFT, accepted-output decode, full request
   time, sampled memory pressure and any decode-residency regression.
5. Qualify broader answer behavior separately, and retain a default-off path and
   existing standard-execution fallback.
6. Only then publish reviewed, pinned research artifacts and give Synapse's own
   agent a surgical integration/test/rollback plan.

Current rollback is simply **no deployment**: preserve installed libraries,
native batch defaults, existing thread policy and the current Synapse installer
pins. Do not enable the experimental parity flag globally. Fast-disk staging
remains a separate explicit workflow; cached generation does not justify
copying this model to a RAM disk.
