# Model-agnostic architecture study

Date: 2026-10-19. Status: staged comparison complete; throughput lead found,
downstream numerical/integration gates remain blocked.

Subsequent work in the [optimized-verification study](optimized-verification-study.md)
removes the native optimized-scalar numerical blocker on bounded tested
windows, including GPT rejected-suffix rollback. Its live GPT candidates are
still slower than scalar; the original canonical failures below remain valid.
No 60 tok/s or rollout gate has passed.

## Objective and boundaries

Improve accepted-output latency/throughput without changing target weights,
dropping selected experts, silently changing precision, or counting aggregate
batch/proposal throughput as single-user generation. Sustained 60 tok/s on
unchanged GPT-OSS-120B remains a goal, not a required positive outcome.

Research belongs in https://github.com/appatalks/unified-memory-scheduler.
https://github.com/appatalks/Synapse is a read-only downstream consumer.
The existing Strata installation and original model files also remain
unchanged. No downloads, archive remounts, cache drops or automatic eviction
are part of this protocol.

## Phases and exit gates

| Phase | Work | Required exit evidence |
| --- | --- | --- |
| 0. Asset/protocol qualification | Verify existing Qwen GGUF shards, Strata pack/native offsets, draft assets, binaries, source revisions, mounts and space | SHA-256/shape/type manifest; actual controls; explicit mismatches or blockers |
| 1. Common target-pass control | Same Qwen weights, token IDs, 16K context, F16 KV, 8/16 native threads, common frozen resident expert IDs | Fresh-process scalar controls, exact counts and zero prompt reuse; separate numerical contracts per engine |
| 2. Speculation/architecture ablations | Scalar versus engine-native MTP, then token-graph versus host-layer launches; keep target placement fixed | Accepted-output speed, draft/verify costs, acceptance, memory, quality and output-state comparisons |
| 3. Attribution | Decompose target-pass latency, expert traffic/reuse, handoffs and accepted tokens per verification | Measured explanation and costed upper bound; no attribution solely from unmatched headline rates |
| 4. Justified shared-policy experiment | Select only a lead with meaningful measured upside; expose applicability through descriptors/capabilities | Unchanged within-engine numerical gates and repeated held-out before/after results on applicable model families |
| 5. Review/handoff | Record findings, limitations, negative cases and reproducible commands | Reviewed machine-readable evidence and a clear general versus specialized recommendation |

An unsupported or failing phase is reported explicitly. It is not replaced
with a proxy, relaxed threshold or a claim of completed implementation.
Dependent phases proceed only when their prerequisites justify them.

## Frozen comparison conditions

- Primary shared model: the existing Qwen3.8-Flash-Next GSQ-RCO IQ3_S shards.
  GPT-OSS-120B MXFP4 remains a separate transfer/qualification target.
- Primary residency control: the existing training-only Qwen 6,144 MiB seed.
  Export that exact expert set to Strata; measure actual payload and allocation
  overhead separately. Do not equate expert count with bytes across formats.
- Context: 16,384; F16 KV; no KV streaming/growth; text only; one request.
- Keep all target experts: explicitly disable Strata tail skipping and
  approximate/control-vector features. No borrowed GPT hot-set plan.
- Disable adaptation, prompt/conversation reuse, prompt lookup and transient
  cold-GPU offload in the attribution control. Adaptive/native-default policies
  may be studied later in a separately labeled track.
- Strata's native-pack startup requires a verify capacity of at least two,
  but its serve loop uses one actual row when MTP/lookup are absent. Validate
  actual proposal/output counters; do not label configured capacity as work.
- Start with a bounded 512-token physical prefill batch where supported.
  If an engine cannot match it, report the difference and do not attribute
  prefill speed solely to architecture.
- Use identical pre-rendered token IDs, fixed template time/timezone,
  greedy sampling and common output caps. Performance and natural-EOS
  sanity/quality tracks remain distinguishable.
- Performance: three fresh alternating pairs per coding/reasoning/prose
  workload, 256 or 512 output tokens selected before a cohort begins.
  One warm functional/qualification run precedes timing promotion.
- Report load-to-ready, prefill, first token/text, accepted-output generation,
  total request time, actual output counts, CPU/memory/swap/physical reads,
  residency and available draft/verification counters.
- Model-native draft implementations may differ in quantization, vocabulary
  subset and hidden-state conventions. Record these as factors rather than
  calling them matched drafter implementations.

## Numerical and quality contracts

Cross-engine floating-point results are not assumed bitwise identical.
Separate unchanged target-byte identity from execution semantics.

Within the research backend, preserve the existing full-vocabulary gates:
maximum absolute error <= 0.01 and relative L2 <= 0.001, with 96 held-out
positions. Within Strata, verify scalar/speculative consistency under its
documented reproducibility controls and retain any divergence.

Cross-engine throughput can be reported as an implementation comparison
without being promoted as a numerically interchangeable speedup. Use the
same bounded arithmetic, structured-output and coding sanity cases; distinguish
these from broad answer-quality evaluation. A failure remains visible.

## Model-agnostic capability policy

Shared policy inputs are tensor shapes/types, expert domains/top-k, bytes,
cache capacity, measured resource costs, numerical compatibility and proposal
acceptance/reuse. They are not checkpoint-name constants.

Specialized capabilities may include GDN versus attention state, PLE lookup,
trained MTP, quantization kernels and compatible scalar/batched reductions.
Unsupported capabilities remain disabled with explicit diagnostics. A positive
Qwen result does not establish GPT support, nor does a single-format kernel
become the generic engine contract.

## Initial asset findings

- Configured archive root is currently unavailable; existing library links
  into it are broken. No mount repair or model relocation is authorized.
- The active Strata settings identify existing Qwen/draft assets on the root
  SSD outside the configured model roots. Use manifest-owned zero-copy
  references under the approved SSD model root, preserving originals.
- Local Strata source is clean at
  `7695859aa13afb248bdb4d36f9b2e95707db079d`; both existing engine binaries hash
  `73b66e36e16a8336c09fd834ae00fe009edc1d3b535f9f65fd0f6fc1c29ee47e`.
  Do not assume the binary's source binding without embedded/build evidence.
- Latest upstream documentation was inspected at
  `f57c0adcb31c00f3f216550c004dac4e713a9f70`, not substituted for the executed build.
- Native backend pin remains
  `620d2225cc436ab0b001e3200e590e3b286ef9ab`.
- Existing Strata target conversion manifest reports no tensor conversions;
  native-expert offsets point directly into the original GGUF. Verify those
  claims before comparison.
- The previous native MTP bridge is not currently discoverable at the checked
  locations. Its historical benchmark is not a new result. Restore/rebuild it
  only from verified existing assets if required.

## Results

### Phase 0 qualification and explicit amendments

Both original GGUF shards were completely SHA-256 verified. The native expert
index covers all 48 layers with matching types/absolute source offsets, and
777 dense tensors match their original bytes. Native and Strata tokenizers
agree on every token of the six frozen prompts.

The first functional smoke loaded 3,991 experts when passed the intended
3,067 count: Strata interprets the argument as a budget of largest-sized blobs
and fills more smaller experts when the profile allows them. The comparison
profile is therefore deliberately partial, containing exactly the shared
3,067 experts. A valid startup must confirm that count and the shared payload;
the initial smoke is not promoted as a matched performance result.

Strata's pool is configured as seven background workers plus its participating
host thread, matching eight compute participants rather than calling eight
background workers plus the host "eight threads." Engine-native coordinator
overheads and actual process CPU use remain measured.

Shared rendering uses low reasoning effort, enabled/preserved thinking,
the same helpful-assistant system message, epoch zero and UTC. This is the
controlled research workload, not Strata's separate community/model-card
benchmark defaults.

The confirmed existing engine reports version 0.1.43; latest upstream docs
are not used to relabel that binary. Completed staged results appear below.

### Frozen cohort amendment before primary timing

The corrected functional gate confirms 3,067 experts in both engines. The
shared logical payload is 6,277,760,000 bytes (5,986.9384765625 MiB); Strata's
integer INFO field rounds this down to 5,986 MiB. The original complete profile
and its manifest remain unchanged; the new partial profile has its own control
manifest.

Primary scalar timing uses 512 output tokens, three alternating engine pairs,
and all three held-out workloads per fresh process. Prompt reuse and residency
adaptation remain off between workloads. Native expert weights retain their
original mmap storage; Strata allocates a resident CPU arena. This allocation
difference is reported rather than hidden by changing native loading policy.
Physical reads, process CPU time and swap are measured separately for the
request and post-first-output generation windows.

The common streaming throughput denominator is output tokens minus one divided
by wall time from the first output token to completion. Engine-native prompt
and decode timings remain available but are not treated as identically defined
cross-engine measurements. Both raw-token adapters enforce prompt and output
counts, scalar zero-draft counters, frozen-residency acknowledgement and owned
process cleanup.

The first 512-token native attempt failed the collector's raw-ID count gate.
The server's streaming implementation omits an intermediate raw ID when it
buffers an incomplete UTF-8 character, although the generated count and text
remain complete. That failed attempt is retained, not silently repaired.
Before restarting the complete cohort, collection is amended to audit every
native request with a full nonstream replay after all timed requests finish.
The complete replay must match every streamed ID at its reported generation
position and return the exact cap/prompt count. Timing uses the original
authoritative accepted count; complete hashes use the audited output.
Missing streamed IDs and replay time are explicit fields, never fabricated
tokens or additional accepted throughput. Any replay mismatch fails the run.

### Ablation gates and bounded follow-up

Source inspection shows that `--no-token-graph` is not a valid graph-disable
ablation for the executed native-pack serve verifier. It controls the separate
token-loop path, while this pack uses verification windows. Do not benchmark
that flag as proof of a whole-token scheduling benefit.

Native canonical block sizes two and four pass the original full-vocabulary
thresholds at all 96 checked positions, with zero measured error. Optimized
scalar versus canonical fails those same thresholds; no threshold is widened.
Scalar/MTP output hashes also differ on some 512-token cases. These modes
therefore remain separate numerical execution contracts, not a transparent
replacement for the optimized runtime.

The four-row MTP screen is mixed in both engines. A bounded two-row follow-up
tests the lower-waste alternative: three fresh alternating Strata scalar/MTP
pairs, 512 tokens per workload, both with capacity two and the same frozen
expert set. This is an exploratory engine-native setting experiment, not a
trained shared controller or a numerically qualified downstream improvement.
Natural-EOS arithmetic, code-evaluation and structured-extraction sanity cases
are collected separately. They cannot substitute for full-vocabulary gates.

The first paired follow-up retains Strata's default dynamic CPU prefill share.
Source/log inspection subsequently identified this as a changing numerical
placement: the measured CPU/GPU split can change prompt activations even when
decode residency is frozen. Those results remain exploratory. A separate final
confirmation cohort sets `STRATA_PREFILL_CPU_SHARE=0` in both arms, records the
observed child environment, and repeats all three 512-token pairs. This changes
the prefill control, not target bytes or the decode expert set. Do not combine
the two cohorts or attribute the prefill-policy effect to MTP.

## Reviewed end results

The [machine-readable record](../../benchmarks/results/model-agnostic-architecture-summary.json)
retains samples, paired percentages, output hashes, physical reads, numerical
failures and promotion limits. The [handoff packet](../synapse-prefill-handoff.md#architecture-packet-a-real-throughput-lead-with-explicit-transfer-gates)
states what Synapse's own session can implement and what must stay blocked.

### Final unchanged-target before/after

Strata 0.1.43, fixed GPU prefill, exactly 3,067 resident experts, 16K/F16 KV,
three fresh alternating pairs, 512 accepted outputs per request:

| Workload | Scalar median | Two-row MTP median | Ratio-of-medians gain | Median paired gain |
| --- | ---: | ---: | ---: | ---: |
| Coding | 47.67 tok/s | 57.20 tok/s | +20.01% | +19.22% |
| Reasoning | 34.37 tok/s | 43.85 tok/s | +27.57% | +28.10% |
| Prose | 38.43 tok/s | 45.26 tok/s | +17.77% | +14.43% |

All nine pairs improve and have identical full 512-token hashes. Each arm also
repeats its output exactly across all three runs. This is a real within-engine
throughput result, not a selected passing layer, reduced expert contribution,
smaller KV/context or counted draft proposal. Whole-request throughput improves
in every pair, with median paired gains of 17.77%, 26.31% and 13.49%.

All 15 separate natural-EOS sanity cases pass: code-result evaluation,
arithmetic and structured extraction across native scalar/canonical/MTP and
Strata scalar/MTP. This is bounded sanity, not broad answer-quality proof.
The first confirmation pair retains small physical generation reads; both
later pairs have zero in both arms and still improve. All owned-process swap
snapshots are zero. No disk-cache drops, RAM disk or energy inference is used.

### Attribution and useful architecture distinction

Two-row windows deliver approximately 1.77/1.89/1.73 accepted outputs per
post-first window for coding/reasoning/prose. They cost approximately 31.0,
43.2 and 38.3 ms per window, about 1.48 scalar steps rather than two.
Verified target rows increase by about 12.7%, 5.7% and 15.4%; fewer target
invocations plus shared-cost amortization outweigh that extra work.
Do not call this a measured CPU/GPU graph-only decomposition or an exact
expert-union byte-saving measurement: those finer attributions are unavailable
from the executed control. Existing parallel CPU/GPU placement alone does not
explain tokens emitted per verification.

Four-row MTP is not universally better. Initial screens improve reasoning but
lose coding/prose, and native MTP screens do not reproduce the final Strata
gain. A policy should minimize
`(draft + verify + commit cost) / expected accepted outputs`, subject to the
numerical contract, rather than maximize window length or proposal acceptance
alone.

### Qualification and phase exits

| Phase | Result / exit |
| --- | --- |
| 0. Assets | Complete: two full shard hashes, 48 native expert layouts, 777 dense tensors, shared tokenizer parity and explicit binary/source caveat |
| 1. Target-pass comparison | Complete: three native/Strata scalar pairs; different outputs/execution semantics remain implementation comparisons |
| 2. Ablations | Positive repeated two-row Strata result; mixed native/four-row screens; graph-disable ablation unsupported |
| 3. Attribution | Target-window/row counts and accepted-output cost measured; finer graph/kernel attribution not established |
| 4. Shared-policy gate | Bounded existing-knob experiment complete; new live generic controller and native transfer not promoted |
| 5. Review/handoff | Reviewed aggregates, reproduction tools and downstream compatibility worklist persisted; no deployment |

Native canonical target blocks two and four are exact at 96 held-out
full-vocabulary positions against canonical scalar. Optimized versus canonical
fails the original thresholds: worst max-absolute 1.97075 and relative L2
0.11157. No tolerance is widened. Matching Strata greedy outputs do not
establish the missing full-vocabulary/committed-state contract, so they cannot
qualify a native port or bypass Synapse's hybrid/speculation guard.

The native draft was restored from 31 SHA-verified existing BF16 tensors using
the pinned upstream transforms. The used F32-router/Q4_0-expert derivative has
34 tensors, hidden width 10,240, 2,469,565,024 bytes and SHA-256
`fd3feffc468282e3222b0136d4a4eff4b4933e01032ce65815eacf6489e5332e`.
Its target embedding/output bytes are unchanged. It is not the historical
derivative and is not the same quantization/vocabulary implementation as the
Strata Q2 draft. A provisional F16-router derivative is retained but unused;
both derivatives have ownership manifests under the approved root. Originals
are preserved and no model is automatically evicted.

Synapse was rechecked read-only at
`545a7ebf4a6dea3dd4a1ac04760057eed2de1b15`. Its current hybrid constructor
rejects speculation. No Synapse, installed runtime, user configuration or
default was changed, and no commit/push was made.

**Scientific conclusion:** there is now a reproducible Qwen throughput lead,
but not a model-agnostic production optimization or a GPT-OSS-120B 60 tok/s
result. Generalize the cost/compatibility policy, not a checkpoint-name rule
or an unconditional two-row setting. The next implementation gate is exact
target verification/rollback compatibility in the downstream backend.

### Verification record

Both native research executables build against matching installed-backend
headers/libraries. All six targeted CTest entries pass: architecture/profile
aggregation tests, adapter/count/audit/cleanup tests, renderer privacy/help
tests, and the existing full-logit checker's input/help tests. The rebuilt
renderer reproduces all six shared token sequences exactly. The pure profile
builder reproduces the frozen binary byte for byte.

Post-cohort CPU/CUDA/base library hashes match the preexisting verified
installation; this is a post-cohort snapshot, not a claimed per-request
cryptographic attestation. Public aggregate privacy invariants pass, raw
tokens/logits/logs remain ignored in a private directory, the external Strata
checkout remains clean, and no owned inference process remains running.
