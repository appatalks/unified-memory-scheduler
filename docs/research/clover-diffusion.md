# Clover block-diffusion research

Date: 2026-11-07. Status: the merged-backend load failed; an older prototype
produced a preliminary autoregressive baseline, but its GPU self-speculative
path crashed. A separate experimental transfer change completes GPU
self-speculative requests but is slower and fails output-hash comparisons.
This is research, not a Synapse release.

The later [candidate speed and GPU follow-up](#candidate-speed-and-gpu-follow-up)
passes a bounded GPU decode contract and measures its speed. Earlier failed
tests below remain valid for the exact candidates that produced them;
concurrent/chunked prefill and production qualification remain blocked.

The [scheduling and family follow-up](#scheduling-and-family-follow-up)
records later bounded workload successes, eight-client longer-stream
failures and the remaining cross-family blockers.

The later [attention-core follow-up](#attention-core-follow-up) resolves the
tested GPU eight-sequence long-stream discrepancy and measures a bounded
aggregate gain. It does not qualify CPU-only longer streams or a release.

## Attention-core follow-up

Reviewed evidence:
[`moe-component-qualification-summary.json`](../../benchmarks/results/moe-component-qualification-summary.json).
Source remains prototype `cb09d4355d8bf968bb907c5de03cf42d8aa073ce`.
The probe's compile-time `VRAMBRIDGE_PROBE_STEPS=320` preserves the default
32-step protocol and binds the step count in its reference header.

The previous scalar-projection candidate matches every tested 2/4-sequence
position over 320 steps, but eight sequences first diverge at step 246.
It passes 1968/2560 positions at width eight; subsequent recurrent state
drifts. This puts the failure beyond the earlier short screen and near
the padded attention-cache transition. The attention-off control in the
new build reproduces exactly the same failure.

The default-off
[`llama-moe-attention-sequences-experimental.patch`](../../third_party/patches/llama-moe-attention-sequences-experimental.patch)
isolates the Flash-Attention core per independent causal stream, not just
its Q/K/V/output projections. It views existing Q/K/V/mask tensors,
preserves F32 accumulation, sinks and soft caps, and concatenates the
results. It does not crop context, alter KV precision, weights or routing.
Coupled sequence rows and incompatible layouts are rejected.
`EXPERIMENT_ATTN_SEQUENCE` confirms actual graph activation.
The measured graph layers the FFN/GDN splits and generic role-diagnostic
scalar projection alternative; its disabled GPU arm agrees with the
previous scalar candidate's long results.

With `VRAMBRIDGE_EXPERIMENT_ATTN_SEQUENCE=1`, the new library
`5209a4aa3f983f1633c11708570d520d5c733d411d7d96c259c3f893543ea8f2`
passes **4480/4480 decode positions** over widths 2/4/8 with zero error.
Use the original prototype GGML/CUDA libraries, not a newly rebuilt mix:
the experiment's frozen libllama-only overlay deliberately retains those
same dependencies. Earlier passing overlays remain untouched.

Real direct-server tests pair it with the prior physical-KV scheduler:
eight 256-token outputs match all serial controls at 2/4/8 clients;
mixed inputs of 4174/2129/72/69 tokens, cancellation/recovery,
arithmetic, AST-only coding and forced tool formatting pass.
Full snapshots of all active sequences at teacher-forced step 280,
three rejected steps and restoration pass 42/42 reference/repeatability
comparisons. This remains a bounded whole-state test, not arbitrary
partial-suffix removal.

Three fresh loads per client level, eight 16K slots, F16 KV, 8/16 threads,
64 excluded warmup tokens and eight 256-output requests give:

| Active clients | Aggregate HTTP tok/s | Serial-matching outputs | Median agent TTFT | Median completion |
|---|---:|---:|---:|---:|
| 1 | 38.98 | 24/24 | 23.60s | 29.37s |
| 2 | 39.40 | 24/24 | 20.50s | 32.32s |
| 4 | 42.09 | 24/24 | 13.85s | 36.06s |
| 8 | 43.97 | 24/24 | 3.48s | 45.82s |

The eight-client aggregate gain is about **12.8%**, with worse individual
completion latency but earlier first output in the same eight-request
burst. It is lower than the old numerically failed eight-client screen;
correctness, not the fastest invalid rate, determines qualification.
These are autoregressive GPU-backbone/CPU-expert results, not diffusion
rates, GPU-hot expert caching or Synapse production concurrency.

The CPU-only 256-output follow-up matches only 4/8 and 3/8 controls at
2/4 clients and is stopped after the negative gate. Its incomplete
report/log are retained. This does not isolate a CPU regression to the
attention patch without a matched CPU off/on control. GPU qualification
must not be generalized to CPU-only operation.
Actual near-capacity input, other platforms, broader family/placement
qualification and production integration remain open. No installed
backend, user settings, profile defaults or Synapse engine pin changes.

## Inverse CPU-backbone / GPU-expert feasibility

The [reviewed inverse-placement screen](../../benchmarks/results/clover-inverse-placement-summary.json)
tests the user's alternative partition without changing engine code or
weights. The unmodified prototype keeps nominal layer devices CUDA so
the native expert cache can target them, but tensor overrides move
non-shared model weights to CPU. Routed weights remain CPU-resident
backing for a 12 GiB GPU cache; shared expert weights are explicitly
CUDA-resident. `--no-kv-offload` places attention and recurrent state on
CPU, and `--no-op-offload` prevents opportunistic host-operation offload.
Some boundary operations can still inherit GPU placement.

The separate placement diagnostic inspects the last live decode graph,
not initialization/reserve graphs. It confirms 120 routed expert matmuls
on CUDA, 341 non-expert/dense matmuls on CPU, 320 MiB CPU KV and 62.81 MiB
CPU recurrent state, 127.5 MiB GPU shared weights and 12285.30 MiB GPU
expert cache. The first two diagnostics were invalid: one used insufficient
debug verbosity and another included reserve graphs. They are retained
privately and are not performance evidence.

With debug logging disabled, one fresh load per arm, a 16K/F16 single
slot, 8/16 threads, 32 excluded warmup tokens and three held-out
128-output requests give:

| Placement | Coding decode | Reasoning decode | Prose decode |
|---|---:|---:|---:|
| Native automatic fit | 63.57 | 63.25 | 63.32 |
| GPU backbone / CPU routed experts | 44.17 | 44.15 | 43.71 |
| CPU backbone/state / GPU expert cache | 12.65 | 12.71 | 12.94 |

All requests complete, but the inverse is about 80% slower than native
fitting in this short screen. CPU dense/recurrent work and additional
device handoffs are plausible costs; phase profiling is needed to
attribute them. This result does not generalize to other architectures,
CPU/GPU combinations or larger concurrent batches, and no numerical or
broad quality contract is established for the inverse placement.
Do not treat it as a qualified model profile or promote it.

## Scheduling and family follow-up

Reviewed evidence:
[`clover-scheduling-family-summary.json`](../../benchmarks/results/clover-scheduling-family-summary.json).
The measurements span November 7 local time / November 8 UTC, 2026.
Earlier failed screens and immutable build identities remain historical
evidence; a later pass never retroactively qualifies another build.

The experimental server isolates one text-causal prefill owner, continuing
that owner before selecting another by FIFO task ID. Decode pauses during
prefill. Logical position buckets, rounded to the pinned 256-cell KV graph
padding, initially passed 64-token short streams at 2/4/8 clients on CPU
and CUDA-backbone/CPU-expert placement. Mixed prompts of
4,174/2,129/72/69 tokens also matched all four serialized controls with
32-token outputs. Cancellation of one stream preserved the follower and
recovery output. Arithmetic, AST-only code and forced tool-format screens
passed; these are bounded checks, not broad quality evaluations.

Longer 256-token streams exposed additional divergence. The later
[`llama-clover-kv-scheduling-experimental.patch`](../../third_party/patches/llama-clover-kv-scheduling-experimental.patch)
forecasts physical KV allocation with the pinned cache's read-only
`find_slot`, includes occupied cells, and selects compatible buckets
round-robin. This improves the eight-client screen but still matches only
5/8 outputs; 2/4-client screens match 8/8 each. It is not production-ready.
All widths 1 through 8 pass short 32-step Clover raw-logit checks, so odd
batch widths alone do not account for this longer-stream failure.

The logical-bucket server, not that later physical-bucket server, was timed
in three fresh runs per client level, eight slots at 16K each, eight
short requests, 256 forced output tokens, F16 KV and 8/16 threads:

| Active clients | Aggregate HTTP tok/s | Outputs matching serial | Median agent latency |
|---|---:|---:|---:|
| 1 | 39.57 | 24/24 | 29.13s |
| 2 | 40.41 | 24/24 | 31.63s |
| 4 | 43.27 | 24/24 | 36.40s |
| 8 | 45.29 | 7/24 | 44.43s |

Rates include prefill, queueing and response overhead; latencies include
queueing from the same eight-request arrival burst. Four-client aggregate
throughput is about 9% higher but individual completion latency is worse.
The eight-client speed is unqualified. Per-load model buffers remain
19,433.31 MiB CPU-mapped and 2,488.81 MiB CUDA; routed experts remain on
CPU, not in an adaptive hybrid cache.

The new [`sequence-rollback-probe.cpp`](../../tools/trace/sequence-rollback-probe.cpp)
captures all active sequences, performs three teacher-forced steps, restores
full per-sequence snapshots, and compares original/reference,
restored/reference and repeatability. Each placement passes all 42
full-vocabulary comparisons with zero error. This does not establish
partial-suffix-removal correctness or arbitrary speculative rollback.

Coder Q4_K_M, the modified SC117 Qwen3.8 IQ3_S two-shard target and
GPT-OSS-120B MXFP4 two-shard target are now fully hashed and available.
Original serialized prefills match 14/14 on each family. Scalar projections
alone pass Qwen3.8 at widths 2/4 but only 128/256 positions at width 8.
The later generic expert-dispatch v2 refinement passes all 448 Qwen3.8
positions with zero error. Coder and GPT-OSS still fail all 448 positions
each, including with that refinement. The retained
[`llama-moe-scalar-dispatch-experimental.patch`](../../third_party/patches/llama-moe-scalar-dispatch-experimental.patch)
is the later role-diagnostic variant, not the exact frozen v2 binary.
It is a standalone graph alternative, not stacked on the earlier Clover
scalar-projection patch; the FFN graph-budget prerequisite remains.

Next correctness investigation should capture the first differing
attention/state tensor under the failed physical-KV schedule and native
transformer batches, not rerun unchanged failing arms. Preserve the raw
logit thresholds (absolute 0.01, relative L2 0.001), all selected experts,
weights and KV precision. Arbitrary chunk-boundary equivalence and broader
occupied-context qualification remain open. No Synapse pin or default
change follows from this checkpoint.

The separate [tandem.cpp study](tandem-cpp-study.md) investigates single-user
expert offload and orchestration alongside this multi-agent work. It has
its own pinned base and accuracy gates; its reported gains are not a fix
for this checkpoint's failed streams.

## What is being studied

[Clover-35B-A3B-Preview](https://huggingface.co/blog/Diffuspace/clover-35b-a3b-preview)
combines an MoE backbone with block-diffusion generation. Its published
configuration describes 40 layers, 256 experts, eight experts per token,
Gated DeltaNet/full-attention layers, a 32-token block, and a 131,072-token
maximum context. The configuration also enables a block coreset; its actual
inference semantics and relation to selected experts must be checked before
making any exactness claims. A context limit is not evidence that the model
fits or works correctly at that length on this machine.

DiffuSpace reports 101.8 output tokens/s on an RTX 5070 Ti in its own setup.
That is a vendor measurement, not a Synapse result or a matched comparison
with our quantization, workload, or execution path.

## Invocation and batching leads

The [official model card](https://huggingface.co/Diffuspace/Clover-35B-A3B-Preview)
and [reference inference code](https://huggingface.co/Diffuspace/Clover-35B-A3B-Preview/blob/4b0402b65e523d0183a14c431955438714cb16ff/modeling_clover_moe.py),
at Hugging Face revision `4b0402b65e523d0183a14c431955438714cb16ff`,
describe two distinct things:

- **Within one request:** low-confidence block diffusion previews up to 32
  positions and iteratively resolves them; the clean recurrent state is not
  mutated during preview, and completed blocks are replayed causally. This is
  different from autoregressive verification of self-speculative proposals.
- **Across requests:** the reference Hugging Face `generate` implementation
  runs batch entries sequentially. A block of positions is not proof of
  concurrent-request batching or a 32-times speedup.

The reference configuration enables a block-level expert coreset at routing
mass 0.9 during denoising. It selects a subset from the union of per-token
top-8 candidates; that can alter selected experts, so it must not be presented
as an exact generic batching optimization. Compare the published reference
semantics separately from full per-token routing and resolve compatibility
with Synapse's no-pruned-selected-experts rule before considering rollout.
Do not silently remove, enable or alter this behavior.

[HyperQwen](https://github.com/syv-ai/HyperQwen/tree/6ab9bd05cb84b538a6595d634f3e337ac8a80856),
inspected at commit `6ab9bd05cb84b538a6595d634f3e337ac8a80856`, offers useful
architecture-aware research leads, not a drop-in backend for this GGUF:

- Its headline aggregate rate is measured across 64 concurrent requests on
  a 24 GiB RTX 3090; that is not one user's decode speed and is not comparable
  with our single-request Clover baseline. Its single-user and batch modes
  make different tradeoffs.
- Its [small-batch GDN patch](https://github.com/syv-ai/HyperQwen/blob/6ab9bd05cb84b538a6595d634f3e337ac8a80856/patches/gdn-ba-pad-small-m.patch)
  reports a cuBLAS kernel-selection cliff for BF16 projection rows 2–4 on
  that 3090/software combination. Padding to five rows improved its measured
  engine steps by approximately 4–5%, while changing projection values by up
  to 0.03 due to BF16 rounding. Those are the authors' results, not a measured
  Clover or RTX 5070 Ti result, and the reported difference is not a
  full-model numerical gate.
- Clover's GGUF has separate BF16 `ssm_alpha` and `ssm_beta` weights of
  `2048 x 32`; this prototype invokes them separately. HyperQwen's tested
  fused projection is a different `5120 x 96` shape on sm86/vLLM, so neither
  its kernel cliff nor its proposed padding transfers by assumption. Check
  actual execution, backend, shapes and precision before measuring an
  equivalent lead on sm120/llama.cpp. Retain the original FP32 recurrent state.
- HyperQwen also uses model-specific drafts, requantization and lower-precision
  cache/activation options. Do not copy those or change target weights,
  precision, selected experts or user settings as a shortcut. Any numerical
  change must independently pass the existing applicable accuracy gates.

The follow-up should separately measure (1) shared-expert multirow work within
one diffusion/verification block, (2) prompt prefill batching and (3)
multi-request serving. Capture actual rows per expert, layer and phase,
router/transfer costs, GPU and CPU time, memory, accepted tokens, and
single-request latency as well as aggregate throughput. Use fixed model
bytes, semantics and placement, and compare scalar versus batched outputs
and recurrent rollback; confirm gains in real requests rather than promoting
an isolated GEMM win. Prior research found an exact packed CPU kernel about
2.1x faster in isolation but 9–12% slower in live generation on GPT; that
does not rule out Clover batching, but is a reason to measure end to end.

## Multi-agent serving protocol

Synapse currently admits one active chat and queues other requests; this
study measures an isolated server, not a change to that application behavior.
Use eight distinct synthetic agent requests in
[`agent-concurrency-corpus.jsonl`](../../benchmarks/agent-concurrency-corpus.jsonl).
Give every arm the same eight backend slots and 16,384-token capacity per
slot, and vary the maximum simultaneous clients across 1, 2, 4 and 8.
This keeps allocated context capacity fixed across those arms; actual
per-load automatic placement must still be checked.

All eight requests arrive at the client harness together; requests beyond
the current active-client limit wait in its queue. Each fresh load excludes
one 64-token warmup, then forces 256 output tokens per request. Use greedy
seed 42, prompt reuse off, 8 generation / 16 prompt threads, F16 KV,
continuous batching, no speculation, and three fresh runs in alternating
forward/reverse/forward order. Throughput is total generated tokens divided
by the whole cohort wall time, including prefill and queueing. Per-agent
time-to-first-text and latency include harness queue time. Record actual
slot context, GPU/CPU model allocation, sampled GPU use, output hashes and
any failed or incomplete streams.

This tests a fixed eight-request burst and one unqualified prototype backend;
it does not establish Synapse's current concurrency, natural-answer quality,
full-context behavior, the hybrid engine's concurrency support, or results
for all agent workloads. Short-lived competing GPU work can escape one-second
monitoring. A separate one-slot control quantifies the cost of reserving
eight slots for this workload.

### Measured throughput: promising, not qualified

The [reviewed aggregate](../../benchmarks/results/clover-agent-concurrency-summary.json)
records all three runs per level. The direct server used the **unmodified**
`cb09d4355d8bf968bb907c5de03cf42d8aa073ce` prototype and the verified
official Q4_K_M model, in autoregressive mode. All 96 requests completed at
the requested length, and the backend logs confirmed the requested peak
active slot count for each trial. The actual model buffer totals matched
across the fixed-eight-slot trials: 10,241.61 MiB CUDA and 11,680.51 MiB
CPU-mapped. Individual tensor placement was not retained.

| Concurrent clients | Aggregate output tok/s | Median agent completion | Median active-agent decode tok/s |
|---|---:|---:|---:|
| 1 | 53.99 | 21.34 s | 59.69 |
| 2 | 68.03 | 18.72 s | 37.02 |
| 4 | 86.55 | 17.59 s | 23.24 |
| 8 | 105.63 | 19.39 s | 14.43 |

These are medians over three fresh cohorts. Completion time includes the
client-side wait for all eight arrivals; active-agent decode rate excludes
that wait. The aggregate rate includes prompt processing and is not the
speed of any one agent. Eight active clients nearly double aggregate output
versus one active client with the **same eight slots reserved**, while four
give the best median completion time for this particular burst.

After hardening both runners to create private artifacts, an independent
repeat gave 53.45/67.68/85.74/104.07 aggregate tok/s for 1/2/4/8 clients.
Its automatic fit alternated between two GPU model-buffer sizes differing by
139.86 MiB, so those comparisons do not have the primary cohort's matching
model-buffer totals. Both complete cohorts are retained. The historical
runner hashes differ from the current scripts only by the privacy hardening;
no measurement logic changed.

A separate, otherwise matched single-slot control gives 61.19 aggregate
tok/s. Reserving eight slots reduces its one-active-client rate to 53.99
tok/s, about 12% lower. In the single-slot control, the CUDA model buffer
was 12,965.61 MiB rather than 10,241.61 MiB: fitting the extra context
capacity displaced model data from GPU to CPU. Do not equate reported
buffer totals with exact individual expert placement or make eight slots
the unmeasured universal default.

The throughput cohort used streamed chat requests. For every concurrency
level 2, 4 and 8, **0 of 24** complete output hashes matched their
same-repetition one-active-client controls; the one-active-client outputs
were repeatable for all eight prompts. Hash differences do not establish
which answer, if any, is incorrect.

### Numerical gate: failed

The separate
[`vrambridge-agent-numerics.py`](../../tools/benchmark/vrambridge-agent-numerics.py)
diagnostic compares one raw-completion output position across the full
248,320-token pre-sampling **log-probability** vector. It uses the same
eight prompts and one 8-slot server, with prompt reuse off. Its sequential
repeat passes all eight checks with zero observed difference. Its GPU
concurrent comparisons use the existing diagnostic thresholds:
maximum absolute log-probability difference <= 0.01 and relative L2 <= 0.001.

| Active clients | Requests passing both thresholds | First token matches sequential |
|---|---:|---:|
| 2 | 6 / 8 | 8 / 8 |
| 4 | 1 / 8 | 7 / 8 |
| 8 | 0 / 8 | 7 / 8 |

In the primary GPU diagnostic the largest absolute difference is 2.65 and
largest relative L2 is 0.0363, across the three levels. A separate CPU-only 8-client
diagnostic also fails all eight comparisons (maximum absolute 2.87;
maximum relative L2 0.0401); its sequential repeats pass. Both diagnostics
confirmed eight slots at 16K each, F16 KV and the requested peak active
slots. An active slot is not proof that every request was evaluated in the
same graph batch.

The independent privacy-hardened repeat again fails GPU comparisons at all
three concurrency levels (0/8, 1/8 and 0/8 pass at 2/4/8) and the CPU-only
8-client comparison (0/8 pass). Its sequential repeats all pass. The
particular requests passing at a given level can depend on scheduling; do
not interpret a partial pass as a safe concurrency setting.

These are normalized log probabilities, **not raw logits**, and one raw
prompt position is not a broad quality assessment. The cause of the
difference is unresolved; output-hash and numerical differences do not
prove incorrect answers or identify cross-request leakage. Nevertheless,
they block claiming that concurrent execution preserves the reference
behavior. Investigate batch shape, router selection, recurrent-state
isolation and placement against raw-logit and state/rollback gates before
considering any Synapse concurrency change.

### Synapse integration checkpoint

The [integration summary](../../benchmarks/results/clover-synapse-integration-summary.json)
records a real-model smoke through an isolated Synapse instance at commit
`4f6fc050364ba989a8f14edbc6d3cf1b726b70d1`, using the separate unmodified
Clover prototype and a temporary XDG configuration. Two clients sent
streaming requests at the same time: one waited in Synapse's existing
queue, and both received 128 output tokens. The backend confirmed one slot
and 16K context. Six existing fake-backend tests also exercised queue
ordering, capacity, cancellation, unloading and streaming.

This confirms the **existing serialized integration**, not simultaneous
model execution or output quality. It did not modify Synapse, the installed
backend, profile or saved state. The real-model smoke used a one-off private
session harness; its source hash is retained in the summary.

A separate one-position diagnostic requested `--ubatch-size 1` on the same
prototype; backend output confirmed `n_ubatch = 1`. Sequential repeats
passed all eight full-vocabulary log-probability checks, but all eight
comparisons failed at both 2 and 4 active clients. First sampled tokens
matched in this diagnostic. Thus limiting physical microbatch size alone
did not resolve the observed discrepancy. This single diagnostic does not
identify the cause or establish whether answers are wrong; it also used a
private one-off harness whose hash is recorded in the summary.

Keep Synapse's serialized path until the research backend passes raw-logit,
recurrent-state and cross-family qualification. The temporary Clover
prototype must not be installed as a production backend.

### Raw-logit localization and a bounded CPU lead

The [sequence-correctness aggregate](../../benchmarks/results/clover-sequence-correctness-summary.json)
traces the discrepancy beyond the HTTP server. With one request active at a
time, the same prompt produces matching first-token log probabilities in
each of the eight explicitly selected slots on CPU and GPU. That rules out a
simple static slot-number explanation for this probe; it does not prove
concurrent state isolation.

A separate CPU-only direct-`llama_decode` probe compares the entire
248,320-entry **raw-logit** vector for a 13-token prompt. Repeating the
prompt in 2, 4 and 8 sequence-major or 8 interleaved sequences fails the
unchanged 0.01 maximum-absolute / 0.001 relative-L2 gate in every comparison.
The single-sequence repeat is exact. Observer-enabled and observer-disabled
controls match exactly for the captured single- and two-sequence cases.

For this one prompt, the first differing active layer output is layer 7.
Router logits, selected expert IDs, gate/up projections and activation agree.
The Q6_K expert down projection differs by at most about `3e-8`; after a
BF16 round, the routed output differs by about `6.1e-5`, and later layers
amplify the difference. The prototype's CPU tiled expert dispatch switches
at eight routed rows. Experiments preserving each sequence's original
dispatch class pass this bounded batch check, strongly implicating
batch-dependent arithmetic for this case. Captured inactive graph branches
are not used to claim a divergence.

Two default-off research patches split the expert FFN and the single-token
GDN alpha/beta projections by sequence. They do not change model weights or
drop selected experts; unsupported ownership/layouts reject explicitly.
Apply the
[FFN patch](../../third_party/patches/llama-clover-sequence-ffn-experimental.patch)
and then the
[GDN patch](../../third_party/patches/llama-clover-sequence-gdn-experimental.patch)
to the pinned `cb09d4355d8bf968bb907c5de03cf42d8aa073ce` prototype only.
Both require explicit experimental environment flags and are **not**
production patches.

After serially prefilling eight different short prompts, a direct CPU test
uses the original single-sequence model's tokens to teacher-force 32 decode
steps. Across 2, 4 and 8 concurrent sequences, the unchanged prototype
passes **0 / 448** full-vocabulary raw-logit checks; the two-patch candidate
passes **448 / 448**, with maximum absolute error `6.68e-6` and maximum
relative L2 `2.08e-7`. All 14 separate prefill comparisons are exact. These
are bounded numerical results, not a throughput or natural-answer claim.
The FFN-only ablation passes just 41 / 448 decode checks. In a separate
observer-controlled eight-sequence trace, the first differing selected active
attention output occurs at layer 34, after small alpha/beta projection
differences; splitting those projections is necessary for this bounded CPU
candidate but does not establish a general solution.

The same candidate is **not qualified on the RTX 5070 Ti**. With the
backbone and KV on CUDA and routed experts on CPU, it fails all 448 decode
comparisons: maximum absolute error is 0.0318 at two and four sequences and
15.67 at eight. A GPU trace with observer controls matching exactly shows
small differences in full-attention projections before later BF16 rounding
amplifies them. Separately splitting attention output and QKV projections
also fails all tested GPU comparisons; that negative ablation is preserved
in the [attention patch](../../third_party/patches/llama-clover-sequence-attention-experimental.patch).

Chunked prefill remains a separate blocker even for one CPU sequence:
splits after tokens 1, 6 and 12 all fail against the original unchunked
reference. Disabling all CPU tiled matmuls makes some batch comparisons
internally consistent but changes the original single-sequence reference,
so it is not an acceptable fix. In the actual server's diverse
concurrent-prefill log-probability screen, the combined two-patch CPU
candidate passes only 5/8, 4/8 and 1/8 requests at 2/4/8 active clients.
FFN markers were observed; GDN markers may come from graph reservation and
do not establish an active GDN decode branch in this one-output test. No
long-input, GPU, rollback, natural-answer, performance or cross-family gate
has passed.

The reusable CPU raw-logit probes are
[`sequence-logit-probe.cpp`](../../tools/trace/sequence-logit-probe.cpp)
and [`sequence-decode-probe.cpp`](../../tools/trace/sequence-decode-probe.cpp).
Their private reference files contain synthetic token IDs and logits. Compile
against the exact prototype headers and libraries, check the model checksum,
and protect references with `umask 077`. Both accept `--self-test`; their
`write` and `read` modes return status 2 when any numerical gate fails.
The decode probe's optional `gpu-cpu-experts` mode pins all routed experts
to CPU and explicitly requires a CUDA device; use separate CPU and GPU
reference files. No tracing result is a speed measurement.

For example, after building the unmodified prototype and the two-patch
candidate separately, set `LLAMA_SRC`, `CANDIDATE`, `MODEL` and `PRIVATE` to
local paths. Both builds must use the same pinned source, compiler and
matching GGML libraries; verify the model's full SHA-256 first.

```sh
umask 077
mkdir -p "$PRIVATE/overlay"
ln -s "$CANDIDATE/build/bin/libllama.so.0" "$PRIVATE/overlay/libllama.so.0"
g++ -O2 -std=c++17 -I"$LLAMA_SRC/include" -I"$LLAMA_SRC/ggml/include" \
  tools/trace/sequence-decode-probe.cpp -L"$LLAMA_SRC/build/bin" \
  -Wl,-rpath,"$LLAMA_SRC/build/bin" -lllama -lggml -lggml-base -o "$PRIVATE/sequence-decode-probe"
"$PRIVATE/sequence-decode-probe" --self-test
CUDA_VISIBLE_DEVICES= GGML_CPU_TILED_MM=1 "$PRIVATE/sequence-decode-probe" \
  "$MODEL" write "$PRIVATE/cpu-reference.bin"
CUDA_VISIBLE_DEVICES= GGML_CPU_TILED_MM=1 VRAMBRIDGE_EXPERIMENT_FFN_SPLIT=1 \
  VRAMBRIDGE_EXPERIMENT_GDN_SPLIT=1 LD_LIBRARY_PATH="$PRIVATE/overlay:$LLAMA_SRC/build/bin" \
  "$PRIVATE/sequence-decode-probe" "$MODEL" read "$PRIVATE/cpu-reference.bin"
```

The original multi-sequence comparison is expected to return status 2; its
reference file is still written. Clear unrelated backend/environment overrides
and check the FFN/GDN activation markers. To reproduce the GPU/CPU-expert
study, append `gpu-cpu-experts` to **both** invocations, omit
`CUDA_VISIBLE_DEVICES=`, and use a separate reference file. It must not be
compared against the CPU reference.

### Reproduction

Both runners use private loopback servers, temporary keys, clean execution
environments and owned-process cleanup. Replace paths with a separately built
prototype and the verified model; raw reports and logs remain ignored.

```sh
python3 tools/benchmark/vrambridge-agent-concurrency.py \
  --backend /path/to/prototype/llama-server --model /models/Clover-Q4_K_M.gguf \
  --output benchmarks/results/clover-agents-private.json \
  --backend-slots 8 --concurrency 1 2 4 8 --requests 8 --tokens 256 --repetitions 3

python3 tools/benchmark/vrambridge-agent-numerics.py \
  --backend /path/to/prototype/llama-server --model /models/Clover-Q4_K_M.gguf \
  --output benchmarks/results/clover-agent-probabilities-private.json \
  --vocabulary 248320 --levels 2 4 8
```

For the separate capacity control, repeat the throughput runner with
`--backend-slots 1 --concurrency 1`. For the CPU numerical check use
`--placement cpu --levels 8`. A completed numerical diagnostic exits with
status 2 when the comparison gate fails; failures are not converted into
successful measurements.

## Candidate speed and GPU follow-up

The [reviewed speed summary](../../benchmarks/results/clover-candidate-speed-summary.json)
uses separately frozen libraries to measure the CPU candidate and a new,
default-off GPU projection experiment. It adds the
[scalar-projection patch](../../third_party/patches/llama-clover-scalar-projections-experimental.patch)
after the FFN and GDN patches. Unlike the earlier attention-only ablations,
it covers all supported `build_lora_mm` projections in independent,
single-token, all-output multi-sequence decode graphs. Unsupported adapters,
layouts and budget configurations reject explicitly. Precision, target
weights and selected experts remain unchanged.

For this bounded GPU/CPU-expert test, the projection experiment preserves
the original single-row projection arithmetic while sharing one model and
context across requests. It passes **448 / 448 full-vocabulary raw-logit
comparisons with zero observed error** across 32 teacher-forced decode steps
at 2, 4 and 8 sequences, plus 14 exact serialized-prefill comparisons.
When disabled, all 463 metric records match the earlier failing GPU
candidate. This isolates a useful projection-shape lead; it does not prove
a general GPU fix or equivalent execution across all workloads.

### Measured speed

Three fresh runs per arm alternate original serial, original batched and
corrected batched order. All use eight short synthetic prompts (7–10 input
tokens), 16K capacity per sequence, eight sequence slots, F16 KV, 8/16
threads, an excluded prefill warmup and 64 forced greedy output tokens per
request. Prefill is **serialized** in every arm. Decode rate counts 63
evaluated generation tokens per request; whole-cohort rate also includes
prefill and memory clearing. The GPU arms put the backbone and KV on CUDA
and all routed experts on CPU, not in Synapse's adaptive expert cache.

| Placement | Original serial decode | Original batch decode, unqualified | Corrected batch decode | Corrected whole-cohort rate |
|---|---:|---:|---:|---:|
| CPU | 11.52 tok/s | 28.30 tok/s | 26.76 tok/s | 24.96 tok/s |
| GPU backbone / CPU experts | 43.68 tok/s | 75.21 tok/s | 54.77 tok/s | 50.67 tok/s |

All rates are **aggregate across requests**, not per-agent streaming rates.
Corrected batching improves aggregate decode by about 132% on CPU and 25%
on GPU versus serial requests in this workload. It costs about 5% on CPU
and 27% on GPU versus the faster but numerically unqualified batch path.
Every corrected 64-token output sequence matches its same-repetition
serial control (24/24 on each placement); the original batch path matches
only 6/24 on each placement. These short results do not establish natural
answer quality or a production throughput target.

An earlier exploratory speed cohort did not bind an immutable candidate
library and is excluded from these results. The repeated measurements use
separate copied library files and retain their hashes, avoiding rebuilds
changing the library loaded by later trials.

### Limits and reproduction

Concurrent/chunked prefill, long occupied context, rollback, cancellation
and the required other MoE families remain unqualified. The experiment
keeps scalar-equivalent projections at the expense of some batching
efficiency; profiling that cost is future work, not a claim that another
optimization will recover it.

[`sequence-speed-probe.cpp`](../../tools/trace/sequence-speed-probe.cpp)
uses the same prototype headers/libraries as the raw-logit probes. It runs
eight requests in waves of the requested size and writes synthetic token
IDs, so keep its output private. For the same prepared paths as above:

```sh
g++ -O2 -std=c++17 -I"$LLAMA_SRC/include" -I"$LLAMA_SRC/ggml/include" \
  tools/trace/sequence-speed-probe.cpp -L"$LLAMA_SRC/build/bin" \
  -Wl,-rpath,"$LLAMA_SRC/build/bin" -lllama -lggml -lggml-base -o "$PRIVATE/sequence-speed-probe"
"$PRIVATE/sequence-speed-probe" "$MODEL" 1 64
"$PRIVATE/sequence-speed-probe" "$MODEL" 8 64
VRAMBRIDGE_EXPERIMENT_FFN_SPLIT=1 VRAMBRIDGE_EXPERIMENT_GDN_SPLIT=1 \
  LD_LIBRARY_PATH="$PRIVATE/overlay:$LLAMA_SRC/build/bin" \
  "$PRIVATE/sequence-speed-probe" "$MODEL" 8 64
```

For GPU tests, build a separate candidate with the scalar-projection patch,
use its own immutable library overlay, set
`VRAMBRIDGE_EXPERIMENT_SCALAR_PROJECTIONS=1` only for that candidate, and
append `gpu-cpu-experts`. Use that same candidate and placement with
`sequence-decode-probe` against a separate original GPU reference before
interpreting speed. Keep unrelated GPU compute idle; the executed cohort
monitored only its own GPU process. No Synapse or installed backend change
is made.

## Upstream and asset identity

- The [official GGUF model card](https://huggingface.co/Diffuspace/Clover-35B-A3B-Preview-GGUF)
  advertises `draft-self` through llama.cpp PR
  [#31306](https://github.com/ggml-org/llama.cpp/pull/31306).
  However, the merged PR explicitly states **autoregressive only**;
  its merge commit is `e4b6c8ca6a6e8da70248da30709e897e7cc8615e`.
- The separate open
  [self-speculation PR #31732](https://github.com/ggml-org/llama.cpp/pull/31732)
  says Clover requires further implementation changes. It reports Nemotron,
  not Clover, measurements. Do not equate this with a working Clover
  diffusion implementation. Resolve the documentation discrepancy against
  actual code and activation markers before measuring the diffusion path.
- An [earlier PR comment](https://github.com/ggml-org/llama.cpp/pull/31306#issuecomment-6541743464)
  reports a Clover `draft-self` experiment on an older, subsequently rewritten
  PR revision: 13.47 versus 8.10 tok/s on a GB10, with prompt caching observed.
  That is an unmatched, two-run external screen, not a local result or a
  general conclusion. High reported acceptance did not guarantee a speedup.
- [SGLang PR #46927](https://github.com/sgl-project/sglang/pull/46927) describes
  native Clover serving and self-speculation; it was open at inspection, with
  head `2243dc6edb416073b60be0765904cabae9eb61cf`. Its support and hardware
  requirements remain to be independently verified.
- Synapse's pinned backend is llama.cpp b11669
  (`620d2225cc436ab0b001e3200e590e3b286ef9ab`), which predates the merged
  Clover support. The canonical hybrid patch targets that older revision;
  no port to the Clover revision or compatibility claim has been made.
- The initial model is official `Clover-35B-A3B-Preview-Q4_K_M.gguf` at
  Hugging Face revision `32e22a02b885e173177b7ce7661ad13448b4f89e`.
  Published size is 22,998,009,248 bytes and the LFS SHA-256 is
  `4d439bb832487fbb91d0a739a5746cf3089bde4137c17050576b24e6dba3539c`.
  The completed local file matches both the size and the full SHA-256.

## Initial local smoke: explicit failure

On the Ryzen 9950X / RTX 5070 Ti host, an isolated CUDA build of merged
upstream commit `e4b6c8ca6a6e8da70248da30709e897e7cc8615e` was exercised
through a private Synapse runtime and the existing owned-baseline runner.
The installed app and its backend were not modified. The test requested
Standard execution, 16K context, 8/16 threads and F16 KV; no generation
started. The backend rejected the verified official GGUF with:

```text
error loading model hyperparameters: key not found in model: clover_moe.expert_shared_count
```

The published model configuration says there is one shared expert. A separate
CPU-only diagnostic supplied that value as a **read-time metadata override**
without changing the file. It still failed before loading weights:

```text
done_getting_tensors: wrong number of tensors; expected 853, got 733
```

In this loader the first number is the tensor count in the file and the second
is the count claimed by the model graph. The 120 unclaimed tensors have not
yet been individually qualified; suppressing the error or dropping them
would not establish a correct conversion.

Consequently that merged-backend attempt has no placement, context,
throughput, or answer-quality result. Its failed private report is retained
under ignored benchmark results; it is not a successful measurement. The
mismatch is between this particular official GGUF revision and this
particular merged backend, not evidence that all Clover GGUFs or other
revisions are incompatible.

The merged build lists no `draft-self` flag. The earlier prototype revision
`cb09d4355d8bf968bb907c5de03cf42d8aa073ce` is a separate experimental
candidate for compatibility investigation, not a production backend or proof
that self-speculation is faster or correct. Do not edit the downloaded model
in place to hide the incompatibility.

## Prototype autoregressive baseline

The verified GGUF loads on prototype commit
`cb09d4355d8bf968bb907c5de03cf42d8aa073ce`. A separate CUDA build served it
through the existing owned Synapse baseline runner, with temporary config and
no installed profile or backend changes. The
[sanitized summary](../../benchmarks/results/clover-initial-summary.json)
records binary and model hashes and the summarized measurements.

Three fresh Standard loads used 16,384 requested context, F16 KV, 8/16
generation/prompt threads, auto-fit with a 1,024 MiB margin, and an excluded
64-token warmup. Each held-out request forced 512 output tokens with greedy
sampling, seed 42, prompt cache off and low reasoning effort. Input lengths
were 104 coding, 125 reasoning and 102 prose tokens.

| Workload | Median wall decode tok/s | Median end-to-end tok/s |
|---|---:|---:|
| Coding | 65.52 | 62.21 |
| Reasoning | 65.49 | 62.06 |
| Prose | 65.92 | 62.66 |

Wall decode rate is `(output tokens - 1) / (request wall seconds - reported
prompt seconds)`. Forced output can continue after EOS. A separate diagnostic
with the same requested fit settings confirmed 16,384 slot context, CUDA
model allocation of 12,825.75 MiB and CPU-mapped tensors; the fit log also
reports 41/41 layers offloaded, which does **not** mean all weights are on
the GPU. The timed runner did not retain per-load placement logs, so identical
tensor placement across the three loads is unverified.

The two small arithmetic and AST-only coding checks passed on the first load.
All three workload output hashes varied across fresh loads despite greedy
sampling. This is an unresolved reproducibility finding, not proof of an
incorrect answer; neither the checks nor the hashes establish full-logit or
broad answer-quality equivalence. The short input does not establish
near-capacity behavior.

The benchmark can be repeated after preparing the same source revision,
CUDA build and verified GGUF, with local paths substituted:

```sh
python3 tools/benchmark/vrambridge-synapse-baseline.py \
  --synapse-source /path/to/Synapse --backend /path/to/prototype/llama-server \
  --profile /path/to/read-only/hybrid-profile.json --model /models/Clover-Q4_K_M.gguf \
  --contexts 16384 --variants standard --tokens 512 --repetitions 3 \
  --split test --answer-checks --output benchmarks/results/clover-private.json
```

The runner owns and stops its own app/backend processes; its output remains
private and ignored until explicitly reviewed.

## Initial self-speculation screens

For the same prototype and model, a separate direct loopback server was
started with `--spec-type draft-self --spec-draft-n-max 2` and the same
16K/8/16 settings. With auto-fit mixed placement the server exited with
SIGSEGV (`-11`) during the first 64-token request; the stream ended without
`[DONE]` or usage and had reported only two predicted tokens. A second
diagnostic with `--n-gpu-layers all --cpu-moe --fit off` also exited with
SIGSEGV during its first request. No draft counter was emitted before either
failure, so activation and performance are unverified for those attempts.
The cause has not been isolated.

With `--n-gpu-layers 0 --device none --fit off`, a one-load-per-mode
CPU-only comparison completed: excluded 64-token warmup, then the same three
held-out prompts with 128 forced tokens, otherwise the same sampling settings.
The self-speculative responses reported positive draft and acceptance counts.
These are single, fixed-order pairs—not repeated results or GPU comparisons.

| Workload | AR tok/s | Self tok/s | Accepted / proposed | Output hashes equal |
|---|---:|---:|---:|---|
| Coding | 11.40 | 10.51 | 77 / 86 | Yes |
| Reasoning | 10.62 | 11.03 | 83 / 84 | Yes |
| Prose | 11.12 | 9.96 | 73 / 85 | No |

These mixed changes and one differing hash do not establish an accuracy
failure or a speed win. The CPU test used the prototype's causally verified
`draft-self`, not pure block-diffusion denoising. Investigate GPU crash and
reproducibility before any promotion or architecture change.

## Router-transfer diagnosis and unqualified experiment

A repeated GPU crash was inspected with `cuda-gdb`. To obtain source
locations, a separate `-O3 -g` build supplied the CPU and base libraries
while the remaining libraries came from the unmodified prototype build.
In the observed 3-row preview at layer 22, the CPU backend executed a
`GGML_OP_CPY` from pinned-host coreset logits into a CUDA-resident
`clover_router_masked` destination and faulted during the CPU copy. This
identifies an observed invalid host/device transfer in that experiment, not a
proof that all other crashes have the same cause.

A separate
[experimental patch](../../third_party/patches/llama-clover-router-transfer-experimental.patch)
against prototype commit `cb09d4355d8bf968bb907c5de03cf42d8aa073ce`
returns the CPU-masked tensor to its consumers rather than explicitly
copying it into the device-backed router tensor. The scheduler then handles
consumer transfers. Neither the original prototype nor the installed Synapse
backend was changed. The patch is research-only: it is **not validated as
correct** and is not the canonical b11669 hybrid patch.

The [sanitized candidate summary](../../benchmarks/results/clover-router-experiment-summary.json)
records three fresh direct-server pairs on this candidate: AR/self,
self/AR, AR/self. Both use auto-fit, 16K requested context, 8/16 threads,
the same verified GGUF and held-out prompts, greedy seed 42, prompt cache
off, a 64-token warmup, and 128 forced output tokens. Each self request
reported nonzero proposed and accepted draft counts; all completed.

| Workload | Median AR tok/s | Median self tok/s | Accepted / proposed | Paired text hashes |
|---|---:|---:|---|---|
| Coding | 67.09 | 35.27 | 77 / 86 | Match in 3/3 |
| Reasoning | 67.76 | 37.14 | 83 / 84 | Differ in 3/3 |
| Prose | 67.02 | 34.28 | 71 / 86 | Differ in 3/3 |

In this short screen self-speculation is about 45–49% slower despite high
acceptance. Output hashes repeat within each mode but differ between modes
for two workloads. Those differences are not by themselves a diagnosis of
bad answers, but they prevent treating the candidate as equivalent. A
separate original-prototype AR control matched candidate AR hashes on two of
the three prompts; previous fresh-load AR tests also showed hash variation.
Per-load auto-fit tensor placement was not retained, so exact placement
matching is unverified.

No full-vocabulary logit comparison, recurrent-state/rollback gate, natural
answer quality, filled-context test, or cross-family qualification has been
completed. The current result warrants correctness investigation and actual
phase profiling, not promotion or a claim that batching is ineffective.

## Research gates

1. **Asset and functional control.** Build an isolated CUDA backend at the
   merged upstream commit; verify the model hash, startup, actual GPU placement,
   chat template, context, prompt/output counts, and a bounded natural-answer
   smoke test. Keep raw logs and machine paths private. Failed startup or
   unexpected fallback is a result, not a successful baseline.
2. **Autoregressive baseline.** Fix quantization, backend build, 16K context,
   F16 KV, 8 generation / 16 prompt threads, greedy sampling, one request,
   and the held-out coding/reasoning/prose corpus. Record loading, prefill,
   first-token, decode, total wall time, token counts, placement, GPU/RAM use,
   and generation reads. Use repeated fresh loads before making performance
   claims. Allocated context is not occupied context.
3. **Native diffusion investigation.** Identify an actual working implementation
   and whether it is block diffusion or causally verified self-speculation.
   Record model bytes, block size, mask/refinement schedule, sampling,
   acceptance/verification counters, generated versus proposed tokens, and
   activation evidence. Pure diffusion and causal verification are distinct
   semantic paths; neither is automatically equivalent to the other.
4. **Matched comparison and attribution.** On identical assets, prompts,
   context and placement where supported, measure accepted output and
   end-to-end latency—not proposal throughput. Compare natural answers,
   structured outputs and appropriate numerical/state checks; document
   any changed inference semantics. Profile MoE routing, block occupancy,
   host execution, transfers and synchronization before proposing a new
   architecture. Retain slower and failed attempts.
5. **Generalization and handoff.** Any shared engine technique must preserve
   selected experts and model weights, use metadata/capabilities rather than
   model names, and pass applicable numerical and held-out gates on Clover
   and the project's required Qwen3-Coder, Qwen3.8 and GPT-OSS families.
   Inapplicable model capabilities must be explicitly identified, not
   fabricated. Only measured and qualified behavior goes to Synapse's
   detect → Auto Tune → remember flow.

Use temporary runtime configuration and separately built binaries. Do not
change the installed backend, saved choices, model weights, or the user's
other research branch. No new default, patch port, or production feature is
authorized by this initial protocol.
