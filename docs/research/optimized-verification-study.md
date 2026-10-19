# Optimized-preserving verification and selective KV staging

Date: 2026-10-19. **Numerical advancement; no qualified throughput gain.
GPT-OSS-120B has not reached sustained 60 accepted-output tok/s.**

Research workspace: https://github.com/appatalks/unified-memory-scheduler.
Read-only downstream consumer: https://github.com/appatalks/Synapse.
The [reviewed aggregate](../../benchmarks/results/optimized-verification-summary.json)
contains sanitized results. Raw prompts, token IDs, logits, addresses, logs and
source-identity manifests remain in the ignored private experiment directory.
No installed backend, original model, Strata installation or Synapse file was
changed. Publishing this research packet does not update downstream installer
pins or authorize a performance rollout.

## Frozen target and interpretation

The target is the original two-shard GPT-OSS-120B MXFP4 model, not its draft or
the earlier Qwen architecture comparison. Target settings remain 16,384
context, F16 K/V, eight generation/sixteen prefill threads, physical prefill
batch 512, all selected experts, and the training-only frozen 944-expert plan.
The requested expert budget is 12,000 MiB; the native allocation is 11,995.16
MiB with 948 bank slots including reserve. Adaptation and decode weight
migration remain off. The backend pin is
`620d2225cc436ab0b001e3200e590e3b286ef9ab`.

The host is a Ryzen 9 9950X with approximately 123.48 GiB RAM and an RTX 5070
Ti. Existing root-SSD weights are used; no downloads, archive remounts, page
cache drops, RAM-disk staging or automatic eviction were performed. The root
SSD had approximately 93 GiB available, above the user's 16 GiB disk reserve.
The separate prototype host-RAM headroom guard is also 16 GiB; it is not disk
space or additional VRAM.

The common generation rate is `(accepted outputs - 1) / elapsed time after
the first output`. Draft proposals and aggregate verification rows are not
accepted output. This phase's live results are **training/calibration screens**,
not three-pair held-out 512-output confirmation cohorts. Older 51-58 tok/s GPT
screens used different transfer/adaptation/rendering conditions and must not
be pooled with these rates.

## 1. Removed the optimized-verification numerical blocker

The older global scalar-attention control disables Flash Attention even for
single-token decoding. Its canonical contract still differs from optimized
scalar; the previous failing result is not withdrawn.

The new default-off [window interposer](../../tools/trace/attention_window.cpp)
instead delegates each verification query to the **original single-query
Flash Attention builder**, with exact query/mask views. Floating dense/router
and applicable device expert products use original per-column builders only
inside bounded multi-token, all-output graphs. Single-output prefill keeps
its original arithmetic and placement.

GPT and Qwen widths two/four pass against their original optimized references,
including the tested long-input cases. Width-three short-input checks also
pass. Observed maximum absolute and relative-L2 logit errors are both zero.
The unchanged acceptance thresholds are absolute <= 0.01 and relative L2
<= 0.001; they were not widened.

This is a research graph-role classifier, not a production inference-phase
API. Prompt-logprob/all-output-prefill and other ambiguous call patterns need
an explicit verification role before integration. The interposer is pinned to
private exported interfaces, not a portable installation mechanism.

### Actual rejected-suffix rollback

The [logit checker](../../tools/trace/logit_check.cpp) now accepts
`--rollback-check`. It submits correct prefixes plus deliberately wrong
speculative suffixes, cycles retained lengths, synchronizes, removes rejected
KV positions, checks the retained position maximum and compares the surviving
full-vocabulary logits to optimized scalar.

The initial GPT rollback matrix covers 1,056 positions and 337 rejected-suffix
windows with zero error. It includes widths two/three/four and a long KV
boundary case. Qwen's tested single-sequence tail-removal-only path reports
unsupported. This does not prove Qwen speculation impossible: recurrent-state
snapshot/copy rollback needs its own qualification.

### Live prefill placement was a separate blocker

The live server leaves a final four-token prompt chunk. Raising the hybrid
verification capacity from one to four also moved this prefill chunk onto
resident GPU experts, changing outputs even when teacher-forced verification
passed. The prototype now restores both the batch-limit getter and slot-map
lookup outside verification, using a scoped original MoE-builder call.

An intermediate implementation fixed only the getter. The slot-map fallback
still used the wider stored limit and caused a retained CUDA illegal-access
warmup failure. That failed run is excluded. The corrected live lookup and
DFlash comparisons preserve the complete audited outputs.

## 2. Reused an existing trained draft; thread policy helped but did not win

After checking both approved roots, historical evidence located an existing
1,582,738,368-byte BF16 GPT DFlash model. A SHA-verified, manifest-owned
zero-copy reference was made under approved SSD model storage; the original
was preserved. The pinned upstream quantizer produced a 454,803,648-byte
Q4_0 **proposal-only** derivative: 57 Q4_0 and 66 F32 tensors. Tensor-name/shape
maps, target-layer metadata and F32 vector bytes were checked. Ordered tensor
metadata was not used as an identity proxy because quantization reorders it.
Target weights/precision were unchanged. No Q5/Q8 derivative was created.

The native runtime ordinarily selects sixteen batch threads for any
multi-token batch, including tiny verification windows. An explicit prototype
policy uses eight generation threads for bounded all-output decode calls and
restores original settings afterward. Full-logit/rollback checks remain exact,
with actual 16-to-8 acknowledgements.

| Q4 draft, two rows, global managed memory | Coding | Reasoning | Prose |
| --- | ---: | ---: | ---: |
| Default sixteen verification threads | 34.06 | 22.93 | 28.71 |
| Eight verification threads | 36.89 | 24.98 | 31.25 |

Rates are accepted-output tok/s at 256 outputs. CPU use falls from about
15.8 to 7.9 core equivalents. This is an improvement over a slower speculative
configuration, **not a win over the unchanged scalar baseline**. It does not
justify changing general prefill thread defaults.

## 3. Exact CPU packing: isolated gain, real-request loss

Ordinary lossless CPU repacking fails the full-logit gate: maximum absolute
error reaches 4.82377. Preserving weight values alone does not preserve
floating reduction semantics.

The separate [exact packed kernel](../../tools/trace/exact_packed_cpu.h)
preserves native integer-dot lanes, separate even/odd FMA chains, horizontal
reduction and Q8 activation preparation. After initializing the native CPU
lookup tables correctly, 3,432 low-level output checks are bit-exact.
The warm single-thread matrix microbenchmark is approximately 2.1x faster;
288 full-vocabulary GPT positions also pass with zero error.

| 256-output calibration | Coding | Reasoning | Prose |
| --- | ---: | ---: | ---: |
| Fresh original scalar control | 42.16 | 25.96 | 35.51 |
| Exact packed CPU | 37.12 | 23.59 | 32.21 |

All complete output hashes match, but real generation loses about 9-12% to
that control. Loading takes about 14.9 seconds and anonymous memory reaches
about 57.3 GiB. Approximately 9.85 million GEMV calls are acknowledged;
GEMM/activation-pack calls are zero in this path. The native raw expert
operator explicitly calls its vector dot with one row, so changing a trait's
row-count field alone does not enable shared multirow computation.

Do not promote the microbenchmark or claim an established cause for the
whole-request slowdown. Bandwidth, frequency, scheduling and preparation
overhead remain hypotheses pending causal profiling.

## 4. Selective cold KV staging: correctness passed; performance did not

Global managed-memory mode in this pinned CUDA backend substitutes host
`MemAvailable` for reported GPU free memory. Logs claiming roughly 116 GiB
free VRAM or negative unaccounted memory are not physical VRAM measurements.
The adapter now captures physical NVML memory and host-memory snapshots for
managed experiments, including a pre-load physical snapshot.

The default-off [KV interposer](../../tools/trace/kv_cold_tier.cpp) instead:

- Requires global unified-memory mode off.
- Recognizes F16 K/V context owners by semantic tensor names and shapes;
  skips views and rejects mixed contexts.
- Changes `cudaMalloc` to `cudaMallocManaged` only during native allocation of
  a recognized KV context. Ordinary model/expert allocations remain device
  allocations.
- Keeps GPU preferred placement; after native bulk clear, completes a CPU
  prefetch. Active pages can demand-migrate back to GPU.
- Retains configured 16K capacity and F16 precision, checks host headroom,
  tracks owners and requires the tracker to drain on free.

The [small actual-GPU test](../../tests/test_kv_cold_gpu.cpp) verifies managed
KV pointers, completed CPU-prefetch location, bit-preserving GPU set/get,
normal and mixed-context device allocations, views, positive allocation/stage
counters and owner drain. It is not only a CPU role-selector test.

The full-model selective-KV matrix passes **864 full-vocabulary positions,
270 rejected-suffix windows and 450 discarded input tokens**, all with zero
error. It includes two/four-row short cases, a 4,099-token prefill with a final
three-token chunk, and 4,048/4,112-token boundary cases. Target KV owners total
603 MiB. Live target-plus-draft KV owners total 859 MiB across three managed
allocations; weights remain ordinary device allocations.

### Fresh live calibration

| 256 outputs, same raw prompts and frozen target | Scalar | DFlash two rows + selective KV | Change |
| --- | ---: | ---: | ---: |
| Coding | 41.03 | 37.49 | -8.63% |
| Reasoning | 25.10 | 24.46 | -2.56% |
| Prose | 35.65 | 30.51 | -14.40% |

All three complete output hashes match. Both arms have zero physical
generation reads and zero owned-process swap; CPU use is approximately eight
core equivalents. Whole-request throughput also loses by approximately
8.25%, 2.38% and 12.20%, respectively. These are one fresh calibration pair,
not repeatability or energy-saving claims.

A four-row coding smoke reaches 32.45 tok/s over 64 outputs, matching the
scalar prefix. It accepts 40 of 65 offered tokens, with reported mean window
length 2.82. It is not a 256/512-output success and is not promoted to a
confirmation cohort. Draft generation totals about 108 ms across 44 calls
including the untimed audit; draft time alone does not explain generation
latency.

Physical NVML memory remains near full: the two-row READY snapshot has about
84.6 MiB free and the four-row smoke about 70.5 MiB. Successful logical CPU
prefetch does **not** establish 859 MiB of physical VRAM savings, lower PCIe
traffic or an improved GPU working set. Requested-stage-byte counters are
logical requests, not measured transfers. No such savings are claimed.

## 5. Expert-product ablation rejected

The follow-up `--expert-products native` ablation keeps split attention,
floating linear/router products, prefill placement guards, eight verification
threads and selective KV unchanged, but removes expert-column splitting.
It fails at both widths two and four: maximum absolute logit error 4.62417,
maximum relative L2 0.148768. Restoring `columns` passes all 288 checked
positions with zero error on the same new build.

Thus expert splitting cannot simply be removed from this correctness-qualified
path to recover batching speed. The ablation remains an explicit research
control, not an optimization or supported downstream setting.

## Decision and next evidence gate

### Installed Synapse is a separate baseline

After the experiment phase, the actual installed package and managed-launcher
receipt were inspected read-only. The launcher uses a verified-format profile
with 16K context, 12,000 MiB expert cache, direct transfers, decode pipeline and
asynchronous expert IDs already enabled. The installed runtime source matches
the local Synapse runtime file inspected at
`f7b1668f83140f1b284eb73beb29fbd1dd40ea59`; this is a file-content comparison,
not proof that the entire package was built from that commit.

The installed runtime still rejects hybrid execution with speculation. No
active deployed-model performance was measured during this inspection, and
GUI/request overrides were not tested. The older asynchronous-ID technique is
therefore already configured, not a new speedup waiting to apply. The current
frozen/staged research controls and Strata/Qwen MTP measurements are not a
matched comparison against this direct-transfer deployment.

**Do not deploy these throughput candidates or bypass Synapse's
hybrid/speculation guard.** Native optimized verification is now numerically
qualified on the bounded tested paths, but that is an enabling result, not
60 tok/s or full integration readiness.

Stop scaling the known-losing warmer/repack/staging variants and do not make
a larger draft merely on the assumption that acceptance fixes verification
cost. The next justified investigation is the cost of preserving scalar
reductions while amortizing multirow execution:

1. Establish the deployed-profile baseline first, using the same target bytes,
   raw prompts, context/KV, output caps and sampling, with owned benchmark
   processes and no edits to Synapse or user settings. Record the actual
   cache/transfer/adaptation controls and distinguish this track from frozen
   attribution. Do not attribute a model/engine switch to a generic speedup.
2. Profile the qualified two/four-row path, separating actual CPU expert work,
   GPU operator/launch work, transfers, draft work and synchronization. Graph
   construction counters are not runtime kernel counts.
3. Test an exact multirow/single-launch operator only where this decomposition
   shows meaningful headroom. Preserve each scalar lane/reduction order,
   causal masks, accumulation precision and KV partition/combine policy.
   A one-query tile alone does not prove the launcher uses scalar semantics.
4. Require the same full-vocabulary, rejected-suffix and live-prefill gates
   before measuring real accepted outputs. Promote only a positive calibrated
   candidate to three fresh alternating held-out 512-output pairs.

No new fused launcher, generic online controller or 60 tok/s confirmation
cohort has been implemented or completed in this phase. Broad answer quality
and energy savings remain unestablished.
