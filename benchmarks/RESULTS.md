# Benchmark Ledger

This ledger is the reviewed index for benchmark evidence committed under `results/`.
It records the conditions required to compare runs, links raw artifacts, and
separates valid measurements from failed-command evidence. New result JSON is
ignored by default and is added here only after review.

Published JSON uses `benchmark-host` for the hostname and `/models` for
machine-specific model storage roots, including paths embedded in commands and
captured output. Model filenames, measurements, timestamps, and diagnostics are
otherwise unchanged. These placeholders are not executable local paths.

## Test Host

- AMD Ryzen 9 9950X: 16 physical cores
- 123.48 GiB host RAM
- NVIDIA GeForce RTX 5070 Ti: 15.45 GiB VRAM
- PCIe 5.0 x8; measured pinned H2D: 28.520 GB/s
- measured sequential RAM read: 42.160 GB/s
- model storage: Samsung 990 PRO NVMe
- llama.cpp: `b21e4de74567f5eef213765c9476a843c2e43f0d`

Unless a row states otherwise, measurements use three repetitions, a 512-token
prompt, a 32-token generation, and a 1,024 MiB auto-fit margin at a 4,096-token
fit context.

## Mistral Native Integration: Experimental Status Only

The [reviewed integrated experiment](results/mistral4-native-experimental-summary.json)
is published as separately labeled research evidence, not a headline speedup
or native-equivalent compatibility claim.

| Same isolated build, 16K/8/16, three exact 512-token samples | Decode median tok/s |
| --- | ---: |
| Native CPU experts/GPU backbone, experimental flags off | 21.09 |
| Automatic placement with experimental kernels | 23.96 |
| Empty-causal 8 GiB hybrid with experimental kernels | 33.01 |

Arithmetic passes; the original strict coding predicate fails in all three
profiles. Strict native-reference gates also fail at the unchanged 0.01 maximum
absolute / 0.001 relative-L2 limits. Mistral remains out of SYNAPSE's four-model
headline table and out of default accelerated installation.

The separate [experimental patch](../third_party/patches/llama-b11669-fp32-experts-experimental.patch)
adds dtype/precision/shape-gated CUDA matvec support, prevents quantized fusions
from bypassing protected expert precision, and optionally reproduces CPU Q8_K
activation semantics with FP32 scales. Defaults and the installed backend are
unchanged. The patch was applied after the existing hybrid patch on a fresh
pinned checkout and critical source components matched the measured build.

Twelve actual-weight primitive cases and padded/broadcast layouts pass the
native-CPU primitive gates. Integrated memcheck reports zero errors; racecheck
zero hazards/errors/warnings. Frozen and adaptive same-placement optimized
checks pass at 96 full-vocabulary positions each with zero error, but those are
not native equivalence. An unchanged SYNAPSE zipapp with a temporary profile
loads/generates successfully, returns the exact integer answer, and exercises
both backends with all top-4 contributions retained. The 128-token API smoke
is not a published throughput measurement.

Diagnostic captures can perturb fusion/scheduling. CPU-empty controls matched
when global CUDA fusion was disabled, but graph-disable and top-k-only controls
did not resolve all failures; mixed CPU/GPU native comparisons still fail.
Small first-layer expert differences can coexist with substantial final-logit
differences. No sole numerical root cause or general model quality is claimed.

The full [SYNAPSE experimental study](https://github.com/appatalks/Synapse/blob/main/docs/benchmarks/mistral4-native-experimental.md)
records source hashes, every retained numerical failure, the prior unqualified
variant and reproduction limits. Original GPT settings were restored and no
release/tag or default backend promotion was performed.

## Mistral Fused Quantized-Weight / FP32-Activation Kernel Probe

The [reviewed standalone follow-up](results/mistral4-fused-fp32-kernel-summary.json)
reuses pinned IQ4_XS/Q6_K dequantizers in an opt-in SM 120 CUDA matvec. It
dequantizes on the fly and accumulates in FP32, avoiding the initial prototype's
128 MiB/position global expanded-weight intermediate. Per-CTA shared memory
is 4,096/2,064 bytes for the two types.

Twenty-four completed timing fixtures pass fixed FP32-reference gates, including
one/three positions, normal/1e6 synthetic activation scales and CPU cache
pressure. Worst relative L2 is 1.88e-7. Routing/bounds fixtures cover masked
zero/-1, repeated IDs, varying token routing, partial rows and explicit invalid
ID/dimension rejection. Four separate instrumented fixtures pass; Compute
Sanitizer memcheck finds zero errors and racecheck zero hazards/errors/warnings.
Instrumented times are excluded.

In the completed refined-kernel/cache-pressure window, single-position medians
range **0.075-0.087 ms** versus **0.464-0.503 ms** for expanded-weight GGML GPU
execution and **0.365-0.500 ms** for the native quantized CPU primitive.
Warm controls are retained. API/timing boundaries differ and uploads are
excluded; these ratios are not Mistral tok/s predictions.

An earlier refined run passed all eight arithmetic fixtures but failed GPT
restoration with CUDA OOM while its process retained context state. It is
retained and excluded from headline timing. After recovery, non-CUDA
parent/CUDA-child isolation ensured context exit before GPT reload; subsequent
timing, memcheck and racecheck runs restored successfully.

This is an integration candidate only: no llama.cpp graph integration,
full-model native-equivalence check or new Mistral decode rate exists.
Installed backends, target weights and app defaults are unchanged. Final GPT
16K/8/16/12,000 MiB settings are verified. The
[app study](https://github.com/appatalks/Synapse/blob/main/docs/benchmarks/mistral4-compatibility.md#fused-fp32-gpu-follow-up)
details the limitations and next gates.

## Mistral Small 4 Compatibility And FP32 GPU Prototype

The [reviewed study](results/mistral4-fp32-compatibility-summary.json) retains
the pinned, verified UD-IQ4_XS target, initial harness failures, corrected native
controls and isolated precision probes. This is **not** a fourth/fifth hybrid
speedup or a full-model compatibility pass.

At actual 16K/8/16 and three exact 512-token samples, native automatic placement
measures **5.64 tok/s** and native CPU experts/GPU backbone **19.89 tok/s**.
Both pass the exact integer check and fail the original AST check: fenced code
also reverses the requested min/max nesting, so fence removal alone does not
pass. Generated code was never executed.

The hybrid loads the fused gate/up descriptor, both quantization groups and
36/36 routed layers, but rejects the protected FP32-input down projection.
CUDA explicitly refuses `MUL_MAT_ID` with source-1 `GGML_PREC_F32`; the model
requests it because activations can exceed FP16 range. The guard was not removed.

An opt-in, isolated GET_ROWS/dequantization plus FP32 batched-matmul prototype
passes 16 actual IQ4_XS/Q6_K weight probes, including synthetic activations at
scale 1e6. Dequantized weights are exact and relative-L2 versus CPU FP32
reference is at most 1.55e-7. It requires 128 MiB selected-weight scratch per
position and does not establish native-CPU equivalence: the native quantized
kernel differs by roughly 0.0032 relative L2.

Warm single-position GPU probes take 0.37-0.38 ms versus native CPU 0.09-0.11 ms.
With an explicit 256 MiB CPU cache-pressure control, GPU takes 0.46-0.48 ms and
CPU 0.28-0.49 ms. These are primitive measurements, not tok/s predictions;
there is no compelling universal win to promote. Next candidates are fused
quantized/FP32 GPU math or per-projection scheduling, gated by complete-model
verification against the faster native control. The installed backend and
target were not changed; GPT settings were restored. See the
[detailed study](https://github.com/appatalks/Synapse/blob/main/docs/benchmarks/mistral4-compatibility.md).

## Qwen3.6-35B-A3B Q8_0 Matched Hybrid Screen

The [reviewed report](results/qwen36-q8-benchmark-summary.json) adds a fourth
larger-than-VRAM target to the matched SYNAPSE table. At actual 16K, 8/16
threads and three exact 512-token samples per profile, stock automatic fitting
measures **45.50 tok/s**, native CPU experts/GPU backbone **34.07**, and the
empty-causal 8 GiB hybrid **77.99**, a **71.4%** gain against the faster control.
Hybrid samples are 73.47/77.99/80.24, with residency retained across requests.

The Apache-2.0 Q8_0 GGUF is pinned and SHA256-verified, staged on the mounted
WD_BLACK SN850XE NVMe, not the RAM disk. Descriptors give `qwen35moe`, 40 routed
layers, 256 experts/top-8, and 34,225,520,640 routed bytes. All selected
contributions are retained: CPU 148,713 plus GPU 608,143 equals eight per
94,607 dispatches. Warm second/third samples have zero process disk reads.

All six exact integer/AST checks pass. Descriptor-seeded frozen base and
optimized-versus-base checks each pass 96 sampled full-vocabulary positions
with zero error and both expert backends exercised. This is same-placement
pipeline agreement, not CPU-reference equivalence or broad quality qualification.
The smaller model's throughput does not meet the GPT-OSS-120B 60 tok/s milestone.
Original settings were restored. See the
[app report](https://github.com/appatalks/Synapse/blob/main/docs/benchmarks/qwen36-q8.md)
for settings, provenance and retained limits.

## Upstream llama.cpp PR 29887 Investigation

[ggml-org/llama.cpp#29887](https://github.com/ggml-org/llama.cpp/pull/29887)
is already in the pinned `620d2225c` backend. This
[reviewed policy/loading screen](results/upstream-pr29887-gpt120b-summary.json)
does not compare commits before and after the PR or claim another multiplier
on SYNAPSE's current results.

On unchanged GPT-OSS-120B MXFP4 at 16K, 8/16 threads, T0/seed42 and three exact
512-token samples per block:

| Configuration | Decode median tok/s |
| --- | ---: |
| Stock automatic placement | 24.42 |
| Automatic placement plus 6 GiB upstream LRU cache | 14.26 |
| CPU experts plus 10 GiB upstream LRU, mapped loading | 17.31 |
| CPU experts plus 10 GiB upstream LRU, copied CUDA-host buffers | 29.36 |
| Restored SYNAPSE 12,000 MiB trained hybrid, three windows | 47.89 / 48.66 / 48.92 |

Copy versus mapped loading is a matched 69.6% improvement for upstream LRU.
The three corresponding output hashes and final LRU counters match, including
66.77% small-batch hits. Backend RSS stays near 58-59 GiB; copied loading uses
`CUDA_Host` rather than `CPU_Mapped` buffers, with 57.5 GiB observed shared RSS.
It changes the upload path and OS memory accounting, not merely resident capacity.
Warm samples read no new disk bytes.

The 12,000 MiB copied upstream-cache attempt failed allocating a 622.15 MiB
prefill compute buffer. The first fit/cache block's missing-statistics failure
at ordinary verbosity is retained; an independent verbosity-4 repeat confirmed
cache activity. All completed modes pass the exact integer/AST checks, not
general answer-quality or CPU-reference equivalence. Screens are sequential,
not randomized or exhaustive. No native defaults were promoted; every original
profile was restored. See the
[detailed app report](https://github.com/appatalks/Synapse/blob/main/docs/benchmarks/upstream-pr29887.md).

## GLM-4.5-Air IQ4_XS And RAM Accounting

The [reviewed third-model screen](results/glm45-air-iq4-xs-benchmark-summary.json)
uses the unchanged106B GLM-4.5-Air community IQ4_XS target at16K/8/16.
Three exact512-token samples per configuration retain both standard controls,
native mapped/copied controls, answer-check failures and the8GiB CUDA OOM.

| Configuration | Decode median tok/s |
| --- | ---: |
| Standard automatic placement, first window | 12.67 |
| Native CPU experts / GPU backbone, mapped | 11.30 |
| Causal direct-DMA,6,144MiB, no trained plan | **17.28** |
| Standard automatic placement, reversed-order window | 12.14 |
| Native CPU experts / GPU backbone, copied into RAM | 11.43 |

The conservative gain against the fastest control block median is **36.4%**.
The candidate keeps every selected expert and does not use MTP, expert
skipping, target conversion or a lower context. Two downloaded shards total
60,480,900,352bytes and match the pinned public LFS hashes. The47-block GGUF
contains46 trunk layers and one unused NextN block;45 routed layers use
128experts/top8, hidden4096, FFN1408 and IQ4_XS/IQ4_NL components.

Same-placement optimized-versus-base full-logit checks pass96positions with
zero error and both CPU/GPU contributions. Original default-thinking failures
remain: an arithmetic request exhausted1024tokens without a final answer,
and fenced Python failed the strict output-format check. Separate thinking-off
diagnostics pass the unchanged integer/AST checks in both modes. This is not
CPU-only/native-reference equivalence or broad quality qualification.

Mapped native and hybrid execution already had about54GiB RSS; warm samples
show zero disk reads. Copy mode changed host allocations and OS-unavailable
RAM substantially without a useful throughput gain. Logical selected routed
weights total3.37GB/token; the native rate implies a38.1GB/s logical read budget,
not a direct measurement of physical memory bandwidth.

The browser's old RAM pie showed pressure (`total - MemAvailable`), omitting
much cached model residency. Occupancy, OS availability, free pages, file cache,
shared pages and overlapping backend RSS are now distinguished; no allocation
headroom check was loosened. **Require hybrid** supports explicit causal cache
budgets without changing Automatic defaults or fixed trained-plan budgets.

GLM-5.3-Flash320B/18B was inspected but not downloaded: the checked IQ2_XXS
was94.85GiB, above available disk space. No5.3 speedup is claimed. See the
[app benchmark guide](https://github.com/appatalks/Synapse/blob/main/docs/benchmarks/glm45-air-iq4-xs.md)
for model provenance, measured profile and limitations.

## Canonical Qwen Verification And Native MTP Research

The [reviewed follow-up](results/moe-canonical-verification-mtp-summary.json)
isolates scalar/batched execution differences without weakening the
0.01 maximum-absolute / 0.001 relative-L2 logit gates. Scalar-shaped learned
BF16/F16 projections, attention products, and resident GPU expert products
make Qwen blocks **2/3/4 exact at 96 sampled positions each**, including mixed
CPU/GPU execution. Native and seeded mixed-residency long-context samples
also pass with zero error. A training-only, group-capacity-aware 6,144 MiB
seed passes a separate 96-position block4 gate.

Initial seeded failures, the unsuccessful CUDA-fusion-disable control,
active-capture observer effects, and a graph-metadata allocation failure
are retained. Extra graph metadata is reserved only for canonical experiments;
ordinary app allocation and execution remain unchanged.

A derived native MTP draft reuses the pinned upstream converter's tensor
transforms, original raw draft tensors, and unchanged target embedding/output
tensors. It is not an architecture-label rename. Only the derived draft's
experts are quantized to Q4_0; original target/draft files are untouched.
The separate MTP context does not inherit the target bank or seed plan.

| Screen | Scalar control | MTP, two proposals | MTP, three proposals |
| --- | ---: | ---: | ---: |
| Native CPU experts/GPU backbone, 4/16 threads, 128 coding tokens | 21.17 | 31.65 | Not tested |
| Frozen 6,144 MiB hybrid, 8/16 threads, 512 coding tokens | 28.35 | 32.13 | 33.62 |

Numbers are accepted-output decode tok/s, not proposal or teacher-forced
positions/s. Drafting, verification and synchronization are inside the
generation interval; total request wall times are retained separately.
All compared output hashes match within each screen. Hybrid two/three
proposal runs accepted **335/350** and **376/402** draft tokens, respectively;
rejected proposals were not counted as output.

These are **single-sample screens**, not cross-workload qualification or a new
overall best. The frozen 6 GiB bank and canonical execution differ from the
earlier 8 GiB adaptive Qwen configuration; do not infer a regression or a
33.62 tok/s hardware ceiling from that unmatched comparison. The 60 tok/s
unchanged-GPT milestone remains unmet. Canonical/MTP app options stay off.

## Shared Preparation, Routing Packets And Per-Model Verification

The [reviewed exploration](results/moe-preparation-packet-verification-summary.json)
records the requested follow-up without promoting unqualified gains.
Shared CPU activation/row-map preparation passed strict primitives and frozen
math, but technical medians were mixed: GPT **46.40 -> 45.94**, Qwen
**43.14 -> 43.50 tok/s**. The latter's sub-1% screen is inconclusive, despite
79,958 actual reuses and a zero-error 96-position scalar comparison.

The complete routing packet reduced filtered D2H calls **170,856 -> 113,904**
with identical logical bytes. Its technical median improved
**45.76 -> 46.31**, but a reverse-order natural screen was slower:

| Workload | Existing handoff | Routing packet |
| --- | ---: | ---: |
| Coding | 46.11 | 43.09 |
| Reasoning | 46.51 | 44.59 |
| Prose | 65.83 | 62.54 |

An initial packet failed to remove separate CPU bias-ID copies; those and
pre-FFN consumers were fixed before the complete test. Both iterations and the
failed exact quantitative check are retained. These measurement windows are
not matched to the earlier 50.06 technical result and do not establish its
cause or a new hardware limit.

Qwen block2/3 verification also failed unchanged strict gates, including a
native block2 control. Its available drafter declares `qwen4exp-mtp`, not the
native loader's canonical format. No adapter, MTP timing claim or speculative
app mode was enabled. Both prototypes remain off. All **19 configured test
targets** passed; seven combined prior-hybrid and seven cold-copy comparisons
passed with the same 14 native-reference failures. The working app was
restored to GPT's 12,000 MiB profile and passed live activation.

## Cold-Work Experiment And Second Trained Large MoE

The [reviewed follow-up](results/moe-cold-work-and-qwen38-summary.json)
tests lessons from Strata without target expert skipping or weight changes.
On GPT-OSS-120B at 11,264 MiB, a bounded transient cold-GPU path was slower:
**48.42** control, **40.93** synchronous-copy reference, **42.96 tok/s**
asynchronous candidate. Matching-placement logit gates passed; extra PCIe
uploads outweighed the reduction in CPU expert calls. Initial CUDA OOM and
scratch-ID bound failures were retained, with the latter fixed.

The already-downloaded Qwen3.8-Flash-Next IQ3_S demonstrated a second real
large-MoE benefit at 16K/8/16: **25.39 -> 43.88 tok/s** median over three
512-token technical samples. The control uses native CPU experts/GPU backbone;
the candidate uses an empty causal plan and 8,192 MiB direct-DMA bank. This
is not stock automatic placement or a matched Strata/MTP comparison.
Both passed the narrow integer/AST checks, not broad quality qualification.

A separate Qwen cold-offload pair was nearly flat/slower (**44.85 -> 44.41**),
so transient offload remains disabled on both tested models. Scalar-shaped
router diagnostics also failed to resolve batch verification. No speculation,
expert pruning or relaxed numerical threshold was enabled. The app now has
separate optional causal budget/transfer settings while preserving the GPT
trained-plan profile. The overall GPT 60 tok/s goal remains unmet.

## Admission Lifetime Research And Larger Resident Bank

The [reviewed investigation](results/gpt120-admission-capacity-research-summary.json)
records negative probation/ranked/anchored admission screens, a fresh CPU
thread sweep, native batch-verification failures and a larger-cache winner.
With the same 16K/8/16/direct/overlap/fusion/asynchronous-ID settings,
11,264 MiB/hot880 versus 12,000 MiB/hot944 technical medians were
**47.33 -> 50.06 tok/s**. Separate three-by-512 natural workload medians were:

| Workload | Previous bank | Larger bank |
| --- | ---: | ---: |
| Coding | 44.33 | 46.71 |
| Reasoning | 47.01 | 49.26 |
| Prose | 62.42 | 65.99 |

Both passed the 12,852-token retrieval/allocation check and narrow integer/AST
checks. Both retained the same exact quantitative-answer failure, so neither
is quality-qualified. Matching-placement frozen gates passed; native scalar
versus multi-position gates still fail and speculative serving stays off.
The local experimental app uses the larger bank; the prior plan and Standard
rollback remain. The overall sustained 60 tok/s goal is still unmet.

## Manual Universe Replay And Negative CPU-Empty Screen

The [reviewed comparison](results/gpt120-universe-cpu-empty-summary.json)
investigates the reported 41.6-versus-52 tok/s difference without treating
unmatched requests as a regression. The recorded 614-token case had the full
updated profile active. Controlled model-default-reasoning, T0/seed42,
three-by-512 replay of the supplied Universe/Earth prompt measured
**45.89 -> 47.65 tok/s** with asynchronous IDs; original UI sampling/history
was not reconstructed and the original difference remains unattributed.

An all-masked raw CPU operator early-zero path passed primitive, frozen and
seven-layout regression checks, but its matched median was **46.60 versus
47.19**, despite 61,983 exercised operators. It remains disabled in the app.
The same 14 native-reference failures remain visible. The 17 configured test
targets passed. New app timing components and an explicit reasoning selector
improve manual comparisons without changing default reasoning or sampling.

## Copy/Wait Breakdown And Event-Protected GPU IDs

The [reviewed handoff summary](results/gpt120-async-id-handoff-summary.json)
records the first follow-up to research checkpoint `068f69e`. On unchanged
GPT-OSS-120B MXFP4 at actual 16K context, 8/16 threads and 11,264 MiB residency,
three matched 512-token technical samples improved the median from
**47.01 to 49.27 tok/s (+4.8%)**. Natural-style coding/reasoning medians improved
**43.84 -> 45.31** and **46.56 -> 48.26**; prose was nearly flat at
**61.31 -> 61.62**. This does not establish the overall 60 tok/s milestone.

The prototype replaces blocking scheduler slot-ID copies with bounded,
event-protected per-layer transfers and explicit graph dependencies.
H2D destination-wait calls decreased by about 1.02 ms per filtered graph, but
CPU submission rose; only the net end-to-end gain is counted. Logical ID
transfers still occur outside the scheduler-copy counters.
All 12 compared generation hashes and all seven prior-hybrid fixture gates
matched controls. Existing native-reference and exact-answer failures remain.
The optional `async_ids` installation setting is enabled in the local
experimental manual profile after validation; Standard rollback is unchanged.

## Decode Overlap And Projection Fusion Screen

The [reviewed summary](results/gpt120-decode-optimization-summary.json) records
GPT-OSS-120B MXFP4 on the 9950X/5070 Ti with unchanged target weights,
actual 16K context, one request, 8 generation/16 prompt threads, and 512 output
tokens per sample. These conditions supersede the older defaults below for
this section only. The final technical three-repeat median is **46.94 tok/s**.

Separate natural-style prompts disabled forced EOS suppression and retained
the exact 512-token cap/count gate. With an 11,264 MiB resident cache:

| Workload | Three samples, tok/s | Median tok/s |
| --- | --- | ---: |
| Coding | 43.18, 43.25, 42.51 | 43.18 |
| Reasoning | 43.74, 46.10, 47.03 | 46.10 |
| Prose | 58.60, 60.54, 61.94 | 60.54 |

The prose median is not the overall 60 tok/s milestone. These capped segments
do not establish full-answer quality. Exact quantitative answers still failed;
two-key retrieval at 12,852 prompt tokens passed. During these screens the new pipeline options were off in the installed
default, and the previous app configuration was restored. The subsequent
experimental manual trial opts into the bundle with installation
`decode_pipeline: true`; this does not remove the failed quality gates.

Q8-drafter speculation was slower. QKV aliasing failed an isolated synthetic
regression and was removed. All seven final fixtures passed comparisons with
the prior hybrid path; 14 broader strict/native-reference failures remain
visible. Read the [investigation](../docs/research/frontier-moe-feasibility.md#guarded-overlap-and-resident-projection-fusion)
before interpreting these measurements as deployment evidence.

## Reviewed Results

| Model | Placement | Prompt tok/s | Generation tok/s | Evidence |
| --- | --- | ---: | ---: | --- |
| Calme 3.2 78B Q4_0 | auto-fit, 26 of 86 dense blocks in VRAM | 185.74 +/- 0.55 | 1.35 +/- 0.02 | [matrix](results/20260822T153837Z-calme-3_2-instruct-78b-Q4_0.json) |
| GPT-OSS 20B MXFP4 | auto-fit, full GPU | 9,289.33 +/- 394.37 | 217.18 +/- 1.52 | [run](results/20260823T143325Z-gpt-oss-20b-MXFP4.json) |
| Qwen3-Coder 30B-A3B Q3_K_L | auto-fit, near/full GPU | 4,048.34 +/- 120.65 | 204.87 +/- 6.82 | [run](results/20260823T143354Z-Qwen3-Coder-30B-A3B-Instruct-Q3_K_L.json) |
| Qwen3.6 35B-A3B Q8_0 | all routed experts on CPU | 313.22 +/- 3.62 | 30.83 +/- 0.17 | [run](results/20260823T143533Z-Qwen3_6-35B-A3B-Q8_0.json) |
| Qwen3.6 35B-A3B Q8_0 | auto-fit GPU/CPU expert placement | 487.95 +/- 7.19 | 41.41 +/- 0.67 | [run](results/20260823T143433Z-Qwen3_6-35B-A3B-Q8_0.json) |
| GPT-OSS 120B MXFP4 | all routed experts on CPU | 232.54 +/- 1.47 | 19.30 +/- 0.12 | [run](results/20260823T151527Z-gpt-oss-120b-MXFP4-00001-of-00002.json) |
| GPT-OSS 120B MXFP4 | auto-fit GPU/CPU expert placement | 305.95 +/- 13.27 | 22.75 +/- 0.31 | [run](results/20260823T151455Z-gpt-oss-120b-MXFP4-00001-of-00002.json) |

The dense 78B result is host-RAM-bandwidth-bound. The large MoE results instead
compute CPU-resident routed experts in place; their generation PCIe traffic is
activation-scale rather than full-expert transfer traffic. See
[frontier-moe-feasibility.md](../docs/research/frontier-moe-feasibility.md) for
the GGUF payload accounting and phase telemetry.

## Context Sweep

Qwen3.6 35B-A3B Q8_0 used auto-fit placement with Q8_0 K/V cache. Each point
uses a prompt half as long as its fit context and generates 32 tokens.

| Fit context | Prompt | Prompt tok/s | Generation tok/s | Evidence |
| ---: | ---: | ---: | ---: | --- |
| 16K | 8,192 | 457.53 +/- 14.27 | 40.96 +/- 0.20 | [run](results/20260823T150402Z-Qwen3_6-35B-A3B-Q8_0.json) |
| 32K | 16,384 | 437.16 +/- 6.69 | 40.55 +/- 0.76 | [run](results/20260823T150701Z-Qwen3_6-35B-A3B-Q8_0.json) |
| 64K | 32,768 | 426.94 +/- 13.07 | 40.55 +/- 0.44 | [run](results/20260823T151238Z-Qwen3_6-35B-A3B-Q8_0.json) |

## Retained Failed Evidence

| Artifact | Reason retained |
| --- | --- |
| [initial dense matrix](results/20260822T153837Z-calme-3_2-instruct-78b-Q4_0.json) | Its nominal CPU run was not strict CPU because CUDA operation/KV offload remained active. Its Unified Memory command used invalid `-ngl all` syntax. Valid manual and auto-fit runs remain comparable. |
| [Qwen3.6 context attempt](results/20260823T150213Z-Qwen3_6-35B-A3B-Q8_0.json) | Pinned `llama-bench` does not accept a general `-c` context override. The corrected context series uses `--fit-ctx` and Q8_0 K/V cache types. |

## October 16: Model-Aware Defaults And Downloaded-Model Matrix

This series uses pinned llama.cpp commit
`620d2225cc436ab0b001e3200e590e3b286ef9ab`, **actual 16,384-token context,
8 generation / 16 prompt threads, and three exact 512-output-token runs**
per model/profile. It is not comparable to the earlier 512-prompt/32-generation
tables without accounting for the changed workload/backend.

The [complete matrix](results/20261016-downloaded-model-matrix.json) contains
all 30 throughput samples, 20 small answer checks, source/patch/plan hashes,
model architecture/type metadata, selected execution profiles, and successful
restoration of the original GPT-OSS-120B app.

| Existing model | Standard median tok/s | Automatic median tok/s | Automatic execution | Arithmetic / AST-only code checks (both profiles) |
| --- | ---: | ---: | --- | --- |
| GPT-OSS-120B MXFP4 | 25.69 | 39.46 | Custom hybrid direct DMA | Pass / pass |
| Gemma 3 12B Q4_K_M | 80.94 | 80.50 | Standard | Exact-integer check failed / pass |
| DeepSeek-R1-0528-Qwen3 8B Q4_K_M | 130.08 | 128.50 | Standard | Arithmetic completion hit 1,024-token limit / pass |
| Qwen3.5 local Q8_0 (`claude-oss-q8_0.gguf`) | 79.48 | 80.26 | Standard | Pass / pass |
| LFM2 1.2B Q8_0 | 485.67 | 486.55 | Standard | Exact-integer check failed / strict AST check failed |

The Qwen3.5 row is identified by actual GGUF architecture `qwen35`; its local
filename is not evidence of an official Anthropic model. The CLIP projector
was excluded because it is not a standalone language model. No models were
downloaded. Only GPT-OSS-120B exceeds this GPU's VRAM among the current five;
this series cannot demonstrate hybrid benefits on a second larger-than-VRAM
architecture.

GPT-OSS-120B's paired median gain is about **53.6%**. Standard mode uses upstream
auto-fit; automatic mode uses the patched resident bank, CPU cold-expert
execution, adaptive admissions, and direct registered-source DMA. Context,
threads, prompt, output count, seed, and sampling settings are matched;
placement/offload are intentionally different and recorded. Loading and source
registration are outside decode timing; end-to-end generation and load
durations are recorded separately. The initial trained bank resets on each
profile load; adaptive residency persists between its repetitions and answer
checks.

All four unsupported architectures selected standard execution as intended.
Their repeated output hashes matched across standard/automatic profiles;
small timing differences are not evidence of an optimization gain. The
standard GPT-OSS repetitions matched one another, but adaptive GPT-OSS hashes
did **not** all match each other or the standard control. Passing the two small
answer checks is not general model/logit equivalence. Strict CPU-reference
diagnostics and the earlier quantitative JSON failure remain unresolved.

Answer checks deliberately require the exact integer and a specific AST with
no extra content. Failures can include formatting or token-budget issues;
answer text is not stored in this matrix, so these booleans must not be
reinterpreted as a broad factual-quality score. The runner retains failed
checks and exits nonzero even when the throughput runs and restoration succeed.
The 512-token throughput prompt is forced past EOS if necessary, not a guarantee
of 512 natural-answer tokens. This is a repeated one-prompt screen, **not**
sustained coding/reasoning/prose qualification. GPT-OSS-120B's 60 tok/s target
is still unmet; the smaller models' higher rates do not satisfy that goal.

### Published Hybrid Research Evidence

| Artifact | Interpretation |
| --- | --- |
| [Direct-source pair](results/20261016-gpt120-direct-source-screen.json) | First paired 35.73 -> 38.96 tok/s result; six output hashes match between copy paths, but the original quantitative JSON check remains failed. |
| [Direct-DMA thread screen](results/20261016-gpt120-direct-thread-screen.json) | Four/six/eight-thread one-prompt samples 38.81/39.63/39.99 tok/s; not three-repeat per-thread qualification. |
| [CPU dual-layout screen](results/20261016-gpt120-dual-layout-screen.json) | Real CPU-repacked storage with lossless GPU exports; adaptive 33.80 tok/s did not beat the raw-layout screen. |
| [Asynchronous staged-copy screen](results/20261016-gpt120-async-copy-screen.json) | Synchronous 35.65 versus async 34.67 tok/s; identical sampled logits, no timing gain, failed quantitative check retained. |
| [Small-block DFlash screen](results/20261016-gpt120-hybrid-block-dflash-screen.json) | 22.81/22.12/24.47 accepted tok/s versus 29.28 control with a smaller bank; includes draft/rejection overhead, not a successful optimization. |
| [Full-vocabulary diagnostic metrics](results/20261016-gpt120-logit-diagnostics.json) | CPU repeat is exact; stock placement and hybrid fail the unchanged strict CPU-reference gates. Direct-DMA versus synchronous hybrid is exact at 96 sampled positions. Block4 remains a failed strict diagnostic. No token IDs/full logits/activations are published. |
| [Training-only initial plan](results/gpt120-trained-hot800.txt) | 800 expert groups from the public routing corpus's training half only; SHA-256 `1198ae0ea67a9b94f1f51f0913f7ade3378bb5ba01e3ac93ace7eccaa77e98e6`. |

Published copies replace private paths with `/models`, `/opt/vrambridge`,
`/workspace/unified-memory-scheduler`, `/session`, and `/home/benchmark-user`.
Raw synthetic answer strings from old screens are replaced with SHA-256 hashes;
measurements, failed checks, diagnostics, and original run status are retained.
Large routing traces, binary references, activation dumps, and raw process logs
remain ignored. Older screen source hashes describe their own prototype state,
not necessarily the final patch. The current matrix identifies the final patch
with SHA-256
`a0f61f4e86aa6468d9b76ed4c9ee801dbee30ff2ec034566e8f86cf25be472a1`.

The app's new default is **automatic model-specific selection**, not a claim of
universal hybrid support. A pinned local installation selects direct DMA only
for the supported GPT-OSS-120B MXFP4 layout; unsupported models select standard
execution with an explicit reason. `--execution-mode standard` bypasses the
installation profile for rollback. Experimental warnings and readiness/resource
checks remain enabled.

## Pending Candidates

These are not benchmark results and must not be compared with the rows above.

| Candidate | Status | Next gate |
| --- | --- | --- |
| GLM-5.2 UD-Q4_K_M | Download in progress; 466 GB expected GGUF, 753B total / 40B active | Verify full shard set and header, then run a fit-only probe. The complete expert pool exceeds current host RAM, so an interactive run is not expected on this host. |
| Qwen3.6 27B Q4_K_M or Q6_K | Candidate dense quality comparison | Run fit-only probe first, then the same 512/32/3 matrix if it fits. |

## Tracking Rules

1. Retain raw JSON for every reviewed conclusion, including invalid commands and failed runs that changed the interpretation.
2. Record model file, quantization, llama.cpp commit, fit margin/context, prompt and generation tokens, repetitions, and K/V cache types.
3. Do not compare token rates across different model architectures as quality scores. Compare placement policies only within an identical model and workload.
4. Add a result to this ledger only after focused validation and a review of placement and telemetry.