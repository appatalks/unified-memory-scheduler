# Roadmap

## Native Handoff Follow-up: Measured Budget, Mixed Control-thread Result

The [critical-path study](../benchmarks/results/handoff-critical-path-summary.json)
uses a warmer-free, installed-library CUDA/CPU trace. Within a 3.068-second
CPU-enclosed diagnostic envelope, CPU expert graphs account for about 1.730
seconds, scalar controls 0.049 seconds, and CUDA API intervals cover 0.418
seconds outside CPU graph/GPU execution. Required GPU waits, API calls,
observer overhead and idle gaps are not interchangeable or additive.

A default-off scalar-control interposer keeps expert work at its original
thread count while executing bounded dispatch/copy-only graphs with one worker.
The original long 96-position reference remains exact. Three matched pairs
per workload give median decode changes of +0.85% coding, +0.13% reasoning
and -0.41% prose. That is not a general speedup or a reason to alter defaults.
The control work itself is too small a budget for the 60 tok/s target.

The native Q/K/V optimizer passes short numerical controls but does not
improve a three-pair coding screen; concurrency metadata is not proof of
actual stream execution. CUDA12 pinned-copy/sync microbenchmarks find no
advantage for explicit spinning over auto; blocking waits lose. Do not repeat
these under new labels.

Further work should cost a genuinely different handoff protocol that reduces
host calls while proving native buffer lifetime/dependencies. Pinned memory,
CUDA graph replay, route packets and guarded CPU/GPU overlap are already
present or tested; neither skipping required fences nor counting all API
elapsed time as savings is justified.

## Live CPU Warmer: Implemented And Correct, No Throughput Win

At the user's request, the [live follow-up](../benchmarks/results/live-cpu-cache-warmer-summary.json)
goes beyond the offline gate below. A separate opt-in reader uses current
router IDs and original host weight buffers, with a latest-only bounded queue,
same-CCD helper affinity, finite job budget, cancellation and safe unload.
It does not wait for warming before executing experts or modify target math.

The final read path matches the original long-prompt full-vocabulary reference
exactly at all 96 coding/prose/reasoning positions. Native/noop/read controls
have matching outputs across 27 repeated 512-output requests. Median live
read throughput is 0.2-1.1% below matched native, not a demonstrated speedup.
An unrestricted coding triplet also fails to show a gain; fixed-CCD placement
is not credited as a warming benefit.

Actual NVTX helper-job intervals overlap GPU kernels by about 0.012%,
and any GPU activity by about 4.6%, mostly transfers. Much of warming instead
coincides with CPU dispatcher/control work or host gaps before GPU expert launches.
These whole-process diagnostic intervals are not per-cacheline timestamps.
The final matrix attests identical installed backend library mappings across
all variants. An earlier cohort with a differing preloaded GGML base was
retained as exploratory and excluded from the clean comparison.

Keep the prototype default-off and uninstalled. Do not increase warm volume
or promote a scheduler from isolated warm-cache ratios. Further work must
change information timing or execution order and pass a new matched gate;
the current causal dispatch trigger does not improve the throughput goal.

## GPU-window CPU-cache Staging: Feasible Primitive, Early Knowledge Unresolved

The [staging investigation](../benchmarks/results/gpu-window-cpu-cache-summary.json)
now has real CUDA/CPU timelines and repeated native CPU cache calibration.
Same-CCD full warming of one original expert reduces isolated FFN compute
about 3.0-3.13x with bit-exact outputs, but serial warming is slower overall.
Warming from the other CCD instead yields about 0.61x compute speed.

An ideal 32 MiB, 40 GB/s, perfect-next-demand model can stage about 27-33%
of logical projection payload in the four sampled traces. This is not a
throughput prediction: it assumes ideal retention, ignores helper overhead
and is measured on a profiler-perturbed execution. Actual observed route-ready
and pre-admission snapshots cover less than 0.3%; the simple previous-cold-set
control covers only about 2-3% while wasting most staged bytes.

Do not implement a learned predictor or live scheduler merely from the
microbenchmark ratio. Next require an earlier causal hint with useful
byte-weighted prediction under the current residency/admission policy, then
cost same-CCD tile warming, pollution, helper wakeup and concurrent DRAM use.
No installed runtime, target arithmetic, residency budget or decoder changes
are authorized by this study. The native control remains the throughput
baseline; this study does not promote the slower low-CPU-use profile below.

## Current Research Priority: Decode Cost And Resource Efficiency

GPU-prefill integration is parked while decode alternatives are evaluated.
The [CPU follow-up](../benchmarks/results/cpu-decode-efficiency-summary.json)
now has a repeatable, numerical-preserving **resource/latency tradeoff**:
GNU OpenMP `GOMP_SPINCOUNT=1000` reduces generation-window process CPU by
27-32% at roughly 4% lower generation throughput. Whole-request savings are
about 20-22% on short prompts but only 3% when prefill dominates. Native control
and candidate remain exact at 96 held-out full-vocabulary positions, with
matching generated outputs across the paired request matrix.

This is a candidate **optional lower-CPU-use mode**, not a faster default,
energy measurement or achievement of 60 tok/s. Diagnostic paired profiles
retain the same ordered CPU matmul/count/byte signatures, supporting idle
spinning rather than omitted expert work as the resource-saving mechanism.

VNNI integer replacement, pool-affinity and non-OpenMP threadpool pilots did
not establish a speed improvement. Full prompt-conditioned seed rebuilds
also fail the first costed comparison against causal admission on the small
held-out dataset. Preserve those negative results instead of repeating them
under new names. Further throughput work needs a measured critical-path
headroom argument before another kernel or scheduler implementation.

## Milestone 1: Baseline Infrastructure

- hardware/CUDA discovery;
- host and CUDA memory bandwidth tests;
- exact upstream llama.cpp audit;
- structured llama-bench experiment harness;
- CPU-only CI and focused tests.

Exit criterion: run a supplied dense GGUF larger than VRAM through all applicable baseline configurations and identify the limiting resource with synchronized telemetry.

## Milestone 2: Usable Local Baseline

The first Linux preview now provides:

- a Python-standard-library launcher opening a dependency-free browser GUI;
- loading/unloading an existing GGUF through a separately installed llama-server;
- upstream automatic fitting or CPU-only execution;
- streaming chat, cancellation, backend status, and explicit failures;
- a versioned loopback control API and OpenAI-compatible chat endpoint;
- fake-backend lifecycle/API tests in the existing CI.

Remaining work before claiming an easy-install release:

- validate real inference on the documented backend and representative models;
- package or guide installation of a verified backend without manual runtime setup;
- add local model discovery and deliberate, cancellable downloads with license/provenance information;
- estimate fit and report hardware/resource use without promising unsupported performance;
- test desktop responsiveness, accessibility, and a clean-machine setup;
- add Windows support only after the Linux flow is validated.

Exit criterion: install, select a supported model, run, chat, and connect an
external client without understanding placement flags. The current preview
requires Python, an existing GGUF, and llama-server; it does not yet meet the
clean-machine installation criterion.

## Milestone 3: Execution Telemetry

Add NVML and process sampling, PCIe counters where available, CUDA event tracing,
and a minimal llama.cpp layer/stage hook. Produce a common timeline keyed to
prompt processing and generation. Measure CPU computation, transfers, and GPU
work to identify the critical path rather than treating utilization as success.

## Milestone 4: Cross-Model MoE Scheduling Experiment

**Primary architectural goal: a MoE model-agnostic engine.** GPT-OSS-120B is
the first difficult workload, not the engine's identity. Routing widths,
projection layouts, dimensions, dtypes, resident plans, and memory budgets
must come from descriptors and measured resource costs, not checkpoint names.
Backend-owned model semantics must be shared with native execution. Fully
resident and dense models should not be forced into slower offload paths just
to claim that a new scheduler ran.

Cross-model acceptance requires more than successful fallback: multiple
trained MoE families, including a second larger-than-VRAM model, must pass
matched correctness/answer checks and repeated end-to-end throughput tests.
Synthetic graph/layout tests are a prerequisite, not a replacement. Different
models may need different placement policies; universal speedup is not a
credible invariant.

First instrument per-token/per-layer expert selection and collect held-out coding,
reasoning, and prose traces. Measure expert sizes, frequency, reuse distance,
temporal locality, and the actual cache capacity left at the intended context.
Trace collection is not a throughput benchmark.

Completed first slice: the pinned collector, strict decode trace validation,
training-only hot-set export, byte-budgeted LRU/LFU, a limited hit oracle,
fractional fixed-residency bounds, causal cost-aware admission, and costed replay are implemented. A real
1,536-token 120B trace and independent repeated prefixes were validated.
The first estimates do not establish a path to 60 tok/s with fixed residency.

Simulate static placement, LRU, LFU, and a future-aware oracle with explicit
PCIe, CPU execution, eviction, and synchronization costs. Cold experts must remain
eligible for direct CPU execution rather than every cache miss becoming a GPU
upload. Use the oracle as a feasibility gate before implementing migration.

The current GPT-OSS-120B goal is at least 60 sustained single-user generation
tokens/s at 16K context, with 90 tokens/s a stretch target. Aggregate batching,
prompt-copy-only acceleration, changed target weights, or ignored draft overhead
do not establish that target. Existing stock/fork/speculation screens did not
meet it; the verified official backend remains the control.

Live execution screening also now covers host-operation offload, polling,
512 MiB reserve, and an upstream operation-batch threshold. Decode stayed near
25 tok/s. A threshold that improved short-prompt latency failed a quantitative
answer check, so it was rejected as a default. The runner keeps those failures
and restores the original model settings; phase-latency wins are not counted
as achievement of the decode-throughput goal.

A separate backend patch now implements whole-model, expert-granular dispatch
for the original GPT-OSS-120B MXFP4 tensors and biases. Frozen residency screened
near 33 tok/s; causal adaptive residency screened near 35-36 tok/s. A packed
CPU dispatcher, masked CPU matmuls, and bounded admission staging are implemented.
CPU repacking and larger residency did not improve these screens. Asynchronous
uploads matched synchronous full logits at 96 teacher-forced positions, but
did not improve throughput in the first paired screen.
Direct DMA from bounded, immutable registered sources improved the paired
512-token sample from 35.73 to 38.96 tok/s, with a 38.81 repeat. Exploratory
six/eight-thread screens reached 39.63/39.99 tok/s; these are not repeated
cross-workload qualifications. The opt-in block-verification path runs, but
DFlash plus the smaller resident bank measured 22-24.5 accepted tok/s against
a 29.3 control, so it was rejected as a performance optimization.

Automatic model-specific execution is now the app default: a configured
installation selects non-speculative direct DMA for matching GPT-OSS-120B
MXFP4 layouts, while other models use clearly labeled standard execution.
Standard mode remains selectable; required-hybrid mode rejects unsupported
layouts. The installation profile persists local backend paths, not chats.
This is
not a qualified performance result or production release. Strict logit
diagnostics against CPU-only placement fail; stock automatic placement also
exceeds that diagnostic tolerance, so the threshold does not by itself isolate
a prototype bug. A quantitative answer check failed and remains a deployment
blocker. Broader paired quality checks, longer repeated workloads, and measured
benefits from any additional transfer optimization are required before rollout.

The downloaded-model matrix validates standard/automatic selection across
existing models at matched context/thread counts, with exact-length repeated
generation and small arithmetic/AST-only checks. This separates compatibility
from optimization coverage: successful standard inference on Gemma, Qwen,
or LFM does not demonstrate cross-architecture hybrid expert dispatch.

The model-agnostic refactor removes fixed 36-layer/128-expert/top-4
assumptions from the cache/dispatcher and reuses the original expert-math
builder for native and hybrid branches. Complete synthetic Llama, Qwen2,
Qwen3, GPT, Llama4, Cohere, and Grok graphs now execute with variable layouts,
seven storage types, shared experts, fused projections, optional biases, and
mixed dense/MoE layers. Cross-device view placement and duplicate float-CUDA
dummy-ID defects were found and fixed. Generic admission supports an empty
causal plan, and asynchronous staging is capped at 64 MiB with safe recycling.
Version-2 installation profiles prevent cross-model trained-plan reuse and
select native fitting when a small weight file makes splitting unnecessary.

Strict numerical failures for quantized cases remain visible, including native
GPU controls; no new generic quality guarantee is claimed. The refactored
real 120B frozen path is bit-exact to its previous hybrid reference at 96
teacher-forced positions. The next real-model screen must retain performance
and correctness results separately, then validate another trained large MoE
before claiming cross-family optimization benefits.

Only if the costed oracle leaves headroom, investigate expert-granular storage
and dispatch in ggml, first with a fixed hot set spread across layers. Keep GPU
hits on the GPU and CPU misses in RAM, combining partial outputs at the correct
graph dependencies. Compare correctness before performance, then test adaptive
admission and migration. Repeat on a second MoE architecture and retain a dense
control. Add bounded staging only when the measurements justify it.

An isolated real-weight hybrid-layer prototype is now implemented. It executes
CPU and GPU expert branches with asynchronous dispatch, bounded pinned partial
outputs, and separate GPU reconciliation. Both per-device weighted partial
vectors and all-GPU weighting/reduction pass tight numerical checks on layers
0 and 35. Serialized/concurrent timing is interleaved; synthetic F32 self-tests
also use an independent scalar reference. These warm-layer measurements are
not a full-model executor or generation-speed result.

The standalone probe alone does not establish whole-model integration:
broader rotating working sets, actual model activations/router weights, and
correct CPU/GPU scheduling must also be tested. The separate whole-model
dispatcher described above now consumes trained plans and performs causal
admission; the probe itself does not. Its small pinned pool does not justify
pinning the full model.
The collector now also replays each request with observation inactive to check
greedy token equivalence. After the user rebooted, all six real requests passed
that gate. Partial traces remain excluded. Post-reboot auto-fit changed expert
placement, so the trace, uninstrumented control, and RAM-read calibration were
refreshed; new conditional estimates remain below 60 tok/s. The optional parallel
RAM probe distinguishes concurrent sequential-read bandwidth from the older
single-thread measurement without claiming expert-kernel throughput.

Exit criterion: a narrowly scoped policy with reproducible benefit across the
tested MoE models, documented regressions/overhead, and upstream fallback when
unsupported or unproven. Do not require dense layer streaming first: its naive
form loses to static placement in the existing measured baseline.

The next critical-path slice implements early CPU-input preparation, guarded
CPU/GPU expert overlap, GPU reconciliation boundaries, resident gate/up fusion,
and optional ranked admission. The final technical median is 46.94 tok/s;
a separate three-repeat prose screen reached a 60.54 median, but coding and
reasoning remained 43.18/46.10. This is not the sustained cross-workload
60 tok/s milestone. Exact quantitative answers and strict native-reference
diagnostics still fail, so new options remain experimental rather than app
defaults. A QKV candidate introduced a cross-layout regression and was removed;
all seven layouts now pass comparisons against the prior hybrid path.
See the [decode investigation](research/frontier-moe-feasibility.md#guarded-overlap-and-resident-projection-fusion)
for retained failures, natural-output timing, and remaining validation gates.

## Milestone 5: Workload-Aware Policies

Immediate decode research order: separate copy waits from transfer/enqueue
intervals with a profiling observer control; test descriptor-derived routing
packets if justified; then investigate shared CPU preparation, empty cold
branches and time-priced admissions. The user's roughly 52 tok/s manual result
is not a benchmark qualification. Recovering about 2.56 ms/token would bridge
52 to 60, but no additional saving is assumed. See the
[optimization queue](research/frontier-moe-feasibility.md#next-optimization-queue-measure-before-removing-waits)
for contracts, rejected approaches and acceptance gates.

The first measured follow-up prioritized event-protected GPU slot-ID handoff
over D2H packing: the matched technical median improved 47.01 -> 49.27 tok/s,
with smaller coding/reasoning gains and nearly flat prose. Seven prior-hybrid
regressions and all 12 compared generation hashes matched; existing numerical
and exact-answer failures remain. The local experimental installation opts
in via `async_ids`; older installations remain unchanged. Next focus is CPU
preparation/empty branches and time-priced residency, not arbitrary removal
of producer/reconciliation waits.

The raw CPU all-masked-operator experiment is now tested but not promoted:
its matched Universe replay median was 46.60 versus 47.19 tok/s, with mixed
paired results, despite 61,983 exercised operators and passing regression
gates. The app retains the faster asynchronous-ID profile. Manual timing now
separates prompt/generation durations and input tokens, and an explicit
reasoning selector preserves Model default unless changed. The original
41.6-versus52 observation is not causally attributed; controlled replay did
not reproduce an asynchronous-ID regression. Shared CPU preparation and
time-priced residency remain the next targets, not assumptions of a gain.

Admission lifetime profiling now separates observed future reuse from censored
live residency. Probation, ranked ties and protected training anchors were
slower despite some traffic reductions; none is promoted. A fresh 4/8/12/16
thread sweep retains eight. Increasing residency to 12,000 MiB/hot944 improved
technical median 47.33 -> 50.06 tok/s and coding/reasoning/prose medians to
46.71/49.26/65.99, with 12,852-token retrieval passing. This is an experimental
app promotion only: exact quantitative and strict native-reference failures
remain, and the general 60 tok/s goal is still unmet.

Multi-position discrepancies also reproduce in native GPU-backbone and fully
CPU controls; forcing Q8 MMQ did not change them. The next architectural route
is separately gated resident-expert self-assisted proposals with full-target
verification, informed by SpecMoE, but it is not implemented and requires
numerical isolation first. See the [admission/capacity research](research/frontier-moe-feasibility.md#admission-lifetimes-larger-residency-and-current-research)
for primary sources, negative results and remaining gates.

The first transient cold-work split preserved all selected experts and reused
an existing bank slot, but was slower on GPT (48.42 -> 42.96 tok/s) and
nearly flat/slower on Qwen (44.85 -> 44.41). It stays off. Scalar-shaped router
projections did not fix full-model batch verification either.

The existing generic executor now has a second trained large-MoE result:
Qwen3.8-Flash-Next IQ3_S, native CPU experts/GPU backbone versus an untrained
8 GiB causal direct-DMA bank, 25.39 -> 43.88 tok/s. The app supports separate
optional causal budget/transfer policies without changing a bound model's
trained plan or budget. Selection remains descriptor-derived, not a Qwen
allowlist. This is not universal acceleration, broad quality qualification,
or the unchanged-GPT 60 tok/s milestone. See the [cold-work and cross-model
record](research/frontier-moe-feasibility.md#cold-work-split-and-a-second-trained-large-moe).

Shared CPU preparation and combined routing packets are now implemented behind
research gates. Their primitive/frozen checks pass, but shared preparation is
mixed/inconclusive and packet timing loses on held-out natural workloads despite
one third fewer D2H calls. Neither is promoted. Qwen-specific block2/3 checks
initially failed, including a native control; its custom MTP artifact needed
a proper native-format bridge. No speculative app mode or approximate expert skipping
is enabled. The final 19-target suite and matching default GPT regression pass;
the same14 native-reference failures remain. See the [three-track record](research/frontier-moe-feasibility.md#shared-cpu-preparation-route-packets-and-model-specific-verification).

The subsequent [canonical/MTP investigation](research/frontier-moe-feasibility.md#canonical-qwen-verification-and-native-mtp)
makes native and seeded mixed-residency Qwen blocks2/3/4 exact under opt-in
scalar-shaped learned/attention/expert products. Long-context samples and a
training-only, group-capacity-aware frozen seed also pass. A proper local
native MTP bridge now generates accepted output: a matched 512-token hybrid
coding screen improves **28.35 -> 32.13/33.62 tok/s** for two/three proposals,
with identical output hashes and real rejected proposals. These are not
repeated cross-workload results or a win over the different adaptive profile.
Next reduce verifier overhead and improve causal resident coverage, then
qualify rejection/state behavior and natural workloads before exposing an
app mode. GPT's original 60 tok/s milestone remains unmet.

Extend only demonstrated mechanisms to prefill, decode, and concurrent-request
objectives. Compare static, size-aware LRU, and measured/adaptive policies. Expose
simple user goals rather than expert-cache parameters; keep ordinary upstream
inference available independently of experimental scheduling.

The stage-aware prefill investigation now has transfer-inclusive real-weight
curves and separate observed routing histograms. After a user reboot restored
CUDA, 96 initial and 54 repeated reports tested rows 1-128 across early/middle/late
GPT layers. Four-row GPU chunks yield passing isolated improvements in selected
splits, including layer 0 at 32 rows (6.569 -> 4.666 ms). Other splits still fail
unchanged numerical gates, so there is no universal crossover or qualified
whole-model prefill gain. Two controlled prompt traces confirm substantial
row reuse and explicitly retain zero/one-row final-layer pruning.
Next resolve numerical differences on representative activations/expert sets,
measure per-expert/cache-pressure/pipelined costs, then qualify an integrated
stage-aware candidate on full-request time and decode-residency effects.
Application defaults remain unchanged. See the
[measured investigation](../benchmarks/RESULTS.md#stage-aware-prefill-calibration-isolated-gains-full-model-unqualified).

The subsequent [MXFP4 activation-alignment study](../benchmarks/RESULTS.md#mxfp4-activation-alignment-isolated-improvement-residual-failures)
finds a numerical contributor in CPU/CUDA activation rounding, scale storage,
and matrix precision. A separate default-off CUDA patch passes 57/60 broader
operator cases and all six zero/tie stress cases; three residual failures
remain. Qualified fixed-set isolated upload timings favor GPU execution at
32/64 rows but lose at one/eight rows. These are not application gains.
The [effective-Q8 replay follow-up](../benchmarks/results/mxfp4-effective-q8-propagation-summary.json)
now reproduces the severe down error on CPU using the GPU chain's effective
input; GPU down on identical effective input is accurate near 1e-7 relative L2.
Both projection propagation and activation-only rounding remain relevant for
that GPU-first prototype.

The [CPU-activation/GPU-down experiment](../benchmarks/results/cpu-activation-gpu-down-summary.json)
now implements the alternative split with CPU-native Q8 canonicalization and
all preparation/copy costs included. It passes all sixty bounded operator
configurations on the unchanged stock backend without the parity patch.
Three-repeat transfer-inclusive isolated timings improve at 32/64/128 rows;
one/eight-row uploads lose and sixteen rows are inconsistent. Scoped memcheck
and racecheck complete without reported errors. This bypasses the unresolved
GPU gate/up/activation differences rather than fixing the original prototype.
The [actual-input follow-up](../benchmarks/results/actual-prefill-activation-replay-summary.json)
now passes nine routed FFN frames across coding/prose/reasoning and early,
middle and pruned final layers. Native CPU replay matches captured output and
GPU-down preserves the original per-token IDs/probabilities. Capture on the
optimized CUDA profile perturbs logits; a separately qualified no-fusion
reference/capture pair is exact at 96 full-vocabulary observer positions.
Replay itself uses normal GPU fusion. This does not qualify candidate
full-model logits or make fusion-off a product recommendation.
The [separate full-model mask control](../benchmarks/results/resident-down-full-model-prototype-summary.json)
now implements explicit prefill, resident-only down weights and unchanged
zero/one-row/decode paths. Final default-off logits match the installed hybrid
exactly at 96 positions; enabled mode fails. CPU-only control is exact under
matched fusion-off execution, but real GPU down and even layer-0-only GPU down
still fail. A live CPU oracle passes all 735 local FFN comparisons near 1e-7
relative L2 while target logits diverge.
Do not move directly to compaction or request-speed claims: first isolate
graph/fusion sensitivity and where small arithmetic changes are amplified
through subsequent quantization/routing. No tolerance widening or tie-snapping
is justified. Full-target numerical, mixed-routing, memory and request-latency
gates remain; compact resident-aware dispatch is still the intended long-term
direction once those correctness issues are resolved.

The [numerical localization](../benchmarks/results/resident-down-numerical-localization-summary.json)
now confirms one amplification path with observer-preserved logits: tiny
layer-0 FFN differences remain small through layer-1 Q/K/V projections, cross
native F16 K/V storage boundaries (19/30 changed values), then grow through
attention and FFN Q8 replay before an expert-set change in layer 2. Query-F16
conversion is only replayed, not captured. The production fusion discrepancy
remains a separate open problem.

Next advancement candidates, not promised speedups:

1. Extend the now-passing
   [CPU-compatible packed-down probe](../benchmarks/results/cpu-compatible-packed-down-summary.json)
   into the research target graph with original CPU post-ops and actual
   quantized-state parity. Scalar and packed-DP4A variants pass 384 total
   checks bit-exact against local CPU controls. DP4A has repeated raw-stage
   gains with Q8 preparation/transfers included, but the all-required-experts
   resident probe is not the model's mixed-placement plan. Fresh uploads
   remain unfavorable or mixed, and shared CPU activation/post-op costs are
   outside the raw-stage comparison. Do not infer request acceleration.
2. Preserve native optimized fusion/kernel choices while changing dispatch.
   Validate the CPU-only control first without intrusive tensor observers.
3. Only after both target gates pass, compact resident token/rank work,
   reuse bounded buffers and test overlap without decoder-residency regressions.
   Then measure actual request latency and accepted-output generation.

The [real-model before/after study](../benchmarks/results/gpt120-prefill-real-workload-summary.json)
also tests native batching independently of that experimental CUDA patch.
Three 3,577-token/256-output pairs show about 3.5% lower prompt latency with
1024 rather than 512 physical rows, but a separate long held-out full-vocabulary
check fails the original gates while its unchanged control is exact. Larger
batching is rejected; thread/logical-batch screens find no new qualified winner.
The [Synapse research handoff](synapse-prefill-handoff.md) is reviewable but
explicitly blocks rollout. Qualify real activations and varying membership
before any full-model stage-aware scheduler. Installed libraries and serving
defaults remain unchanged.

## Deferred

KV cache tiering, NVMe expert execution, and multi-device scheduling remain
deferred until a narrower execution/residency policy is correct and beneficial.
No universal acceleration or transparent RAM-to-VRAM conversion is promised.