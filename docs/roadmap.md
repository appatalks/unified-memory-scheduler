# Roadmap

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

Extend only demonstrated mechanisms to prefill, decode, and concurrent-request
objectives. Compare static, size-aware LRU, and measured/adaptive policies. Expose
simple user goals rather than expert-cache parameters; keep ordinary upstream
inference available independently of experimental scheduling.

## Deferred

KV cache tiering, NVMe expert execution, and multi-device scheduling remain
deferred until a narrower execution/residency policy is correct and beneficial.
No universal acceleration or transparent RAM-to-VRAM conversion is promised.