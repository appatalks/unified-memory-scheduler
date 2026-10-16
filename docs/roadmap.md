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

Still required before whole-model integration: broader rotating working sets,
actual model activations/router weights, and CPU/GPU scheduling that preserves
correctness and improves the uninstrumented critical path. The exported hot-set
plan is not yet wired into expert dispatch or admission/migration. The probe's
small pinned pool does not justify pinning the full model.
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

## Milestone 5: Workload-Aware Policies

Extend only demonstrated mechanisms to prefill, decode, and concurrent-request
objectives. Compare static, size-aware LRU, and measured/adaptive policies. Expose
simple user goals rather than expert-cache parameters; keep ordinary upstream
inference available independently of experimental scheduling.

## Deferred

KV cache tiering, NVMe expert execution, and multi-device scheduling remain
deferred until a narrower execution/residency policy is correct and beneficial.
No universal acceleration or transparent RAM-to-VRAM conversion is promised.