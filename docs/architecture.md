# Architecture

## Current Milestone

The implemented runtime and measurement components are:

```text
vrambridge-info
    -> Linux proc/sysfs discovery
    -> CUDA runtime discovery
    -> RAM and CUDA copy benchmarks
    -> human or JSON report

vrambridge-benchmark
    -> experiment matrix
    -> llama-bench JSON subprocesses
    -> provenance and failure capture
    -> benchmarks/results/*.json

local browser app
    -> versioned local control API
    -> managed llama-server process
    -> OpenAI-compatible chat proxy

vrambridge-trace (optional pinned-backend adapter)
    -> original expert tensor sizes and placement
    -> actual post-evaluation decode routing IDs
    -> complete, versioned local JSONL trace

vrambridge-routing
    -> trace/domain/completeness validation
    -> held-out residency and migration replay
    -> conditional estimates, bounds, and a non-executable hot-set plan

vrambridge-hybrid-layer (optional pinned-backend prototype)
    -> real selected GPT-OSS expert weights, synthetic activations
    -> independent CPU and GPU expert graphs
    -> bounded pinned activation staging
    -> separate GPU reconciliation graph
    -> numerical gates and isolated warm-layer timings

experimental whole-model backend patch (default for matching installed profiles)
    -> immutable original GPT-OSS-120B expert tensors
    -> training-only resident bank and packed CPU routing
    -> CPU cold / GPU resident branches with original probabilities
    -> causal admissions, bounded staging or read-only registered-source DMA
    -> opt-in frozen-residency target-verification blocks

vrambridge-logit-check (optional patched-backend diagnostic)
    -> aligned teacher-forced full-vocabulary comparison
    -> single-position or small-block evaluation
    -> explicit failed gates, not generation-throughput claims
```

The original official executable remains intact. The supported experimental
default changes the expert graph through a separate pinned patch; llama.cpp
still supplies GGUF parsing, tensor semantics, kernels, sampling, and the
standard backend path. Default selection is not production qualification.

## Hybrid Expert Prototype Boundary

The standalone layer probe executes all five partitions of four chosen experts,
with original MXFP4 weights, biases, and OpenAI SwiGLU semantics. The CPU graph
and asynchronous GPU graph run independently. A separate backend stream receives
the CPU result through a small reusable pinned buffer, establishes the GPU branch
copy dependency, combines contributions, and synchronizes before reuse.

### Stage-aware Prefill Calibration

The isolated layer probe can calibrate 1-256 synthetic rows per selected expert
and optionally include synchronous pageable weight uploads in its timed path.
Its same-expert row reuse is an explicit microbenchmark assumption, not measured
prefill routing. Optional GPU row chunks share weights while the CPU reference
retains its full-row graph. Per-mode and per-partition gates include all warm-up
and timed executions; failed fast variants are not qualified improvements.
The trace collector separately supports observed prefill row histograms with
explicit final-layer pruning coverage. Instrumented collection is not a speed
measurement. Neither probe changes the whole-model dispatch batch limit,
installed backend, or application defaults. Stage-aware prefill scheduling still
requires representative numerical checks and complete target-graph qualification;
see the [calibration boundary](../benchmarks/README.md#stage-aware-prefill-calibration).

### CPU-compatible Packed-down Research

An explicitly enabled CUDA build of the layer probe also tests native packed
Q8_0 inputs with a host-specific MXFP4 dot contract. Scalar and packed-integer
implementations retain the pinned CPU's paired FMA lanes and horizontal sum;
original bias, probability weighting and ordered reduction remain on CPU.
This avoids treating a small operator L2 error as sufficient model-state
parity. The probe checks raw bits and native FFN boundary bits, and charges
quantization, transfers and synchronization in raw-stage timings.

The [evidence](../benchmarks/results/cpu-compatible-packed-down-summary.json)
does not change runtime dispatch, installed plugins or engine capability
detection. Its all-required-experts GPU replay is not the deployed residency
plan. Full-model normal-fusion, KV/Q8 state, mixed-routing and request-latency
gates still precede compact dispatch or a Synapse integration recommendation.

## Model-Agnostic Engine Contract

The scheduler is not allowed to encode a particular checkpoint's layer count,
expert count, top-k, projection dimensions, or quantization as global constants.
Those come from GGUF metadata for planning and the loaded tensors for execution.
Architecture-specific routing and FFN semantics remain owned by the backend;
native and hybrid expert branches share its original expert-math builder.
Choosing a different placement must not introduce a second approximate model,
drop experts, or renormalize routing probabilities per device.

Current generic dispatch derives per-layer expert/key ranges, optional biases,
fused/separate gate/up storage, per-layer active counts, cache groups, and
admission history/staging from those descriptors. Shared experts and residual
work stay outside the routed branch and are not duplicated. The upstream
activation/clamp, grouped gating, original weighting order, and precision rules
are reused rather than reimplemented as a GPT-only activation path.

GPU misses use **distinct reserved zero-expert IDs**, one per active position.
Some float CUDA expert kernels assume unique IDs within a token; using one
zero ID for every miss was only safe on the originally tested quantized path.
Graph views retain the backend/storage of their source. Explicit placement
applies to computation, not an alias of a device pointer masquerading as CPU
memory. Both rules were exposed by cross-layout fixtures and are regression
requirements, not model-name exceptions.

Version-2 installation profiles bind a trained residency plan to a canonical
model descriptor. A different MoE layout cannot inherit that checkpoint's hot
experts: it starts with an empty causal plan and bounded asynchronous staging.
Cold-start state is not trained on held-out requests. Small unsharded weight
files prefer native fitting when they fit the configured residency budget;
forcing CPU/GPU splitting on an already resident model is not automatically
beneficial. Dense models continue through native execution.

Metadata detection means **candidate capability**, not a quality or speed
certification. Loaded operations and complete routed-layer coverage must
verify before Ready. Scaled expert tensors, LoRA/Hadamard transforms, and
multi-GPU hybrid execution remain explicit unsupported cases, not successes
hidden behind fallback. A version-1 installation is migrated as its original
model-bound descriptor only; that compatibility data is not an engine-wide
architecture allowlist.

Benefit must be measured against a matched native control per model and
workload. Synthetic fixtures establish graph/storage behavior, not trained
quality or generation speed. The separate 60 tok/s 120B goal remains unmet;
no universal speedup follows merely from model-agnostic dispatch.

### Guarded Decode Overlap

The [handoff follow-up](../benchmarks/results/handoff-critical-path-summary.json)
partitions native CPU graph, GPU execution and CUDA API intervals instead of
equating API wait time with removable latency. The native path already uses
pinned intermediates and CUDA graph replay. A separate default-off interposer
tests one worker only for small scalar dispatch/copy graphs, retaining the
original expert plans and all native operations. It passes exact long-prompt
logit checks but has mixed cross-workload timings, so is not installed or
recommended as a general improvement. Existing Q/K/V graph optimization and
CUDA wait-policy screens also establish no new speedup.

The [GPU-window CPU-cache study](../benchmarks/results/gpu-window-cpu-cache-summary.json)
separately investigates warming original CPU expert weights while the GPU is
busy. Same-CCD warming substantially reduces isolated cold FFN latency, but
serial warming loses and cross-CCD warming is harmful. Timestamp-aligned
CUDA/CPU traces leave optimistic oracle headroom, not an executable scheduler:
current route-ready hints arrive too late and previous-cold-set prediction is
poor. No warmer, predictor, new placement or cache flush is enabled in the
runtime. CPU L2/L3 residency is distinct from Linux page cache and GPU VRAM.

The subsequent [live prototype](../benchmarks/results/live-cpu-cache-warmer-summary.json)
implements dispatch-triggered warming in a separate opt-in CPU interposer.
It reads bounded prefixes of original host expert buffers, never changes
graph math or routing, and cancels at CPU expert entry without waiting for
completion. Buffer release drains the reader before source memory is freed.
Current pre-admission hints can include experts later admitted to GPU; those
bytes are accounted separately. Native/noop/read request controls and exact
logit gates pass, but repeated requests show no throughput improvement.
Actual helper intervals mostly precede GPU compute, so this is not a
GPU-activity-gated scheduler or an installed optimization.

CPU utilization is not a measurement of useful expert computation. The
[CPU decode follow-up](../benchmarks/results/cpu-decode-efficiency-summary.json)
finds GNU OpenMP workers consuming CPU between graph calls. An opt-in child
`GOMP_SPINCOUNT=1000` policy preserves the checked logits/output sequences
while reducing CPU-seconds at a small measured latency cost; it is not a
faster default or an energy claim. Native pthread polling is a different
control. Diagnostic graph shims, negative VNNI kernels and alternative CPU
builds remain research tools, not runtime capabilities or installed defaults.

An optional pipeline prepares the exact three-dimensional CPU hidden-input
view as a dispatcher dependency. The scheduler can then reuse that host copy
instead of waiting for resident GPU FFN work before making another one.
This is a graph dependency change, not removal of arbitrary synchronization.

Overlap is limited to one-copy CUDA-to-CPU transitions with no new CPU split
inputs and a pure expert-operation whitelist. After allocation, all GPU
tensors/sources must be device-only on the same device; CPU tensors/sources
must be host-only, retain valid view ownership, and not write weight buffers.
Custom callbacks and unsupported operations are excluded. Evaluation callbacks
retain serialization. Pending asynchronous host-source copies are fenced
before CPU scratch reuse; already synchronous copies are complete. Existing
merge-input dependencies still reconcile the branches.

Eligible resident gate/up projections share an interleaved parent bank and
use native fused math, with original separate biases applied after splitting.
Initial and adaptive uploads distinguish original source stride from the
interleaved bank stride. Original target weights, selected experts, routing
probabilities, and CPU cold-expert math remain unchanged. Optional ranked
admission preserves recent frequency as the primary criterion; seed priority
only breaks zero-frequency ties, and active-frequency ties use recency.
All dimensions and eligibility tests remain tensor-derived.

Host-side profiling distinguishes copy/wait time, synchronous CPU submission,
GPU launch submission, actual overlap, and logical GPU projection count.
It does not measure CUDA kernel elapsed time or prove continuous occupancy.
Backend classification is cached at scheduler initialization to avoid
per-submit CUDA property-query overhead. The optional installation boolean
`decode_pipeline` enables the tested bundle for experimental manual use.
Readiness requires a configuration acknowledgement; actual overlap and
fused-projection graph observations are reported separately. Standard rollback
strips all pipeline flags. Older installation files remain unchanged.

Default reconciliation sums weighted per-device partial vectors. The alternative
keeps individual expert outputs and performs weighting/reduction on the merge
backend. Neither branch separately renormalizes router weights. Tests check both
strategies against an unsplit CPU graph and check synthetic F32 graph math against
an independent scalar implementation.

A separate stream is not a physical GPU partition, priority guarantee, or proof
of continuous GPU occupancy. Serialized/concurrent schedules are interleaved,
with CPU-only/all-GPU controls. Host-side reconciliation intervals include
transfer and dependency waits, not merely the reduction kernel.

The probe reads selected real weight slices but uses synthetic activations,
fixed expert IDs, and repeated warm weights. It does not consume live routing,
run attention/router/shared-expert/residual graphs, admit/evict experts, read
weights during timed decode, or integrate into llama-server. No model tok/s is
derived from these layer timings. A whole-model hybrid executor still requires
graph integration and representative cold/rotating-working-set validation.

## Routing Evidence Boundary

The collector uses the existing scheduler evaluation callback and a pinned
internal model tensor-map accessor. It reads sizes and host/device storage from
original model tensors, not temporary graph-stage copies. Tensor components may
have mixed placement; GPU residency cannot be reduced to a whole-layer boolean.

Only real decode selections are emitted. Prefill pruning, graph warm-up, and
sampling are not mislabeled as complete decode events. The callback honors tensor
strides when reading I32 expert IDs and checks per-layer expert domains, uniqueness,
shape, and complete token coverage. The collector changes observation overhead,
not model weights or execution policy; instrumented elapsed time is not a benchmark.

The simulator separates measured placement from hypothetical policies. Misses
execute on the CPU before any admission becomes a future GPU hit. Uploads consume
both PCIe and shared host-memory bandwidth, with explicit staging assumptions.
Future-aware hit oracles are not cost-optimal schedulers; fractional fixed-residency
bounds are not deployable expert allocations. GPU kernel/synchronization costs,
prefill cache effects, and an invariant baseline residual remain explicit modeling
limitations until measured.

The analyzer itself does not migrate individual experts or combine CPU/GPU
outputs; its plan and performance estimates remain offline. A separate,
opt-in backend patch now implements expert-ID-to-slot mapping, byte-accurate
storage, and CPU/GPU branch combination for GPT-OSS-120B. Its text plan is
exported from training references; it does not automatically execute an analyzer
JSON report. Router weights are not renormalized per device.

### Whole-Model Experimental Boundary

The whole-model prototype uses branches in the target's GGML graph, not the
layer probe's independent synthetic-input graphs. Original CPU weights remain
available for cold experts. Reserved zero GPU slots supply misses, while
masked CPU expert ID `-1` skips resident matmuls and clears their output.
Original biases and probabilities are retained. Graph dependencies order
combination before downstream computation; no physical GPU slice is reserved.

Causal admission protects all current selections. Synchronous staging needs
4,406,400 pinned bytes; the optional asynchronous pool needs 53,015,040 bytes,
with completion before reuse. Generic staging is capped at 64 MiB and recycled
only after queued DMA completes; large raw expert components are transferred
in bounded chunks. CPU-repacked components exceeding that cap are explicitly
unsupported. Direct DMA instead registers immutable source
pages with a 64 GiB cap and 24 GiB RAM-headroom checks. Stream ordering places
uploads before consumers, and cache destruction precedes backend destruction
so DMA can finish before source unregistration. CPU-repacked storage has a
lossless native-byte export for GPU uploads, not a second target quantization.

Small-block mode freezes residency throughout a multi-position call, protecting
every position's inputs. Zero-output prefill stays upstream. The DFlash opt-in
does not make block positions equivalent to accepted generation tokens.
The browser app now defaults to automatic model-specific selection. A pinned
machine-local installation profile provides the patched executable, plan, and
CUDA plugin. Version-2 profiles use bounded, architecture-prefixed GGUF
descriptors and model-bound plans, with a generic causal/staged policy for other
routed MoE layouts. Native fitting is preferred for small weight files and
dense models, with an explicit selection reason. Missing installation means standard-only
operation; malformed installation/metadata is an error, not a silent fallback.
Explicit plan/plugin CLI arguments still support a required-hybrid launch.
The hybrid process fixes the tested placement
settings, rejects incompatible API options, isolates research environment
overrides, and requires backend activation/context markers before Ready.
Its banner makes the experimental mode and unresolved quality gates visible;
the standard backend remains selectable for rollback. Small-block/DFlash hybrid flags are not
exposed by this app profile.
Strict model diagnostics and an answer screen remain failed; all measured
timings are research screens, not deployment qualification. See the
[experimental workflow](../benchmarks/README.md#experimental-whole-model-expert-dispatch).

## Local Application Boundary

`tools/vrambridge/app.py` serves dependency-free browser assets and `/api/v1`
control endpoints on loopback. It starts one `llama-server` process on a separate
loopback port with a private API key. `/v1/models` and `/v1/chat/completions` proxy
the ready backend without exposing its key or port in application responses.
The service, not the browser, owns backend lifetime and admission of one chat at
a time. Loading, ready, stopping, stopped, and error states are explicit.

Loading is asynchronous and cancellable; readiness checks, exit monitoring, and a
deadline distinguish a working backend from a process that merely started. SSE
chat is forwarded incrementally, and client disconnection closes the upstream
connection. Ordinary inference stays upstream; the UI does not contain scheduling
logic. The local control API is versioned separately from chat compatibility.

This Linux preview requires a separately installed backend and an existing GGUF.
It has no backend installer, catalog, chat persistence, or RAM-fit estimator.
The optional hybrid installation profile persists backend paths and a resident
plan selection locally; model paths, chat history, and per-run settings are not
persisted. Profile defaults are selected anew on each model load, so browser
settings from an earlier architecture are not reused by Automatic mode.
Hybrid context planning takes the minimum of installation and model limits.
The actual backend slot must match before Ready; native caps update the
reported actual context with an explicit warning. Required-hybrid model
switches omit disabled context/bank fields so a previous model's effective
limits cannot leak into the next load.
Host/origin checks and loopback binding prevent ordinary cross-site browser access;
they do not authenticate other local processes. See the
[local app guide](../tools/vrambridge/README.md).

## Intended Data Plane

Future work will introduce explicit, non-equivalent tiers:

```text
GGUF / optional NVMe
          |
   pageable system RAM
          |
 bounded pinned staging pool
          |
   PCIe asynchronous copies
          |
  VRAM resident working set
          |
    ggml CUDA execution
```

The scheduler will describe tensor identity and future use separately from physical allocation. Placement decisions must account for size, next-use distance, transfer time, reuse, allocation pressure, and correctness constraints. Host-resident weights may be computed directly by the CPU rather than transferred to the GPU. The objective is minimum critical-path latency for the user's workload, not GPU utilization alone.

## Control Plane

The control plane will consume:

- static GGUF/model architecture metadata;
- predictable dense transformer layer order;
- MoE routing observations;
- current VRAM and pinned-pool occupancy;
- measured PCIe and compute timelines;
- KV age and context ownership when KV experiments begin.

It will produce explicit execution-placement, prefetch, retention, and eviction decisions. Different policies are needed for dense decode, sparse MoE decode, and batched prefill; unsupported optimizations must preserve upstream execution. A policy interface may begin with static residency and size-aware LRU for comparison, but measured critical-path cost must determine whether an adaptive policy is enabled.

## Asynchronous Pipeline

Where transfer is justified by the workload, a candidate pipeline is double- or triple-buffered:

```text
compute stream:   layer N       layer N+1     layer N+2
copy stream:      prefetch N+1  prefetch N+2  prefetch N+3
eviction:                       retire N-1    retire N
```

CUDA events establish dependencies between copy completion and graph execution. The pinned pool is bounded and configurable; pinning all system RAM is explicitly out of scope. Adaptive prefetch depth must be based on measured copy time versus layer compute time.

Dense streaming is not a required intermediate milestone. The measured dense
baseline favors static placement; MoE execution splitting and expert residency
are the next candidates for a transferable policy. CPU/GPU partial results must
be combined at the correct graph dependencies and checked against upstream output.

## Integration Boundary

The preferred progression is linked llama API, existing tensor buffer overrides, thin instrumentation, then a custom ggml backend extension if dynamic residency cannot be expressed cleanly. Any patch must be isolated, documented against an exact upstream commit, and tested for output equivalence.

The backend scheduler is the likely long-term control point because it already reasons about backend assignment and cross-backend copies. Model-loading staging is not sufficient: VRAMBridge needs inference-time lifetime control and overlap.

## Correctness

Every placement policy must compare deterministic token/logit output with an unmodified llama.cpp baseline under equivalent seeds and sampling settings. Performance results are invalid until correctness passes within expected floating-point and quantization tolerances.

## Non-Goals for Milestone 1

- no custom CUDA kernels;
- no llama.cpp fork;
- no transparent oversubscription claims;
- no layer or expert cache;
- no KV migration;
- no NVMe inference tier.