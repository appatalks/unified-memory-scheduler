# Architecture

## Current Milestone

The implemented system has two deliberately small components:

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
```

No inference data path has been replaced. llama.cpp remains the source of GGUF parsing, tensor semantics, graph construction, kernels, sampling, and correctness.

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

The scheduler will describe tensor identity and future use separately from physical allocation. Placement decisions must account for size, next-use distance, transfer time, reuse, allocation pressure, and correctness constraints. VRAM is the compute working set; RAM is backing storage.

## Control Plane

The control plane will consume:

- static GGUF/model architecture metadata;
- predictable dense transformer layer order;
- MoE routing observations;
- current VRAM and pinned-pool occupancy;
- measured PCIe and compute timelines;
- KV age and context ownership when KV experiments begin.

It will produce explicit prefetch, retention, and eviction decisions. A policy interface may begin with size-aware LRU for comparison, but transformer-aware next-use scheduling is expected to be the serious baseline.

## Asynchronous Pipeline

For dense layers, the target pipeline is double- or triple-buffered:

```text
compute stream:   layer N       layer N+1     layer N+2
copy stream:      prefetch N+1  prefetch N+2  prefetch N+3
eviction:                       retire N-1    retire N
```

CUDA events establish dependencies between copy completion and graph execution. The pinned pool is bounded and configurable; pinning all system RAM is explicitly out of scope. Adaptive prefetch depth must be based on measured copy time versus layer compute time.

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