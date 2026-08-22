# Roadmap

## Milestone 1: Baseline Infrastructure

- hardware/CUDA discovery;
- host and CUDA memory bandwidth tests;
- exact upstream llama.cpp audit;
- structured llama-bench experiment harness;
- CPU-only CI and focused tests.

Exit criterion: run a supplied dense GGUF larger than VRAM through all applicable baseline configurations and identify the limiting resource with synchronized telemetry.

## Milestone 2: Telemetry

Add NVML and process sampling, PCIe counters where available, CUDA event tracing, and a minimal llama.cpp layer/stage hook. Produce a common timeline keyed to prompt processing and generation.

## Milestone 3: Bounded Staging Prototype

Implement a configurable pinned host pool and a standalone copy/compute overlap microbenchmark. Prove overlap on representative layer-sized buffers before integrating model execution.

## Milestone 4: Dense Layer Streaming

Add explicit layer residency, correctness comparisons, and fixed-depth prefetch. Compare against manual offload and CUDA Unified Memory; then test adaptive depth.

## Milestone 5: MoE Expert Cache

Instrument routing, establish static CPU-MoE baseline, and test whether selected experts can arrive before execution. Begin with measured LRU/size-aware policies before predictive routing.

## Deferred

KV cache tiering and NVMe storage remain deferred until weight/expert streaming is correct and faster than upstream baselines.