# Benchmarking

## Principles

Change one placement variable at a time, retain failures, run warm-ups, and report distributions rather than only best results. Keep model, prompt token count, generation token count, context, batch sizes, threads, CPU affinity, sampling, and background load fixed within a comparison.

Every result must identify:

- model path, byte size, and fingerprint;
- llama.cpp commit and build options;
- exact command and environment delta;
- host/GPU hardware report;
- prompt and generation workload;
- repetitions and warm-up policy;
- raw backend measurements and stderr.

## Milestone 1 Matrix

1. CPU: all model layers and compute in system RAM (`-ngl 0 -dev none`).
2. Manual offload: a fixed `--n-gpu-layers` that fits without fallback.
3. Auto-fit: current upstream `--fit-target` behavior.
4. Unified Memory: all layers requested with managed CUDA allocations.
5. Tensor placement: explicit, documented regular-expression overrides.
6. MoE: static CPU expert layers, when the model contains experts.

Run prompt processing and token generation in the same matrix because their compute/transfer balance differs substantially.

## Required Metrics

- prompt tokens/s;
- generation tokens/s;
- time to first token when using an interactive runner;
- model load time;
- GPU utilization and memory-controller utilization;
- VRAM and system RAM high-water marks;
- PCIe RX/TX throughput;
- CPU utilization;
- major/minor faults and managed-memory migrations where observable.

`llama-bench` supplies throughput and load measurements. Milestone 2 telemetry will correlate those with NVML, `/proc`, sysfs, perf, and CUDA events.

## Tool

`tools/benchmark/vrambridge-benchmark.py` asks llama-bench for native JSON and wraps it with provenance. By default it runs CPU, manual GPU offload, fit, and Unified Memory. While each run is active it also samples process RSS/CPU time and, when available, `nvidia-smi dmon` GPU utilization, memory-controller utilization, framebuffer use, power, and PCIe RX/TX. Results are atomic JSON files under `benchmarks/results/`.

The partial model fingerprint hashes the first and last MiB plus records size. It avoids reading a very large model solely for provenance; it is not a cryptographic identity for the complete GGUF.

## Interpreting Bandwidth

Pinned versus pageable transfer results include host-side staging behavior because timing starts before the CUDA copy call and ends after stream synchronization. This is intentional: it represents application-visible transfer cost.

Device-to-device copy reports payload bytes divided by elapsed time. GDDR theoretical bandwidth counts memory transfers, while a copy reads and writes each payload byte. Compare conventions explicitly.

## Measured Baseline

The first 78B dense-model matrix and bottleneck analysis are recorded in [research/baseline-calme-78b.md](research/baseline-calme-78b.md). It combines only successful comparable slices while retaining the original invalid CPU and failed Unified Memory attempts as evidence.

The local frontier-MoE matrix is recorded in [research/frontier-moe-feasibility.md](research/frontier-moe-feasibility.md). It measures full-GPU, static CPU-expert, and automatic GPU/CPU expert placement through the same llama.cpp benchmark and telemetry path.