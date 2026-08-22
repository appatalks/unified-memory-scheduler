# VRAMBridge

VRAMBridge explores **LLM-aware heterogeneous memory tiering for discrete GPUs**. The project asks whether GPU VRAM can act as a high-speed working set for a much larger pool of system RAM while running GGUF language models through llama.cpp/ggml.

VRAMBridge does not convert RAM into VRAM and does not assume coherent, Apple-style unified memory. On a PCIe-attached NVIDIA GPU, host memory has different latency, bandwidth, allocation, and migration behavior. Those costs are the subject of the experiment.

## Status

Milestone 1 establishes measurement infrastructure only:

- `vrambridge-info` reports Linux, CPU, NUMA, RAM, hugepage, IOMMU, CUDA, GPU, VRAM, PCIe, and Resizable BAR information.
- The same tool measures sequential RAM bandwidth plus pageable, pinned, and device CUDA copies.
- `vrambridge-benchmark` runs reproducible llama-bench placement experiments and stores structured JSON.
- The llama.cpp integration surface is documented at an exact upstream commit.

No dynamic tensor cache, layer streamer, expert cache, or KV migration is implemented yet. Optimization starts only after a GGUF baseline identifies the actual bottleneck.

The first measured baseline is complete: for a 44.34 GB dense Qwen2 model on a 15.45 GiB RTX 5070 Ti, llama.cpp auto-fit static placement is fastest and autoregressive generation reaches the system-RAM bandwidth ceiling. CUDA Unified Memory is dominated by bidirectional migration. See [docs/research/baseline-calme-78b.md](docs/research/baseline-calme-78b.md) for the measurements and next-step decision.

## Build

Requirements are CMake 3.24+, a C++20 compiler, and optionally the CUDA toolkit.

```bash
cmake --preset release
cmake --build --preset release
ctest --preset release
```

For a portable build without CUDA:

```bash
cmake --preset cpu-release
cmake --build --preset cpu-release
ctest --preset cpu-release
```

## Hardware Report

```bash
./build/release/vrambridge-info
./build/release/vrambridge-info --json
./build/release/vrambridge-info --size-mib 512 --iterations 10
./build/release/vrambridge-info --json --output benchmarks/results/hardware.json
```

Transfer tests allocate only the requested buffer size in pageable RAM, pinned RAM, and VRAM. The default is 256 MiB with five measured copies after a warm-up copy.

Reported GB/s uses decimal gigabytes. Device-to-device copy bandwidth reports bytes copied per second, not aggregate read-plus-write traffic. The theoretical VRAM figure is aggregate DDR transfer rate and is therefore not directly comparable without accounting for this convention.

## llama.cpp Baseline

Build a CUDA-enabled llama.cpp checkout and retain its exact Git commit. Then run:

```bash
python3 tools/benchmark/vrambridge-benchmark.py /models/model.gguf \
  --llama-bench /path/to/llama.cpp/build/bin/llama-bench \
  --llama-source /path/to/llama.cpp \
  --gpu-layers 30
```

The default matrix contains:

- CPU placement: `-ngl 0 -dev none`
- conventional manual GPU offload: `-ngl <N>`
- automatic fitting: `-fitt 1024 -fitc 4096`
- CUDA managed allocation fallback: `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 -ngl -1`

Optional experiments include current tensor buffer overrides and CPU MoE placement:

```bash
python3 tools/benchmark/vrambridge-benchmark.py model.gguf \
  --llama-bench /path/to/llama-bench \
  --config tensor-placement \
  --tensor-override 'blk\.0\..*=CPU'

python3 tools/benchmark/vrambridge-benchmark.py model.gguf \
  --llama-bench /path/to/llama-bench \
  --config cpu-moe --cpu-moe-layers 20
```

Each run records the exact command, environment delta, model size and partial fingerprint, upstream commit, stdout, stderr, status, and native llama-bench JSON. Results are written atomically under `benchmarks/results/`. A failed configuration is evidence and is retained rather than discarded.

## Scientific Questions

The first experiment is to determine the fastest current llama.cpp method for a model substantially larger than VRAM, then locate its limiting resource using tokens/s, prompt processing, load time, utilization, memory, PCIe traffic, and page migration measurements.

Only after that baseline will VRAMBridge test whether bounded pinned staging and asynchronous transformer-aware prefetch can outperform static CPU/GPU placement and CUDA Unified Memory. MoE expert caching follows only after dense layer streaming is correct and measured.

See [docs/architecture.md](docs/architecture.md), [docs/benchmarking.md](docs/benchmarking.md), [docs/research/upstream-llama.md](docs/research/upstream-llama.md), and [docs/research/baseline-calme-78b.md](docs/research/baseline-calme-78b.md) for the design, upstream findings, and measured baseline.

## License

VRAMBridge is available under the MIT License.