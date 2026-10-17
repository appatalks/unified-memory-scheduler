# VRAMBridge

**The user-facing application is now [SYNAPSE](https://github.com/appatalks/synapse).**
This repository retains the full inference research, native backend patches,
benchmarks and experimental history. SYNAPSE has a clean application snapshot,
cross-platform release-candidate packaging, sidebar navigation and live monitoring.
Its standard app does not bundle or production-qualify the experimental engine.

VRAMBridge explores **LLM-aware heterogeneous memory tiering for discrete GPUs**. The project asks whether GPU VRAM can act as a high-speed working set for a much larger pool of system RAM while running GGUF language models through llama.cpp/ggml.

VRAMBridge does not convert RAM into VRAM and does not assume coherent, Apple-style unified memory. On a PCIe-attached NVIDIA GPU, host memory has different latency, bandwidth, allocation, and migration behavior. Those costs are the subject of the experiment.

**Design goal: a MoE model-agnostic execution engine, not a GPT-OSS-only
accelerator.** Expert layout, routing width, quantization, activation semantics,
and resource budgets must be data-driven. The engine should select native
full-residency or heterogeneous execution according to the model's bottleneck,
not force every model through one offload policy. This is a goal of broad
benefit, not a promise that every model is faster on every machine.

## Status

The measurement baseline is implemented, alongside a Linux-first local browser preview:

- `vrambridge-info` reports Linux, CPU, NUMA, RAM, hugepage, IOMMU, CUDA, GPU, VRAM, PCIe, and Resizable BAR information.
- The same tool measures sequential RAM bandwidth plus pageable, pinned, and device CUDA copies.
- `vrambridge-benchmark` runs reproducible llama-bench placement experiments and stores structured JSON.
- The llama.cpp integration surface is documented at an exact upstream commit.
- An optional pinned-backend routing collector and standard-library cache analyzer provide real expert traces, held-out policy replay, and explicit cost/feasibility bounds.
- The local app manages a `llama-server` subprocess, streams chat, exposes a loopback OpenAI-compatible chat endpoint, and reports actual process CPU/RAM usage with explicit tuning controls.

The browser app defaults to automatic model-specific selection.
A separate, opt-in backend patch now prototypes expert-granular CPU/GPU
dispatch, a training-only resident bank, and causal admissions for GPT-OSS-120B.
With a [version-2 local hybrid installation profile](tools/vrambridge/README.md#installing-the-default-hybrid-profile),
model-bound training plans remain isolated to their descriptors; other routed
MoE layouts can use causal cold-start admission with bounded staging.
Small weight files prefer native fitting to avoid unnecessary scheduling
overhead. Dense models still use native execution. Without the installation
profile, standard execution remains available. The prototype is still not quality-qualified for deployment. There is no implemented
disk executor, general layer streamer, or KV migration scheduler.

The first measured baseline is complete: for a 44.34 GB dense Qwen2 model on a 15.45 GiB RTX 5070 Ti, llama.cpp auto-fit static placement is fastest and autoregressive generation reaches the system-RAM bandwidth ceiling. CUDA Unified Memory is dominated by bidirectional migration. See [docs/research/baseline-calme-78b.md](docs/research/baseline-calme-78b.md) for the measurements and next-step decision.

Sparse MoE has a materially different feasibility boundary: the local 36.90 GB Qwen3.6-35B-A3B Q8_0 model generates at 41.41 tok/s with static GPU/CPU expert placement, while the full-GPU gpt-oss-20b MXFP4 reaches 217.18 tok/s. See [docs/research/frontier-moe-feasibility.md](docs/research/frontier-moe-feasibility.md) for the measured placement, active-expert accounting, and follow-up experiments.

The reviewed result index is [benchmarks/RESULTS.md](benchmarks/RESULTS.md). It links committed evidence and retains invalid-command artifacts that affected later experiment definitions.

The [downloaded-model matrix](benchmarks/RESULTS.md#october-16-model-aware-defaults-and-downloaded-model-matrix)
now tests five existing models with matched 16K context, 8/16 threads, and three
512-token generations per profile. GPT-OSS-120B's median improved from
**25.69 to 39.46 tok/s** with hybrid direct DMA. The other four architectures
ran through standard execution, not the custom dispatcher; this establishes
compatibility, not cross-model hybrid gains. Failed exact-answer/AST checks
and GPT adaptive output-hash variation are retained in the evidence.

The subsequent [generic MoE graph tests](benchmarks/README.md#generic-moe-graph-compatibility)
exercise seven untrained synthetic model families, variable expert/top-k counts,
seven storage types, fused/unfused projections, shared experts, optional biases,
and mixed dense/MoE layers. Numerical failures are retained, including quantized
cases that also fail native-placement comparisons. These are graph/correctness
tests, not evidence of trained-model quality or cross-model speedups. The
refactored real 120B frozen path matched its earlier hybrid reference exactly
at all 96 teacher-forced positions.

## Routing And Cache Feasibility

The trace-first research path is implemented: collect real decode expert IDs,
validate complete layer/token coverage, and compare measured static placement,
training-only hot sets, LRU/LFU, and explicitly limited oracle models. Reports
include migration and shared-RAM-bus costs, warm-up, partial tensor placement,
source hashes, and a trained residency plan. They never claim measured acceleration.
Replay also includes causal cost-aware admission: migration must repay its
estimated transfer cost and the benefit lost through eviction.

An initial 1,536-token GPT-OSS-120B collection is complete and reproducible.
It shows useful locality, but a fixed hot set alone does not clear the 60 tok/s
goal under the measured bandwidth inputs. See the [collection/replay guide](benchmarks/README.md#routing-collection-and-cache-replay)
and [measured trace findings](docs/research/frontier-moe-feasibility.md#real-routing-and-costed-replay).

An optional [hybrid-layer execution probe](benchmarks/README.md#isolated-cpugpu-expert-execution)
now runs real expert weights on CPU and GPU, stages CPU outputs through a bounded
pinned buffer, and reconciles them on a separate GPU stream. It checks partial
vector merging and full GPU reduction against an unsplit reference. This is an
isolated warm-cache prototype with synthetic activations, not an integrated
model executor or a demonstrated 60 tok/s result.

The separate [whole-model experimental patch](benchmarks/README.md#experimental-whole-model-expert-dispatch)
has generated actual 512-token responses at roughly 33 tok/s with frozen
residency, 35-36 tok/s with staged adaptive residency, and 39-40 tok/s with
direct-DMA admissions. These are short screening
measurements, not a qualified result across workloads. Full-logit diagnostics
against CPU-only placement still fail their strict numerical gate, and an
answer-quality screen found a quantitative error. **The 60 tok/s goal is not
met. The default-supported hybrid path remains visibly experimental and requires
independent answer checking; it is not a qualified production replacement.**

## Local Browser App (Linux Preview)

Requires Python 3.10+, a browser, an existing local GGUF model, and a compatible
`llama-server` build. No Node.js, Python packages, or VRAMBridge C++ build is required.

```bash
python3 tools/vrambridge/app.py --llama-server /path/to/llama.cpp/build/bin/llama-server
```

The launcher opens `http://127.0.0.1:8080`. Enter your GGUF path, click **Load model**,
then chat or copy the API base URL into an external client. Use `--port 0` to choose
a free port, or `--no-browser` to open it manually. `LLAMA_SERVER` can also select
the executable. Missing backends and loading errors are shown explicitly.

Chat supports separate reasoning output, an adjustable 8,192-token default
generation budget, token-limit warnings, and backend-reported live tokens/second.
The output budget includes reasoning and is separate from the loaded context.

This preview does not install llama.cpp, download models, or predict RAM fit.
Standard automatic placement uses upstream fitting; CPU-only mode is also
available. The installed hybrid profile selects the separate patched
executor and does not qualify it for production.
[Setup, API, and limitations](tools/vrambridge/README.md).

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

The original RAM read probe is single-threaded. To also measure disjoint,
concurrent reads of the same total-sized buffer:

```bash
./build/cpu-release/vrambridge-info --json --size-mib 1024 --iterations 10 --ram-threads 4
```

This adds `ram_parallel_read` and reports CPU thread counts without changing the
default benchmark list. Neither read microbenchmark proves sparse expert-kernel
bandwidth or the machine's absolute memory ceiling.

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

The next scheduling experiments will test measured CPU/GPU execution placement and MoE expert residency against static baselines. Dense streaming is not a prerequisite: the dense baseline already shows that naive per-token weight transfer would lose on the measured hardware. A common framework should support workload-specific policies and retain ordinary llama.cpp behavior when an optimization is unsupported or unproven.

See [docs/architecture.md](docs/architecture.md), [docs/benchmarking.md](docs/benchmarking.md), [docs/research/upstream-llama.md](docs/research/upstream-llama.md), [docs/research/baseline-calme-78b.md](docs/research/baseline-calme-78b.md), and [docs/research/frontier-moe-feasibility.md](docs/research/frontier-moe-feasibility.md) for the design, upstream findings, and measured dense and MoE baselines.

## License

VRAMBridge is available under the MIT License.