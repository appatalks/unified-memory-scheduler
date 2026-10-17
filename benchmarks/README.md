# Benchmarks

Schemas are versioned under `schema/`. Generated hardware and llama.cpp results go under `results/` and are not committed by default.

Tests never download models. Benchmark commands require paths to local GGUF files.

[RESULTS.md](RESULTS.md) is the reviewed benchmark ledger. It links each committed evidence artifact, distinguishes valid measurements from retained failed commands, and records the fixed comparison conditions. New generated JSON remains ignored until it has been reviewed and added to that ledger.

Raw files under `results/`, including nested output and logs, are ignored unless explicitly allowlisted in [`.gitignore`](../.gitignore). Before allowlisting evidence, review hostnames, model paths, command arguments, stdout, and stderr for credentials and private machine or user identifiers. Use neutral placeholders for identifying details without changing measurements. Only sanitized environment examples should be committed; local environment files, model weights, and user-specific CMake presets are ignored.

Ignore rules do not remove already tracked content or historical copies. Before publishing the repository, review both the current files and Git history, including commit author and committer identities.

## Routing Collection And Cache Replay

The optional `vrambridge-trace` collector uses the official evaluation callback
to copy actual `ffn_moe_topk` I32 expert IDs after evaluation. It inspects the
loaded model's original expert tensors, including biases, to record sizes and
host/device placement. Inspecting temporary graph copies would incorrectly count
prefill staging as permanent GPU residency.

This adapter currently requires official llama.cpp commit
`620d2225cc436ab0b001e3200e590e3b286ef9ab` and matching C++ shared libraries.
It uses a pinned internal tensor-map accessor because the stable API does not
expose that placement map. Do not mix fork headers, other revisions, or incompatible
C++ ABIs. The normal CPU build and browser app do not require these dependencies.
Replace the example paths below with real local source/library/model paths:

```bash
cmake -S . -B build/trace-release \
  -DCMAKE_BUILD_TYPE=Release -DVRAMBRIDGE_ENABLE_CUDA=OFF \
  -DVRAMBRIDGE_LLAMA_SOURCE=/path/to/llama.cpp \
  -DVRAMBRIDGE_LLAMA_LIBRARY_DIR=/path/to/matching/shared-libraries
cmake --build build/trace-release --target vrambridge-trace --parallel 2
```

Unload the app's model before collecting so two processes do not compete for
VRAM. The corpus contains distinct training and held-out prompts in coding,
reasoning, and prose, with a 256-token generation limit per request:

```bash
umask 077
./build/trace-release/vrambridge-trace \
  --trace-corpus benchmarks/routing-corpus.jsonl \
  --trace-output benchmarks/results/routing/run.jsonl \
  --model /models/gpt-oss-120b-MXFP4-00001-of-00002.gguf \
  --ctx-size 16384 --n-gpu-layers auto --fit on --fit-target 1024 \
  --threads 4 --threads-batch 16
```

The collector fixes greedy sampling, a stable system message/date, and low
reasoning effort. LoRA adapters, expert caches, and speculative drafts are rejected.
Prompts use the model's Jinja template; raw prompt/output text and generated token
IDs are not stored. Generated routing includes reasoning tokens.

Only decode is captured. Prefill can prune the final FFN to requested output
positions, so it must not be falsely presented as a complete per-token routing
trace. Each sampled non-EOS token is evaluated, including the last token at the
limit, giving exactly the declared number of routed generated tokens.
All expected MoE layers must appear once per token. Unsupported tensor layouts,
bad IDs, missing observations, and I/O failures are explicit errors.
GPU collection checks for an initialized GPU device, not just a CUDA-capable
library. If initialization fails, it refuses to load the model; intentional
CPU-only collection requires explicit `--n-gpu-layers 0` and is not a GPU-fit
baseline.

After each request, the collector clears context memory and replays the same
prompt with observation inactive. Every generated non-EOS token and the stop
condition must match before `token_sequence_equivalent: true` is emitted.
This checks observer impact in that context, not answer quality, logit tolerance,
or equivalence with a separately created callback-free context. Older traces
without the field remain analyzable but report equivalence as unverified.
Long prompts are processed in bounded logical batches.

Collection adds synchronization and I/O: **its elapsed time is not a throughput
benchmark**. Final traces are published atomically without overwriting existing
files. A failed collection leaves a `.part` file, which analysis rejects.
Reload the app's model after collection.

Replay with this host's previously measured inputs, replacing them for other
hardware:

```bash
python3 tools/benchmark/vrambridge-routing.py \
  benchmarks/results/routing/run.jsonl \
  --output benchmarks/results/routing/cache-report.json \
  --ram-gbps 42.16 --pcie-gbps 28.52 \
  --baseline-tok-s 22.12 --target-tok-s 60
```

`vrambridge-routing` uses only the Python standard library. It validates complete
traces and byte accounting before replaying:

- **Measured static:** the actual component-level host/device placement, including
  partially offloaded expert groups.
- **Training-static:** a fixed full-expert hot set selected from training references
  only. The report exports the exact byte-budgeted plan; it is **not** an executable
  llama.cpp offload configuration.
- **LRU/LFU:** byte-capacity caches with repeat admission (`--admit-after`, default
  two observations). Current misses execute on the CPU. Net cache changes are
  committed after the layer, never counted as hits for that same access.
- **Cost-aware admission:** causal recent-use counts over `--payback-tokens`
  generated decode steps (default 64, excluding prefill). A candidate is admitted
  only when estimated CPU-read savings minus supplied GPU execution cost exceed
  its serial upload/staging cost plus predicted lost savings of the evicted
  experts. The same recent window is extrapolated into the future; no future
  trace accesses are read. Window size is an assumption, not a trained predictor
  or a promise that admissions actually pay back. Rejected admissions, training
  uploads, and every accepted copy are accounted for separately.
- **Belady hits:** a future-aware hit oracle for equal-size experts and end-of-layer
  demand admission. It can bypass a miss, but it is not a latency-optimal or
  proactively prefetching oracle. Unequal expert sizes are explicitly unsupported
  for this policy rather than mislabeled as optimal.
- **Fractional fixed oracle:** a future-aware fractional knapsack bound for fixed
  residency, including variable expert sizes. Fractional experts are not executable,
  and this bound does not rule out all adaptive algorithms.

Costs include cold CPU reads, admitted weight bytes, PCIe transfer, staging RAM
traffic, and optional measured GPU/synchronization costs. The default
`--staging-ram-passes 2` represents one read+write staging copy; DMA adds another
RAM read. Optimistic overlap includes CPU and DMA contention for the shared RAM
bus instead of treating their bandwidth as independent. `--staging-ram-passes 0`
is an explicitly optimistic already-pinned sensitivity case, not an implemented
copy-elision mechanism.

`--gpu-expert-us` and `--layer-sync-us` accept measured costs. If omitted, they are
zero only as a stated optimistic assumption. Optional baseline calibration fits
an unchanged residual from training accesses; it does not validate a new policy
and can be wrong in either direction. Warm-up uploads, initial cache transfers,
per-request/category results, reuse distances, and token reuse gaps are reported.
Prefill cache effects and live kernel equivalence are not modeled.

No report can declare the production target met. The formal schemas are:

- [Corpus record](schema/vrambridge-routing-corpus.schema.json)
- [Trace record](schema/vrambridge-routing-trace.schema.json)
- [Analysis report](schema/vrambridge-routing-analysis.schema.json)

JSONL record schemas do not replace cross-record validation. Unit tests include
incomplete traces, mixed component placement, variable expert sizes, byte budgets,
admission, shared-bus traffic, and a small exhaustive check of the hit oracle:

```bash
python3 tests/test_routing_analysis.py tools/benchmark/vrambridge-routing.py
ctest --test-dir build/trace-release -R 'routing-analysis|trace-' --output-on-failure
```

Raw traces and generated reports are ignored under `results/`. Even without text,
routing, model names, and request labels can reveal information; keep private
corpora local and review artifacts before publication.

### Refresh Calibration After Restart Or Placement Changes

Auto-fit can change tensor placement when desktop/driver VRAM usage changes.
Record a new trace and remeasure the uninstrumented control rather than assuming
an old layout or routing sequence still applies. Source and input hashes in the
report identify which collection actually supplied its routing.

`vrambridge-info --ram-threads N` optionally adds a parallel sequential-read
probe. It keeps the total buffer fixed and divides it into disjoint contiguous
chunks; the reported byte count is total bytes, not bytes per worker.
Worker creation and readiness are outside timing; wake-up and joins are included.
Each worker consumes every assigned element on every pass and the aggregate
checksum is verified. Threads are OS-scheduled, not pinned, and memory is
initially touched by the allocating thread. Use a buffer larger than LLC and
repeat runs. The unchanged default still records single-thread read/write only.

For example, on the CPU-only build:

```bash
./build/cpu-release/vrambridge-info --json --size-mib 1024 --iterations 10 --ram-threads 4
```

A new read microbenchmark is a calibration input, not a measured model-kernel
speedup. Keep old and refreshed estimates separately, with their respective
layout, bandwidth, and uninstrumented control measurements.

## Live Execution Profile Screening

`vrambridge-execution-sweep` is an opt-in local-app runner, separate from routing
collection. It does not modify backend source or global CPU/GPU settings.
Requires an idle official-backend model loaded in the local app, with automatic
placement, no draft, and no explicit expert cache. Save ongoing work first:
the sweep reloads that model between configurations and temporarily occupies
the one chat slot.

```bash
umask 077
python3 tools/benchmark/vrambridge-execution-sweep.py \
  --allow-reloads --tokens 512 --repetitions 2 \
  --output benchmarks/results/routing/execution-screen.json
```

The runner compares a control, CPU execution of host-tensor operations,
zero CPU polling, their combination, and 512 MiB fit reserves with/without
host-operation offload, plus a raised minimum host-operation GPU batch of 128.
Select a smaller set with `--profiles control batch128`.
Target model weights, requested context, generation/
prompt threads, and other original settings are preserved. No KV quantization
or model download is enabled. Smaller reserves increase OOM risk and are never
selected automatically.

Each load must report the required actual `n_ctx_slot` in backend output before
sampling starts. Every run requires a complete stream with exactly the requested
output-token count, backend timings, and independent wall-clock timing. The
fixed prompt, temperature zero, seed 42, low reasoning effort, disabled prompt
caching, and EOS-ignore setting make this a screening workload rather than a
natural-answer quality test. The report records repeated text hashes and
whether they match the control; numerical-path changes are not hidden.

The runner restores the original model settings in a `finally` block, including
after ordinary candidate failures or Ctrl+C. It refuses to interrupt an active
chat. Restoration errors are explicit and make the command fail. Force-killing
the runner, restarting the computer, or using the app concurrently can prevent
restoration; this is not a multi-user scheduler.

Outputs are written after each sample so failed candidates remain evidence.
Original model paths are not published in the report; the model basename and
explicit overrides are retained. The installed name is
`vrambridge-execution-sweep`. A profile is never auto-enabled or marked as meeting
the production target based on this short fixed-prompt screen. Qualify candidates
separately on coding, reasoning, prose, and longer inputs.

The official CUDA threshold control comes from the pinned backend's
`ggml_cuda_op_offload_min_batch()` implementation, not a new VRAMBridge kernel.
Short-prompt latency benefits must not be described as a decode-speed improvement.
In the first qualification, threshold 128 failed the quantitative-answer check;
retain the default control and the failure evidence rather than selecting solely
by the best first-token time.

## Isolated CPU/GPU Expert Execution

The optional `vrambridge-hybrid-layer` target uses the same pinned source/library
configuration as the routing collector. Unlike the simulator, it actually
executes expert matmuls on CPU and GPU, transfers the CPU result, and combines
the contributions on a separate GPU backend stream. It does not reserve SMs
or integrate a new executor into llama-server.

```bash
cmake --build build/trace-release --target vrambridge-hybrid-layer --parallel 2
ctest --test-dir build/trace-release -R hybrid-layer --output-on-failure
umask 077
./build/trace-release/vrambridge-hybrid-layer \
  --model /models/gpt-oss-120b-MXFP4-00001-of-00002.gguf \
  --layer 0 --threads 4 --samples 5 --repetitions 50 \
  --output benchmarks/results/routing/hybrid-layer0.json
```

The real-weight adapter deliberately supports the verified 2880-wide
GPT-OSS-120B MXFP4 layout only. It reads four expert groups (IDs 0, 7, 42, 113)
including their original F32 biases from the selected layer across GGUF shards.
It does not load the whole model or modify its weights. Keep inference idle
while timing; the working browser model need not be unloaded when enough GPU
memory is free. Allocation failures and unavailable GPU/backend operations are
explicit errors, not CPU fallbacks.

**Scope:** activations are deterministic synthetic vectors and routing weights
are deliberately asymmetric `[0.05, 0.15, 0.30, 0.50]`. This is a warm, isolated
layer probe, not actual full-model hidden states, adaptive residency, cold-cache
bandwidth, or a generation-throughput benchmark. Selected weights may fit CPU
or GPU caches; never multiply the measured layer time by 36 and call it tok/s.
The fixed residency plan exported by the simulator is not consumed here.

All five CPU/GPU splits are tested with serialized and concurrent schedules.
Concurrent mode submits the GPU graph asynchronously before computing the CPU
branch. Serial mode waits for GPU completion first. Cross-backend copies
establish the dependency before GPU reconciliation; they may internally fall
back to synchronization. No unfenced buffer reuse is assumed.

The default strategy applies the original routing weights and reduces within
each device, copies one CPU vector through a reusable 11,520-byte pinned buffer,
then adds both vectors on the GPU. It never normalizes weights separately in
the two branches.

`--gpu-reduce` instead copies unreduced CPU expert outputs (up to 46,080 bytes)
and performs weighting and reduction on the reconciliation backend:

```bash
./build/trace-release/vrambridge-hybrid-layer \
  --model /models/gpt-oss-120b-MXFP4-00001-of-00002.gguf \
  --layer 35 --threads 4 --samples 5 --repetitions 50 --gpu-reduce \
  --output benchmarks/results/routing/hybrid-layer35-gpu-reduce.json
```

The probe preserves expert biases and the OpenAI SwiGLU alpha/clamp semantics.
Every warm-up and timed execution is compared with an unsplit CPU reference,
using both maximum absolute error and relative L2 gates:

```text
max_abs_error <= 0.00001 + 0.0001 * reference_max_abs
relative_l2_error <= 0.0001
```

`--self-test` uses tiny F32 tensors and additionally checks an independent scalar
reference, including bias, activation, routing weights, and expert summation.
The default self-test reduces two CPU partitions manually; `--self-test
--gpu-reduce` tests the reconciliation graph on CPU. Neither requires GPU
compute. These checks do not establish full-model token/logit equivalence.

Timing order alternates serial/concurrent first within each repetition. Reports
include per-iteration wall times, CPU branch time, GPU input/dispatch time, and
reconciliation-plus-dependency-wait time. These are host intervals, not isolated
CUDA kernel times; component medians need not sum to the total median.
Input copies, dispatch, CPU computation, staging, merge, and final synchronization
are included. Weight loading/graph construction and validation readback are not.
All-GPU and all-CPU controls distinguish schedule/synchronization overhead from
benefits that actually require mixed execution.

Reports use the [hybrid-layer schema](schema/vrambridge-hybrid-layer.schema.json),
retain the compiled source hash, and refuse to overwrite final or partial files.
Failed numerical checks produce a report with `all_checks_passed: false` and a
nonzero exit status. No report claims full-model acceleration.

## Experimental Whole-Model Expert Dispatch

This is a research workflow. Version-2 installations select a model-bound
trained plan when its descriptor matches, or a generic causal/staged plan
for other routed MoE layouts. Native fitting remains preferable for small
resident weight files. The experimental path is not quality-qualified.
The dispatcher now derives layer/expert/top-k counts, dimensions, component
sets, and quantizations from loaded tensors. It reuses the backend's original
expert math, including fused gate/up, optional biases, activation variants,
and original weighting order. It does not prune experts, change target weights,
or implement disk computation. Read the
[measurements and failed quality gates](../docs/research/frontier-moe-feasibility.md#experimental-whole-model-expert-dispatch)
before using it. The 60 tok/s goal is not met.

The generated
[hybrid patch](../third_party/patches/llama-b11669-hybrid-experts.patch)
is against llama.cpp commit
`620d2225cc436ab0b001e3200e590e3b286ef9ab`. The upstream
[MIT license](../third_party/patches/LLAMA-LICENSE) applies. Apply it only to a
separate checkout at that exact commit; do not replace the working app's
executable or mix different-revision libraries:

```bash
git -C /path/to/experimental/llama.cpp apply --check \
  "$PWD/third_party/patches/llama-b11669-hybrid-experts.patch"
git -C /path/to/experimental/llama.cpp apply \
  "$PWD/third_party/patches/llama-b11669-hybrid-experts.patch"
```

Build that checkout using its existing CUDA build workflow. Alternatively, the
tested setup built the shared CPU libraries and server with
`-DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DGGML_NATIVE=OFF`
and loaded the **matching official** CUDA plugin through
`GGML_BACKEND_PATH=/path/to/libggml-cuda.so`. That variable names a library file,
not a directory. The CPU plugin must be from the patched build: the ordinary
CPU matmul does not understand masked expert ID `-1`. This patch does not
require a modified CUDA kernel. The separate DFlash attention-bias patch is
not required for non-speculative experiments.

Configure the optional tools against the patched source/libraries:

```bash
cmake -S . -B build/hybrid-tests \
  -DCMAKE_BUILD_TYPE=Release -DVRAMBRIDGE_ENABLE_CUDA=OFF \
  -DVRAMBRIDGE_LLAMA_SOURCE=/path/to/experimental/llama.cpp \
  -DVRAMBRIDGE_LLAMA_LIBRARY_DIR=/path/to/patched/shared-libraries \
  -DVRAMBRIDGE_TEST_MASKED_IDS=ON
cmake --build build/hybrid-tests --parallel 4
ctest --test-dir build/hybrid-tests --output-on-failure
```

The opt-in test configuration additionally builds `vrambridge-logit-check` and
the raw/repacked masked-matmul test. No models are downloaded by these tests.
The repacked primitive test requires the x86 AVX2 eight-row MXFP4 layout.

A plan is a UTF-8 text file with `layer expert` integer pairs, one per line,
and optional whole-line `#` comments. Duplicates, out-of-range entries, malformed
rows, and plans exceeding the allocated bank are errors. A band of distinct
zero slots is reserved for misses (one per maximum active expert position),
and allocation includes a guard slot; the entire nominal budget
is not available for plan entries.

For the measured 800-group starting plan, first validate a complete routing
trace with the analyzer above, then export **training references only**:

```bash
umask 077
python3 - benchmarks/results/routing/run.jsonl benchmarks/results/routing/hot800.txt <<'PY'
from collections import Counter
import json
import sys

counts = Counter()
split = None
with open(sys.argv[1]) as source:
    for line in source:
        record = json.loads(line)
        if record["type"] == "request_start":
            split = record["split"]
        elif record["type"] == "route" and split == "train":
            counts.update((record["layer"], expert) for expert in record["experts"])
if len(counts) < 800:
    raise SystemExit("Insufficient distinct training expert groups")
with open(sys.argv[2], "x") as output:
    for (layer, expert), _ in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:800]:
        output.write(f"{layer} {expert}\n")
PY
```

Unload the normal app before running another large model process. A manual
frozen-bank invocation is:

```bash
GGML_BACKEND_PATH=/path/to/matching/libggml-cuda.so \
VRAMBRIDGE_HYBRID_PLAN="$PWD/benchmarks/results/routing/hot800.txt" \
/path/to/patched/llama-server \
  --model /models/gpt-oss-120b-MXFP4-00001-of-00002.gguf \
  --host 127.0.0.1 --port 8081 --ctx-size 16384 --parallel 1 \
  --n-gpu-layers all --cpu-moe --fit off --no-repack --no-op-offload \
  --threads 4 --threads-batch 16 --moe-cache-mib 10240
```

Use a private per-launch API key when controlling a research server, never a
public bind. The example budget is hardware-specific, not a fit guarantee;
automatic fitting is deliberately disabled. Restore the original app/model
after the private process closes, including on failed tests.

Optional environment switches, scoped to the private process:

| Switch | Behavior |
| --- | --- |
| `VRAMBRIDGE_HYBRID_ADAPT=1` | Causal recent-use admissions; cold experts otherwise execute on CPU |
| `VRAMBRIDGE_HYBRID_ASYNC=1` | With adaptive mode, tensor-sized staging capped at 64 MiB, safe pool recycling, and ordered asynchronous uploads |
| `VRAMBRIDGE_HYBRID_REPACK=1` | Allow CPU repacking; omit `--no-repack`, add `--no-host --load-mode none`; lossless native-layout export for GPU uploads |
| `VRAMBRIDGE_HYBRID_DIRECT=1` | With adaptive/asynchronous raw layout and `--no-host`, direct DMA from immutable registered source pages, without staging copies |
| `VRAMBRIDGE_HYBRID_PLAN=auto` | Empty causal initialization; no checkpoint-specific trained plan is reused |
| `VRAMBRIDGE_HYBRID_BATCH=4` | Allow up to four target positions per custom graph (range 1-32); residency is frozen for every multi-position call |
| `VRAMBRIDGE_HYBRID_DFLASH=1` | Allow the specifically selected `--spec-type draft-dflash` experiment; requires the separate attention-bias patch and a smaller resident bank to leave draft-model headroom |
| `VRAMBRIDGE_HYBRID_PROFILE=1` | Host-side split/copy/submit counters; GPU submit time is launch latency, not GPU execution time |
| `VRAMBRIDGE_HYBRID_COPY_PROFILE=1` | Requires `PROFILE=1`; separates copy directions and host intervals for destination waits, source synchronization, synchronous copy calls, asynchronous attempts and callbacks |
| `VRAMBRIDGE_HYBRID_PAYBACK=1` | Bounded admission-lifetime accounting: future hits, evicted reuse and censored live entries; combined metadata cap 64 MiB |
| `VRAMBRIDGE_HYBRID_ASYNC_IDS=1` | Experimental CUDA-only handoff using private per-layer GPU ID storage and bounded pinned staging; completion events protect host staging reuse |
| `VRAMBRIDGE_HYBRID_CPU_EMPTY=1` | Raw CPU matmul experiment: all `-1` IDs clear output before conversion/row-map work; read once at CPU initialization, disabled by default after a negative speed screen |
| `VRAMBRIDGE_HYBRID_CPU_PREPARE=1` | Research-only reuse of exact activation conversion and row grouping between compatible raw CPU expert projections within one graph |
| `VRAMBRIDGE_HYBRID_ROUTE_PACKET=1` | Research-only combined GPU-to-CPU IDs/probabilities; CPU bias IDs and pre-FFN weights reuse the decoded packet too; hidden input retains its early copy |
| `VRAMBRIDGE_HYBRID_COLD_GPU=1` | Research-only scalar dispatch: move one cold expert to an existing extra bank slot only when at least two misses remain; preserve residency and all selected contributions |
| `VRAMBRIDGE_HYBRID_COLD_SYNC=1` | Matching-placement synchronous-copy reference for `COLD_GPU`; requires that option, not an app optimization |
| `VRAMBRIDGE_HYBRID_SCALAR_ROUTER=1` | Diagnostic scalar-shaped router projections for blocks of 2-32 positions; does not fix the current full-model batch gates or enable speculation |
| `VRAMBRIDGE_HYBRID_SCALAR_LINEAR=1` | Research-only scalar-shaped F16/BF16 learned 2D projections for 2-32 columns, preserving projection precision policies and adapter contributions |
| `VRAMBRIDGE_HYBRID_SCALAR_ATTN=1` | Research-only per-query KQ/V products for windows up to 32 queries; use plain attention for scalar and small-window decode, retaining ordinary large prefill |
| `VRAMBRIDGE_HYBRID_SCALAR_EXPERT=1` | Research-only per-token GPU expert products for 2-32 positions; preserve native CPU execution, expert IDs, scaling, precision and adapter contributions |
| `VRAMBRIDGE_HYBRID_MTP=1` | Explicit research opt-in for native `draft-mtp`; the MTP context must not inherit the target's resident bank or seed plan |
| `VRAMBRIDGE_HYBRID_STAGE_INPUT=1` | Prepare the exact CPU hidden-input view before resident GPU expert computation |
| `VRAMBRIDGE_HYBRID_OVERLAP=1` | Overlap only proven device-only CUDA expert work and independent host-only CPU expert work; callbacks and unsupported graphs retain barriers |
| `VRAMBRIDGE_HYBRID_BOUNDARY=gpu` | Reconcile the expert contributions on GPU; default remains CPU |
| `VRAMBRIDGE_HYBRID_FUSE_GATE_UP=1` | Interleave eligible resident gate/up banks without increasing weight storage; reuse native fused projection math |
| `VRAMBRIDGE_HYBRID_POLICY=ranked` | Prefer weaker training seeds only for zero-frequency eviction ties; active ties use recency; default is `classic` |
| `VRAMBRIDGE_HYBRID_POLICY=probation` | Raise the causal admission margin after a prior low-reuse lifetime; experimental heuristic, slower in the current screen |
| `VRAMBRIDGE_HYBRID_POLICY=anchored` | Protect the strongest initial seeds up to one quarter of usable group slots; remaining slots adapt; slower in the current screen |
| `VRAMBRIDGE_HYBRID_WINDOW=32` | Recent-history window in routed-token rounds, derived using routed-layer count; allowed range 8-256, default 64 |

The switches remain research options. An installation can opt into the tested
overlap/fusion bundle using `decode_pipeline: true`; the updated experimental
manual profile does so, while omission preserves the previous pipeline.
Admission-policy/window tuning and payback profiling are not part of that
bundle; classic remains default. A live entry has an unfinished observation
window, so zero future reuse is not reported as a wasted lifetime. Payback
accounting includes initialization/partial-prefill observations, unlike the
filtered scheduler counters.
Pair input staging with guarded overlap and verify
nonzero `overlap_splits` rather than treating an environment variable as proof.
The scheduler fences pending asynchronous host-source reads before scratch
reuse; synchronous input copies need no redundant fence. Therefore a zero
`copy_fences` counter can be correct even when overlap is active. Evaluation
callbacks disable overlap. Unsupported operations, buffer domains, or
multi-copy schedules preserve the ordinary barriers.

Transient cold offload uses raw registered sources and the same GPU stream as
its consumers. The extra slot is outside the residency map, never evicts a
cached expert, and is exposed consistently to bank views and ID validation.
It allocates no additional expert bank, but CUDA module/workspace and host
registration resources still need headroom. Source registration keeps its
64 GiB cap and 24 GiB RAM headroom. Adaptive use requires direct DMA; CPU
repacked sources are rejected. Multi-position calls retain frozen residency
and do not use transient offload. It was slower on the GPT screen and is
disabled in the app.

Shared CPU preparation is local to each worker's graph invocation, never
reused across requests or graph calls. The input/ID tensor identities,
dot-input type and expert count must match. Only views and disjoint raw
`ADD_ID` operations may intervene; other operations, aliases and extra-buffer
handlers invalidate or exclude reuse. Native tiled expert kernels read the
prepared prefix and use a separately reserved scratch suffix; larger-row
Q4_K controls cover this path without changing the selected math kernels.
Masked destinations are cleared on every projection, and scheduling counters
are reset even when immutable preparation is reused. CPU reference mode keeps
its original path. The process reuse counter is modulo 2^32. This saves no
weight reads and does not imply a speedup.

Routing packets represent bounded expert IDs exactly as F32, then validate and
decode them into per-layer CPU I32 storage capped at 64 MiB. Selected
probabilities remain F32. CPU consumers use packet-derived original IDs and
weights as well as masked IDs/weights; otherwise bias operations would cause
a duplicate GPU-to-CPU copy. The host output has six lanes in packet mode
versus four in the original protocol. Asynchronous slot-ID upload uses their
actual token stride. Packet construction uses native GPU operations, so fewer
copies can still cost more kernel/dependency time. Enable only for experiments
until end-to-end comparisons qualify it.

Copy breakdowns use host/device buffer domains (`H2H`, `H2D`, `D2H`, `D2D`,
or `unknown`) and logical tensor bytes. They do not measure PCIe hardware
counters. An event-wait call can enqueue a dependency without blocking the
host, and a synchronous copy call can include internal driver waits. Compare
profiling-off, coarse-profile and detailed-profile controls before attributing
changes to an optimization. Detailed timings retain the same unbatched hybrid
graph filter as the coarse counters.

CPU-empty primitive tests run with the option explicitly off/on, including
seven storage types, one/four threads, mixed/empty/full selections and strided
IDs with non-masked padding. An exposed process counter confirms the path ran
(modulo 2^32). Partial or invalid non-`-1` IDs still take the existing path and
validation; existing workspace/stride assertions remain. This does not skip a
whole FFN or shared/residual work, and does not cover the separate CPU-repack
kernel. The real-model screen did not establish a speed gain, so the app does
not enable it. Changing the environment requires a new backend process.

The asynchronous-ID prototype keeps an explicit graph dependency: a CPU
custom operation enqueues the ID upload into its preallocated device output,
then the GPU expert kernels consume that output on the same backend stream.
The callback does not dereference its device output from the CPU. Private ID
buffers are outside graph scratch; host staging is not overwritten until its
previous copy-completion event finishes. GPU stream ordering protects device
storage reuse, and teardown synchronizes before freeing staging. Storage
dimensions derive from each layer's active count and the bounded batch limit;
device allocation is charged to the cache budget and host staging is capped
at 64 MiB. Original IDs, distinct miss IDs and expert math are unchanged.
The ordinary ID-cast/copy path remains the default unless explicitly enabled.
These direct callback uploads are counted separately from scheduler copies;
fewer scheduler-copy bytes do not mean that ID transfers disappeared.

Cached backend classifications are essential: querying CUDA device properties
per submission caused a severe profiling observer effect. Keep that invalid
probe separate from corrected timings. Logical GPU expert-projection counts
verify active gate/up fusion. The QKV-alias experiment was removed after
isolated cross-layout regression failure; requesting its former switch now
fails explicitly rather than silently changing attention math.

For natural-length workload checks, the shared Python `sample()` helper accepts
keyword-only `prompt=` and `ignore_eos=False`. These preserve the exact requested
output-count gate: an answer ending early fails instead of being extrapolated
to 512 tokens. Optional authentication headers are sent but not serialized.

Direct mode also requires `GGML_CUDA_REGISTER_HOST=1`. It is Linux/CUDA-only,
page-aligns and coalesces source ranges, rejects already CUDA-pinned/repacked
sources, caps registration at 64 GiB, and checks for 24 GiB of available-RAM
headroom before and after registration. Registering the source model can pin
roughly 57 GiB of RAM until process/context teardown; this is a substantial
temporary resource commitment, not a lightweight default. Startup failures
unregister completed ranges; teardown waits for DMA before unregistering.
Registration failures are errors, never a silent pageable-memory fallback.
Staged admission is capped at 64 MiB, with raw components copied in chunks
and completion before pool reuse. Repacked components larger than the pool
cap are rejected. Budgets/top-k are not assumed to match the 120B measurement.

The checker has two distinct modes:

```bash
./build/hybrid-tests/vrambridge-logit-check \
  --corpus benchmarks/routing-corpus.jsonl --steps 32 \
  --reference-out benchmarks/results/routing/reference.bin \
  --output benchmarks/results/routing/reference.json \
  --model /models/gpt-oss-120b-MXFP4-00001-of-00002.gguf \
  --ctx-size 16384 --n-gpu-layers all --cpu-moe --no-repack \
  --no-op-offload --fit off --threads 4 --threads-batch 16
```

Run reference creation without hybrid variables for a CPU-expert reference.
For comparison, use `--reference-in` instead, a new report path, and the chosen
candidate's environment/model arguments, including its cache budget. For a
copy-only equivalence experiment, create the reference using the synchronous
candidate, then compare an otherwise identical asynchronous/direct candidate.
These are different questions: copy equivalence does not establish equivalence
to CPU-only placement or general answer quality.

Reference creation reports `mode: reference_creation` and
`all_checks_passed: null`, not a quality pass. Comparisons retain the strict
maximum-absolute and relative-L2 gates. They teacher-force the reference tokens
even when candidate argmax differs. Reference magic/header/length/trailer and
prompt alignment are checked; malformed references fail before model loading.
Reports refuse to overwrite final or partial files. This is not a generation
throughput benchmark. Native binary references contain token IDs and full
logits; diagnostic outputs contain activations. Keep them private under the
ignored results directory, with restrictive permissions.

Use `TZ=UTC` and `--template-epoch UNIX_SECONDS` to pin date-bearing templates.
The timestamp must be nonnegative and representable by the platform clock;
the default is Unix epoch zero. The backend patch honors the requested
template time, rather than silently using wall time. For a historical
reference, select its original template date: prompt-token alignment remains
mandatory, and the checker never substitutes or ignores a mismatched prompt.
Reports record the selected timestamp. Date-free templates are unaffected.

The checker requests low reasoning effort by default. Use `--reasoning-effort
default` to omit that template override, or choose `none`, `low`, `medium`, or
`high` when supported by the model template. References and reports record
non-default settings; legacy references retain their original low setting.
The checker rejects mismatched reasoning settings before backend initialization.

`--eval-callback-check` installs an inactive callback; it does **not** serialize
execution. The old `--serialized-check` spelling is a deprecated alias with an
explicit warning. `--debug-output` and `--sync-hybrid-gpu` are diagnostics,
not optimized serving settings.
`--debug-all-layers` requires `--debug-output` and extends the selected
intermediate tensors to every layer. `--debug-experts` additionally captures
bounded gate/up, activation and down-projection intermediates; it also requires
`--debug-output`. Active capture can perturb scheduling and is diagnostic only.
Capture remains bounded to one million
elements and a 16 MiB raw span per tensor, with a 64 MiB cumulative raw-span
limit. Strided F32/I32 views are gathered logically; unsupported types and
capture failures are errors, not silently omitted tensors. Serialized JSON
and its in-memory representation can be larger than these raw-byte limits.
`--debug-layer-limit N` instead selects the first N layers (1-1024), requires
`--debug-output`, and cannot be combined with `--debug-all-layers`.
`--debug-attention` requires diagnostic output and includes Q/K/V, indexer,
selection and attention intermediates. Captures retain the legacy `shape`
and add full `shape4` and a per-name `occurrence` index for repeated HC names.
Compare logical token axes, not a flat prefix of a multi-head tensor.
Active capture can change graph materialization and fusion: use it to locate
divergence, then repeat the numerical gate without capture callbacks.
`--cpu-reference` selects the pinned CPU backend's existing reference-mode
hook for diagnostic isolation; it does not change CUDA kernels or enable a
serving optimization. References made in this mode carry an extra header
field and cannot be silently compared with ordinary references. Missing
backend support is an explicit error.

`--batch-check 4` compares full teacher-forced logits from four-position decode
blocks against a single-position reference; it is valid only in comparison
mode. All block outputs are requested, aligned to the same reference tokens,
and bounded by the actual batch/ubatch capacities. A full-vocabulary comparison
can still fail due to changed execution paths; do not treat a successfully
executed block or its positions/second as accepted speculative tokens/second.
Speculative timing must include drafting, verification, rejections, migration,
and synchronization. No small-block/DFlash hybrid path is enabled in the app.
The scalar-linear/attention/expert switches and native MTP opt-in are also
off in the app. They reserve additional host graph metadata for the expanded
small-window graphs; this is separate from tensor-capture limits. Use a
model's own scalar reference, exercise both CPU and resident GPU contributions,
and test real sparse/long-context behavior before treating block verification
as qualified. Native MTP additionally needs canonical NextN metadata/tensors,
compatible hidden width, rejection/rollback checks, and a residency bootstrap
that does not train on held-out prompts. Set the hybrid batch limit to cover
the maximum verification window (draft count plus one). An empty frozen bank,
teacher-forced positions/s, or proposed draft tokens are not optimized accepted
generation throughput.
The non-speculative direct-DMA path can now be selected with the app's explicit
[manual hybrid launch profile](../tools/vrambridge/README.md#manual-hybrid-direct-dma-trial),
which verifies activation and fixes the measured configuration. It remains
experimental; an installed profile also enables automatic model-specific
selection on ordinary launches, with an explicit standard rollback.

## Downloaded-Model Compatibility And Throughput Matrix

The repository-only `vrambridge-model-matrix.py` runner compares standard and
automatically selected execution on models already on disk. It does not
download or convert weights. Both profiles use identical context/thread
settings. Automatic selection can use a bound training plan or a generic
causal plan for other complete MoE descriptors. Record the actual selected
profile: native fallback must not be credited with a nonexistent hybrid
speedup. Vision projectors are not standalone models.

```bash
python3 tools/benchmark/vrambridge-model-matrix.py \
  --allow-reloads --context 16384 --threads 8 --batch-threads 16 \
  --tokens 512 --repetitions 3 \
  --output benchmarks/results/routing/model-matrix.json \
  /models/gpt-oss-120b-MXFP4-00001-of-00002.gguf \
  /models/gemma-3-12b-it-Q4_K_M.gguf \
  /models/DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf
```

The app must be ready/idle before the test; the explicit reload-consent flag
permits unloading models. Never run this while manually chatting. The runner
restores the original model, policy, and execution settings in `finally`, even
after a failed answer check or interruption. Readiness includes actual context
confirmation. Every throughput sample requires exactly the requested output
count, a complete stream, backend timings, and finite positive rate/duration.
Reports contain wall/decode and end-to-end timing, repeated hashes, actual
profile/verification, model metadata, source/plan/patch hashes, and restore state.

The one technical-explanation throughput prompt is forced to the token limit;
it is not necessarily 512 natural answer tokens or multi-category sustained
qualification. Separate arithmetic and AST-only clamp checks report failures
without discarding valid timings. Generated code is never executed, answer text
is hashed rather than published, and no quality ranking follows from comparing
different model token rates. Keep raw output local until reviewed/sanitized.
The [matrix schema](schema/vrambridge-model-matrix.schema.json) and
[reviewed ledger](RESULTS.md) distinguish compatibility, numerical/answer gates,
and measured performance. Existing strict diagnostic failures are not relaxed.

## Generic MoE Graph Compatibility

The implementation goal is architecture-independent scheduling, not one hot
plan stretched across unrelated models. The shared native/hybrid expert
builder preserves routing normalization, optional biases, fused/separate
projections, activation/clamp variants, shared experts, and pre/post-FFN routing
weights. The cache derives per-layer key ranges, active counts, history, and
staging from tensors. Scaled expert tensors, LoRA/Hadamard transformations,
and multiple hybrid GPUs remain explicit unsupported cases.

The deterministic [fixture generator](../tools/trace/make_moe_fixture.py) uses
the pinned checkout's GGUF writer and matching `libggml-base` quantizer. It
requires that developer environment's NumPy/gguf tooling, not extra application
dependencies. No trained weights are downloaded. It bounds model dimensions,
refuses existing final/partial files, uses a fixed seed, and marks every model
**synthetic and untrained**. These are correctness probes, never quality or
speed benchmarks. The default CTest suite checks fixture input safety without
importing NumPy/gguf or initializing a GPU.

The complete tiny-model screen used these layouts:

| Architecture fixture | Storage | Blocks | Hidden / FFN | Experts / active | Important coverage |
| --- | --- | ---: | ---: | ---: | --- |
| Llama/Mixtral-style | F32 | 3 | 64 / 128 | 8 / 2 | Bias-free SwiGLU |
| Qwen2-MoE-style | Q8_0 | 3 | 96 / 192 | 7 / 3 | Shared experts, non-power-of-two expert count |
| Qwen3-MoE-style | Q4_K | 3 | 256 / 256 | 16 / 4 | K-quantized projections |
| GPT-style | MXFP4 | 3 | 64 / 128 | 4 / 2 | Optional routed biases and OpenAI SwiGLU |
| Llama4-style | F16 | 4 | 64 / 96 | 4 / 1 | Mixed dense/MoE layers, shared experts, weights before FFN |
| Cohere-MoE-style | BF16 | 3 | 64 / 128 | 8 / 2 | Fused gate/up, leading dense block, sigmoid routing |
| Grok-style | Q4_0 | 3 | 64 / 128 | 8 / 2 | GeGLU and normalization variants |

For each layout, the screen created CPU-expert and native-GPU references,
then checked native-GPU versus CPU, frozen mixed residency, three-position
blocks, causal staged admission, causal direct admission, and all-resident
hybrid versus native GPU: **56 executions, including 14 reference creations
and 42 full-vocabulary comparisons**, 16 teacher-forced positions per comparison.
All seven execute after the startup defects were corrected.

Llama, Qwen2, Llama4, and Cohere fixtures passed all six strict comparisons.
The GPT fixture passed its four mixed/block/admission checks against the CPU
reference, while some native-GPU and all-resident comparisons failed.
Quantized Qwen3 and Grok retained strict relative-L2 failures, including their
native-GPU controls. In total 28 comparisons passed and 14 failed. Thresholds
remain `max_abs <= 0.01` and `relative_l2 <= 0.001`; no failed check is relabeled
as a correctness pass. Native-placement variation is evidence to investigate,
not a general exemption. None of this proves trained-model accuracy or a
cross-model speedup.

The first screen exposed two implementation defects: forcing views onto a
different backend aliased device storage as CPU memory, and repeated zero
expert IDs violated unique-ID assumptions of float CUDA expert kernels.
Views now keep source placement; distinct zero IDs cover each active position.
Incorrect fixture tensor names and insufficient Llama4 layer-pattern coverage
were also corrected before rerunning; failed initial artifacts remain local.

The expanded CPU primitive test covers F32, F16, BF16, Q4_0, Q4_K, Q8_0, and
MXFP4; expert counts 4/7/8/16; active counts 1/2/4/8; token counts 1/2/3;
partial/all/no active IDs; stale-output clearing; and one/four threads.
MXFP4 repacking still has a separate byte-exact export test.

A separate F32 stress fixture (two blocks, width 512, FFN 1,024, 16 experts
and top-16 routing) exercised actual pool reuse: 32 admissions uploaded
201,326,592 bytes through a verified 67,108,864-byte pool and passed the strict
full-logit comparison. That is a staging-cap/correctness result, not throughput.
An app-level Qwen2-MoE fixture test also verified both policies: Automatic
preferred native fitting for the small file, while Require-hybrid selected
verified causal staged dispatch without a trained-plan hash. Both produced
16 output tokens with matching answer hashes at the model's actual 2K context;
the original 120B direct-DMA app was restored. These untrained answers are not
quality evidence.

The first app smoke trial exposed the server capping a synthetic 2K model
while the installation requested 16K. Context planning now respects the model
limit, readiness rejects any further mismatch immediately, and native context
adjustments are displayed explicitly. That initial partial trial is retained,
not counted as a successful 16K test.

For a small reproducible example, unload the app's model first:

```bash
python3 tools/trace/make_moe_fixture.py \
  --llama-source /path/to/pinned/llama.cpp \
  --ggml-base /path/to/patched/libggml-base.so \
  --architecture qwen2moe --expert-type Q8_0 \
  --embedding 96 --feed-forward 192 --experts 7 --active 3 \
  --output benchmarks/results/routing/synthetic-qwen2.gguf

# Create this only if it does not already exist.
(set -o noclobber; printf '%s\n' \
  '{"id":"synthetic","split":"test","category":"reasoning","prompt":"abc"}' \
  > benchmarks/results/routing/synthetic-corpus.jsonl)

GGML_BACKEND_PATH=/path/to/matching/libggml-cuda.so \
./build/hybrid-tests/vrambridge-logit-check \
  --model benchmarks/results/routing/synthetic-qwen2.gguf \
  --corpus benchmarks/results/routing/synthetic-corpus.jsonl --steps 16 \
  --ctx-size 512 --threads 4 --threads-batch 4 \
  --n-gpu-layers all --cpu-moe --fit off --no-op-offload --no-repack --no-host \
  --reference-out benchmarks/results/routing/synthetic-reference.bin \
  --output benchmarks/results/routing/synthetic-reference.json

GGML_BACKEND_PATH=/path/to/matching/libggml-cuda.so \
VRAMBRIDGE_HYBRID_PLAN=auto VRAMBRIDGE_HYBRID_ADAPT=1 VRAMBRIDGE_HYBRID_ASYNC=1 \
./build/hybrid-tests/vrambridge-logit-check \
  --model benchmarks/results/routing/synthetic-qwen2.gguf \
  --corpus benchmarks/results/routing/synthetic-corpus.jsonl --steps 16 \
  --ctx-size 512 --threads 4 --threads-batch 4 \
  --n-gpu-layers all --cpu-moe --fit off --no-op-offload --no-repack --no-host \
  --moe-cache-mib 64 \
  --reference-in benchmarks/results/routing/synthetic-reference.bin \
  --output benchmarks/results/routing/synthetic-staged-comparison.json
```

Keep generated GGUFs, binary references, activations, and logs under ignored
results. Reload the original app/model afterward. The reference stores token
IDs/full logits; do not publish it. For native-GPU controls omit `--cpu-moe`.
For frozen/direct cases use a dimension-valid `layer expert` plan; never borrow
a different model's trained plan. `--batch-check 3` plus
`VRAMBRIDGE_HYBRID_BATCH=3` checks three-position target blocks, not accepted
generation throughput.