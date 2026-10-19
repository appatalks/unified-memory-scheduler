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

### Real-model prefill configuration qualification

The Linux I/O study can also screen native logical/physical batching without
changing launcher defaults:

```bash
python3 tools/benchmark/vrambridge-io-study.py \
  --llama-server /path/to/standard/llama-server \
  --hybrid-profile /path/to/hybrid-profile.json \
  --model /models/gpt-oss-120b-MXFP4-00001-of-00002.gguf \
  --prompt-file /path/to/frozen-context.py \
  --context 16384 --threads 8 --batch-threads 16 \
  --batch-size 2048 --ubatch-size 512 \
  --tokens 256 --repetitions 1 \
  --output benchmarks/results/prefill-control.json
```

The paired options are bounded to 32-2048, with physical <= logical <= context.
They are research-only CLI controls, backed by an optional `Runtime`
constructor argument, not app/API/UI settings. Normal launches append neither
batch override nor extra verbosity. Explicit studies use verbosity 4 and retain
the native context's observed `n_batch`/`n_ubatch` even after startup lines leave
the log ring; missing or mismatched acknowledgments fail before generation.

Freeze the context before editing source. This runner wraps the bounded UTF-8
file in a fixed technical-review prompt and records its hash, not its text.
Use prompts longer than the candidate physical batch, fixed context/bank/thread
settings, and fresh report names. Alternate separately loaded baseline/candidate
runs rather than comparing a newly loaded candidate with an adapted live control.
Keep exactly counted output tokens, prompt-cache disabling, and the low-reasoning
sampler settings unchanged. Loading wall time includes launch through readiness;
cache-warm startup is not cold-disk loading. Memory snapshots are endpoints,
not sampled or continuous peaks.

The [real-workload record](results/gpt120-prefill-real-workload-summary.json)
and [Synapse handoff](../docs/synapse-prefill-handoff.md) retain three pairs on
the real 120B model. The apparent 512 -> 1024 physical-batch latency gain is
**not qualified**: generated hashes differ and a separate long held-out
full-vocabulary check fails the unchanged gates. Identical output hashes in
other screens are not a substitute for those checks.

### Stage-aware prefill calibration

The isolated expert probe also accepts `--rows N` (1-256) and
`--upload-weights`. These are research calibration controls, not a prefill
scheduler or an application setting. Each synthetic row selects the same four
original experts with unchanged asymmetric probabilities; only its activation
vector varies. This deliberately measures concentrated expert reuse, not a real
prompt's routing distribution.

For example, compare resident execution with the cost of transferring the
expert weights on every invocation:

```bash
./build/trace-release/vrambridge-hybrid-layer \
  --model /models/gpt-oss-120b-MXFP4-00001-of-00002.gguf \
  --layer 0 --threads 8 --samples 3 --repetitions 12 --rows 32 \
  --output benchmarks/results/prefill-layer0-rows32-resident.json
./build/trace-release/vrambridge-hybrid-layer \
  --model /models/gpt-oss-120b-MXFP4-00001-of-00002.gguf \
  --layer 0 --threads 8 --samples 3 --repetitions 12 --rows 32 \
  --upload-weights \
  --output benchmarks/results/prefill-layer0-rows32-upload.json
```

Sweep rows 1/2/4/8/16/32/64/128, then repeat representative early/middle/late
layers and alternate resident/upload run order. Each invocation retains all
five CPU/GPU partitions and interleaves serial/concurrent schedules. Only
correctness-passing samples may contribute a break-even result. Report the
entire curve: a first crossing in a noisy screen is not a scheduling rule.
`--reverse-splits` reverses partition order for repeated controls. Reports
include per-mode and per-partition gates across warm-up and timed executions,
not just the displayed first few correctness samples.

`weight_upload_bytes_per_iteration` includes all selected weight and bias
components in the GPU branch. Uploads use synchronous copies from pageable
host vectors and are inside the measured wall interval; their host duration is
reported separately. CPU branch input/compute, GPU input/dispatch, staging,
reconciliation, and final synchronization remain included. This is not pinned
asynchronous DMA, and the same source bytes can remain in CPU caches between
iterations. Graph allocation, initial loading, and validation readback are
outside the interval. The largest permitted merge staging is 11.25 MiB.

CPU-only `--self-test --rows N` checks every output row against an independent
scalar implementation, including both reduction strategies and reconciliation
layouts. Adding
`--upload-weights` to that self-test exercises CPU tensor recopies only; the
report explicitly labels their destination `cpu_self_test`. These are not GPU
measurements.

The first real-weight attempt on October 18 failed CUDA initialization before
loading weights. Direct driver initialization returned error 999, and the
unchanged stock backend also found no usable CUDA device. CUDA recovered after
the user's reboot. The first multi-row GPU attempt then exposed a reduced-output
layout mismatch, which was fixed without changing expert math; CPU self-tests
now also exercise that reconciliation layout. Subsequent resident/upload
curves retain numerical failures rather than qualifying their faster timings.
The
[investigation record](results/stage-aware-prefill-investigation-summary.json)
retains the initial failures and measured follow-up.

The native Blackwell MXFP4 matrix path can select Q4 activations. A
process-scoped `GGML_CUDA_MMQ_PREC=q8` is an existing backend diagnostic control;
the probe records its value, and it must not be treated as a global app setting.
This reduces some observed differences but does not make every layer/split
pass the unchanged gate. `--gpu-row-limit N` is a separate bounded control that
shares original weights across smaller GPU row chunks, while retaining the
full-row CPU reference. It preserves expert selection and probabilities but
adds launch/concatenation overhead and is not a proven numerical fix.
`--cpu-row-limit N` applies the corresponding row grouping only to the candidate
host branch, keeping the unsplit CPU reference full-row. It is a numerical
isolation control, not a production scheduling option. The
[CPU isolation follow-up](results/prefill-cpu-isolation-summary.json) records zero
observed CPU-chunk error in its tested cases while retaining GPU failures.

`--operator-check` adds bounded diagnostic graphs for at most 64 rows. It compares
gate/up projections, biases, OpenAI SwiGLU, down projection, weighting and reduction.
Separate unobserved/observed CPU and GPU graphs detect output changes caused by
materializing intermediates. Identical-input replays distinguish an operator's
own behavior from errors propagated from earlier stages. MXFP4-only controls
also replay CPU-effective Q8_0 inputs while explicitly requesting GPU Q8
projections; these are diagnostic input substitutions, not a qualified serving
optimization. The report contains aggregate errors and diagnostic FNV-1a output
fingerprints, not raw activations. Fingerprints are non-cryptographic identities,
not model integrity checks.

Operator mode always marks timings unqualified. Ordinary reports also require
their numerical gates before the overall timing flag can qualify. Never present
CPU-side canonical preparation performed outside timing as free acceleration.

Effective-Q8 diagnostics compare aggregate code/scale changes and replay the
down projection on CPU with the chain's effective input. A separate lesion uses
GPU SwiGLU with the CPU's gate/up input to distinguish activation-only from
projection-propagated differences. They publish no raw codes or activations.
The [propagation follow-up](results/mxfp4-effective-q8-propagation-summary.json)
retains all three failing expert/row configurations: the severe down error is
reproduced by CPU replay, whereas GPU down with the same effective input has
relative-L2 error near 1e-7. This narrows the cause; it does not fix it.

The separate
[experimental MXFP4 CPU-Q8 patch](../third_party/patches/llama-b11669-mxfp4-q8-parity-experimental.patch)
adds a default-off `VRAMBRIDGE_HYBRID_MXFP4_Q8_PARITY=1` mode to the pinned
CUDA backend. It changes activation integer rounding to the native x86 Q8_0
nearest-even convention, rounds stored activation scales through FP16, and
requests Q8 rather than Blackwell's default Q4 matrix activations for MXFP4.
Original target weight blocks, expert selections and router probabilities are
unchanged. An explicit conflicting Q4 request or invalid flag value fails
instead of silently choosing another mode.

This is an optional CPU-reference-consistency experiment, not a finding that
all native CUDA arithmetic is incorrect. Use a separate source/library build;
do not replace installed backends or export the flag globally. The recorded
research build uses CUDA 13.1, SM 120a and an x86 AVX2/AVX512 CPU backend.
Other CPUs, GPUs and model layouts are not qualified by these tests.
The application launcher strips inherited `VRAMBRIDGE_HYBRID_` variables, so
the research flag is not an application setting.

Initial default-off controls retain clean-baseline output fingerprints and
metrics. Wider tests retain three failures among sixty operator configurations,
including one final-output failure; this is not universal numerical parity.
Keep both the original primitive and full-model gates. Performance numbers
must exclude diagnostic captures and include transfers/preparation where
applicable, and cannot be converted into full-model tok/s from layer timings.
The [reviewed operator/parity record](results/mxfp4-activation-parity-summary.json)
retains source hashes, default-off comparisons, adversarial inputs, all remaining
failures, scoped sanitizer coverage and the transfer-inclusive timing window.
`--expert-ids A,B,C,D` selects four distinct real expert slices, and
`--activation-pattern zero|q8-ties` supplies explicit stress inputs; neither
stress pattern is a natural inference workload.

#### CPU activation with GPU down projection

`--cpu-activation` is a separate probe execution split. All gate/up projections,
their original biases and OpenAI SwiGLU remain on CPU. For MXFP4, a CPU-native
Q8_0 cast followed by F32 reconstruction preserves the input the native CPU down
projection would consume. The GPU down operation explicitly requests Q8 inputs.
This mode passes the recorded stock-backend checks **without** the experimental
CUDA parity patch or flag.

```bash
env -u VRAMBRIDGE_HYBRID_MXFP4_Q8_PARITY -u GGML_CUDA_MMQ_PREC \
  ./build/trace-release/vrambridge-hybrid-layer \
  --model /models/gpt-oss-120b-MXFP4-00001-of-00002.gguf \
  --layer 17 --threads 16 --rows 64 --samples 3 --repetitions 24 \
  --cpu-activation --upload-weights \
  --output benchmarks/results/cpu-activation-down-layer17-rows64.json
```

Use matching stock libraries at the pinned backend revision. Omit
`--upload-weights` for resident down weights; add `--operator-check` for a bounded,
untimed nine-operator/observer check (at most 64 rows). In split mode the CPU row
limit also applies to activation preparation; the GPU row limit applies only to
down projections. `--gpu-reduce` still tests ordered per-expert reconciliation.

CPU Q8 quantization/reconstruction, activation input/copy, GPU requantization,
down compute, remaining CPU experts, reconciliation and synchronization are
inside the wall interval. Optional uploads include **only actual down weights
and biases**, not gate/up weights that remain on CPU. The activation copy is
synchronous F32 effective values, not a packed-Q8 or pinned-DMA optimization.
Copy time is a subset of GPU input/dispatch time; do not add both to the total.
Setup, file loading and verification readback remain outside timing.

Read `execution_split` and each partition's `expert_partition_scope`: in this
mode the split index and `gpu_experts` count refer to down projections, not
complete GPU FFNs. Explicit gate/up/down CPU/GPU counts and activation-copy bytes
remove that ambiguity; CPU self-tests identify CPU-only copies separately.

The [measured follow-up](results/cpu-activation-gpu-down-summary.json) records
all sixty stock-backend operator configurations passing and 108 repeated
resident/upload timing reports. Cold one/eight-row uploads lose; sixteen rows
are inconsistent; 32/64/128 rows improve in all recorded pairs. These are
concentrated-reuse layer batches, not a universal dispatch threshold or a
demonstrated prefill/generation improvement. The actual-input gate below adds
bounded real routing/activation checks; integrated logits and decode-residency
effects remain unqualified.

#### Actual prefill activation and routing gate

The logit checker accepts `--prefill-capture-output` only with `--reference-in`
and an existing private parent directory (mode 0700). It exports sensitive
activations with mode 0600, not public benchmark JSON. It supports at most three
held-out GPT-OSS-120B requests and layers 0/17/35. Each frame samples the last
64 rows of that layer's first nonempty FFN microbatch, or all rows if fewer.
Tail windows avoid merely reusing a common system/template prefix.

Inputs and IDs come from the native CPU up projection's original source
tensors; probabilities come from the actual weighting input, not a reconstructed
router. Outputs come from the native FFN boundary. Strides, sizes, per-row IDs,
pruning and alignment are checked. Empty batches are counted; final-layer
one-row pruning is preserved rather than expanded into fake rows.

First create a native CPU-expert reference using a frozen held-out corpus:

```bash
umask 077
mkdir -m 700 benchmarks/results/actual-replay
GGML_BACKEND_PATH=/path/to/matching/libggml-cuda.so \
GGML_CUDA_DISABLE_FUSION=1 \
./build/hybrid-tests/vrambridge-logit-check \
  --corpus /path/to/frozen-heldout-corpus.jsonl --steps 32 \
  --reference-out benchmarks/results/actual-replay/reference.bin \
  --output benchmarks/results/actual-replay/reference.json \
  --template-epoch 0 --ctx-size 16384 --threads 8 --threads-batch 16 \
  --model /models/gpt-oss-120b-MXFP4-00001-of-00002.gguf \
  --n-gpu-layers all --cpu-moe --no-repack --no-host --fit off \
  --moe-cache-mib 0 --no-op-offload --batch-size 2048 --ubatch-size 512
```

Keep inherited hybrid/placement overrides unset. Repeat with the same settings,
replace `--reference-out` with `--reference-in`, choose a new report, and add
`--prefill-capture-output benchmarks/results/actual-replay/private-ffn.json`.
This is deliberately a native CPU-expert numerical reference with no MoE bank,
not a substitute for a fixed-bank hybrid serving comparison. Disabling CUDA
fusion is a diagnostic observer control, **not an application recommendation**.

Replay a record with normal GPU fusion:

```bash
env -u GGML_CUDA_DISABLE_FUSION -u VRAMBRIDGE_HYBRID_MXFP4_Q8_PARITY \
  GGML_BACKEND_PATH=/path/to/matching/libggml-cuda.so \
  ./build/hybrid-tests/vrambridge-hybrid-layer \
  --cpu-activation --activation-capture benchmarks/results/actual-replay/private-ffn.json \
  --capture-record 0 --threads 16 \
  --model /models/gpt-oss-120b-MXFP4-00001-of-00002.gguf \
  --output benchmarks/results/actual-replay/replay-0.json
```

This separate mode is always untimed/operator-checked. It loads exactly the
original expert slices needed by the captured per-token IDs, remaps them
bijectively to the replay bank, and preserves each probability without
renormalization. Explicit layer/row/expert/timing overrides are rejected.
It compares native CPU replay and GPU-down output with the actual captured
boundary, not just with each other. Unqualified observed captures are rejected.
The [routed replay schema](schema/vrambridge-routed-ffn-replay.schema.json)
references shared comparison definitions from the existing probe schema.

The [reviewed actual-input study](results/actual-prefill-activation-replay-summary.json)
passes nine frames across three categories, including all operator/observer
gates. With CUDA fusion disabled in both reference and capture, 96
full-vocabulary observer-control positions are exact. Earlier optimized-profile
active captures fail; unchanged/inactive controls are exact. That failure is
retained, not relabeled as a pass. The 96 positions qualify the **observer
profile**, not an integrated GPU-down candidate. No real-request speed result
follows from these replays.

#### Full-model resident-down mask control

The separate [experimental backend patch](../third_party/patches/llama-b11669-resident-down-prefill-experimental.patch)
is stacked on the published hybrid patch at pinned llama.cpp
`620d2225cc436ab0b001e3200e590e3b286ef9ab`. **Do not install or promote it**:
the [full-model gate](results/resident-down-full-model-prototype-summary.json)
fails even though default-off preservation and local FFN checks pass.

The user selected a mask-based correctness control before compact dispatch.
This reuses the existing protected zero GPU slots and supported negative CPU
IDs. Every required cold expert still executes on CPU; every resident expert's
down projection executes on GPU. Inactive GPU rows perform real extra work
against zero slots. That cost is not free and no speed claim is made.

The additive `llama_set_prefill_phase` API explicitly distinguishes prompt
processing from decode/verification. Its bit participates in graph-cache
reuse. The server derives phase from every token's `is_prompt` tag, not batch
size alone. Zero/one-row FFNs retain their original path; speculative execution
is not qualified in enabled research mode. CPU gate/up/SwiGLU, native-Q8
canonicalization, live per-token routes/probabilities, CPU boundary/reduction
and the existing single-token decoder are preserved in the tested design.
Bulk-prefill dispatch makes no admissions or evictions.

Use a separate source/build. The research build reuses the installed CPU/CUDA
plugin files read-only; no CUDA source patch or library replacement is needed.
After applying the base hybrid patch followed by the experimental patch to a
separate checkout, the tested backend build configuration is:

```bash
cmake -S /path/to/pinned-research-source -B build/prefill-down-backend \
  -DCMAKE_BUILD_TYPE=Release -DGGML_BACKEND_DL=ON \
  -DGGML_CPU=OFF -DGGML_CUDA=OFF -DGGML_NATIVE=OFF \
  -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_OPENSSL=OFF
cmake --build build/prefill-down-backend --target llama-common llama-server -j4
```

This configuration needs matching existing CPU/CUDA dynamic plugins available
in the research backend directory; the experiment used read-only symlinks to
the verified installed files. It does not build those plugins itself.
Configure the checker against that source/library directory with
`VRAMBRIDGE_TEST_MASKED_IDS=ON` and `VRAMBRIDGE_TEST_PREFILL_PHASE=ON`. Build the
checker and `vrambridge-prefill-phase-test`, then run the focused CTest targets.
An ordinary backend rejects the feature flag before model/GPU initialization.

Create a fresh installed-hybrid reference with the full-logit guide, fixed
16K context, 12,000 MiB bank, 2048/512 batches, 8/16 threads and the trained
944-expert plan. Compare the separate build with
`VRAMBRIDGE_HYBRID_PREFILL_DOWN=0` first, then `=1`, retaining the same fusion
policy and configuration. The checker signals prefill explicitly. Do not add
an evaluation callback: enabled mode rejects it.

Diagnostic-only controls, all requiring enabled research mode:

- `VRAMBRIDGE_HYBRID_PREFILL_DOWN_CPU_CONTROL=1` retains the new graph but sends
  all required down work to CPU and only zero-slot work to GPU.
- `VRAMBRIDGE_HYBRID_PREFILL_DOWN_AUDIT=1` computes an additional native CPU
  weighted-down oracle, logs aggregate errors and returns the candidate data
  unchanged. It is not a performance mode.
- `VRAMBRIDGE_HYBRID_PREFILL_DOWN_LAYER=N` permits real GPU down work only at
  diagnostic layer 0-34; all other bulk layers use the CPU control. This is
  localization, not a deployment layer-selection policy.

Matched fusion-off controls are a separate diagnostic question, never a
replacement for the optimized-profile gate. The final default-off build is
exact at 96 positions. Enabled mode fails all three workloads. Fusion-off
CPU control is exact, but real GPU down still fails; a layer-0-only experiment
also fails. All 735 audited local weighted-FFN comparisons pass the unchanged
primitive gate, with worst relative L2 about 1.17e-7. This supports further
investigation of fusion sensitivity and downstream amplification of small
arithmetic differences, not a claim that the exact amplifying operator is known.
Latency benchmarks and compact dispatch are blocked pending that numerical work.

#### Observer-qualified numerical localization

Fixed-date checker commands must set `TZ=UTC` as well as
`--template-epoch 0`. The pinned Jinja `strftime_now` uses `std::localtime`;
the epoch alone does not fix the timezone. The CPU study retains two
unaligned-reference rejections caused by omitting this environment control.

The [localization study](results/resident-down-numerical-localization-summary.json)
adds three bounded diagnostic focuses to the checker:

- `--localize-focus ffn`: first nonempty FFN inputs, dense router scores,
  original IDs/probabilities and outputs for all 36 layers (180 records).
- `--localize-focus attention1`: layer-0 residual and layer-1 attention
  normalization, Q/K/V projection/bias, attention context/output and post-norm
  checkpoints (12 records).
- `--localize-focus rounding1`: rotated Q/K, biased V, actual native F16 K/V
  cache stores and attention context (six records).

Use `--localize-output private.bin` to write a reference, or
`--localize-reference private.bin` to compare against it. The modes require one
held-out GPT-OSS-120B request, at most 512 physical rows, a full-logit comparison
reference, a private directory and matched CUDA fusion-off controls. All 512
rows are observed, not just a shared prompt prefix. Zero-row work is skipped
until the first actual FFN row; final-layer one-row pruning remains explicit.
Sensitive binary traces stay private and reports contain aggregate errors only.
The [aggregate schema](schema/vrambridge-prefill-localization.schema.json)
distinguishes actual cache storage from hypothetical query-F16 rounding replay.

For the enabled research backend, apply the
[diagnostic overlay](../third_party/patches/llama-b11669-prefill-localization-experimental.patch)
after the base hybrid and resident-down patches, and explicitly set
`VRAMBRIDGE_HYBRID_PREFILL_DOWN_LOCALIZE=1`. It permits an evaluation callback
only in that deliberate fusion-off diagnostic scope; normal enabled mode still
rejects unqualified observers. Do not enable it globally or use trace timings.

Observer preservation needs two distinct references. First qualify the CPU
trace against the native CPU-control logits. Then run the unobserved layer-0
candidate with the original continuation and
`--teacher-forced-reference-out private-candidate.bin`; it exports candidate
logits with the supplied reference's continuation, without pretending the
candidate passes the native model gate. Compare the observed candidate against
that preservation reference. Top-one matches use the reference logits' argmax,
while continuation tokens remain teacher-forced; those can legitimately differ.

The measured CPU and candidate traces are exact against their respective
capture-free references at all 32 checked positions for each focus. The candidate
still fails the original native-reference gate. The diagnostic overlay also
retains exact default-off preservation at all 96 original optimized-profile
positions.

The first directly observed large/discrete amplification is in attention's
F16 working state: the small layer-0 FFN difference reaches layer-1 Q/K/V,
then 19 native stored keys and 30 stored values change at F16 boundaries.
Every sampled actual K/V store is checked against its original F32 source's
IEEE-F16 rounding, with aligned cache indices; values are read, never replaced.
Query-F16 replay predicts 195 differences, but actual query conversion is not
captured and must not be presented as a measured query-cache result.

Layer-1 FFN input Q8 replay subsequently changes 1,307 codes and 448 scales.
Layer 2 has five changed ordered-route rows, but only one changed expert set
(first-batch row 138, zero-based). This distinction prevents mistaking a rank
permutation for selection of different experts.

The [packed-down follow-up](results/cpu-compatible-packed-down-summary.json)
now implements and times a CPU-reference-compatible raw projection in the
isolated probe. Fusion-preserving full-model integration and actual native
KV/Q8 state checks remain before compact dispatch. Do not infer a remedy from
a larger cache: native Flash Attention may convert F32 K/V back to F16 internally.

#### CPU-compatible packed-Q8 down probe

This separate, default-off CUDA research target reproduces the pinned x86
MXFP4/Q8 dot's paired eight-lane even/odd FMA accumulation and horizontal sum.
It consumes original CPU-generated Q8_0 blocks directly: no F32 reconstruction
or GPU requantization. A second integer implementation uses byte permutations
and DP4A without changing the FP32 accumulation contract. Neither is installed
as a backend kernel or used by the application.

```bash
cmake -S . -B build/compatible-down-probe \
  -DCMAKE_BUILD_TYPE=Release -DVRAMBRIDGE_ENABLE_CUDA=ON \
  -DVRAMBRIDGE_TEST_COMPATIBLE_DOWN=ON \
  -DVRAMBRIDGE_LLAMA_SOURCE=/path/to/pinned-source \
  -DVRAMBRIDGE_LLAMA_LIBRARY_DIR=/path/to/matching/libraries
cmake --build build/compatible-down-probe --target vrambridge-hybrid-layer -j4
ctest --test-dir build/compatible-down-probe \
  -R 'vrambridge-(hybrid-layer|compatible-down)' --output-on-failure

# GPU MXFP4/Q8 fixtures, not the ordinary CPU F32 self-test:
build/compatible-down-probe/vrambridge-hybrid-layer \
  --self-test --compatible-down --compatible-dp4a \
  --rows 32 --samples 3 --repetitions 24

# Reuse a qualified private capture; do not publish its source tensors:
build/compatible-down-probe/vrambridge-hybrid-layer \
  --model /models/gpt-oss-120b-MXFP4-00001-of-00002.gguf \
  --compatible-down --compatible-dp4a --cpu-activation \
  --activation-capture /private/three-category-private-ffn.json \
  --capture-record 0 --threads 16 --repetitions 24 \
  --output benchmarks/results/compatible-actual-0.json
```

Omit `--compatible-dp4a` for the scalar integer control. Alternate
`--compatible-gpu-first` across fresh invocations to reverse CPU/GPU timing
order. Existing real-weight `--layer`, four `--expert-ids`, 1-64 `--rows`,
`--samples` and `--activation-pattern` select synthetic-input tests.
`--upload-weights` additionally charges every down-bank upload. Capture replay
retains its original per-row expert IDs/probabilities; do not pass row/layer
overrides. Existing ordinary builds reject the research mode before backend
initialization.

The raw-stage GPU interval includes native CPU Q8 casting/readback, pageable
H2D packed inputs and IDs, projection, D2H **all per-rank raw outputs** and
synchronization. Allocation and initial resident-bank setup are outside timing.
Shared CPU gate/up/SwiGLU and CPU bias/weighting/reduction are excluded from
both raw-stage comparisons. CPU post-ops are nevertheless executed separately
and checked bit-for-bit against the unobserved native CPU FFN. The independent
captured boundary retains the original primitive tolerance; it is not claimed
bit-exact. All warm-up and timed outputs must be repeatable.

Both variants pass 192 checks apiece, including nine actual frames repeated
three times, sixty real-weight configurations, upload controls and adversarial
fixtures. DP4A improves the raw stage, but all captured required experts are
loaded into the probe GPU bank. This is **not** a 12,000 MiB mixed-residency
full-model comparison, a KV-state parity check, or a request-latency result.
The arithmetic contract is host/pin/format-specific; it is not a new global
architecture assumption. No tolerance changes, expert dropping, state snapping,
global fusion disable or decoder changes are authorized by this result.

Actual prefill work can be observed separately with the routing collector:

```bash
./build/trace-release/vrambridge-trace \
  --trace-corpus benchmarks/prefill-corpus.jsonl --trace-output decode-trace.jsonl \
  --prefill-output prefill-observations.json \
  --model /models/gpt-oss-120b-MXFP4-00001-of-00002.gguf \
  --n-gpu-layers all --cpu-moe --fit off --no-op-offload \
  --no-repack --no-host --ctx-size 4096 \
  --batch-size 512 --ubatch-size 512 --threads 8 --threads-batch 16
```

The optional sidecar uses the
[prefill-observation schema](schema/vrambridge-prefill-observations.schema.json).
The included [controlled corpus](prefill-corpus.jsonl) reproduces the two
observed prompt workloads and requests one continuation token each; it is not
a representative workload or answer-quality evaluation.
It records each observed routing tensor's rows and expert histogram without
prompt text, generated text, or token IDs. Coverage is explicit: zero-row
pruned observations are retained, and a final layer with only the last prompt
position is not labeled full coverage. Missing layers, duplicate/out-of-range
expert IDs, invalid layouts, and output-path collisions fail explicitly.
The ordinary decode trace format remains unchanged. The collector verifies a
greedy continuation against a callback-inactive run; this is an observer check,
not broad answer-quality or full-logit qualification. Captured timings are not
serving performance. `--prefill-self-test` checks strided CPU histogram reads
and invalid/zero-row cases without loading model weights.

Before integrating any policy, collect actual prefill expert row histograms
and validate the complete target graph. Compare matched prompt throughput,
time to first token, accepted-output decode rate, total request wall time, and
peak memory. Do not increase the existing hybrid batch limit as a substitute:
that also changes numerical execution paths and does not implement an
expert-workload-aware prefill scheduler.

## Explicit SSD Model Staging

The Linux research helper uses the existing root-drive SSD, not the Samsung
NTFS volume. The current staging directory is
`/home/mj420/Desktop/TOR/llm/model-staging`, owned by the user with mode 0700.
Its marker records the user-selected **16 GiB minimum free-space reserve**.
This is a working-copy cache, not a backup or a downstream Synapse feature.

```bash
python3 tools/benchmark/vrambridge-model-staging.py init
python3 tools/benchmark/vrambridge-model-staging.py plan /archive/model-00001-of-00002.gguf
python3 tools/benchmark/vrambridge-model-staging.py stage /archive/model-00001-of-00002.gguf
python3 tools/benchmark/vrambridge-model-staging.py list
python3 tools/benchmark/vrambridge-model-staging.py verify CACHE_ID
# Unload the model and close users of the staged files first:
python3 tools/benchmark/vrambridge-model-staging.py evict CACHE_ID --confirm-unloaded
```

`--root PATH` before the subcommand selects another explicitly initialized
directory; unmarked nonempty directories are never adopted. The helper checks
the first GGUF metadata's split count and requires complete standard shard
names. It is not a checkpoint semantic validator. Sources already on the
staging filesystem are returned for direct reuse with zero copied bytes;
GPT-OSS-120B therefore stays at its existing SSD path.

Cross-filesystem copies are written to a private partial directory, checked for
source changes, fsynced and SHA256-readback-verified before publication. The
space reserve plus a 16 MiB metadata allowance is checked before and during copying; other processes can still
consume disk space independently, so this is not a system-wide reservation.
Staging operations serialize with a nonblocking lock. Completed manifests
record source paths, file sizes and checksums. Interrupted unrecognized entries
are listed explicitly, never automatically deleted.

Eviction requires explicit confirmation that inference has been unloaded,
verifies both copies and surviving archive originals, and checks open/mapped
staged files with `fuser -I`. Missing originals, checksum mismatches, symlinks,
unexpected files and uncertain/in-use checks block deletion. Only listed
staged files and their manifest/directory are removed; originals and unrelated
directories are never eviction targets. The confirmation is still necessary:
uncooperative processes can open files after an external in-use check, and
this helper does not integrate a lease with every possible inference server.

The [root-drive setup record](results/root-drive-model-staging-summary.json)
includes a small cross-filesystem fixture test, real open-file eviction
rejection, cleanup, the preserved 16 GiB policy and zero-copy GPT reuse.
An initial `fuser --` incompatibility failed safely; the supported inode-check
form is now covered by a real process test. No full model was duplicated and
no cold-load or generation improvement is claimed from setup alone.

## Storage I/O And RAM-disk Hypothesis

The Linux-only `vrambridge-io-study.py` research runner reuses the existing
backend launcher and exact-count streaming sampler. It launches and stops its
own backend on a private loopback port; it does not attach to or unload an
existing application. Run only while inference is idle and adequate RAM/VRAM
is available. Run it from this checkout, not as an installed application.

```bash
python3 tools/benchmark/vrambridge-io-study.py \
  --llama-server /path/to/llama-server \
  --model /models/gpt-oss-120b-MXFP4-00001-of-00002.gguf \
  --context 16384 --threads 8 --batch-threads 16 \
  --tokens 256 --repetitions 3 \
  --output benchmarks/results/io-control.json
```

Optionally supply `--hybrid-profile /path/to/hybrid-profile.json` to diagnose
that installed experimental configuration separately. It is not automatically
equivalent to the standard placement control or broadly quality-qualified.

Reports separate Linux physical `read_bytes` from read-syscall `rchar`, capture
minor/major faults and endpoint RSS/swap/memory counters, and partition requests
at the first received content/reasoning event. This is an instrumented
stream-receipt boundary, not an exact backend decode-only interval. Loading is
measured after the first owned-PID snapshot. The runner requires complete SSE
output and exact usage/backend token counts; observation failures invalidate
the run. Failure reports are explicit, credentials are redacted, and only the
owned backend is stopped.

No RAM-disk copy, cache drop, mount modification or model relocation is
performed. Check real backing files and file-cache residency separately, for
example with `findmnt -T /actual/model/path` and `fincore --bytes --json shard.gguf`.
Cached file pages and process RSS overlap. A tmpfs capacity limit is not extra
RAM, and a RAM-disk copy can compete with model/KV/pinned allocations.

The [reviewed warm-control study](results/ram-staging-io-investigation-summary.json)
retains small startup/first-request reads but finds zero physical reads in six
generation windows. It does not reproduce historical UI behavior, qualify
every model, or establish any RAM-disk acceleration.

## Matched Architecture And Accepted-output Study

The [protocol/results](../docs/research/model-agnostic-architecture-study.md),
[reviewed aggregates](results/model-agnostic-architecture-summary.json) and
[Synapse packet](../docs/synapse-prefill-handoff.md#architecture-packet-a-real-throughput-lead-with-explicit-transfer-gates)
separate genuine Strata/Qwen MTP gains from native numerical incompatibility.
This is a single-user, unchanged-target study, not aggregate batching or a
claim that Qwen results establish GPT-OSS-120B performance.

Use existing weights and an already verified pack. Inspect both approved model
roots, real symlink targets, mounts and free space first. Do not download,
remount an archive, copy to a RAM disk, change installed defaults or run the
user's GUI launcher. Use a fresh private result directory and preserve the
16 GiB reserve:

```bash
umask 077
mkdir -m 700 "$RUN"
PYTHONDONTWRITEBYTECODE=1 python3 tools/benchmark/vrambridge-architecture-assets.py \
  --model "$SOURCE_GGUF" --pack "$SOURCE_PACK" --mtp "$SOURCE_MTP" \
  --seed "$TRAINING_ONLY_SEED" --gguf-python "$PINNED_SOURCE/gguf-py" \
  --references "$NEW_APPROVED_ROOT_REFERENCES" --output "$RUN/private-assets.json"
```

The [asset verifier](../tools/benchmark/vrambridge-architecture-assets.py) hashes
complete shards, verifies native offsets/types and dense bytes, and creates
owned zero-copy references. Its partial profile contains only the selected
experts. A complete ranking is **not** a count ceiling: Strata sizes by actual
blob bytes and otherwise admits extra small experts. Verify actual READY
counts, payload and controls before timing; never overwrite the historical
incorrect profile or relabel its original manifest.

Build the [shared renderer](../tools/trace/render_corpus.cpp) with matching
pinned headers/libraries, not experimental headers declaring unavailable APIs:

```bash
cmake -S . -B build/architecture-native-check -DVRAMBRIDGE_ENABLE_CUDA=OFF \
  -DVRAMBRIDGE_TEST_MASKED_IDS=ON \
  -DVRAMBRIDGE_LLAMA_SOURCE="$PINNED_SOURCE" \
  -DVRAMBRIDGE_LLAMA_LIBRARY_DIR="$MATCHING_LIBRARY_DIR"
cmake --build build/architecture-native-check \
  --target vrambridge-render-corpus vrambridge-logit-check
TZ=UTC build/architecture-native-check/vrambridge-render-corpus \
  --model "$MODEL" --corpus benchmarks/cpu-decode-corpus.jsonl \
  --output "$RUN/private-shared-tokens.json"
```

Check tokenizer parity separately. Feed these same IDs directly to both
engines; do not independently render chat endpoints. The
[comparison adapter](../tools/benchmark/vrambridge-engine-comparison.py) owns
each process, freezes cache controls, cleans inherited research overrides,
checks actual residency/counts, measures process resource intervals and closes
the backend even on failure. Strata uses a direct protocol adapter imported
read-only with bytecode writes disabled. An example confirmation arm is:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/benchmark/vrambridge-engine-comparison.py \
  --engine strata --mode mtp --window 2 --tokens 512 \
  --model "$MODEL" --pack "$PACK" --mtp "$MTP_RUNTIME_PACK" \
  --strata-source "$STRATA_SOURCE" --strata-binary "$STRATA_BINARY" \
  --strata-profile "$RUN/common-strata-profile.bin" \
  --token-corpus "$RUN/private-shared-tokens.json" \
  --output "$RUN/strata-mtp-fixed-prefill-r0.json"
```

Use `--mode scalar` for its control, preserving capacity two. Collect three
fresh pairs in A/B, B/A, A/B order across all three workloads. The adapter
explicitly disables dynamic CPU prefill sharing, tail skipping, adaptive
residency, prompt reuse, lookup drafts and cold-GPU offload. Earlier default
prefill-share runs are exploratory and must not be pooled with confirmation.

For native scalar use `--engine native --mode scalar`, supplying
`--native-binary`, the exact `--cuda-plugin`, and `--native-plan`.
`--mode canonical` is a different execution contract; `--mode mtp` additionally
requires `--native-draft`. A restorable native draft can be produced with the
[MTP derivative tool](../tools/benchmark/vrambridge-restore-mtp.py) from the
verified existing raw tensors. It preserves target embedding/head bytes, reuses
pinned tensor transforms, retains F32 routers, writes only to approved model
storage and produces an ownership manifest. Structural loadability is not
quality qualification. No installed Strata/native files are rewritten.

Native streamed raw IDs can omit a UTF-8-buffered token. The adapter retains
reported generation positions and audits every timed request using a complete
nonstream replay **after all timed requests**. Every streamed position and full
prompt/output count must agree; complete hashes come from the audited output.
Replay time is not generation throughput. The common rate is accepted outputs
minus one over wall time after the first output, not engine-specific timing
fields whose first-token conventions differ.

Render [bounded sanity cases](architecture-sanity-corpus.jsonl) into a separate
private token corpus and use `--natural-eos`. Check the exact expected JSON
answers separately; report stop/cap failures and do not execute generated code.
Natural-EOS and forced-cap results must not be pooled. Raw tokens, logits and
logs stay private. The [aggregate publisher](../tools/benchmark/vrambridge-architecture-summary.py)
accepts the fixed confirmation cohort and rejects mismatched counts, hashes,
dynamic prefill or target-row accounting before producing an allowlisted
summary. Its CLI refuses to overwrite reviewed evidence.

The executed serve verifier does not support a meaningful `--no-token-graph`
ablation. Do not use that flag as a proxy. Matching greedy output hashes and
passing bounded sanity cases also do not replace the unchanged full-vocabulary
gate or demonstrate that native optimized execution is interchangeable.

## Optimized-preserving Native Verification And Cold KV Experiment

The [study](../docs/research/optimized-verification-study.md) and
[reviewed aggregate](results/optimized-verification-summary.json) distinguish
new exact native verification from the older incompatible canonical mode.
**These are default-off research tools, not a faster default or a 60 tok/s
result.** Exact CPU repacking and selective-KV DFlash both lost real generation
throughput in calibration.

Use the original matching patched source headers and installed libraries, not
experimental headers declaring APIs absent from those libraries:

```bash
cmake -S . -B build/architecture-native-check -DVRAMBRIDGE_ENABLE_CUDA=OFF \
  -DVRAMBRIDGE_TEST_MASKED_IDS=ON \
  -DVRAMBRIDGE_TEST_ATTENTION_WINDOW=ON \
  -DVRAMBRIDGE_TEST_KV_COLD_TIER=ON \
  -DVRAMBRIDGE_TEST_EXACT_PACKED_CPU=ON \
  -DVRAMBRIDGE_CPU_REFERENCE_LIBRARY="$MATCHING_CPU_LIBRARY" \
  -DVRAMBRIDGE_LLAMA_SOURCE="$PINNED_SOURCE" \
  -DVRAMBRIDGE_LLAMA_LIBRARY_DIR="$MATCHING_LIBRARY_DIR"
cmake --build build/architecture-native-check --target \
  vrambridge-logit-check vrambridge-attention-window \
  vrambridge-attention-window-test vrambridge-kv-cold-tier \
  vrambridge-kv-cold-tier-test vrambridge-kv-cold-gpu-test \
  vrambridge-exact-packed-cpu vrambridge-exact-packed-cpu-test -j4
ctest --test-dir build/architecture-native-check --output-on-failure \
  -R 'vrambridge-(engine-comparison|attention-window-views|kv-cold-tier-roles|exact-packed-cpu|logit-check-inputs)$'
PYTHONDONTWRITEBYTECODE=1 python3 tests/test_kv_cold_gpu.py \
  build/architecture-native-check/vrambridge-kv-cold-gpu-test \
  "$CUDA_PLUGIN" build/architecture-native-check/libvrambridge-kv-cold-tier.so \
  "$MATCHING_LIBRARY_DIR/libggml-base.so"
```

The KV module uses the explicit CUDA12 runtime/header installation, without
building CUDA kernels or modifying the installed plugin. The opt-in actual-GPU
test requires a compatible GPU; it is invoked separately from CPU-only role
tests. It verifies managed KV, completed CPU prefetch, exact GPU roundtrip,
normal/mixed device allocations, views and owner drain.

For full-vocabulary qualification, preload the window module, set
`VRAMBRIDGE_ATTN_WINDOW=split`,
`VRAMBRIDGE_ATTN_WINDOW_LIBRARY="$MATCHING_LIBRARY_DIR/libllama.so"`,
`VRAMBRIDGE_ATTN_WINDOW_PRODUCTS=columns` and an exclusive private
`VRAMBRIDGE_ATTN_WINDOW_OUT`. Keep global scalar/canonical flags off.
Use the logit checker's original optimized scalar `--reference-in`, fixed
`--template-epoch 0` with `TZ=UTC`, `--steps 96`,
`--batch-check 2`, `3` or `4`, and `--rollback-check`. Preserve all frozen
target/model/thread/KV controls. Tail-removal rollback can be unsupported;
do not substitute a success-shaped fallback.

To additionally test selective KV, compose its preload, set
`VRAMBRIDGE_KV_COLD_MODE=stage`,
`VRAMBRIDGE_KV_COLD_BASE_LIBRARY="$MATCHING_LIBRARY_DIR/libggml-base.so"`
and an exclusive private `VRAMBRIDGE_KV_COLD_OUT`. Global
`GGML_CUDA_ENABLE_UNIFIED_MEMORY` must be absent or zero. Only recognized
F16 K/V contexts with one backing buffer are supported; mixed contexts are
not made managed. Require positive owner/stage/managed-allocation counters,
zero outstanding owners, and original full-logit thresholds.

After numerical/rollback qualification, a bounded live research arm is:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/benchmark/vrambridge-engine-comparison.py \
  --engine native --mode dflash --window 2 --tokens 256 --split train \
  --cache-mib "$CACHE_MIB" --experts "$EXPERT_COUNT" \
  --attention-window build/architecture-native-check/libvrambridge-attention-window.so \
  --verification-threads decode --expert-products columns \
  --kv-cold-tier build/architecture-native-check/libvrambridge-kv-cold-tier.so \
  --model "$TARGET_GGUF" --native-draft "$VERIFIED_PROPOSAL_ONLY_DRAFT" \
  --native-binary "$MATCHING_SERVER" --native-plan "$FROZEN_TRAINING_PLAN" \
  --cuda-plugin "$CUDA_PLUGIN" --token-corpus "$RUN/private-shared-tokens.json" \
  --output "$RUN/new-dflash-cold-kv.json"
```

The adapter composes the opt-in preloads, captures physical NVML/host memory,
checks source/header identities and counters, audits complete outputs outside
timing, and owns backend cleanup. Compare with fresh `--mode scalar` using the
same raw prompts and frozen controls. The GPT study uses 12,000 MiB and 944
selected experts; do not reuse those counts for other tensor layouts.

`--expert-products native` is an explicit **numerically failing ablation**,
not a qualified fast path. `--cpu-repack --exact-packed-cpu ...` is a separate
exact but slower CPU experiment. `--managed-memory` is a separate global mode
and is incompatible with selective KV. Neither the backend's global-managed
free-memory log nor logical prefetch-byte counters establish physical VRAM
savings or measured PCIe transfers. Promote only a calibrated real gain to
fresh alternating held-out confirmation; the known-losing screens were not
promoted. Keep raw tokens/logits/logs private and do not change user defaults.

## GPU-window CPU-cache Staging

The [staging study](results/gpu-window-cpu-cache-summary.json) separates two
questions: do measured GPU windows provide bandwidth headroom, and does warming
the same CPU cache domain help unchanged CPU expert computation? It does not
enable live staging, predict routes or replace expert weights.

### Aligned private CUDA/CPU capture

Build the explicitly enabled CPU profiler with optional header-only NVTX:

```bash
cmake -S . -B build/cache-window-probe \
  -DCMAKE_BUILD_TYPE=Release -DVRAMBRIDGE_ENABLE_CUDA=OFF \
  -DVRAMBRIDGE_CPU_RESEARCH=ON -DVRAMBRIDGE_CPU_PROFILE_NVTX=ON \
  -DVRAMBRIDGE_LLAMA_SOURCE=/path/to/pinned-source \
  -DVRAMBRIDGE_LLAMA_LIBRARY_DIR=/path/to/matching/libraries
cmake --build build/cache-window-probe \
  --target vrambridge-cpu-graph-profile vrambridge-hybrid-layer \
           vrambridge-cpu-mxfp4-vnni vrambridge-cpu-profile-hint-test -j4
ctest --test-dir build/cache-window-probe \
  -R 'vrambridge-(cache-window|cpu-cache-staging|cpu-research|hybrid-layer)' \
  --output-on-failure
```

NVTX annotations do not require compiling CUDA kernels. Nsight CUDA tracing
was available without CPU perf-event permissions. Do not change machine
permissions or profiler policy to reproduce this study.

Scope `LD_PRELOAD` to the new profile library, set the actual native CPU
plugin in `VRAMBRIDGE_CPU_PROFILE_LIBRARY`, and use a new private output path
in `VRAMBRIDGE_CPU_PROFILE_OUT`. `VRAMBRIDGE_CPU_PROFILE_EXPERT_IDS=1` requires
an owned 0700 parent directory. It records actual cold IDs, pre-admission
selected IDs/slot-map misses and opaque tensor identities; **do not publish
this file or the Nsight report/database**. Output is 0600 and only complete
after a clean process exit.

Wrap the existing owned request runner with:

```bash
# The /private parent must already be owned and mode 0700.
umask 077
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --cuda-graph-trace=node --wait=all --force-overwrite=false \
  --output=/private/new-capture \
  python3 tools/benchmark/vrambridge-io-study.py [matched request arguments]
nsys export --type=sqlite --force-overwrite=false \
  --output=/private/new-capture.sqlite /private/new-capture.nsys-rep
chmod 600 /private/new-capture.nsys-rep /private/new-capture.sqlite

python3 tools/benchmark/vrambridge-cache-window.py \
  --cpu-profile /private/cpu-events.jsonl \
  --timeline /private/new-capture.sqlite \
  --request-report /private/request.json \
  --cache-mib 1 8 16 32 64 --warm-gbps 20 40 --tile-kib 64 \
  --output benchmarks/results/window-headroom.json
```

Nsight can ignore the caller's umask for its report file; the private parent
must protect it throughout capture, and final file permissions are explicit.

Use `TZ=UTC`, native fastest wait behavior, an unloaded GPU and the same
16K context/12,000 MiB bank/8+16 threads/2048+512 batches as the control.
Compare exact output hashes first. Private CPU schema 3 and complete matching
NVTX sequence/process coverage are required; name-only ancestry is rejected.
NVTX separates graph execution from observer metadata/recording overhead.
The analyzer intersects actual kernel/copy intervals, debits shared DRAM
traffic and bounds next-demand bytes by cache capacity, bandwidth and tile
granularity. Unknown copies or unobservable cold work fail explicitly.

Perfect-next-demand is an oracle, not a policy. Fresh exact IDs and earlier
pre-admission snapshots are separate causal scenarios. The previous cold-set
control spends bandwidth on wrong predictions; no future filtering is allowed.
Bounds assume ideal cache retention and zero helper overhead and do not
produce throughput predictions. Instrumentation materially slows the request.

### Owned-buffer cache calibration

The optional layer-probe mode uses original selected real weights, native
CPU FFN operations and one synthetic activation row. It needs no GPU:

```bash
taskset -c 0-7 build/cache-window-probe/vrambridge-hybrid-layer \
  --model /models/gpt-oss-120b-MXFP4-00001-of-00002.gguf \
  --cpu-cache-staging --cpu-experts 1 --cache-warm-mib 16 \
  --warm-cpu 0 --threads 8 --repetitions 24 --layer 17 \
  --output benchmarks/results/cache-calibration.json
```

Verify actual cache-sharing topology before selecting CPUs. On this host,
0-7 share a 32 MiB L3, while CPU 8 belongs to the other CCD. Warming on CPU 8
is a deliberately separate negative control, not extra uniform cache capacity.
The helper affinity is explicit; its target can be outside the compute mask.
Only the probe's owned native projection buffers are CLFLUSHed. Persistent
original files, kernel page cache and system mappings are unchanged.

The warmer touches a prefix of gate/up/down weights in graph order, one word
per cacheline; the budget is 0-64 MiB and expert count 1-4. Zero budget measures
helper handoff. Cold/warm order alternates and every output must match native
CPU bits. `--self-test --cpu-cache-staging` validates F32 fixtures. Cache
options require the opt-in research build and reject incompatible split,
capture, batching and GPU modes before backend initialization.

Reports keep cold compute, warm compute, helper touch/handoff and serial
warm-plus-compute separate. Flush/allocation/loading/helper construction are
excluded. A warm compute speedup is **not** a whole-workload speedup: serial
warming loses, actual windows are fragmented and same-CCD residency is vital.
No predictor or runtime prefetch is promoted by these calibrations.

## Native CPU/GPU Handoff Follow-up

The [handoff results](results/handoff-critical-path-summary.json) use native
installed libraries, no warmer, original context/bank/thread settings and
ordinary CPU affinity. Use the private CUDA/CPU capture procedure above, but
build the profiler against the actual installed server's library directory.
Backend mappings must match the no-preload control.

```bash
python3 tools/benchmark/vrambridge-critical-path.py \
  --cpu-profile /private/native-cpu.jsonl \
  --timeline /private/native.sqlite \
  --request-report /private/native-request.json \
  --output benchmarks/results/native-critical-path.json
```

The analyzer reuses strict timeline alignment and interval helpers, partitions
CPU-only/GPU-only/overlap/neither wall time, and separately reports CUDA API
and observer subsets. Kernel/copy/memset activity is included. It never sums
API elapsed time into saved latency or predicts model tok/s. Kernel stream
IDs and time-range overlap alone do not prove actual concurrency.

The optional research build also supplies a scalar-control interposer:

```bash
cmake --build build/critical-path-probe \
  --target vrambridge-cpu-control-threads vrambridge-cpu-control-threads-test -j4
ctest --test-dir build/critical-path-probe \
  -R 'vrambridge-(cpu-control-threads|critical-path|handoff-study)' --output-on-failure

python3 tools/benchmark/vrambridge-handoff-study.py \
  --model /models/gpt-oss-120b-MXFP4-00001-of-00002.gguf \
  --profile /path/to/checked/hybrid-profile.json \
  --library build/critical-path-probe/libvrambridge-cpu-control-threads.so \
  --cpu-library /path/to/installed/libggml-cpu-zen4.so \
  --variant cpu-one --pairs 3 --label control-one \
  --output-dir /private/new-handoff-results
```

Configure that build with `VRAMBRIDGE_CPU_RESEARCH=ON`, the pinned headers and
the exact installed library directory, as for the live-warmer build below.
The helper only changes the local plan's worker count for small scalar
slot-map dispatch/copy graphs. The expert plans, native operations and caller's
8/16 settings remain unchanged. `VRAMBRIDGE_CPU_CONTROL_THREADS=one`,
`VRAMBRIDGE_CPU_CONTROL_LIBRARY` and a new private
`VRAMBRIDGE_CPU_CONTROL_OUT` activate the low-level preload;
`VRAMBRIDGE_CPU_CONTROL_THREADS=off` is pass-through and writes no report.
Do not combine it with the warmer or active CPU profiler.

The paired runner reuses the owned-request measurement API, reverses pair and
category order, restores the parent environment and cleans temporary prompts.
It checks exact hashes, context/bank, runtime environment, matching backend
libraries and actual changed-plan/PID markers. Its default CPU-one variant
passes the original long 96-position gate but has mixed request timings, so
is not a default recommendation.

`--variant qkv --categories coding --pairs 3 --label qkv-screen` separately
screens native `GGML_CUDA_GRAPH_OPT=1` against zero. No CPU preload is used
for that screen. This option did not demonstrate a speedup and its
concurrency metadata must not be presented as a successful parallel executor.

## Live CPU Cache Warmer

The [live follow-up](results/live-cpu-cache-warmer-summary.json) implements the
actual dispatch-triggered helper as a separate research preload, not an
installed plugin replacement. It passes numerical checks but has **no
demonstrated end-to-end speedup**. Keep it default-off.

```bash
cmake -S . -B build/live-warmer-probe \
  -DCMAKE_BUILD_TYPE=Release -DVRAMBRIDGE_ENABLE_CUDA=OFF \
  -DVRAMBRIDGE_CPU_RESEARCH=ON \
  -DVRAMBRIDGE_LLAMA_SOURCE=/path/to/pinned-source \
  -DVRAMBRIDGE_LLAMA_LIBRARY_DIR=/path/to/matching/libraries
cmake --build build/live-warmer-probe \
  --target vrambridge-cpu-live-warmer vrambridge-live-warmer-test -j4
ctest --test-dir build/live-warmer-probe \
  -R 'vrambridge-live-warmer($|-study)' --output-on-failure
```

On the measured host, use the same 16-logical-CPU CCD mask for every variant,
not eight logical CPUs while still requesting 16 batch threads:

```bash
taskset -c 0-7,16-23 python3 tools/benchmark/vrambridge-live-warmer-study.py \
  --model /models/gpt-oss-120b-MXFP4-00001-of-00002.gguf \
  --profile /path/to/checked/hybrid-profile.json \
  --warmer build/live-warmer-probe/libvrambridge-cpu-live-warmer.so \
  --cpu-library /path/to/matching/libggml-cpu-zen4.so \
  --helper-cpu 16 --warm-kib 1024 --maximum-us 200 \
  --category coding --pair 0 \
  --output-dir /private/new-live-results
```

Verify local topology before reusing that mask. The helper shares an existing
physical core, and binding can affect prefill; it is not free compute capacity.
The driver requires an owned private output directory and the checked
16K/12,000 MiB profile. It runs owned native/noop/read requests, reverses order
for pair 1, attests the helper PID/configuration, preserves exact prompts/counts,
and refuses overwrites. It also requires identical mapped backend libraries
across every triplet. Build the warmer against the **actual installed server's
library directory**, not just another ABI-compatible research build: preload
dependencies can otherwise replace the control's GGML base. The first cohort
exposed that mismatch and was rerun rather than credited.
Repeat pairs 0/1/2 for coding/reasoning/prose.
Omit `taskset` and choose a fresh label to screen unrestricted behavior, without
calling pinned/unpinned differences warming gains.

The lower-level controls are scoped only to the owned backend child:

| Environment variable | Meaning |
| --- | --- |
| `LD_PRELOAD` | Absolute research warmer shared-library path; do not combine with the CPU graph profiler |
| `VRAMBRIDGE_CPU_WARM_MODE` | `off` passes native graph calls through; `noop` runs a no-read helper; `read` touches cachelines |
| `VRAMBRIDGE_CPU_WARM_LIBRARY` | Explicit already-loaded matching native CPU library for original graph dispatch |
| `VRAMBRIDGE_CPU_WARM_OUT` | New private aggregate JSON path; complete only after clean shutdown |
| `VRAMBRIDGE_CPU_WARM_CPU` | Explicit helper logical CPU, verified through thread affinity |
| `VRAMBRIDGE_CPU_WARM_KIB` | 0-16,384 KiB per-job payload cap; tested default 1,024 KiB |
| `VRAMBRIDGE_CPU_WARM_MAX_US` | 1-5,000 us wall budget checked at chunk boundaries, not a real-time completion guarantee |

The helper uses current exact selected IDs and the pre-admission slot map.
Its source registry stores bounded descriptors of original contiguous host
MXFP4 tensors learned from native CPU graphs; missing first-use sources are
explicitly counted, not fabricated as warm. No raw IDs, pointers or prompts
are written to the aggregate report.

A latest-only queue bounds pending work. Reads are one byte per 64-byte
cacheline of a bounded native-layout prefix. No buffers are modified, copied
or flushed. Any CPU expert graph cancels outstanding work without waiting,
but up to a 64 KiB chunk can finish afterward; those tails and active-reader
entries are charged. Original buffer deallocation cancels and drains reads
before release. The pinned model owns mappings longer than its tensor buffers.
No helper accesses graph metadata after a graph call or after model unload.

Post-admission snapshots sample completed chunks for actual CPU IDs versus
GPU-admitted IDs. Bytes touched do not prove cache residency or saved time.
Timeout/cancellation and whole-request CPU/latency effects remain in timings.
`off` creates no helper/report; native with no preload remains the control.
The prototype does not gate reads on observed GPU activity or predict experts.

Optional `VRAMBRIDGE_CPU_PROFILE_NVTX=ON` adds live helper-job and original
CPU expert/control ranges. Capture with the private Nsight procedure above.
Those runs are diagnostic only. In the observed live trace, about 0.012%
of helper-job time overlaps GPU kernels, explaining why the isolated warm-cache
ratio does not become a request speedup. Do not compare instrumented tok/s
against the uninstrumented matrix.

## CPU Decode Roofline And OpenMP Wait Policy

The [CPU follow-up](results/cpu-decode-efficiency-summary.json) measures actual
request CPU consumption separately from latency. It does not install a new
backend or change application defaults. The checked CPU plugin uses GNU
OpenMP; its idle-worker spin count is not the native pthread-pool `--poll`
parameter.

For a deliberate lower-CPU-use experiment, scope the setting to the owned
backend/client process, clear other pool overrides and keep all model settings
matched:

```bash
env -u LD_PRELOAD -u OMP_WAIT_POLICY -u OMP_PLACES -u OMP_PROC_BIND \
  -u GOMP_CPU_AFFINITY TZ=UTC GOMP_SPINCOUNT=1000 \
  python3 tools/benchmark/vrambridge-io-study.py \
  --llama-server /path/to/pinned/llama-server \
  --hybrid-profile /path/to/checked/hybrid-profile.json \
  --model /models/gpt-oss-120b-MXFP4-00001-of-00002.gguf \
  --context 16384 --threads 8 --batch-threads 16 \
  --batch-size 2048 --ubatch-size 512 --tokens 512 --repetitions 1 \
  --verbatim-prompt --prompt-file /private/heldout-user-prompt.txt \
  --output benchmarks/results/low-cpu-request.json
```

The native control removes `GOMP_SPINCOUNT` as well; alternate baseline and
candidate order across fresh processes. `--verbatim-prompt` uses the file as
the exact user message, rather than adding the older technical-review wrapper.
Reports attest the selected CPU environment from the backend process and
record mapped CPU modules without recording other environment values.
`--cpu-poll 0-100` selects the native backend pool parameter only; it is not a
portable replacement for the GNU OpenMP environment setting.

Three paired coding/reasoning/prose workloads and a separate long-prompt
matrix retain exact output hashes and the 96-position logit gate. The tested
low-spin setting reduces generation-window process CPU by 27-32% while median
decode throughput falls about 4%. Whole-request CPU falls roughly 20-22% on
short prompts but only 3% on the prefill-heavy long prompt. CPU-seconds are
not joules, instantaneous utilization, or a general hardware energy result.
Keep the fastest default; this is an optional resource/latency tradeoff.

### Privilege-free CPU graph diagnostics

The optional Linux x86 build provides an explicit preload profiler and an
experimental MXFP4/VNNI dot replacement. The latter is a negative performance
result, not an implementation recommendation. It requires the tested
AVX2/FMA/F16C/AVX512-VNNI/VL host and paired 32-value blocks with `nrc=1`;
do not preload it on arbitrary models or machines.

```bash
cmake -S . -B build/decode-cpu-probe \
  -DCMAKE_BUILD_TYPE=Release -DVRAMBRIDGE_ENABLE_CUDA=OFF \
  -DVRAMBRIDGE_CPU_RESEARCH=ON \
  -DVRAMBRIDGE_LLAMA_SOURCE=/path/to/pinned-source \
  -DVRAMBRIDGE_LLAMA_LIBRARY_DIR=/path/to/matching/libraries
cmake --build build/decode-cpu-probe \
  --target vrambridge-cpu-graph-profile vrambridge-cpu-mxfp4-vnni \
           vrambridge-hybrid-layer vrambridge-cpu-profile-hint-test -j4
ctest --test-dir build/decode-cpu-probe \
  -R 'vrambridge-(cpu-research|prefill-seed)' --output-on-failure
```

To profile, use process-scoped `LD_PRELOAD` pointing to the graph-profile
library, `VRAMBRIDGE_CPU_PROFILE_LIBRARY` pointing to the actual native CPU
plugin and `VRAMBRIDGE_CPU_PROFILE_OUT` to a new private JSONL path. The
profiler interposes the original CPU graph entry without adding a tensor
observer or changing graph operations. It buffers bounded records and writes
them on clean exit. Missing activation, incomplete output or abnormal exit
invalidates the trace. Profile timings are diagnostic, not throughput evidence.

Read-only external expert IDs are counted only when not produced inside the
current graph and stored in host memory. Unknown accounting remains null,
never fabricated as zero. Raw IDs/activations are not recorded. CPU graph spans
may overlap GPU work; their sum is not marginal critical-path savings.

### Prompt-conditioned seed replay

The [CPU decode corpus](cpu-decode-corpus.jsonl) has three training and three
distinct held-out requests. Collect both `--prefill-output` and `--trace-output`
with the existing collector, explicitly load the pinned CUDA plugin and retain
the callback-inactive continuation checks. The native CPU-expert reference
does not represent installed hybrid placement or inference timing.

```bash
python3 tools/benchmark/vrambridge-prefill-seed.py \
  --trace /private/native-decode-routing.jsonl \
  --prefill /private/native-prefill-observations.json \
  --baseline-plan benchmarks/results/gpt120-trained-hot944.txt \
  --ram-gbps 44.18902426 --pcie-gbps 28.520 --gpu-expert-us 0 \
  --output benchmarks/results/prefill-seed-replay.json
```

The analyzer reuses the strict routing reader and costed admission simulator.
It compares external trained, current-training-only, prompt-only, fixed
50/50 prompt/training and empty seeds at the same payload-byte budget.
Partial/pruned prefill layers use training priors. Held-out decode cannot
select the seeds. Replacement uploads are charged before decode; original
initial loading is common/excluded. The causal replay is not a native-policy
emulator, and zero GPU cost is deliberately optimistic. Modeled expert-stage
milliseconds are never reported as model tok/s or request latency.

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

The checker caps both total and per-sequence output reservations to its decode
group (one by default). This avoids reserving a logical batch of vocabulary
outputs and exhausting VRAM at the serving bank/context. It still reads every
vocabulary entry at every requested position; do not shrink the bank or context
to make a comparison fit. Reports record actual logical/physical batch sizes
and rendered prompt-token counts. Its fixed helpful-assistant system message
and template epoch differ from the user-only HTTP sampler unless deliberately
aligned; label a held-out check accordingly, not as the identical HTTP render.

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