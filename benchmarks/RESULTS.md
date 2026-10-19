# Benchmark Ledger

This ledger is the reviewed index for benchmark evidence committed under `results/`.
It records the conditions required to compare runs, links raw artifacts, and
separates valid measurements from failed-command evidence. New result JSON is
ignored by default and is added here only after review.

Published JSON uses `benchmark-host` for the hostname and `/models` for
machine-specific model storage roots, including paths embedded in commands and
captured output. Model filenames, measurements, timestamps, and diagnostics are
otherwise unchanged. These placeholders are not executable local paths.

## Test Host

- AMD Ryzen 9 9950X: 16 physical cores
- 123.48 GiB host RAM
- NVIDIA GeForce RTX 5070 Ti: 15.45 GiB VRAM
- PCIe 5.0 x8; measured pinned H2D: 28.520 GB/s
- measured sequential RAM read: 42.160 GB/s
- model storage: Samsung 990 PRO NVMe
- llama.cpp: `b21e4de74567f5eef213765c9476a843c2e43f0d`

Unless a row states otherwise, measurements use three repetitions, a 512-token
prompt, a 32-token generation, and a 1,024 MiB auto-fit margin at a 4,096-token
fit context.

## Optimized Native Verification: Numerical Blocker Removed, No Throughput Win

The [reviewed aggregate](results/optimized-verification-summary.json) and
[full study](../docs/research/optimized-verification-study.md) qualify an
optimized-preserving native verification path without the older global
canonical-attention change. Tested GPT/Qwen windows pass with zero logit
error; GPT rejected-suffix rollback passes. Qwen's tested tail-removal-only
rollback remains unsupported, not universally impossible.

Selective managed allocation of cold F16 KV, with ordinary device weight
allocations and global unified memory off, passes a real GPU roundtrip and
864 full-vocabulary positions including 270 rejected-suffix windows. It
retains 16K context and original precision. Physical NVML memory remains
near full: logical prefetch bytes are not measured migration or VRAM savings.

One fresh 256-output training/calibration pair gives:

| Workload | Original scalar tok/s | Two-row DFlash + selective KV tok/s | Change |
| --- | ---: | ---: | ---: |
| Coding | 41.03 | 37.49 | -8.63% |
| Reasoning | 25.10 | 24.46 | -2.56% |
| Prose | 35.65 | 30.51 | -14.40% |

All complete outputs match, with zero physical generation reads and zero
owned-process swap in both arms. Exact CPU packing also passes 3,432
low-level checks and full logits, but its approximately 2.1x isolated gain
becomes a 9-12% real-generation loss. Removing expert-column splitting fails
the unchanged numerical gate at widths two and four; restoring it is exact.

These screens are not promoted to held-out 512-output confirmation cohorts.
**No sustained GPT-OSS-120B 60 tok/s result or Synapse rollout approval.**
Installed backends, original weights, Strata and Synapse remain unchanged.
Earlier canonical failures and the intermediate CUDA failure are retained.

## Shared Architecture Study: Repeated Qwen MTP Gain, Native Transfer Blocked

The [reviewed summary](results/model-agnostic-architecture-summary.json) and
[frozen protocol with amendments](../docs/research/model-agnostic-architecture-study.md)
compare the same original Qwen3.8-Flash-Next IQ3_S bytes and exactly 3,067
resident experts (5,986.94 MiB logical payload). These are 16K/F16-KV,
single-request, 512-output-token tests on the Ryzen 9950X / RTX 5070 Ti host,
using existing root-SSD weights, not the older default host/storage conditions
above. Native uses 8/16 threads; Strata uses seven pool workers plus its host.

The final Strata 0.1.43 cohort disables dynamic CPU prefill sharing, tail
skipping, adaptation, lookup drafting, prompt reuse and cold-GPU offload.
Three fresh alternating scalar/two-row-MTP pairs give:

| Workload | Scalar median tok/s | Two-row MTP median tok/s | Ratio-of-medians gain | Paired gain range |
| --- | ---: | ---: | ---: | ---: |
| Coding | 47.67 | 57.20 | +20.01% | +16.79% to +22.26% |
| Reasoning | 34.37 | 43.85 | +27.57% | +25.58% to +29.46% |
| Prose | 38.43 | 45.26 | +17.77% | +13.19% to +18.49% |

The common rate is `(accepted outputs - 1) / wall time after first output`,
not draft proposals or aggregate batching. All nine pairs have identical full
512-token output hashes; each configuration also repeats the same hash across
all three runs. Median paired gains differ from ratios of medians: +19.22%,
+28.10% and +14.43%, respectively. Median paired whole-request throughput
improvements are +17.77%, +26.31% and +13.49%. Every paired gain is positive.
The second and third pairs have zero physical generation reads in both arms;
the first pair's nonzero reads remain visible. All owned-process swap snapshots
are zero. No energy-saving claim is made.

The mechanism has measurable cost evidence: two-row windows deliver about
1.77/1.89/1.73 outputs per post-first window while costing about 1.48 scalar
steps, not two. Target rows actually increase by about 12.7%/5.7%/15.4%, yet
accepted-output latency falls. This supports shared-cost amortization, not
expert dropping or a claim that disk performs computation. A graph-only speed
attribution is unavailable: `--no-token-graph` does not disable this native-pack
serve verifier.

**Not a drop-in Synapse optimization.** Native optimized scalar medians are
37.94/29.70/31.93 tok/s under their separate frozen control. Native MTP screens
are mixed, and canonical versus optimized fails the unchanged full-vocabulary
gate (worst max-absolute 1.97075, relative L2 0.11157). Canonical block sizes two
and four are exact at 96 checked positions against their own canonical reference.
Strata's matching greedy hashes do not establish its full-vocabulary gate.
All 15 separate natural-EOS code-evaluation/arithmetic/extraction sanity cases
pass, but broad quality remains unestablished.

Retained controls include the invalid 3,991-expert initial smoke, dynamic-prefill
exploratory cohorts, the native streaming UTF-8-ID omission and its nonstream
audit, and the failed native MTP opt-in/collector stop-reason attempts. Native
and Strata use different draft quantization/vocabulary conventions. The general
lead is cost-aware speculation gated by capabilities and numerical contracts,
not a universal two-row constant. GPT-OSS-120B 60 tok/s remains unmet.

## Native Handoff Critical Path: Small Exact Candidate, Mixed Request Results

The [handoff study](results/handoff-critical-path-summary.json) keeps the warmer
disabled and uses the exact installed CPU/CUDA/base libraries. It separates
useful compute, synchronization/API calls and unassigned gaps rather than
assuming every interval between CPU graphs is GPU work.

### Warmer-free diagnostic partition

A native 128-output CUDA/NVTX trace has a 3.068-second envelope bounded by
fully contained CPU graph calls inside the first-text generation interval:

| Disjoint wall-time category | Milliseconds |
| --- | ---: |
| CPU graph execution without overlapping GPU work | 1,304.8 |
| GPU work without overlapping CPU graph execution | 711.5 |
| CPU graph and GPU work overlap | 471.5 |
| Neither CPU graph nor GPU work | 580.0 |

CPU expert graphs total 1,729.6 ms; small scalar control graphs total 48.9 ms.
Inside the last category, CUDA API intervals cover about 417.8 ms and observer
bookkeeping about 73.9 ms. Subsets may overlap and must not be added.
This is a perturbed diagnostic, not a prediction that 417.8 ms is removable.
The native code already uses pinned CPU intermediates and CUDA graph replay.

### One-worker scalar controls

A separate opt-in interposer copies the native CPU plan and requests one worker
only for bounded, scalar dispatch/copy graphs with the known slot-map custom
operation. Matrix/expert/arithmetic graphs retain their original plans and
workspaces. No operation, weight, probability or synchronization is removed.

It matches the original long coding/prose/reasoning reference exactly at all
96 full-vocabulary positions. Three fresh alternating 512-output pairs per
workload retain 16K/12,000 MiB/8+16/2048+512, no affinity override, normal
fusion and identical installed backend mappings:

| Workload | Native median tok/s | One-worker controls | Change |
| --- | ---: | ---: | ---: |
| Coding | 51.46 | 51.90 | +0.85% |
| Reasoning | 51.74 | 51.81 | +0.13% |
| Prose | 57.52 | 57.29 | -0.41% |

All 18 request output hashes/counts match within workload. These small, mixed
results do not establish a general optimization, a robust workload selector or
a 60 tok/s result. Keep the source as an uninstalled research tool.

### Other bounded screens

- `GGML_CUDA_GRAPH_OPT=1` registers Q/K/V concurrency metadata and preserves
  the short-corpus 96-position reference exactly. The observed trace still
  reports one kernel stream and more kernel records; metadata registration
  is not execution proof. Three alternating coding pairs show no win:
  about 51.70 native versus 51.36 candidate tok/s. No longer target
  qualification or promotion is pursued after this screen.
- Fresh-process CUDA12 16-byte pinned-copy/stream-sync microbenchmarks measure
  roughly 5.80 us auto, 5.86 us spin, 5.76 us yield and 14.20 us blocking,
  including Python call overhead. There is no spin advantage over auto;
  these are not model throughput measurements or new runtime defaults.

The next meaningful target is a different CPU/GPU handoff protocol with
measured call reduction and proven lifetime/order, not wider CPU instruction
selection, repeated flag tuning or deletion of necessary synchronization.

## Live CPU Cache Warmer: Exact Outputs, No Measured Speedup

The [live prototype study](results/live-cpu-cache-warmer-summary.json) implements
and measures current-route CPU cache warming, rather than relying on the
offline headroom estimate below. Installed backends, GGUF files, Synapse and
application defaults remain unchanged.

The opt-in shared-library interposer learns original CPU projection buffers
from native graphs, snapshots real selected IDs and the pre-admission slot map,
and launches a single bounded helper. It reads a prefix in gate/up/down
consumption order; there is no model copy, cache flush, expert substitution
or new quantization. CPU expert execution cancels the helper at 64 KiB chunk
boundaries and never waits for its completion. Model-buffer release drains
active/pending readers before native deallocation.

### Matched real requests

Final tests retain 16K context, 12,000 MiB bank, 8/16 threads, 2048/512
batches, direct DMA, overlap pipeline and asynchronous IDs. All variants use
the same CCD mask `0-7,16-23`; the helper is on logical CPU 16, sharing
physical core 0, not an extra physical core. This mask preserves 16 logical
CPUs for the unchanged batch-thread count; an initial eight-logical-CPU
attempt was correctly rejected by the launcher and is retained as failed.

Each workload has three fresh native/noop/read triplets, reversing order in
the middle triplet. Read uses a 1 MiB per-job cap and a 200 us checked wall
budget. The no-op helper executes queue/control work without reading weights.

| 512-output workload | Native median tok/s | No-op median tok/s | Live-read median tok/s | Read change vs native |
| --- | ---: | ---: | ---: | ---: |
| Coding | 51.74 | 51.47 | 51.48 | -0.51% |
| Reasoning | 51.92 | 51.82 | 51.80 | -0.23% |
| Prose | 57.60 | 57.20 | 56.97 | -1.09% |

These small differences do not establish either a speedup or a precise
regression outside run variability. All 27 requests have identical output
hashes within workload and exact 512-output counts. Native mean CPU resources
are not reduced by the helper. The unrestricted coding screen also loses:
51.20 native versus 50.35 live-read tok/s in one triplet; do not attribute
differences between pinned and unpinned runs to warming.

Earlier single-sample screens cover 256 KiB, 1/4/8 MiB, helper CPUs 0/16
within the compute CCD and CPU 8 in the other CCD. None demonstrates a clear
win. Those V1 source hashes remain historical. Final provenance checks also
found that the initial preload linked a research GGML base while the no-preload
control used the installed base. That earlier cohort is excluded from clean
timing claims. The warmer was rebuilt against the exact installed library
set, the long logit gate and all 27 requests were repeated, and mapped backend
libraries are now required to match across every triplet.

### Actual work and overlap

Whole-process logical cacheline payload and post-admission read snapshots
are retained for every read run. They distinguish experts actually required
on CPU from experts subsequently admitted to GPU. These are payload and
snapshot counts, not measured DRAM traffic, LLC hits or saved execution time.
Cancellation tails and active-reader-at-expert-entry counts are retained.

In a separate 128-output NVTX/CUDA diagnostic, helper-job intervals overlap:

- GPU kernels: about 0.012%;
- any GPU activity: about 4.6%, mostly copies;
- CPU dispatcher/control graph calls: about 21%.

Categories can overlap, and job ranges include checks and canceled/no-read
jobs rather than individual cacheline timestamps. This is whole-process
diagnostic timing including any scalar initialization/prefill work, excluded
from performance comparisons. The result identifies why this dispatch-triggered
prototype is not the envisioned GPU-window scheduler: most warming occurs
before GPU expert kernels are executing.

### Correctness and safety

Earlier off/noop/read paths preserve a short-corpus 96-position reference exactly.
The final installed-library read path matches the original **long-prompt 96-position**
full-vocabulary reference exactly across coding/prose/reasoning at the
unchanged gates and normal fusion. This is not general answer-quality proof.

Four helper fixture methods cover exact output bits, zero budget, expiry,
invalid configuration, no-dispatch shutdown and repeated model unload/reload.
Post-admission snapshot fixtures charge GPU-admitted work rather than
retrospectively removing it. Address/undefined-behavior sanitizer cases pass
for active, immediate-unload and idle fixtures; native plugins are not
instrumented and leak detection is disabled. Five study-runner methods cover
order, child configuration/PID, cleanup, overwrite rejection and timeout
process-group termination, including mismatched library-set rejection.

**Decision:** keep the live research tool, but do not install, enable or
recommend this warmer as a throughput optimization. The isolated cache effect
is genuine; a useful live result still requires earlier information or a
different execution schedule, not simply more warming bytes.

## GPU-window CPU-cache Staging: Locality Helps, Current Hints Are Late

The [cache-window study](results/gpu-window-cpu-cache-summary.json) tests a
different hypothesis from GPU expert admission or RAM-disk staging: warm
original CPU expert-weight cachelines during GPU work, then execute unchanged
native CPU expert math. No runtime cache warmer or predictor is installed.

### Hardware cache calibration

Twenty-two initial screens and sixty-three repeated real-weight reports
cover layers 0/17/35, one to four selected experts and 1-64 MiB warming
budgets. Repeated reports use three fresh invocations, two warm-ups and
twenty-four paired cold/warm iterations; their middle invocation reverses
case order. Compute is restricted to physical CPUs 0-7, sharing one 32 MiB
L3 domain. A helper warms either CPU 0 in that domain or CPU 8 in the other.

| Calibration | Repeated warm/cold compute speed ratio | Typical warming handoff |
| --- | ---: | ---: |
| One expert, 4 MiB prefix, same CCD | 1.25-1.31x | 0.09 ms |
| One expert, full 13.22 MB projections, same CCD | 3.00-3.13x | 0.28 ms |
| One expert, full projections, other CCD | 0.61-0.62x | 0.27 ms |
| Two experts, 16 MiB prefix, same CCD | 1.65-1.78x | 0.33 ms |
| Three experts, 16 MiB prefix, same CCD | 1.33-1.42x | 0.33 ms |
| Four experts, full 52.88 MB, same CCD | 1.11-1.32x | 1.09 ms |

All cold/warm outputs are bit-exact. Weights/biases, native Q8 conversion,
SwiGLU, probabilities and reduction stay unchanged. Synthetic one-row
activations are explicitly used; this is not actual routed activation replay.
Only owned copied projection buffers are CLFLUSHed, never original files or
the OS page cache. Allocation, initial loading and flush are excluded; helper
wakeup and acknowledgment are charged separately. One helper touches one
word per cacheline, not a second full model copy.

Serial warming plus compute loses: warming must be usefully hidden to help.
No-op helper wake/ack medians are 7.3-8.8 us, already longer than many GPU
fragments. Cross-CCD warming is a substantial negative result, not evidence
that both L3 caches form one uniform 64 MiB pool.

### Actual GPU-window headroom

Private Nsight CUDA/NVTX traces align to CPU graph calls by sequence IDs;
opaque tensor identities avoid repeated-name producer confusion. Analysis
excludes CPU expert execution and observer bookkeeping from candidate windows.
GPU-copy windows debit host-memory traffic; CPU-control-overlap scenarios
remain optimistic about bookkeeping bandwidth.

At an ideal 32 MiB cache budget and assumed 40 GB/s warming:

| 128-output request | Perfect next-demand oracle: payload stageable | Current actual cold-ID readiness | Pre-admission dispatcher snapshot | Previous cold-set control |
| --- | ---: | ---: | ---: | ---: |
| Technical | 27.5% | 0.20% | 0.18% | 2.6% |
| Coding | 32.2% | 0.24% | 0.23% | 3.1% |
| Reasoning | 28.8% | 0.21% | 0.19% | 1.9% |
| Prose | 33.4% | 0.23% | 0.23% | 3.3% |

Pre-admission snapshots use real selected IDs and current slot maps; experts
subsequently admitted to GPU consume modeled warming bytes but save no CPU
work. The previous-cold predictor uses only earlier CPU misses and stages
without filtering by future demand. Its staged-byte precision is only 6-9%.
With the 20 GB/s assumption, oracle coverage falls to about 14-19%.

These are **logical payload bounds**, not saved DRAM bytes, realized cache
hits, removed critical-path milliseconds or predicted tok/s. Ideal models
ignore retention/pollution, helper overhead and fragmented-window scheduling.
The native timed requests keep the original 16K/12,000 MiB/8+16/2048+512
configuration; calibration's fixed-CCD placement is not substituted for it.
Profiler runs are about 10-17% slower and are not performance comparisons.

All four traced requests match their uninstrumented 128-output hashes.
A separate short coding/reasoning/prose preservation check is exact at 96
full-vocabulary positions. The attempted historical long-prompt check timed
out after two exact 32-position requests; it is retained as incomplete, not
declared a 96-position pass. No model-quality or full-model staging gain is
established.

**Decision:** retain CPU-cache staging as a conditional research candidate,
but do not build a live warmer or learned predictor from this evidence alone.
First prove a useful earlier hint under the real residency policy and account
for same-CCD tile lifetime, warming cost, wrong predictions and DRAM contention.

## CPU Decode: Measured Resource Tradeoff, Not A Faster Default

The [CPU decode study](results/cpu-decode-efficiency-summary.json) follows
the CPU cold-expert recommendation instead of continuing GPU-prefill
integration. It identifies a GNU OpenMP wait-policy tradeoff in the installed
CPU backend. No installed backend, model, app default or Synapse file changed.

At fixed 16K context, 12,000 MiB bank, 8/16 threads, 2048/512 batches and the
same trained/adaptive direct-DMA pipeline, `GOMP_SPINCOUNT=1000` reduces CPU
resource use but also slightly reduces generation throughput. Each row is
the median of three fresh-process pairs in A/B, B/A, A/B order.

| Workload | Outputs | Native tok/s | Low-spin tok/s | Generation-window process CPU reduction | Whole-request process CPU reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Coding | 512 | 50.08 | 47.77 | 28.5% | 20.8% |
| Reasoning | 512 | 49.50 | 47.54 | 28.7% | 19.9% |
| Prose | 512 | 55.50 | 53.21 | 31.9% | 22.3% |
| Long technical prompt, 3,577 input tokens | 256 | 45.32 | 43.68 | 27.0% | 3.1% |

The short-request median throughput cost is 3.9-4.6%, with request wall time
4.2-4.5% higher. Long-request median wall time rises 1.7%; its large prefill
explains why a decode-window CPU reduction does not become a 20% whole-request
reduction. The criterion is a per-workload median, not a guarantee that every
individual run loses at most 5%.

All 24 requests retain exact output counts and matched output hashes within
their workload (10,752 output tokens total). The native control and low-spin
setting are exact at all 96 full-vocabulary held-out positions. No tolerance
is widened. This qualifies the observed resource/latency tradeoff, not general
model quality, other runtimes or an energy saving. Package energy counters
are permission-restricted; no privilege or kernel setting was changed.

A diagnostic 512-output request has 6.39 seconds of synchronous CPU graph
spans inside a 10.57-second generation window; routed graphs account for 6.26
seconds. External native IDs account for 247.82 GB of logical projection
weights across 54,861 matmuls. About 485 MB per streamed transition and
38.8 GB/s per CPU-graph second are logical accounting, not measured DRAM
traffic or a hardware ceiling. An 8-thread streaming-read comparison is
44.19 GB/s. Default process use is nearly eight CPU cores throughout; the
low-spin policy reduces that average substantially.

Negative results retained:

- A VNNI integer replacement passes 240 raw bit tests and exact full-model
  96-position logits, but three paired requests lose about 1.3% throughput.
- Close OpenMP placement is flat and spread slower in bounded pilots.
- A separately built non-OpenMP CPU plugin is checked in a byte-identical,
  isolated server runtime. Native-pool polling 50/0/1 shows no robust
  improvement; it is not promoted or full-logit qualified.
- Six native CPU-reference requests supply 768 training and 768 held-out
  routing tokens, with callback-inactive continuations matching. Full
  prompt-conditioned cache rebuilds lose net modeled cost against trained
  initialization plus causal admission on all three held-out requests, even
  with zero marginal GPU execution cost. Better static hit rates are not
  evidence of beating the live adaptive cache.

The recommendation is an **optional lower-CPU-use profile**, GNU OpenMP only,
with the measured speed penalty disclosed. Keep the native fastest default.
Do not ship the diagnostic/VNNI shims, replace CPU plugins or claim 60 tok/s.
Fixed-date logit reproduction must also set `TZ=UTC`: the pinned template
uses local time, so epoch zero alone is not a portable date contract.

## CPU-compatible Packed Down: Bit-exact Raw Arithmetic, Stage Gains Only

The [compatible packed-down study](results/cpu-compatible-packed-down-summary.json)
implements two isolated CUDA raw projections with original CPU Q8_0 inputs,
paired eight-lane FMA accumulation and the native x86 horizontal sum. The
second variant uses byte permutations and DP4A for the integer dot. Original
CPU bias/weighting/reduction preserves the FFN boundary.

Both variants pass **192 checks each**: sixty real-weight layer/expert/row
configurations with two samples, nine actual routed frames repeated three
times, nine fresh-upload controls and thirty-six adversarial fixture samples.
Raw down, native CPU replay and post-processed FFN outputs are bit-exact to
their respective CPU controls. The historical captured full-model boundary
still has the previously measured replay discrepancy, worst relative L2
5.504e-8; it is not relabeled bit-exact.

These are 16-thread CPU versus GPU **raw-stage** times, not full FFN or request
latencies. Medians below are across three fresh invocations, each with two
warm-ups and twenty-four timed iterations. CPU/GPU timing order alternates.

| Actual 64-row frame | Required expert bank | CPU raw ms | Packed-DP4A stage ms, preparation/transfers included | Repeated stage speedup |
| --- | ---: | ---: | ---: | ---: |
| Coding, layer 0 | 90 | 10.323 | 3.249 | 3.18-3.25x |
| Coding, layer 17 | 54 | 6.825 | 2.988 | 2.22-2.29x |
| Prose, layer 0 | 94 | 10.407 | 3.333 | 3.07-3.20x |
| Prose, layer 17 | 54 | 6.682 | 2.981 | 2.24-2.33x |
| Reasoning, layer 0 | 92 | 10.038 | 3.332 | 3.01-3.11x |
| Reasoning, layer 17 | 53 | 6.905 | 2.891 | 2.34-2.43x |

Resident four-expert synthetic-input controls also improve with DP4A:
32-row ratios are 1.36-1.53x and 64-row ratios 1.32-1.50x. The first scalar
integer implementation loses on those concentrated-reuse controls. Fresh
17,625,600-byte down-weight uploads still lose at 16/32 rows; 64-row repeated
ratios span 0.97-1.09x and do not demonstrate a robust upload win. Preserve
existing one-row/decode paths despite tiny isolated one-row timings.

GPU times include native CPU Q8 casting/readback, pageable H2D input/IDs,
projection, D2H all per-rank raw outputs and synchronization. Shared CPU
gate/up/SwiGLU and post-ops are excluded from both raw-stage times, as are
allocation and initial resident-bank upload. Returning raw outputs for CPU
post-ops means the 3.76x input-format byte reduction is not a total-transfer
or throughput multiplier.

All required replay weights are GPU-loaded, not only the original trained
resident subset. No production 12,000 MiB mixed-residency comparison,
full-model KV/Q8 parity or new target-logit test is implied. No change to
Synapse, installed plugins, weights or application defaults was made.
Scoped memcheck/racecheck/synccheck passes for both variants on the actual
94-expert frame; sanitizer timings are excluded. Five focused CUDA-build
CTest targets pass, including fourteen existing/extended CLI methods and six
new report-contract subcases; ordinary non-CUDA probe tests also pass.

Next resolve the optimized CPU-only graph mismatch, integrate the compatible
dot with original CPU post-ops in the separate research backend and rerun
unchanged full-target gates before compact dispatch or request benchmarks.

## Numerical Localization: FP16 Attention State Amplifies the Seed Difference

The [observer-qualified localization](results/resident-down-numerical-localization-summary.json)
finds a concrete propagation path for the layer-0-only GPU-down experiment.
It uses one held-out 3,590-token coding request, first 512-row batches, unchanged
16K context / 12,000 MiB bank / 8+16 threads and matched fusion-off controls.
No latency benchmark or model-quality pass is claimed.

The CPU trace is exact against its capture-free CPU control. Each candidate
trace is exact against a capture-free candidate reference with the original
teacher-forced continuation, at all 32 full-vocabulary positions. All three
trace focuses preserve their respective reference logits. That qualifies the
observer, not the candidate: native-reference max logit error remains 6.332.
The latest diagnostic backend remains exact default-off at 96 optimized-profile
positions.

| Observed checkpoint | Maximum absolute difference |
| --- | ---: |
| Layer 0 FFN output | 5.72e-6 |
| Layer 1 attention normalization | 3.81e-6 |
| Layer 1 raw Q / K / V projections | 1.91e-6 / 1.53e-5 / 9.54e-7 |
| Layer 1 actual F16 K / V cache stores | 0.001953125 / 0.0009765625 |
| Layer 1 attention context | 0.001845 |
| Layer 1 attention output projection | 0.009821 |
| Layer 1 FFN input after normalization | 0.001081 |
| Layer 1 FFN output | 0.046009 |

The Q/K/V weight tensors are original Q8_0. Their FP32 outputs remain close;
the first large directly observed discrete step is **native FP16 K/V storage**.
Nineteen keys and thirty values differ after rounding. Actual stored values
are read from the native cache at verified first-batch positions and match
predicted IEEE-F16 rounding cell-for-cell in both runs. No state is substituted.
Query-F16 rounding replay predicts 195 changes, but the actual internal query
conversion is not captured; the contributions of Q, K and V are not separately
ablated.

Later, native CPU-Q8 FFN input replay changes 1,307 codes and 448 scales.
Layer-1 expert IDs stay unchanged while probabilities differ. Layer 2 first
changes routing: five rows have a changed rank order, only one has a changed
expert set, at zero-based row 138. The path is therefore not merely a final
logit discrepancy or a wrong manual routing substitution.

This narrows the next engineering work to CPU-reference-compatible down
arithmetic/quantized-state preservation and fusion-preserving integration.
It does not show that a higher-precision cache alone fixes the model or that
compaction solves precision. Twenty-two checker-input/self-test methods pass;
raw activations, routing and token/logit references remain private.

## Full-model Resident-down Prototype: Default-off Exact, Enabled Rejected

The [full-target mask prototype](results/resident-down-full-model-prototype-summary.json)
tests the next integration gate, not another isolated layer estimate. A
separate pinned backend applies the unchanged published hybrid patch plus an
[experimental resident-down patch](../third_party/patches/llama-b11669-resident-down-prefill-experimental.patch).
All fourteen baseline hybrid source files matched the installed source before
the experiment. Installed CPU/CUDA plugin files are reused read-only.

The prototype has an explicit prefill API and graph-cache phase key, resident
GPU down weights, native CPU gate/up/SwiGLU and Q8 preparation, original
per-token routes/probabilities, a CPU reduction boundary and unchanged
zero/one-row/decode paths. The user approved protected zero-slot/mask control
before compact dispatch. Padded GPU work is retained, not counted as a saving.

Three held-out prompts (3,590/3,781/2,275 tokens), 32 teacher-forced positions
each, actual 16K context, a 12,000 MiB bank and 8/16 threads give:

| Full-vocabulary comparison | Worst max absolute error | Worst relative L2 | Result |
| --- | ---: | ---: | --- |
| Final default-off build / installed optimized hybrid | 0 | 0 | Pass, 96/96 top-one matches |
| Enabled resident GPU down / optimized hybrid | 3.861 | 0.1813 | Fail |
| CPU-only routing control / optimized hybrid | 3.603 | 0.1613 | Fail |
| CPU-only control / matched fusion-off baseline | 0 | 0 | Pass, 96/96 top-one matches |
| GPU down / matched fusion-off baseline | 10.007 | 0.3719 | Fail |
| Only layer 0 GPU down / fusion-off baseline | 6.332 | 0.1868 | Fail |

Gates remain max absolute <= 0.01 and relative L2 <= 0.001. No context/cache
shrink, active evaluation callback, expert substitution or tolerance change
is used. The new bulk path reports 735 batches, 337,610 rows and 1,350,440
required expert selections, with zero admissions; its enabled run assigns
320,597 selections to resident GPU down and 1,029,843 to CPU down.

A diagnostic CPU oracle compares the live weighted mixed-device FFN outputs
with original CPU down on the same activations. All **735** local comparisons
pass, with worst relative L2 **1.17e-7**, while full-model logits still fail.
The CPU-only control becomes exact when fusion is disabled, separating that
graph/fusion interaction from the remaining GPU-path mismatch. A layer-0-only
failure shows that moving many layers is not necessary to trigger it.
These controls support small-error/downstream-amplification investigation;
the exact first amplifying quantization/routing operator is not established.

Eight focused CTest targets pass, including phase/cache isolation and existing
masked/empty expert primitives. The owned authenticated server passes a
functional smoke request with exactly sixteen output tokens and positive
prefill dispatch counters, then stops. That is neither a quality pass nor a
latency benchmark. Full-prototype sanitizer/peak-memory qualification is not
claimed from earlier operator tests.

**No latency benchmark or compact-dispatch promotion follows a failed target
gate.** The earlier isolated speed results remain scoped to their experiments.
Synapse, installed backends and serving defaults remain unchanged.

## Actual Prefill Inputs: Routed FFN Gates Pass Under Qualified Observer Controls

The [actual-input follow-up](results/actual-prefill-activation-replay-summary.json)
replays original per-token inputs, expert IDs, probabilities and FFN outputs
from coding/prose/reasoning prompts of 3,590/3,781/2,275 rendered tokens. The
native CPU-expert reference uses 16K context, 2048/512 logical/physical batches,
8/16 threads and no MoE bank. This is an explicit numerical reference, not a
shrunk-bank performance comparison.

Capture itself needed qualification. Unchanged and inactive-observer controls
match exactly, but initial active captures on the optimized CUDA profile reach
4.166 maximum logit error and 0.16275 relative L2. Moving reads to CPU source
tensors does not remove that perturbation. With CUDA fusion disabled for both
reference and capture, all **96 full-vocabulary positions match exactly**.
This controls the observer/fusion interaction; it does not diagnose its exact
kernel cause or justify disabling fusion in the application.

All nine accepted routed replays pass native CPU versus captured output,
GPU-down versus native CPU/captured output, nine operator gates and observer
controls. Replay uses ordinary GPU fusion and no experimental parity flag.

| Category | Layer | Sampled rows / native rows | Original expert bank needed | Distinct ordered routes |
| --- | ---: | ---: | ---: | ---: |
| Coding | 0 / 17 / 35 | 64/512, 64/512, 1/1 | 90 / 54 / 4 | 63 / 63 / 1 |
| Prose | 0 / 17 / 35 | 64/512, 64/512, 1/1 | 94 / 54 / 4 | 62 / 62 / 1 |
| Reasoning | 0 / 17 / 35 | 64/512, 64/512, 1/1 | 92 / 53 / 4 | 61 / 63 / 1 |

The 64-row windows are rows 448-511, avoiding the shared prompt prefix.
Across 387 sampled rows, all 1,548 expert selections retain their original IDs
and probabilities. Final-layer captures explicitly retain seven/seven/four
empty batches followed by one output row. CPU replay relative L2 versus actual
captured outputs is at most 5.51e-8; GPU-down versus CPU is at most 1.52e-7.
Original primitive thresholds remain unchanged.

Scoped memcheck/racecheck on the 94-expert actual replay complete with zero
errors/leaks and zero reported hazards. Thirteen probe regressions pass on
three builds; sixteen checker-input methods pass. Original GPU-first default
controls retain their exact fingerprints/operator checks and failures.

**This passes a bounded actual-FFN gate, not candidate full-model inference.**
Captures come from the qualified no-fusion reference profile, not an
observer-neutral optimized production profile. Replays do not measure
attention, subsequent layers, full-model logits, mixed residency or request
speed. Raw captures/references stay private and ignored. Next implement and
qualify a separate default-off, prefill-only research candidate against the
unchanged serving baseline, preserving context, bank and decoding.

## CPU Activation / GPU Down: Stock Numerical Gates Pass, Isolated Gains

The [projection-split follow-up](results/cpu-activation-gpu-down-summary.json)
keeps gate/up projections, biases and SwiGLU on CPU and sends CPU-effective Q8
values to GPU down projections. CPU quantization/reconstruction and F32 transfer
are timed, not prepared for free. Original weights and routing probabilities
stay unchanged. It uses the unchanged stock backend, without the experimental
CUDA parity patch or flag.

All **60** layer/expert/row configurations pass the nine operator gates,
observer controls, five partitions and both schedules. These use the same
selected cases as the earlier 57/60 GPU-first study, but a different placement
strategy; the original prototype's failures are not relabeled as fixed.
Additional zero/tie, 128/256-row, chunked and ordered-reconciliation cases pass.
Default-mode controls retain the original three failures' output fingerprints
and chained operator checks. Ten Python regression methods pass against all
three probe builds.

The 108 timing reports use three invocation repetitions per
layer/row/resident-upload point, 24 iterations per schedule, three synthetic
inputs and sixteen CPU threads. Invocation medians for the fixed all-CPU versus
all-four-GPU-down serial comparison give:

| Layer | Rows/expert | CPU ms | CPU activation / GPU down ms, including down-weight uploads | Ratio |
| --- | ---: | ---: | ---: | ---: |
| 0 | 32 | 6.830 | 5.854 | 1.17x |
| 17 | 32 | 6.826 | 5.926 | 1.15x |
| 35 | 32 | 6.964 | 5.848 | 1.19x |
| 0 | 64 | 13.819 | 10.475 | 1.32x |
| 17 | 64 | 13.977 | 10.424 | 1.34x |
| 35 | 64 | 13.759 | 10.360 | 1.33x |

All recorded invocation pairs improve at 32/64/128 rows. Fresh uploads lose at
one/eight rows; sixteen rows are inconsistent. Resident 32/64-row down weights
give about 1.45-1.49x in this fixed-set control. Down-only uploads are 17,671,680
bytes for four experts, versus the full FFN's 53,015,040 bytes. The corresponding
F32 activation transfers are 1,474,560 bytes at 32 rows and 2,949,120 at 64.

The scoped 64-row stock-backend operator run completes memcheck with zero errors
and leaked bytes, and racecheck with zero reported hazards/errors/warnings.
Those results qualify only this tested split/path, not all kernels or the
earlier GPU-first prototype. GPU compute consumers were checked between timing
points, but graphics activity was present and no locked-clock/idle-GPU claim
is made.

**This is a passing isolated experiment, not a real-request improvement.**
Inputs remain synthetic with concentrated reuse of four fixed experts. Loading,
graph allocation and verification readback are excluded; no router, attention,
rotating full-model working set or decode-cache interference is measured.
Next qualify actual activations and a default-off prefill-only integration;
resident down weights are a useful first hypothesis because they avoid new
weight transfers/evictions. Installed libraries, serving defaults and Synapse
remain unchanged. See the [updated research handoff](../docs/synapse-prefill-handoff.md).

## Real-model Prefill Before/After: Small Latency Gain Rejected

The [real-workload study](results/gpt120-prefill-real-workload-summary.json)
uses actual GPT-OSS-120B MXFP4 weights, the unchanged installed hybrid backend,
16K context, 12,000 MiB expert bank, eight generation/sixteen prompt threads,
and a frozen 3,577-token technical-review prompt. Each separately loaded run
generates exactly 256 tokens with prompt caching disabled. Three pairs use
A/B, B/A, A/B order. Native logs confirm both requested batch sizes.

| Median of three loads/requests per configuration | Baseline 2048/512 | Candidate 2048/1024 |
| --- | ---: | ---: |
| Cache-warm launch through ready, seconds | 4.228 | 4.225 |
| Backend prompt processing, seconds | 29.321 | 28.283 |
| First received output text, seconds | 29.461 | 28.433 |
| Complete request, seconds | 35.300 | 34.097 |
| Backend decode tokens/s, unqualified output | 42.736 | 43.700 |

The candidate wins the raw latency comparison in all three pairs, but this is
**not an accepted-output speedup**. Its deterministic generated-text hash differs
from the baseline. A separate 3,590-token held-out render compares 32
teacher-forced positions across the full vocabulary: the unchanged 512 control
matches exactly, but 1024 fails at position zero and reaches maximum absolute
error **12.253**, relative L2 **0.3343**, and 29/32 top-one matches. Gates remain
0.01 and 0.001; no rollout or default change is justified.

Single-sample prompt-thread screens at 8/12/16/20/24/32 find no improvement over
16. Logical-batch screens at 512/1024 with physical 512 do not improve prompt
latency; logical 512 also changes generated output. These are screens, not
cross-workload qualifications. No new generation-speed recommendation follows.

All six paired request windows show zero physical reads, zero major faults and
zero endpoint process swap. The files are on the root Samsung 870 EVO SATA SSD;
this is warm inference, not a cold-disk or RAM-disk comparison. Endpoint RSS is
not peak memory. Installed libraries and Synapse were not changed.

An initial checker run exhausted CUDA memory while reserving 2048 output slots.
Capping total and per-sequence reservations to its one-position decode group
permits the same full-vocabulary check at the original context and bank.
The failed log is retained; no smaller workload is substituted.

The [handoff packet](../docs/synapse-prefill-handoff.md) records provenance,
reproduction, current downstream pins, no-change guidance and remaining
qualification gates. It is ready for research review, **not optimization rollout**.

## MXFP4 Activation Alignment: Isolated Improvement, Residual Failures

The [operator/parity study](results/mxfp4-activation-parity-summary.json)
isolates a contributor to CPU/GPU numerical differences. Native x86 MXFP4
uses Q8_0 activation rounding and FP16-stored scales; native CUDA vector and
matrix paths use different conventions, and Blackwell defaults to Q4 matrix
activations. These are different approximations, not evidence that all native
CUDA arithmetic is wrong.

Identical-input tests pass bias/weighting exactly and SwiGLU within the fixed
gate. CPU-effective Q8 inputs reduce gate/up/down relative-L2 errors to about
1.2-1.9e-7. Twelve fusion/capture controls retain zero observer output deltas;
disabling fusion alone does not remove the original projection difference.

A [default-off research patch](../third_party/patches/llama-b11669-mxfp4-q8-parity-experimental.patch)
aligns MXFP4 activation integer rounding/scales with native x86 Q8_0 and requests
Q8 matrix activations. Target weights, selected experts and probabilities stay
unchanged. The build is separate, pinned, CUDA 13.1/SM 120a; no installed library
or application default was changed. Nine current default-off controls retain
clean-baseline fingerprints and error metrics.

All nine initial enabled configurations pass. A broader 60-configuration
screen passes **57**, retaining three failures, including one final-output
failure; smaller chunks and fusion-off retries do not fix them. Six zero/tie
stress cases pass with the mode enabled and fail with it disabled. This is
not complete native equivalence or a qualified model executor.

Seventy-two uninstrumented resident/upload timing reports pass their final
partition gates on the original expert set. Transfer-inclusive all-GPU versus
all-CPU isolated expert timings are:

| Layer | Rows/expert | CPU ms | GPU ms including uploads | Ratio |
| --- | ---: | ---: | ---: | ---: |
| 0 | 32 | 7.223 | 5.002 | 1.44x |
| 17 | 32 | 7.370 | 4.357 | 1.69x |
| 35 | 32 | 7.163 | 4.666 | 1.54x |
| 0 | 64 | 14.218 | 5.169 | 2.75x |
| 17 | 64 | 14.627 | 5.202 | 2.81x |
| 35 | 64 | 14.167 | 5.223 | 2.71x |

Values are medians of three invocation medians, 24 repetitions/schedule,
16 CPU threads and three synthetic activations. Every paired invocation
improves at these points; one/eight-row uploads lose to CPU execution.
The fixed four-expert batch uploads 53,015,040 bytes synchronously from
pageable host vectors. Input/compute/staging/reconciliation/synchronization
are included, but file loading/setup/readback are excluded. OBS was observed
on the GPU before timing; this is not an idle-GPU claim or an application
speedup, and these windows are not matched to earlier backend screens.

Full operator-probe memcheck reports zero errors/leaked bytes. A targeted
quantizer racecheck reports zero hazards/errors/warnings; full racecheck
timed out and is not counted as a pass. Invalid/conflicting flags reject
explicitly. Seven Python regression methods pass against all three probe
builds. Next isolate remaining pre-down activation differences before
real-routing/full-model prefill integration. Original numerical gates remain.
The [effective-Q8 follow-up](results/mxfp4-effective-q8-propagation-summary.json)
finds that changing a few codes/scales upstream can reproduce the residual down
error on CPU. GPU down on that same effective input is accurate near 1e-7
relative L2. Activation-only and complete-chain lesions differ substantially in
the severe case, so a SwiGLU-only fix is not established. The 57/60 status remains.

## NVMe Staging Candidate: Samsung Wins Read Test, Writable Setup Pending

The [read-only validation](results/nvme-staging-read-validation-summary.json)
compares the actual Samsung 990 PRO and WD_BLACK SN850XE partitions after the
user mounted the Samsung read-only and ran the privileged benchmark. Both links
negotiate PCIe 4.0 x4. Twelve serialized fio 3.39 jobs use 1 MiB direct reads,
three repetitions per drive/queue depth, four measured seconds plus one ramp
second, and an 8 GiB region beginning at 4 GiB. All job errors, write bytes
and trim bytes are zero.

| Device | QD1 median | QD16 median |
| --- | ---: | ---: |
| Samsung 990 PRO | 2,292.4 MiB/s | 5,750.4 MiB/s |
| WD_BLACK SN850XE | 173.9 MiB/s | 255.0 MiB/s |

The Samsung is 13.18x/22.55x faster in these tests and is the recommended
**candidate** fast staging device. This is not measured model-load speed or
filesystem write performance. The WD result is unexpectedly low; raw reads
bypass its exFAT filesystem, so the filesystem alone does not explain it.
Both controllers are live, post-test temperatures are about 35.85 C, and a
short post-test device-counter observation is idle. None establishes a cause
or rules out throttling/background activity during the benchmark.

The Samsung remains read-only under a root-owned mount path inaccessible to
the research user; free space and writable staging are not verified. Do not
force dirty/hibernated NTFS writable, reformat either device, or move models
based solely on this read test. The proposed staging policy retains originals,
copies/verifies all shards, tracks ownership, and explicitly evicts only unused
staged copies. No model copies or deletions were performed.

The subsequent user-authorized persistent NTFS setup was blocked: although
`ntfs-3g.probe --readwrite` returned success, the NTFS3 kernel refused the
write mount because the volume is dirty and recommended `chkdsk`. No force
option, repair, formatting or fstab edit was applied. The volume was temporarily
unmounted after that failure. UDisks removed its temporary mount directory during the initial
unmount; the setup script's rollback now recreates that validated directory.
Native Windows checking and clean shutdown are the preferred next steps,
not another driver or a cleared dirty flag used to bypass the refusal.
The user subsequently chose the existing root-drive SSD instead. Samsung was
remounted read-only for inventory and left unchanged; it is not the active
staging target. The
[root-drive staging record](results/root-drive-model-staging-summary.json)
documents the private working-copy directory, user-selected 16 GiB reserve,
verified explicit copy/eviction workflow, and zero-copy reuse of existing SSD
models. A real open-file test blocks eviction, and archive originals survive.

The [job file](nvme-readonly-comparison.fio) was syntax-checked with
`fio --readonly --parse-only`. Before reproducing it, recheck device identities
and use the `--readonly` guard. The input is raw device data that is discarded,
not file contents captured into the result. There is no generation-speed claim:
cache-warm generation already showed zero physical reads below.

## Prefill Numerical Follow-up: CPU Row Grouping Does Not Reproduce Error

The [targeted CPU isolation](results/prefill-cpu-isolation-summary.json) adds a
candidate-host row-chunk control while preserving the original full-row CPU
reference. Twelve reports cover three real GPT layers, 32 rows, CPU chunk
limits 0/1/4 and GPU limits 0/1/4 with three synthetic activations and six
repetitions per schedule. The six new chunked all-CPU controls have zero
observed maximum-absolute and relative-L2 error; all twelve all-CPU partitions
pass. This excludes CPU row grouping as the demonstrated cause in these cases,
not across every layout or workload.

Single-row GPU chunks reduce all-GPU relative L2 from 0.02097/0.05602/0.04126
to about 0.0000333/0.0000873/0.0000567 for layers 0/17/35. Layers 17/35 still
fail the maximum-absolute gate, and only three of twelve complete reports
pass every partition. Both gates remain unchanged. Next isolate GPU projection,
activation/quantization and fusion behavior with observer effects controlled;
no new prefill speedup or complete numerical root cause is claimed.

## RAM-staging Hypothesis: Warm Generation Has No Measured Disk Reads

The [reviewed storage investigation](results/ram-staging-io-investigation-summary.json)
checks current resolved mounts and read-only downstream telemetry, then runs
GPT-OSS-120B at actual 16K context, 8/16 threads, with three exact 256-token
requests in each standard/hybrid mode. The existing 64 GiB tmpfs stays empty;
no cache drops, mounts, drivers, original weights or application defaults change.

Both model shards, **59.034 GiB** in total, are fully resident in the Linux
file cache before and after the controls. Backend completion RSS is about
59.2 GiB, overlapping that cache rather than representing a separate model copy.
Current primary model storage resolves to a SATA SSD; the archive is a spinning
HDD. These are current mounts, not an amendment to historical NVMe benchmarks.

| Control | Load-window physical reads | First request before first text | Physical reads after first text, three requests |
| --- | ---: | ---: | --- |
| Standard automatic placement | 7,577,600 bytes | 2,318,336 bytes | 0 / 0 / 0 bytes |
| Installed experimental hybrid | 6,213,632 bytes | 36,864 bytes | 0 / 0 / 0 bytes |

All six generation windows have zero major faults, and sampled process swap
is zero. Small initialization reads remain unattributed to specific files.
The first-text boundary is instrumented stream receipt, not an exact internal
decode boundary. These measurements do not reproduce the user's past UI
observation, establish behavior for every model, or qualify model quality.

The [phase-aware I/O probe](README.md#storage-io-and-ram-disk-hypothesis) and
stream-observer regression tests pass against both existing builds; exact output
counts and read-counter deltas were checked. Both owned backends exited.
There is no demonstrated RAM-disk generation benefit: the tested weights are
already in RAM, so retain normal file-backed caching and revisit staging only
if sustained physical reads can be reproduced with memory-pressure evidence.

## Stage-aware Prefill Calibration: Isolated Gains, Full Model Unqualified

The [initial investigation](results/stage-aware-prefill-investigation-summary.json)
traces the existing prefill bypass and extends the isolated expert probe with
1-256 rows per expert, timed weight uploads, reversed partition controls and
bounded GPU row chunks. Original weights, biases, probabilities, numerical
thresholds, installed backends, and application defaults are unchanged.

The first real-weight attempt on October 18 failed before weight loading:
CUDA initialization returned unknown error 999. A direct `cuInit(0)` call also
returned 999, and the unchanged stock backend found no usable CUDA devices,
despite `nvidia-smi` working. CUDA recovered after the user's reboot. The first
multi-row attempt then exposed a probe output-layout mismatch; it was fixed
and added to CPU reconciliation checks. Neither failure is hidden.

The completed screen contains **96 reports** across layers 0/17/35, rows
1/2/4/8/16/32/64/128, resident/upload modes and native auto/Q8 MMQ controls.
Only 40 pass every numerical variant. Another **54 transfer-inclusive reports**
repeat rows 32/128 with auto, Q8 and four-row GPU chunking, three invocations,
24 repetitions per schedule, 16 CPU threads, alternating schedule order and
reversed experiment/split order on the middle invocation.

The following are exploratory best **passing** partitions with four-row GPU
chunks, not a runtime policy. Values are medians of three invocation medians;
every retained candidate passes both original gates in both schedules:

| Layer | Rows/expert | GPU / CPU experts | CPU control ms | Candidate ms | Ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 32 | 3 / 1 | 6.569 | 4.666 | 1.408x |
| 0 | 128 | 3 / 1 | 27.039 | 10.392 | 2.602x |
| 17 | 32 | 1 / 3 | 6.739 | 5.941 | 1.134x |
| 17 | 128 | 1 / 3 | 26.701 | 22.782 | 1.172x |
| 35 | 32 | 1 / 3 | 6.571 | 5.974 | 1.100x |
| 35 | 128 | 1 / 3 | 26.971 | 21.877 | 1.233x |

These timings include synchronous pageable uploads and final reconciliation.
The same four real experts and synthetic activations are reused; CPU cache
pressure, arbitrary routing weights and full-model quality are not qualified.
Default wide-row MXFP4 CUDA math fails strict checks in the faster region.
The existing process-scoped Q8 MMQ control and smaller GPU row chunks reduce
some differences, but do not qualify all layers/partitions: only three of the
54 repeated reports pass every variant. Faster failing variants are excluded.

Separate instrumented routing observations on two controlled 628/755-token
prompts show that, in their full 512-row microbatches, **66.8%/67.4%** of routed
work belongs to experts with at least 32 rows. Thirty-five layers have full
coverage; the final layer has only one observed row due to output pruning.
Both one-token greedy continuations match callback-inactive controls.
An initial observer rejection of zero-row pruning was fixed and retained.
These histograms are not serving timings or representative-workload qualification.

The [reproduction guide](README.md#stage-aware-prefill-calibration) distinguishes
concentrated synthetic row reuse and synchronous pageable recopies from actual
prefill routing and asynchronous scheduling. Five targeted CTest targets pass
against both existing builds; all 150 curve/repeat reports pass schema/status
validation, not all numerical gates. The ordinary decode trace still passes
the unchanged analyzer. **No integrated scheduler, universal break-even rule,
full-model prefill gain, or generation speedup is established.** Next resolve
representative numerical cases and measure per-expert, cache-pressure and
pipelined costs before changing whole-model dispatch.

## Mistral Native Integration: Experimental Status Only

The [reviewed integrated experiment](results/mistral4-native-experimental-summary.json)
is published as separately labeled research evidence, not a headline speedup
or native-equivalent compatibility claim.

| Same isolated build, 16K/8/16, three exact 512-token samples | Decode median tok/s |
| --- | ---: |
| Native CPU experts/GPU backbone, experimental flags off | 21.09 |
| Automatic placement with experimental kernels | 23.96 |
| Empty-causal 8 GiB hybrid with experimental kernels | 33.01 |

Arithmetic passes; the original strict coding predicate fails in all three
profiles. Strict native-reference gates also fail at the unchanged 0.01 maximum
absolute / 0.001 relative-L2 limits. Mistral remains out of SYNAPSE's four-model
headline table and out of default accelerated installation.

The separate [experimental patch](../third_party/patches/llama-b11669-fp32-experts-experimental.patch)
adds dtype/precision/shape-gated CUDA matvec support, prevents quantized fusions
from bypassing protected expert precision, and optionally reproduces CPU Q8_K
activation semantics with FP32 scales. Defaults and the installed backend are
unchanged. The patch was applied after the existing hybrid patch on a fresh
pinned checkout and critical source components matched the measured build.

Twelve actual-weight primitive cases and padded/broadcast layouts pass the
native-CPU primitive gates. Integrated memcheck reports zero errors; racecheck
zero hazards/errors/warnings. Frozen and adaptive same-placement optimized
checks pass at 96 full-vocabulary positions each with zero error, but those are
not native equivalence. An unchanged SYNAPSE zipapp with a temporary profile
loads/generates successfully, returns the exact integer answer, and exercises
both backends with all top-4 contributions retained. The 128-token API smoke
is not a published throughput measurement.

Diagnostic captures can perturb fusion/scheduling. CPU-empty controls matched
when global CUDA fusion was disabled, but graph-disable and top-k-only controls
did not resolve all failures; mixed CPU/GPU native comparisons still fail.
Small first-layer expert differences can coexist with substantial final-logit
differences. No sole numerical root cause or general model quality is claimed.

The full [SYNAPSE experimental study](https://github.com/appatalks/Synapse/blob/main/docs/benchmarks/mistral4-native-experimental.md)
records source hashes, every retained numerical failure, the prior unqualified
variant and reproduction limits. Original GPT settings were restored and no
release/tag or default backend promotion was performed.

## Mistral Fused Quantized-Weight / FP32-Activation Kernel Probe

The [reviewed standalone follow-up](results/mistral4-fused-fp32-kernel-summary.json)
reuses pinned IQ4_XS/Q6_K dequantizers in an opt-in SM 120 CUDA matvec. It
dequantizes on the fly and accumulates in FP32, avoiding the initial prototype's
128 MiB/position global expanded-weight intermediate. Per-CTA shared memory
is 4,096/2,064 bytes for the two types.

Twenty-four completed timing fixtures pass fixed FP32-reference gates, including
one/three positions, normal/1e6 synthetic activation scales and CPU cache
pressure. Worst relative L2 is 1.88e-7. Routing/bounds fixtures cover masked
zero/-1, repeated IDs, varying token routing, partial rows and explicit invalid
ID/dimension rejection. Four separate instrumented fixtures pass; Compute
Sanitizer memcheck finds zero errors and racecheck zero hazards/errors/warnings.
Instrumented times are excluded.

In the completed refined-kernel/cache-pressure window, single-position medians
range **0.075-0.087 ms** versus **0.464-0.503 ms** for expanded-weight GGML GPU
execution and **0.365-0.500 ms** for the native quantized CPU primitive.
Warm controls are retained. API/timing boundaries differ and uploads are
excluded; these ratios are not Mistral tok/s predictions.

An earlier refined run passed all eight arithmetic fixtures but failed GPT
restoration with CUDA OOM while its process retained context state. It is
retained and excluded from headline timing. After recovery, non-CUDA
parent/CUDA-child isolation ensured context exit before GPT reload; subsequent
timing, memcheck and racecheck runs restored successfully.

This is an integration candidate only: no llama.cpp graph integration,
full-model native-equivalence check or new Mistral decode rate exists.
Installed backends, target weights and app defaults are unchanged. Final GPT
16K/8/16/12,000 MiB settings are verified. The
[app study](https://github.com/appatalks/Synapse/blob/main/docs/benchmarks/mistral4-compatibility.md#fused-fp32-gpu-follow-up)
details the limitations and next gates.

## Mistral Small 4 Compatibility And FP32 GPU Prototype

The [reviewed study](results/mistral4-fp32-compatibility-summary.json) retains
the pinned, verified UD-IQ4_XS target, initial harness failures, corrected native
controls and isolated precision probes. This is **not** a fourth/fifth hybrid
speedup or a full-model compatibility pass.

At actual 16K/8/16 and three exact 512-token samples, native automatic placement
measures **5.64 tok/s** and native CPU experts/GPU backbone **19.89 tok/s**.
Both pass the exact integer check and fail the original AST check: fenced code
also reverses the requested min/max nesting, so fence removal alone does not
pass. Generated code was never executed.

The hybrid loads the fused gate/up descriptor, both quantization groups and
36/36 routed layers, but rejects the protected FP32-input down projection.
CUDA explicitly refuses `MUL_MAT_ID` with source-1 `GGML_PREC_F32`; the model
requests it because activations can exceed FP16 range. The guard was not removed.

An opt-in, isolated GET_ROWS/dequantization plus FP32 batched-matmul prototype
passes 16 actual IQ4_XS/Q6_K weight probes, including synthetic activations at
scale 1e6. Dequantized weights are exact and relative-L2 versus CPU FP32
reference is at most 1.55e-7. It requires 128 MiB selected-weight scratch per
position and does not establish native-CPU equivalence: the native quantized
kernel differs by roughly 0.0032 relative L2.

Warm single-position GPU probes take 0.37-0.38 ms versus native CPU 0.09-0.11 ms.
With an explicit 256 MiB CPU cache-pressure control, GPU takes 0.46-0.48 ms and
CPU 0.28-0.49 ms. These are primitive measurements, not tok/s predictions;
there is no compelling universal win to promote. Next candidates are fused
quantized/FP32 GPU math or per-projection scheduling, gated by complete-model
verification against the faster native control. The installed backend and
target were not changed; GPT settings were restored. See the
[detailed study](https://github.com/appatalks/Synapse/blob/main/docs/benchmarks/mistral4-compatibility.md).

## Qwen3.6-35B-A3B Q8_0 Matched Hybrid Screen

The [reviewed report](results/qwen36-q8-benchmark-summary.json) adds a fourth
larger-than-VRAM target to the matched SYNAPSE table. At actual 16K, 8/16
threads and three exact 512-token samples per profile, stock automatic fitting
measures **45.50 tok/s**, native CPU experts/GPU backbone **34.07**, and the
empty-causal 8 GiB hybrid **77.99**, a **71.4%** gain against the faster control.
Hybrid samples are 73.47/77.99/80.24, with residency retained across requests.

The Apache-2.0 Q8_0 GGUF is pinned and SHA256-verified, staged on the mounted
WD_BLACK SN850XE NVMe, not the RAM disk. Descriptors give `qwen35moe`, 40 routed
layers, 256 experts/top-8, and 34,225,520,640 routed bytes. All selected
contributions are retained: CPU 148,713 plus GPU 608,143 equals eight per
94,607 dispatches. Warm second/third samples have zero process disk reads.

All six exact integer/AST checks pass. Descriptor-seeded frozen base and
optimized-versus-base checks each pass 96 sampled full-vocabulary positions
with zero error and both expert backends exercised. This is same-placement
pipeline agreement, not CPU-reference equivalence or broad quality qualification.
The smaller model's throughput does not meet the GPT-OSS-120B 60 tok/s milestone.
Original settings were restored. See the
[app report](https://github.com/appatalks/Synapse/blob/main/docs/benchmarks/qwen36-q8.md)
for settings, provenance and retained limits.

## Upstream llama.cpp PR 29887 Investigation

[ggml-org/llama.cpp#29887](https://github.com/ggml-org/llama.cpp/pull/29887)
is already in the pinned `620d2225c` backend. This
[reviewed policy/loading screen](results/upstream-pr29887-gpt120b-summary.json)
does not compare commits before and after the PR or claim another multiplier
on SYNAPSE's current results.

On unchanged GPT-OSS-120B MXFP4 at 16K, 8/16 threads, T0/seed42 and three exact
512-token samples per block:

| Configuration | Decode median tok/s |
| --- | ---: |
| Stock automatic placement | 24.42 |
| Automatic placement plus 6 GiB upstream LRU cache | 14.26 |
| CPU experts plus 10 GiB upstream LRU, mapped loading | 17.31 |
| CPU experts plus 10 GiB upstream LRU, copied CUDA-host buffers | 29.36 |
| Restored SYNAPSE 12,000 MiB trained hybrid, three windows | 47.89 / 48.66 / 48.92 |

Copy versus mapped loading is a matched 69.6% improvement for upstream LRU.
The three corresponding output hashes and final LRU counters match, including
66.77% small-batch hits. Backend RSS stays near 58-59 GiB; copied loading uses
`CUDA_Host` rather than `CPU_Mapped` buffers, with 57.5 GiB observed shared RSS.
It changes the upload path and OS memory accounting, not merely resident capacity.
Warm samples read no new disk bytes.

The 12,000 MiB copied upstream-cache attempt failed allocating a 622.15 MiB
prefill compute buffer. The first fit/cache block's missing-statistics failure
at ordinary verbosity is retained; an independent verbosity-4 repeat confirmed
cache activity. All completed modes pass the exact integer/AST checks, not
general answer-quality or CPU-reference equivalence. Screens are sequential,
not randomized or exhaustive. No native defaults were promoted; every original
profile was restored. See the
[detailed app report](https://github.com/appatalks/Synapse/blob/main/docs/benchmarks/upstream-pr29887.md).

## GLM-4.5-Air IQ4_XS And RAM Accounting

The [reviewed third-model screen](results/glm45-air-iq4-xs-benchmark-summary.json)
uses the unchanged106B GLM-4.5-Air community IQ4_XS target at16K/8/16.
Three exact512-token samples per configuration retain both standard controls,
native mapped/copied controls, answer-check failures and the8GiB CUDA OOM.

| Configuration | Decode median tok/s |
| --- | ---: |
| Standard automatic placement, first window | 12.67 |
| Native CPU experts / GPU backbone, mapped | 11.30 |
| Causal direct-DMA,6,144MiB, no trained plan | **17.28** |
| Standard automatic placement, reversed-order window | 12.14 |
| Native CPU experts / GPU backbone, copied into RAM | 11.43 |

The conservative gain against the fastest control block median is **36.4%**.
The candidate keeps every selected expert and does not use MTP, expert
skipping, target conversion or a lower context. Two downloaded shards total
60,480,900,352bytes and match the pinned public LFS hashes. The47-block GGUF
contains46 trunk layers and one unused NextN block;45 routed layers use
128experts/top8, hidden4096, FFN1408 and IQ4_XS/IQ4_NL components.

Same-placement optimized-versus-base full-logit checks pass96positions with
zero error and both CPU/GPU contributions. Original default-thinking failures
remain: an arithmetic request exhausted1024tokens without a final answer,
and fenced Python failed the strict output-format check. Separate thinking-off
diagnostics pass the unchanged integer/AST checks in both modes. This is not
CPU-only/native-reference equivalence or broad quality qualification.

Mapped native and hybrid execution already had about54GiB RSS; warm samples
show zero disk reads. Copy mode changed host allocations and OS-unavailable
RAM substantially without a useful throughput gain. Logical selected routed
weights total3.37GB/token; the native rate implies a38.1GB/s logical read budget,
not a direct measurement of physical memory bandwidth.

The browser's old RAM pie showed pressure (`total - MemAvailable`), omitting
much cached model residency. Occupancy, OS availability, free pages, file cache,
shared pages and overlapping backend RSS are now distinguished; no allocation
headroom check was loosened. **Require hybrid** supports explicit causal cache
budgets without changing Automatic defaults or fixed trained-plan budgets.

GLM-5.3-Flash320B/18B was inspected but not downloaded: the checked IQ2_XXS
was94.85GiB, above available disk space. No5.3 speedup is claimed. See the
[app benchmark guide](https://github.com/appatalks/Synapse/blob/main/docs/benchmarks/glm45-air-iq4-xs.md)
for model provenance, measured profile and limitations.

## Canonical Qwen Verification And Native MTP Research

The [reviewed follow-up](results/moe-canonical-verification-mtp-summary.json)
isolates scalar/batched execution differences without weakening the
0.01 maximum-absolute / 0.001 relative-L2 logit gates. Scalar-shaped learned
BF16/F16 projections, attention products, and resident GPU expert products
make Qwen blocks **2/3/4 exact at 96 sampled positions each**, including mixed
CPU/GPU execution. Native and seeded mixed-residency long-context samples
also pass with zero error. A training-only, group-capacity-aware 6,144 MiB
seed passes a separate 96-position block4 gate.

Initial seeded failures, the unsuccessful CUDA-fusion-disable control,
active-capture observer effects, and a graph-metadata allocation failure
are retained. Extra graph metadata is reserved only for canonical experiments;
ordinary app allocation and execution remain unchanged.

A derived native MTP draft reuses the pinned upstream converter's tensor
transforms, original raw draft tensors, and unchanged target embedding/output
tensors. It is not an architecture-label rename. Only the derived draft's
experts are quantized to Q4_0; original target/draft files are untouched.
The separate MTP context does not inherit the target bank or seed plan.

| Screen | Scalar control | MTP, two proposals | MTP, three proposals |
| --- | ---: | ---: | ---: |
| Native CPU experts/GPU backbone, 4/16 threads, 128 coding tokens | 21.17 | 31.65 | Not tested |
| Frozen 6,144 MiB hybrid, 8/16 threads, 512 coding tokens | 28.35 | 32.13 | 33.62 |

Numbers are accepted-output decode tok/s, not proposal or teacher-forced
positions/s. Drafting, verification and synchronization are inside the
generation interval; total request wall times are retained separately.
All compared output hashes match within each screen. Hybrid two/three
proposal runs accepted **335/350** and **376/402** draft tokens, respectively;
rejected proposals were not counted as output.

These are **single-sample screens**, not cross-workload qualification or a new
overall best. The frozen 6 GiB bank and canonical execution differ from the
earlier 8 GiB adaptive Qwen configuration; do not infer a regression or a
33.62 tok/s hardware ceiling from that unmatched comparison. The 60 tok/s
unchanged-GPT milestone remains unmet. Canonical/MTP app options stay off.

## Shared Preparation, Routing Packets And Per-Model Verification

The [reviewed exploration](results/moe-preparation-packet-verification-summary.json)
records the requested follow-up without promoting unqualified gains.
Shared CPU activation/row-map preparation passed strict primitives and frozen
math, but technical medians were mixed: GPT **46.40 -> 45.94**, Qwen
**43.14 -> 43.50 tok/s**. The latter's sub-1% screen is inconclusive, despite
79,958 actual reuses and a zero-error 96-position scalar comparison.

The complete routing packet reduced filtered D2H calls **170,856 -> 113,904**
with identical logical bytes. Its technical median improved
**45.76 -> 46.31**, but a reverse-order natural screen was slower:

| Workload | Existing handoff | Routing packet |
| --- | ---: | ---: |
| Coding | 46.11 | 43.09 |
| Reasoning | 46.51 | 44.59 |
| Prose | 65.83 | 62.54 |

An initial packet failed to remove separate CPU bias-ID copies; those and
pre-FFN consumers were fixed before the complete test. Both iterations and the
failed exact quantitative check are retained. These measurement windows are
not matched to the earlier 50.06 technical result and do not establish its
cause or a new hardware limit.

Qwen block2/3 verification also failed unchanged strict gates, including a
native block2 control. Its available drafter declares `qwen4exp-mtp`, not the
native loader's canonical format. No adapter, MTP timing claim or speculative
app mode was enabled. Both prototypes remain off. All **19 configured test
targets** passed; seven combined prior-hybrid and seven cold-copy comparisons
passed with the same 14 native-reference failures. The working app was
restored to GPT's 12,000 MiB profile and passed live activation.

## Cold-Work Experiment And Second Trained Large MoE

The [reviewed follow-up](results/moe-cold-work-and-qwen38-summary.json)
tests lessons from Strata without target expert skipping or weight changes.
On GPT-OSS-120B at 11,264 MiB, a bounded transient cold-GPU path was slower:
**48.42** control, **40.93** synchronous-copy reference, **42.96 tok/s**
asynchronous candidate. Matching-placement logit gates passed; extra PCIe
uploads outweighed the reduction in CPU expert calls. Initial CUDA OOM and
scratch-ID bound failures were retained, with the latter fixed.

The already-downloaded Qwen3.8-Flash-Next IQ3_S demonstrated a second real
large-MoE benefit at 16K/8/16: **25.39 -> 43.88 tok/s** median over three
512-token technical samples. The control uses native CPU experts/GPU backbone;
the candidate uses an empty causal plan and 8,192 MiB direct-DMA bank. This
is not stock automatic placement or a matched Strata/MTP comparison.
Both passed the narrow integer/AST checks, not broad quality qualification.

A separate Qwen cold-offload pair was nearly flat/slower (**44.85 -> 44.41**),
so transient offload remains disabled on both tested models. Scalar-shaped
router diagnostics also failed to resolve batch verification. No speculation,
expert pruning or relaxed numerical threshold was enabled. The app now has
separate optional causal budget/transfer settings while preserving the GPT
trained-plan profile. The overall GPT 60 tok/s goal remains unmet.

## Admission Lifetime Research And Larger Resident Bank

The [reviewed investigation](results/gpt120-admission-capacity-research-summary.json)
records negative probation/ranked/anchored admission screens, a fresh CPU
thread sweep, native batch-verification failures and a larger-cache winner.
With the same 16K/8/16/direct/overlap/fusion/asynchronous-ID settings,
11,264 MiB/hot880 versus 12,000 MiB/hot944 technical medians were
**47.33 -> 50.06 tok/s**. Separate three-by-512 natural workload medians were:

| Workload | Previous bank | Larger bank |
| --- | ---: | ---: |
| Coding | 44.33 | 46.71 |
| Reasoning | 47.01 | 49.26 |
| Prose | 62.42 | 65.99 |

Both passed the 12,852-token retrieval/allocation check and narrow integer/AST
checks. Both retained the same exact quantitative-answer failure, so neither
is quality-qualified. Matching-placement frozen gates passed; native scalar
versus multi-position gates still fail and speculative serving stays off.
The local experimental app uses the larger bank; the prior plan and Standard
rollback remain. The overall sustained 60 tok/s goal is still unmet.

## Manual Universe Replay And Negative CPU-Empty Screen

The [reviewed comparison](results/gpt120-universe-cpu-empty-summary.json)
investigates the reported 41.6-versus-52 tok/s difference without treating
unmatched requests as a regression. The recorded 614-token case had the full
updated profile active. Controlled model-default-reasoning, T0/seed42,
three-by-512 replay of the supplied Universe/Earth prompt measured
**45.89 -> 47.65 tok/s** with asynchronous IDs; original UI sampling/history
was not reconstructed and the original difference remains unattributed.

An all-masked raw CPU operator early-zero path passed primitive, frozen and
seven-layout regression checks, but its matched median was **46.60 versus
47.19**, despite 61,983 exercised operators. It remains disabled in the app.
The same 14 native-reference failures remain visible. The 17 configured test
targets passed. New app timing components and an explicit reasoning selector
improve manual comparisons without changing default reasoning or sampling.

## Copy/Wait Breakdown And Event-Protected GPU IDs

The [reviewed handoff summary](results/gpt120-async-id-handoff-summary.json)
records the first follow-up to research checkpoint `068f69e`. On unchanged
GPT-OSS-120B MXFP4 at actual 16K context, 8/16 threads and 11,264 MiB residency,
three matched 512-token technical samples improved the median from
**47.01 to 49.27 tok/s (+4.8%)**. Natural-style coding/reasoning medians improved
**43.84 -> 45.31** and **46.56 -> 48.26**; prose was nearly flat at
**61.31 -> 61.62**. This does not establish the overall 60 tok/s milestone.

The prototype replaces blocking scheduler slot-ID copies with bounded,
event-protected per-layer transfers and explicit graph dependencies.
H2D destination-wait calls decreased by about 1.02 ms per filtered graph, but
CPU submission rose; only the net end-to-end gain is counted. Logical ID
transfers still occur outside the scheduler-copy counters.
All 12 compared generation hashes and all seven prior-hybrid fixture gates
matched controls. Existing native-reference and exact-answer failures remain.
The optional `async_ids` installation setting is enabled in the local
experimental manual profile after validation; Standard rollback is unchanged.

## Decode Overlap And Projection Fusion Screen

The [reviewed summary](results/gpt120-decode-optimization-summary.json) records
GPT-OSS-120B MXFP4 on the 9950X/5070 Ti with unchanged target weights,
actual 16K context, one request, 8 generation/16 prompt threads, and 512 output
tokens per sample. These conditions supersede the older defaults below for
this section only. The final technical three-repeat median is **46.94 tok/s**.

Separate natural-style prompts disabled forced EOS suppression and retained
the exact 512-token cap/count gate. With an 11,264 MiB resident cache:

| Workload | Three samples, tok/s | Median tok/s |
| --- | --- | ---: |
| Coding | 43.18, 43.25, 42.51 | 43.18 |
| Reasoning | 43.74, 46.10, 47.03 | 46.10 |
| Prose | 58.60, 60.54, 61.94 | 60.54 |

The prose median is not the overall 60 tok/s milestone. These capped segments
do not establish full-answer quality. Exact quantitative answers still failed;
two-key retrieval at 12,852 prompt tokens passed. During these screens the new pipeline options were off in the installed
default, and the previous app configuration was restored. The subsequent
experimental manual trial opts into the bundle with installation
`decode_pipeline: true`; this does not remove the failed quality gates.

Q8-drafter speculation was slower. QKV aliasing failed an isolated synthetic
regression and was removed. All seven final fixtures passed comparisons with
the prior hybrid path; 14 broader strict/native-reference failures remain
visible. Read the [investigation](../docs/research/frontier-moe-feasibility.md#guarded-overlap-and-resident-projection-fusion)
before interpreting these measurements as deployment evidence.

## Reviewed Results

| Model | Placement | Prompt tok/s | Generation tok/s | Evidence |
| --- | --- | ---: | ---: | --- |
| Calme 3.2 78B Q4_0 | auto-fit, 26 of 86 dense blocks in VRAM | 185.74 +/- 0.55 | 1.35 +/- 0.02 | [matrix](results/20260822T153837Z-calme-3_2-instruct-78b-Q4_0.json) |
| GPT-OSS 20B MXFP4 | auto-fit, full GPU | 9,289.33 +/- 394.37 | 217.18 +/- 1.52 | [run](results/20260823T143325Z-gpt-oss-20b-MXFP4.json) |
| Qwen3-Coder 30B-A3B Q3_K_L | auto-fit, near/full GPU | 4,048.34 +/- 120.65 | 204.87 +/- 6.82 | [run](results/20260823T143354Z-Qwen3-Coder-30B-A3B-Instruct-Q3_K_L.json) |
| Qwen3.6 35B-A3B Q8_0 | all routed experts on CPU | 313.22 +/- 3.62 | 30.83 +/- 0.17 | [run](results/20260823T143533Z-Qwen3_6-35B-A3B-Q8_0.json) |
| Qwen3.6 35B-A3B Q8_0 | auto-fit GPU/CPU expert placement | 487.95 +/- 7.19 | 41.41 +/- 0.67 | [run](results/20260823T143433Z-Qwen3_6-35B-A3B-Q8_0.json) |
| GPT-OSS 120B MXFP4 | all routed experts on CPU | 232.54 +/- 1.47 | 19.30 +/- 0.12 | [run](results/20260823T151527Z-gpt-oss-120b-MXFP4-00001-of-00002.json) |
| GPT-OSS 120B MXFP4 | auto-fit GPU/CPU expert placement | 305.95 +/- 13.27 | 22.75 +/- 0.31 | [run](results/20260823T151455Z-gpt-oss-120b-MXFP4-00001-of-00002.json) |

The dense 78B result is host-RAM-bandwidth-bound. The large MoE results instead
compute CPU-resident routed experts in place; their generation PCIe traffic is
activation-scale rather than full-expert transfer traffic. See
[frontier-moe-feasibility.md](../docs/research/frontier-moe-feasibility.md) for
the GGUF payload accounting and phase telemetry.

## Context Sweep

Qwen3.6 35B-A3B Q8_0 used auto-fit placement with Q8_0 K/V cache. Each point
uses a prompt half as long as its fit context and generates 32 tokens.

| Fit context | Prompt | Prompt tok/s | Generation tok/s | Evidence |
| ---: | ---: | ---: | ---: | --- |
| 16K | 8,192 | 457.53 +/- 14.27 | 40.96 +/- 0.20 | [run](results/20260823T150402Z-Qwen3_6-35B-A3B-Q8_0.json) |
| 32K | 16,384 | 437.16 +/- 6.69 | 40.55 +/- 0.76 | [run](results/20260823T150701Z-Qwen3_6-35B-A3B-Q8_0.json) |
| 64K | 32,768 | 426.94 +/- 13.07 | 40.55 +/- 0.44 | [run](results/20260823T151238Z-Qwen3_6-35B-A3B-Q8_0.json) |

## Retained Failed Evidence

| Artifact | Reason retained |
| --- | --- |
| [initial dense matrix](results/20260822T153837Z-calme-3_2-instruct-78b-Q4_0.json) | Its nominal CPU run was not strict CPU because CUDA operation/KV offload remained active. Its Unified Memory command used invalid `-ngl all` syntax. Valid manual and auto-fit runs remain comparable. |
| [Qwen3.6 context attempt](results/20260823T150213Z-Qwen3_6-35B-A3B-Q8_0.json) | Pinned `llama-bench` does not accept a general `-c` context override. The corrected context series uses `--fit-ctx` and Q8_0 K/V cache types. |

## October 16: Model-Aware Defaults And Downloaded-Model Matrix

This series uses pinned llama.cpp commit
`620d2225cc436ab0b001e3200e590e3b286ef9ab`, **actual 16,384-token context,
8 generation / 16 prompt threads, and three exact 512-output-token runs**
per model/profile. It is not comparable to the earlier 512-prompt/32-generation
tables without accounting for the changed workload/backend.

The [complete matrix](results/20261016-downloaded-model-matrix.json) contains
all 30 throughput samples, 20 small answer checks, source/patch/plan hashes,
model architecture/type metadata, selected execution profiles, and successful
restoration of the original GPT-OSS-120B app.

| Existing model | Standard median tok/s | Automatic median tok/s | Automatic execution | Arithmetic / AST-only code checks (both profiles) |
| --- | ---: | ---: | --- | --- |
| GPT-OSS-120B MXFP4 | 25.69 | 39.46 | Custom hybrid direct DMA | Pass / pass |
| Gemma 3 12B Q4_K_M | 80.94 | 80.50 | Standard | Exact-integer check failed / pass |
| DeepSeek-R1-0528-Qwen3 8B Q4_K_M | 130.08 | 128.50 | Standard | Arithmetic completion hit 1,024-token limit / pass |
| Qwen3.5 local Q8_0 (`claude-oss-q8_0.gguf`) | 79.48 | 80.26 | Standard | Pass / pass |
| LFM2 1.2B Q8_0 | 485.67 | 486.55 | Standard | Exact-integer check failed / strict AST check failed |

The Qwen3.5 row is identified by actual GGUF architecture `qwen35`; its local
filename is not evidence of an official Anthropic model. The CLIP projector
was excluded because it is not a standalone language model. No models were
downloaded. Only GPT-OSS-120B exceeds this GPU's VRAM among the current five;
this series cannot demonstrate hybrid benefits on a second larger-than-VRAM
architecture.

GPT-OSS-120B's paired median gain is about **53.6%**. Standard mode uses upstream
auto-fit; automatic mode uses the patched resident bank, CPU cold-expert
execution, adaptive admissions, and direct registered-source DMA. Context,
threads, prompt, output count, seed, and sampling settings are matched;
placement/offload are intentionally different and recorded. Loading and source
registration are outside decode timing; end-to-end generation and load
durations are recorded separately. The initial trained bank resets on each
profile load; adaptive residency persists between its repetitions and answer
checks.

All four unsupported architectures selected standard execution as intended.
Their repeated output hashes matched across standard/automatic profiles;
small timing differences are not evidence of an optimization gain. The
standard GPT-OSS repetitions matched one another, but adaptive GPT-OSS hashes
did **not** all match each other or the standard control. Passing the two small
answer checks is not general model/logit equivalence. Strict CPU-reference
diagnostics and the earlier quantitative JSON failure remain unresolved.

Answer checks deliberately require the exact integer and a specific AST with
no extra content. Failures can include formatting or token-budget issues;
answer text is not stored in this matrix, so these booleans must not be
reinterpreted as a broad factual-quality score. The runner retains failed
checks and exits nonzero even when the throughput runs and restoration succeed.
The 512-token throughput prompt is forced past EOS if necessary, not a guarantee
of 512 natural-answer tokens. This is a repeated one-prompt screen, **not**
sustained coding/reasoning/prose qualification. GPT-OSS-120B's 60 tok/s target
is still unmet; the smaller models' higher rates do not satisfy that goal.

### Published Hybrid Research Evidence

| Artifact | Interpretation |
| --- | --- |
| [Direct-source pair](results/20261016-gpt120-direct-source-screen.json) | First paired 35.73 -> 38.96 tok/s result; six output hashes match between copy paths, but the original quantitative JSON check remains failed. |
| [Direct-DMA thread screen](results/20261016-gpt120-direct-thread-screen.json) | Four/six/eight-thread one-prompt samples 38.81/39.63/39.99 tok/s; not three-repeat per-thread qualification. |
| [CPU dual-layout screen](results/20261016-gpt120-dual-layout-screen.json) | Real CPU-repacked storage with lossless GPU exports; adaptive 33.80 tok/s did not beat the raw-layout screen. |
| [Asynchronous staged-copy screen](results/20261016-gpt120-async-copy-screen.json) | Synchronous 35.65 versus async 34.67 tok/s; identical sampled logits, no timing gain, failed quantitative check retained. |
| [Small-block DFlash screen](results/20261016-gpt120-hybrid-block-dflash-screen.json) | 22.81/22.12/24.47 accepted tok/s versus 29.28 control with a smaller bank; includes draft/rejection overhead, not a successful optimization. |
| [Full-vocabulary diagnostic metrics](results/20261016-gpt120-logit-diagnostics.json) | CPU repeat is exact; stock placement and hybrid fail the unchanged strict CPU-reference gates. Direct-DMA versus synchronous hybrid is exact at 96 sampled positions. Block4 remains a failed strict diagnostic. No token IDs/full logits/activations are published. |
| [Training-only initial plan](results/gpt120-trained-hot800.txt) | 800 expert groups from the public routing corpus's training half only; SHA-256 `1198ae0ea67a9b94f1f51f0913f7ade3378bb5ba01e3ac93ace7eccaa77e98e6`. |

Published copies replace private paths with `/models`, `/opt/vrambridge`,
`/workspace/unified-memory-scheduler`, `/session`, and `/home/benchmark-user`.
Raw synthetic answer strings from old screens are replaced with SHA-256 hashes;
measurements, failed checks, diagnostics, and original run status are retained.
Large routing traces, binary references, activation dumps, and raw process logs
remain ignored. Older screen source hashes describe their own prototype state,
not necessarily the final patch. The current matrix identifies the final patch
with SHA-256
`a0f61f4e86aa6468d9b76ed4c9ee801dbee30ff2ec034566e8f86cf25be472a1`.

The app's new default is **automatic model-specific selection**, not a claim of
universal hybrid support. A pinned local installation selects direct DMA only
for the supported GPT-OSS-120B MXFP4 layout; unsupported models select standard
execution with an explicit reason. `--execution-mode standard` bypasses the
installation profile for rollback. Experimental warnings and readiness/resource
checks remain enabled.

## Pending Candidates

These are not benchmark results and must not be compared with the rows above.

| Candidate | Status | Next gate |
| --- | --- | --- |
| GLM-5.2 UD-Q4_K_M | Download in progress; 466 GB expected GGUF, 753B total / 40B active | Verify full shard set and header, then run a fit-only probe. The complete expert pool exceeds current host RAM, so an interactive run is not expected on this host. |
| Qwen3.6 27B Q4_K_M or Q6_K | Candidate dense quality comparison | Run fit-only probe first, then the same 512/32/3 matrix if it fits. |

## Tracking Rules

1. Retain raw JSON for every reviewed conclusion, including invalid commands and failed runs that changed the interpretation.
2. Record model file, quantization, llama.cpp commit, fit margin/context, prompt and generation tokens, repetitions, and K/V cache types.
3. Do not compare token rates across different model architectures as quality scores. Compare placement policies only within an identical model and workload.
4. Add a result to this ledger only after focused validation and a review of placement and telemetry.