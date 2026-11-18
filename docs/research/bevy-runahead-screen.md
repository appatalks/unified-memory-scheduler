# Bevy runahead: fixed-residency routing screen

## Paper and applicability

[Bevy](https://arxiv.org/abs/2611.04794) schedules concurrent requests at
different layers, fetching known missing experts while ready work executes.
It groups ready jobs at the lowest layer, pins experts until their consumers
complete and gates runahead on sufficient cache slots to prevent deadlock.
Its implementation is in llama.cpp, but the paper does not link a Bevy source
repository; a targeted GitHub search found none. The paper's measured systems
are unified-memory M6 and Jetson Thor using SSD expert fetching. Compared with
its strongest prefetching baseline, reported gains range from 1.01 to 1.55x;
these are aggregate decode results for its tested models and workloads.

The paper explicitly leaves CPU expert computation to future work. Our hybrid
executes cold experts on the CPU and uses asynchronous GPU admissions; it does
not simply block all misses for an SSD-to-GPU fetch. Our earlier single-request
GPT CPU/CUDA profile already shows overlap and substantial CPU-only time. That is
motivation to investigate scheduling, not evidence that Bevy's published
speedup applies. Extra CPU work can compete for RAM bandwidth and GPU time;
layer-granular graphs may increase launch, state and shared-weight costs.

Existing Clover research provides a separate warning: the isolated attention
candidate achieved 24/24 matching 256-output HTTP responses at each of 2/4/8
clients and about 12.8% higher eight-client aggregate throughput than its
serial cohort, but eight-client median completion increased from 29.37 to
45.82 seconds. Other variants fail numerical or longer-stream checks. These
are Clover results, not Bevy or hybrid-engine benchmarks.

## Reproducible routing screen

[`vrambridge-bevy-screen.py`](../../tools/benchmark/vrambridge-bevy-screen.py)
loads complete observer-verified, decode-only routing traces, pairs held-out
requests at equal token and layer indices and evaluates fixed residency. It
counts a request as fully resident only when every routed expert is in the
fixed set. A mixed frame has both a fully resident and a non-fully-resident
request. This definition is intentionally narrower than runnable hybrid work:
partially resident requests can simultaneously use GPU and CPU experts.

The GPT seed is the existing 944-expert plan, SHA-256
`6a679bec76f1a410ed65fa79c186d164a388fc76419d96dcad5978782343949c`.
The tool also selects a byte-budgeted training-only hotset for each trace.
The verified traces record filenames but not model hashes, so the seed's
identity cannot be independently established from trace metadata. The
[aggregate](../../benchmarks/results/bevy-routing-screen-summary.json) retains
trace, source and plan hashes plus individual category-pair counts.

| Fixed residency | Aligned pair frames | Mixed pair frames | Aligned triple frames | Mixed triple frames |
| --- | ---: | ---: | ---: | ---: |
| GPT-OSS-120B, 944 seed experts | 27,648 | 26.31% | 9,216 | 39.46% |
| GPT-OSS-120B, 12,000 MiB training-static | 27,648 | 26.53% | 9,216 | 39.79% |
| Qwen3.8 GSQ-RCO IQ3_S, 8,192 MiB training-static | 9,216 | 14.50% | 3,072 | 21.74% |

There is one coding, reasoning and prose held-out request per trace. Pair
frames pool all three distinct pairs; the triple is one combination. GPT
contributes 256 and Qwen 64 generated tokens per request. Positions are
artificially synchronized from serially captured requests. No prefill,
dynamic residency, actual group capacities, CPU or GPU durations, transfers,
cache contention, job graph, runahead progress, or timing is modeled. These
fractions are not a throughput estimate, bound on achievable overlap, or
evidence that a hybrid miss blocks. No verified Coder routing trace was found.

## Live hybrid numerical gates

The [sanitized aggregate](../../benchmarks/results/bevy-live-numerical-summary.json)
includes model, backend, seed and private-artifact hashes and the individual
diagnostic comparisons.

Copied b11669 Hybrid core/CPU libraries and the existing, read-only pinned
CUDA plugin were tested on Coder Q4_K_M, SC117 Qwen3.8 IQ3_S and GPT-OSS-120B
MXFP4 with two 4,096-token slots, F16 KV, a 4 GiB
expert bank and frozen synthetic seeds. Coder and Qwen use expert IDs 0-15
in each routed layer; GPT uses IDs 0-5. These are deliberately simple controls,
not trained or production residency plans. Full tensor coverage and CPU and
GPU expert execution are verified for every run.

Against a separate scalar reference, two distinct prompts and eight
teacher-forced continuation steps give the following full-vocabulary raw-logit
results. The unchanged limits are max absolute 0.01 and relative L2 0.001.
The serial repeat and token-by-token interleave each pass all 16 positions
with zero error. A two-position decode with the experimental hybrid batch
limit set to two fails all 16 positions:

| Model | Serial repeat | Serial interleave | Width-two batched decode | Batched worst absolute / relative L2 |
| --- | ---: | ---: | ---: | ---: |
| Coder | 16/16 | 16/16 | 0/16 | 1.356 / 0.1367 |
| SC117 Qwen3.8 | 16/16 | 16/16 | 0/16 | 2.497 / 0.1693 |
| GPT-OSS-120B | 16/16 | 16/16 | 0/16 | 0.947 / 0.0639 |

Each serial prefill also matches 2/2. With the physical microbatch forced to
one, alternating single-token prefill and subsequent scalar decode pass for all
three families, but submitting the two decode positions together still fails
16/16 per family. In that microbatch-one control the hybrid batch limit is
one, so the two-position logical batch can use the backend fallback; the
result does not isolate a single numerical cause. This is a bounded diagnostic,
not a proof of all contexts, long-stream isolation or Bevy behavior. The
width-two table enables the experimental frozen multi-position hybrid path;
its scalar reference uses the same build, placement and seed. Reported
hybrid-dispatch counters cover the whole probe, not each individual graph.

Separately, the raw HTTP completion diagnostic confirms two active slots.
Frozen-seed sequential repeats match exactly on two requests per model, while
both concurrent first-token full-vocabulary log-probability comparisons fail
for each model. All first sampled tokens still match. This compares log
probabilities, not raw logits, and active slots do not prove which requests
shared an evaluation graph. An adaptive Coder attempt also fails its serial
repeat, so it cannot isolate concurrency from residency drift. In a separate
Coder screen with physical microbatch one, four serial repeats pass and two
of four concurrent comparisons pass; reducing microbatch size alone is not a
qualification.

As a possible corrective control, enabling the existing scalar-shaped linear,
expert and attention experimental bundle for Coder changes 2/2 scalar prefill
and 16/16 scalar decode logit vectors versus the original reference. It is not
an acceptable preserving fix. Two attempts to capture a new Coder routing trace
also stopped on the observer-equivalence check after the first training request;
their partial traces are retained privately and were not used as routing data.

These gates justify **no concurrent hybrid rollout or batch-two default**.
They do not show Bevy itself is inaccurate or ineffective: no Bevy scheduler
was implemented or benchmarked, and its independently scheduled layer jobs
could avoid this batching path. The exact serialized results support studying
that narrower direction, subject to longer, cross-family numerical and
end-to-end latency gates.

For reproduction, compile `sequence-decode-probe.cpp` against the matching
copied b11669 headers and libraries with `VRAMBRIDGE_PROBE_STEPS=8`. Set the
explicit CPU and CUDA plugin paths in `VRAMBRIDGE_PROBE_CPU_PLUGIN` and
`VRAMBRIDGE_PROBE_CUDA_PLUGIN`, select the fixed plan with
`VRAMBRIDGE_HYBRID_PLAN`, and disable adaptation, asynchronous admissions and
direct admission. The tests enable staged input, guarded overlap, GPU boundary,
gate/up fusion and asynchronous ID handoff. Set
`VRAMBRIDGE_PROBE_PARALLEL_CACHE=1`, `VRAMBRIDGE_PROBE_MOE_CACHE_MIB=4096`,
`VRAMBRIDGE_PROBE_SLOTS=2`, `VRAMBRIDGE_PROBE_CONTEXT=4096`,
`VRAMBRIDGE_PROBE_PROMPTS=2`, `VRAMBRIDGE_PROBE_REPEAT_SERIAL=1` and
`VRAMBRIDGE_PROBE_SERIAL_INTERLEAVE=1`. For the table use
`VRAMBRIDGE_HYBRID_BATCH=2`, `VRAMBRIDGE_PROBE_UBATCH=512`; for the
microbatch control set both to one and additionally enable
`VRAMBRIDGE_PROBE_ALTERNATING_PREFILL=1`. Invoke the probe with `MODEL write
PRIVATE_REFERENCE gpu-cpu-experts`. A numerical failure returns status 2 while
preserving the private reference and per-position results.

## Scalar token-interleave timing control

The [sanitized timing aggregate](../../benchmarks/results/bevy-live-scalar-speed-summary.json)
retains all six accepted samples per model, artifact hashes and the separate
graph-reuse ablation.

This separate control runs eight distinct synthetic requests through one
copied Hybrid backend, serializing prefill and each individual token decode.
It compares one active request with two requests alternated token by token;
both reserve two 4K/F16 slots, use the same 4 GiB frozen synthetic bank and
8/16 threads. It is not Bevy's layer-runahead scheduling and has no HTTP
server. Every active sequence receives 32 excluded warm-up decode steps; three
fresh-load repetitions run in 1/2, 2/1, 1/2 order.

| Model / outputs per request | Aggregate decode tok/s, 1 -> 2 | Whole-cohort tok/s, 1 -> 2 | Median first-token seconds, 1 -> 2 | Median completion seconds, 1 -> 2 |
| --- | ---: | ---: | ---: | ---: |
| Coder / 128 | 37.58 -> 16.04 | 36.43 -> 15.89 | 12.61 -> 24.47 | 15.95 -> 40.34 |
| SC117 Qwen3.8 / 64 | 26.70 -> 13.10 | 25.19 -> 12.83 | 9.17 -> 15.16 | 11.48 -> 24.92 |
| GPT-OSS-120B / 64 | 21.98 -> 13.99 | 20.55 -> 13.48 | 11.24 -> 14.65 | 14.06 -> 23.76 |

First-token time is for a raw token, not streamed text, and includes queueing
from the eight-request arrival burst. All eight complete token sequences match
within each of the three pairs and across fresh loads, for all three models. These
are synthetic, fixed-seed, short-context results, not model performance
ceilings or Bevy measurements.

The earlier Coder timing pilot warmed only the first slot and is retained
separately; the quoted cohort warms both active slot paths. The corrected Coder
run checked the GPU once per second, but did not record per-process swap or
generation-time physical reads. The Qwen and GPT runs additionally sampled
host-memory headroom and owned-process swap; neither observed process swap,
and the minimum available memory was 115.82 and 116.19 GiB respectively.
No run measures generation-time physical reads. GPT's first two pairs completed;
its third attempt ended with a missing `VmSwap` observation as the process exited.
That attempt is retained but excluded. A corrected exit-race guard completed
the replacement pair. The six accepted samples have matching model, plan,
backend, probe and source hashes; runner hashes differ only across the guard
correction. No accepted output token sequence changes across the six loads.
Timing alone does not qualify the
untested layer-level scheduler.

## Graph-reuse ablation

An independent 64-output Coder diagnostic uses the same fixed setup and one
fresh load per arm. The instrumented probe counts llama.cpp graph reuse only
during decode; each arm executes 504 single-position decode calls. With reuse
allowed, one active sequence reuses its graph **496 times** at **36.27 tok/s**;
two alternating sequences reuse it **zero times** at **15.93 tok/s**. With the
existing `LLAMA_GRAPH_REUSE_DISABLE=1` control, one sequence also has zero
reuses and measures **16.23 tok/s**; two measure **16.06 tok/s**. All eight
output token sequences match across the four runs. Same-size and changed-size
graph-reallocation counters are zero during these decode windows. Host headroom,
GPU ownership and zero sampled owned-process swap were verified.

This is one diagnostic sample per arm, not a three-run performance cohort or
proof of a specific CUDA-kernel cost. It strongly supports lost llama.cpp graph
reuse as a contributor to this Coder slowdown: disabling reuse on the serial
control reproduces nearly the same rate. The copied backend retains only one
active previous graph and checks participating sequence IDs for certain
layouts, consistent with the observation; the exact rejection condition was
not instrumented. No safe per-sequence graph pool or Bevy scheduler was built.

## Next qualification gate

Before a Bevy-style backend patch, collect concurrent per-request layer timing,
CPU/GPU execution and actual cache residency with the existing isolated
backend, including graph boundaries and copy dependencies. Establish that the
baseline itself passes unchanged per-position raw-logit and state-isolation
gates for Coder, Qwen3.8 and GPT-OSS. Only then prototype a default-off
layer-job scheduler in a copied backend, preserving all native routing and CPU
fallbacks. Pin all in-flight expert slots through CUDA completion, account for
per-group capacity without assuming expert sharing, and test cancellation,
KV/recurrent state and deadlock recovery.

Compare 1/2/4/8-client matched cohorts after warm-up over three fresh runs;
report median aggregate throughput, per-agent TTFT/completion, memory, CPU/GPU
overlap and upload bytes. Keep profiles and installed binaries unchanged.
Promote nothing without cross-family numerical and held-out workload results.
