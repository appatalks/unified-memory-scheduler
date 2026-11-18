# Commit-late expert staging: measured entry gate

## Decision

Do not build or enable the proposed staging path from the predictors tested
here. The evidence is specific to GPT-OSS-120B, one technical training trace,
three short held-out requests and an optimistic scratch simulation. It does
not prove that a richer hidden-state predictor, another schedule or another
model cannot benefit. No backend patch, model weight, installed profile or
Synapse setting was changed.

## Baseline semantics and evidence

The original design assumed a newly admitted miss always executes on CPU for
the current layer. That is not how the current Classic path works: `admit()`
updates the slot map before `write_dispatch()`, so a newly admitted expert can
execute on GPU in that same layer. Any future candidate must preserve the real
per-rank execution placement, original router outputs and reduction order; it
cannot simply defer that expert to CPU without a new numerical qualification.

Four existing private Nsight/CPU profiles use GPT-OSS-120B MXFP4 with 16K
context, a 12,000 MiB trained bank, direct asynchronous admissions and 8/16
threads. One technical request is used only for training a small predictor;
the other requests correspond to the existing held-out coding, reasoning and
prose prompts. Each requested 128 outputs. Their CPU-contained envelopes cover
123 complete decode tokens plus an initial partial token. We discard the first
64 complete tokens to establish the engine's full 63-token prior history and
score the remaining 59; no held-out labels are used for training.

The profiler observes the pre-admission selected IDs and slot-map misses, then
the subsequent CPU expert masks. For the tested Classic configuration, the
pre-miss set minus the CPU-cold set infers same-layer promotions. Every inferred
promotion is corroborated by the corresponding three expert-projection H2D
`cudaMemcpyAsync` launches inside that exact dispatcher range. The profiler
does not record all original model-shard or backend library hashes in every
request report; matching filenames and recorded profile/plan hashes do not
replace those identities.

| Request | Inferred promotions | Correlated projection copies | Copy union / envelope, ms | Copy time without other recorded CPU/GPU work, ms |
| --- | ---: | ---: | ---: | ---: |
| Technical | 728 | 2,184 | 374.74 / 3,067.70 | 132.90 |
| Coding | 645 | 1,935 | 329.60 / 2,799.33 | 143.71 |
| Reasoning | 633 | 1,899 | 324.10 / 2,991.06 | 133.37 |
| Prose | 628 | 1,884 | 319.45 / 2,833.86 | 131.45 |

The copies and observed kernels use the same CUDA stream in these traces and
do not overlap. The last column excludes recorded CPU graphs, kernels, other
copies, memsets and observer ranges. It may still contain unrecorded work and
is **not** a measurement of causally exposed or removable latency. Nsight and
the CPU profiler perturb execution; these intervals are not throughput results.

## Causal prediction and scratch simulation

The simplest predictor stages, for each layer, the highest-ranked expert that
ran on CPU in that layer on the preceding token and appeared at least three
times in the prior 63 selections. That floor is consistent with the existing
four-observation admission margin; no scored admission violated it. The
predictor only uses earlier observations, not the current token's real route
or admission outcome.

For each prediction, release is conservatively placed at the preceding
same-layer CPU graph's end; deadline is the next same-layer dispatcher start.
A simulated four-slot scratch pool uses first-free admission, one speculative
copy queue and cancellation of requests still queued at their deadline. Copy
service is the sum of the measured median H2D projection durations. Scratch
starts empty at the scoring boundary. The model ignores baseline H2D traffic,
DRAM contention, biases, prediction overhead, device-to-device commit and
changes to future deadlines; it is an optimistic scenario for this specific
policy, not a bound for every scheduler.

| Held-out request | Actual admissions | Staged predictions | Useful before deadline | Unused projection bytes |
| --- | ---: | ---: | ---: | ---: |
| Coding | 239 | 186 | 27 | 1.96 GiB |
| Reasoning | 209 | 211 | 16 | 2.40 GiB |
| Prose | 245 | 181 | 17 | 2.02 GiB |
| **Total** | **693** | **578** | **60** | **6.38 GiB** |

These are modeled transfers, not physical measurements. Four slots contain
about 50.4 MiB of projection weights before biases and allocator overhead.
The full unbounded predictor covers 108/693 admissions but makes 1,280
predictions. A broader two-expert, frequency-ranked known-cold predictor
captures roughly 67-75% per workload in the unlimited model, at the cost of
42-48 GiB of unused projection copies **per 59-token request**. It is not a
reasonable reason to skip the bandwidth and capacity gates.

A separate binned predictor was trained only on the technical trace: 12,007
candidate examples and 325 positive admissions after warmup. It uses past
selection count, age and previous router rank, with a fixed Bayesian smoothing
rule and minimum bucket support. At the predeclared 0.1 threshold, it predicts
129 experts across the three holdouts, matches 14 of 693 admissions and
models 1.42 GiB of unused projection copies. At 0.2 it matches seven; at 0.3
it makes no predictions. Adding the already observed preceding-layer route at
its pre-dispatch snapshot yields 14 matches from 127 predictions at 0.1; only
11 have enough independent elapsed time for one median expert copy. This
independent timing ignores scratch and concurrent DMA. These are exploratory
small-sample screens, not calibrated precision estimates or evidence that
other predictors cannot work.

## Reproduction and follow-up

[`vrambridge-commit-late-screen.py`](../../tools/benchmark/vrambridge-commit-late-screen.py)
accepts a private CPU profile, matching Nsight SQLite file and owned-request
report. It rejects missing layer observations, unsupported masks, unmatched
CUDA runtime calls and per-dispatch copy-count inconsistencies; it emits only
aggregate counts and hashes. The companion
[`vrambridge-commit-late-learned.py`](../../tools/benchmark/vrambridge-commit-late-learned.py)
requires a separate training trace and distinct held-out prompt hashes with the
same recorded model/profile. The [sanitized aggregate](../../benchmarks/results/commit-late-staging-summary.json)
contains the source/input hashes and every evaluated predictor configuration;
raw routing, prompts, Nsight data and private reports remain local.

The next defensible entry gate would require a substantially better causal
signal—potentially from a qualified hidden-state observer or trained adapter—
that identifies real admissions early enough while limiting extra bytes. It
would also require independently verified Coder and Qwen3.8 admission traces.
Only after that screen should a bounded, default-off staging prototype be
considered, followed by unchanged full-vocabulary and output-identity checks
and three-family held-out timing.
