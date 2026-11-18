# Prompt checkpoint raw-logit attribution

**Decision: checkpoint promotion remains blocked.** This is a numerical
attribution experiment, not a throughput benchmark or an implementation of
FLE. The unchanged full-recomputation control itself can diverge, so the
snapshot results cannot establish whether checkpointing adds an error.

## Contract

The [probe](../../tools/trace/prompt-checkpoint-logits-probe.cpp) uses the
copied b11669 Hybrid library, SC117 Qwen3.8 IQ3_S weights, one sequence,
16,384 allocated context, F16 KV, 2,048 logical / 512 physical batch,
8/16 threads and a 4,096 MiB frozen, empty expert bank. Hybrid reported all
48/48 routed layers covered, no admissions and zero GPU expert executions.
It uses the same synthetic 8,192-token prompt, edit at position 4,353 and
10-token append as the previous [server study](prompt-reuse-checkpoints.md).
The two model shard hashes are
`40f369eca301ec6ea318dbb671b3132c26a2f0b89aa5458a7051b33215ee296c` and
`316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113`.

Each comparison uses the full 248,320-entry **raw-logit** vector at prompt
completion and four teacher-forced continuation positions. The reference
continuation is selected once and reused for all candidates. Both maximum
absolute error <= 0.01 and relative L2 <= 0.001 are required at every
position. `split` is at the 4,096 or 8,192 prefix; `ragged_split` uses
4,353 or 8,188. Full and partial snapshots restore the corresponding prefix
and explicitly check both tail-removal success and the resulting sequence
position before processing the target suffix. All original model weights and
the installed backend were untouched.

## Initial attribution run

The maxima below are over the five positions; absolute and relative maxima
need not occur at the same position. Matching argmax in all 50 comparisons
is not a correctness pass.

| Case | Passing positions | Maximum absolute error | Maximum relative L2 |
| --- | ---: | ---: | ---: |
| Middle edit: cold repeat | 0/5 | 2.320655 | 0.147044 |
| Middle edit: aligned split | 0/5 | 2.295743 | 0.148509 |
| Middle edit: ragged split | 0/5 | 1.864561 | 0.108906 |
| Middle edit: full restore | 0/5 | 1.232626 | 0.079947 |
| Middle edit: partial restore | 0/5 | 1.908944 | 0.127582 |
| Append: cold repeat | 0/5 | 0.630167 | 0.045945 |
| Append: aligned split | 5/5 | 0 | 0 |
| Append: ragged split | 0/5 | 2.877516 | 0.203056 |
| Append: full restore | 0/5 | 0.630167 | 0.045945 |
| Append: partial restore | 0/5 | 0.630167 | 0.045945 |

Full snapshot sizes were 244,065,636 and 370,091,364 bytes at the two
prefixes; partial snapshots were 118,039,428 bytes at both. All four restore
calls reported successful tail removal and the expected maximum position.
Those position checks do not prove the restored contents are correct. Equal
summary errors do not establish equality between candidate distributions.

An independent fresh-process follow-up produced an **exact 5/5 same-context
cold repeat**, unlike the initial 0/5. Its newly allocated context failed
5/5 against the original reference, but that context lacked the original
128-token warmup, so this is not a controlled context-lifetime conclusion.
In a third fresh process, with identical 128-token warmups in both contexts,
the same-context cold repeat failed 0/5 (maximum absolute 0.980807,
relative L2 0.053360); the fresh context also failed 0/5 (2.109135,
0.126440). These are separate comparisons against that process's original
reference, not proof that context recreation caused the mismatch. The mixed
cold-repeat results must be investigated rather than averaged or treated as
proof that any one mechanism causes the discrepancy.

A single matched-warmup ablation with `GGML_CUDA_DISABLE_FUSION=1` still
failed the same-context cold repeat 0/5 (maximum absolute 3.653924,
relative L2 0.224307); the fresh-context comparison happened to match 5/5.
Thus disabling that CUDA fusion path is not a demonstrated repeatability
fix. This flag does not disable every possible fusion in the Hybrid backend.

A separate single 16K matched-warmup run with `LLAMA_GRAPH_REUSE_DISABLE=1`
matched 5/5 for both same-context and fresh-context comparisons. The backend
confirmed graph reuse was disabled. Subsequent 96K controls below contradict
the idea that disabling graph reuse is a sufficient fix.

## 96K repeatability control

At 98,304 context and a frozen 6,144 MiB bank, two fresh-process runs used
the same 8,192-token edited prompt and matched 128-token warmups. Each
compared three consecutive full recomputations and a newly created context
against its own initial full raw-logit reference. The second run disabled
graph reuse and the backend confirmed it. Both runs reported full 48/48
Hybrid coverage with zero GPU expert executions.

| Execution | Same-context repeats passing all five positions | Fresh context | Largest absolute / relative error |
| --- | ---: | ---: | ---: |
| Normal graph-reuse setting | 1/3 | 0/5 | 1.908944 / 0.127582 |
| Graph reuse disabled | 1/3 | 0/5 | 1.232626 / 0.079947 |

One of three repeats was exactly equal in each run, but both also had two
5/5 failures. These are small, sequential diagnostic samples, not paired
performance trials. Disabling graph reuse and disabling CUDA fusion each
failed to establish consistent repeatability; neither is a proven remedy or
a demonstrated root cause. The 96K allocation matches the previous server
capacity, but the bank remains frozen and this is still not the server's
complete request/scheduler/sampling execution contract.

The direct probe is not the patched server. The full snapshot attribution
uses a smaller context and bank than the 96K/6 GiB server benchmark. It does not reproduce the server's
complete request history, scheduler or sampling stack. No claim is made
about raw-logit equality of the server variants or other model families.

## Next hypothesis: pooled-indexer selection

The subsequent [QSA determinism screens](qsa-selection-determinism.md)
confirmed that synthetic cutoff ties can vary selected sets, but did not
find a live equal-score finite-tie cause. A bounded deterministic QSA
intervention still failed the fresh-context raw-logit gate.

In the copied source, SC117's `qwen4exp` graph applies `ggml_top_k` to QSA
pooled-indexer scores; the model reports a four-token compression ratio and
2,048 selected token positions. The copied CUDA top-k source includes a
multi-row radix path whose gather uses atomic counters to choose among equal
cutoff keys. If the deployed plugin takes that path and valid pooled blocks
tie at the cutoff, different selected blocks could change attention even
when token IDs and graph shapes match. The later screens did not identify
that as the live cause; the inspected source alone also does not prove the
deployed plugin was built from that exact file.

A bounded follow-up should first check observer equivalence, then compare
`indexer_top_k` sets, cutoff scores and masks at matching layers and ubatches
for repeated inputs. Only if it observes differing valid selections at tied
cutoffs should a deterministic-tie experiment be built in an isolated
backend. Changing attention selections is not automatically equivalent and
must meet raw-logit and cross-family gates; do not promote a workaround from
this hypothesis alone.

## Reproduction

Compile the probe against the matching copied b11669 headers and shared
libraries, defining `VRAMBRIDGE_PROBE_STEPS=4`; the existing sequence probe
provides its batch and comparison helpers. Explicitly load compatible CPU and
CUDA plugins using `VRAMBRIDGE_PROBE_CPU_PLUGIN` and
`VRAMBRIDGE_PROBE_CUDA_PLUGIN`. Use the frozen `auto` Hybrid plan with
adaptation, asynchronous admissions and direct admission disabled, and the
same staged-input, overlap, GPU boundary, gate/up fusion and asynchronous-ID
settings as the prior numerical controls. Invoke with the first model shard.
The output contains numeric metrics only; keep backend logs private. The
`--self-test` exercises the comparison gate. Set
`VRAMBRIDGE_PROBE_FRESH_CONTEXT=1` for the matched-warmup context diagnostic.
`VRAMBRIDGE_PROBE_CONTEXT=98304`, `VRAMBRIDGE_PROBE_MOE_CACHE_MIB=6144`,
and `VRAMBRIDGE_PROBE_EXTRA_REPEATS=2` reproduce the 96K repeated-control
shape. The probe returns 2 for a numerical failure, 3 for unsupported tail
removal, and 1 for an operational error; it never converts a failed gate
into success. Before using 96K, verify actual GPU and host headroom.

The initial probe source SHA-256 was
`93a1d91c672ddb9dc14d70100fc0119d19e34f42933efbbf9619949f37697668`;
its later change adds the fresh-context mode without changing the original
attribution path. The copied `libllama.so` hash was
`b5c5203b346768964c8cd3b9cdd366329ef6d0f9508ee0d94df25f5171ffda09`,
CPU plugin `d6393353c9c99a97016231e09fb278bc05a852a5ece6085f4645f0b4ee31cb35`,
and CUDA plugin `a9b36c7c5ae20b5c25f6ec0303016c9dafdd63118bab68aba1c874a56cc4757e`.
