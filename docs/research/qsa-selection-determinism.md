# QSA top-k determinism: synthetic and live screens

**Decision: the CUDA operator can vary valid selections at tied cutoffs, but
we have not shown that this causes SC117's raw-logit discrepancy.** Do not
change attention selections or promote a workaround from these screens.

## CUDA operator control

The [bounded probe](../../tools/trace/topk-tie-probe.cpp) runs the exact
installed CUDA plugin used by the prior copied backend, without loading a
model or changing that plugin. Synthetic F32 scores have 2,048 columns,
512 rows and top-k 512. The 64 repetitions compare every selected set against
the first; the first iteration is included in the row count. The probe checks
index bounds, uniqueness and that no excluded score exceeds the cutoff.

| Input | Changed sets / 32,768 row comparisons | Same set, different order | Finite cutoff ties |
| --- | ---: | ---: | ---: |
| Unique scores | 0 | 32,256 | 0 |
| Ties entirely within the selected set | 0 | 32,255 | 0 |
| Ties across the selection boundary | 298 | 31,956 | 32,768 |
| Boundary ties with `-inf` masked columns | 188 | 32,062 | 32,768 |

The largest changed set replaced 256 of 512 indices. All reported selections
were valid top-k choices for the synthetic scores. Changed ordering alone is
not a changed set and does not by itself establish a different QSA mask.
In a second fresh process, boundary-tie and masked-boundary-tie sets changed
558 and 382 times, respectively; a mixed-score tied-cutoff fixture changed
13,653 sets. Unique-score sets again never changed. These counts are from
32,768 row comparisons per case, including the first self-comparison.
An independent [vLLM issue](https://github.com/vllm-project/vllm/issues/67192)
reports order sensitivity in a different sparse-attention implementation;
our dense-mask construction must be evaluated on its own rather than assumed
to have that behavior.
This microtest has no model, timing or workload-quality conclusion. The
plugin SHA-256 is
`a9b36c7c5ae20b5c25f6ec0303016c9dafdd63118bab68aba1c874a56cc4757e`;
the copied source's multi-row radix gather uses atomics, but the experiment
does not prove exactly which branch was compiled into that plugin.

## SC117 live observer

The diagnostic mode of the [checkpoint probe](../../tools/trace/prompt-checkpoint-logits-probe.cpp)
captures one QSA layer's score and top-k tensors across the identical
8,192-token edited prompt, twice, and compares sets as well as output order.
It checks each selected set against its score vector and distinguishes
finite-score selected pools from `-inf` masked pools. This uses the copied
b11669 backend, 16K context, F16 KV and a frozen 4 GiB empty Hybrid bank;
both captures have 16 frames / 8,192 rows and run on the GPU.

| Layer | Bit-identical score rows | Changed sets | Changed visible sets | Changed sets with identical scores and finite cutoff tie | Finite cutoff ties in each capture |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3 | 8,192/8,192 | 22 | 0 inferred from identical scores, valid top-k and `-inf` cutoff | 0 | 1 / 1 |
| 7 | 8,192/8,192 | 28 | 0 measured | 0 | 0 / 0 |
| 23 | 8,192/8,192 | 24 | 0 | 0 | 1 / 1 |
| 47 | 2,142/8,192 | 5,963 | 5,938 | 0 | 1 / 1 |

At layer 3, the source sends invisible selected pools to dump rows rather
than unmasking them. This narrows the original hypothesis for this one prompt
and layer; it does not prove all downstream masks or layers are equal.
Layer 7's two observed passes happened to match all five output positions,
so it cannot localize a divergent pass. Layer 23's observed logits failed
0/5 despite identical scores and visible sets at that layer. In the layer-47
run, 5,938 rows changed visible selections, but no changed set with identical
scores had a finite cutoff tie: the score vectors had already changed. These
are different runs, not a simultaneous layer-by-layer causal trace.

Observer equivalence is a serious limitation. The backend evaluation callback
synchronizes and disables guarded overlap even when unarmed. In the layer-3
run, one observed/unobserved comparison matched 5/5 positions while both
unobserved and observed cold repeats failed 0/5. In the layer-7 run the
observer/unobserved comparison failed 0/5, while each cold repeat happened to
match 5/5. Baseline repeatability is itself mixed, so neither result proves
the callback is neutral or responsible. In the layer-23 run the unobserved
repeat and the observed/unobserved comparison both matched 5/5, but the
observed repeat failed 0/5. In the layer-47 run the unobserved repeat matched
5/5 while both other comparisons failed 0/5. Results are provisional, not a
server reproduction or causal proof. Only this model and synthetic input have
been inspected.

## Deterministic-selection diagnostic

An isolated preload-only [CUDA diagnostic](../../tools/trace/qsa-deterministic-topk.cu)
sorts supported QSA top-k shapes by score and then index, forwarding all
other operations to the original plugin. It changes no installed backend.
On five synthetic 2,048-by-512 score matrices, including mixed scores, masked
cells and boundary ties, all 64 repetitions matched a canonical CPU oracle;
the fallback shape also matched. This is diagnostic only: selecting a
different equal-scored pool can change attention, and the kernel does not
establish accuracy or production suitability.
The diagnostic has not been validated for NaN scores, arbitrary tensor
layouts, other GPU architectures or backend ABIs.

In one fresh 16K SC117 process the diagnostic was confirmed active. Three
same-context full recomputations each matched all five raw-logit positions,
but a new, identically warmed context failed 0/5 against that reference
(maximum absolute 2.109135, relative L2 0.126440). Thus this limited
intervention did **not** make the complete experiment repeatable. It only
intercepts QSA top-k with more than 1,024 columns and multiple rows; it does
not replace single-row decoding or every possible nondeterministic path.
Baseline-to-candidate full-logit equivalence has not been established, and
the preload shim depends on this particular copied backend's internal ABI.
It is not a production patch or default.

## Next gate

Capture downstream selected masks and other QSA layers under repeated,
observer-qualified conditions, including divergent observed pairs. If valid
selected sets differ with identical scores at a finite cutoff, compare an
isolated deterministic selection experiment against an explicit oracle and
the original raw logits. Tie-breaking may change attention content even if
the scores are equal, so require unchanged accuracy and the existing model
family gates; deterministic output is not automatically equivalent output.

Compile the GPU microprobe against matching copied ggml headers and
libraries; invoke it with the explicit CUDA plugin path or `--self-test`.
For live observation, compile the checkpoint probe as documented in its
[attribution study](prompt-checkpoint-logits-attribution.md), set
`VRAMBRIDGE_PROBE_QSA_OBSERVER=1`, and select an attention layer with
`VRAMBRIDGE_PROBE_QSA_LAYER`. Captured tensors remain in process memory;
the output contains aggregate counts and raw-logit error metrics only.
