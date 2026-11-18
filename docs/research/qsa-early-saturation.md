# QSA saturation bypass: experimental research

**Default off; not qualified for release.** This project-specific experiment
asks whether an early QSA indexer can avoid computing a selection when the
selection budget covers *every* pooled candidate. It is an application of an
attention invariant, not a claim that the general idea is new. The reviewed
[aggregate](../../benchmarks/results/qsa-saturation-research-summary.json)
contains the timing, numerical and backend identities.

## Mechanism

SC117's `qwen4exp` graph groups four cached positions per pool and selects up
to 2,048 positions. When the total padded pool count is no larger than the
pool selection budget, top-k contains every pool; its visible members plus
the tail may be equivalent to the ordinary causal mask. The
[experimental patch](../../third_party/patches/llama-b11669-qsa-saturated-mask-experimental.patch)
uses that conservative graph-shape condition and returns the existing causal
mask instead of scoring, selecting and scattering pools. It still constructs
and writes the raw and pooled indexer keys needed by later, unsaturated
ubatches. It does not intentionally change weights, router logic or KV
quantization; expert-choice identity has not been separately traced.

The opt-in `VRAMBRIDGE_QSA_SATURATED_MASK=1` is implemented only in a copied
b11669 backend; it is absent from the canonical patch and the installed
backend. The flag is read once per process. Changes of graph shape invalidate
reuse through the existing pooled-indexer input checks.

## Initial exactness screen

In one SC117 synthetic 2,048-token prompt at 16K context, F16 KV and a
frozen 4 GiB bank, the unmodified backend's observed QSA and causal masks
matched bitwise across all 12 QSA layers, 48 physical batches, 24,576 query
rows and 31,457,280 mask entries. The unobserved cold repeat and the observed
versus unobserved full raw-logit checks each matched 5/5 positions. The
evaluation callback synchronizes and disables guarded overlap; one matching
comparison does not establish observer neutrality in general. A second screen
found four output/input aliases, so it captured the causal mask before any
possible in-place reuse and compared all layers to that snapshot: the same
31,457,280 entries matched, with no causal-source differences. Both raw-logit
controls again matched 5/5 positions.

For a separate 8,192-token edited prompt, a rebuilt unmodified control wrote a
private full-vocabulary reference. A fresh control, opt-in candidate and
default-off candidate each matched its first raw-logit vector and four
teacher-forced continuation vectors exactly, 5/5 positions apiece. The
opt-in activation marker and 48/48 frozen-bank Hybrid coverage were present;
GPU expert execution was zero. This exercises the transition to unsaturated
ubatches and subsequent decode, but it is one synthetic prompt, not proof
for checkpoints, adaptive expert residency or other contexts and families.

The first mask-observer attempt was invalid: a later graph callback renamed
the tensor, so the intended capture did not occur. The corrected probe checks
all expected layers and fails on missing coverage rather than counting it as
a pass. An intermediate alias-unaware screen is also not a valid complete
comparison; the final observer accounts for the reused storage explicitly.

## 8K timing and numerical limitation

Three fresh control/candidate pairs alternated order on the same SC117 16K,
4 GiB frozen-bank contract. Each process excluded a full 8,192-token warmup;
the measured prompt timer includes completion of queued GPU work and excludes
model loading and decode. The same saved full-vocabulary reference was used
for numerical checks.

| Repetition | Control prompt | Candidate prompt | Paired candidate change |
| --- | ---: | ---: | ---: |
| 1 | 40.097 s | 40.234 s | +0.34% |
| 2 | 38.995 s | 40.422 s | +3.66% |
| 3 | 38.562 s | 38.250 s | -0.81% |
| Median | 38.995 s | 40.234 s | +0.34% median paired change |

The paired changes are mixed and within the project's approximate 5% noise
band: **no speedup is established**. Two of three unchanged controls and all
three candidates failed the 5-position raw-logit comparison to the saved
reference after full-length warmup. Since the reference itself was captured
without that full-length warmup, these are not matched-history numerical
controls; the unchanged failures prevent attribution to the prototype.
Neither a faster result nor numerical qualification is established.

## 2K matched-warmup control

Three additional alternating fresh-process pairs used a 2,048-token edited
prompt with a separately saved reference captured after an identical full
warmup. All control and candidate comparisons matched all five raw-logit
positions exactly; the opt-in marker and 48/48 frozen-bank coverage were
confirmed. Prompt timing again excludes the matching warmup and decode.

| Repetition | Control prompt | Candidate prompt | Paired candidate change |
| --- | ---: | ---: | ---: |
| 1 | 9.720 s | 9.261 s | -4.73% |
| 2 | 9.325 s | 9.566 s | +2.58% |
| 3 | 9.563 s | 9.499 s | -0.67% |
| Median | 9.563 s | 9.499 s | -0.67% median paired change |

The paired changes are mixed and within the noise band, even though this
shorter input gives the bypass a larger share of the work. **Do not promote
the tested saturation guard:** no convincing gain has been measured, and the
long-prompt baseline and broader correctness gates remain unresolved.

## Gates

The mask equivalence is not a universal proof for sequence holes, shifts,
shared or multimodal positions, multi-sequence batches, nonfinite indexer
scores, different attention layouts or other models. The prototype's guard
must be tested against those cases and preserve the pooled cache after the
short-circuit. Validate long full-vocabulary raw logits and exact text, then
repeat on the required model families. Compare prompt and complete request
latencies with warmup excluded and alternating matched controls, including
realistic adaptive-bank behavior and post-prefill decode. If an unchanged
control diverges, treat the comparison as inconclusive rather than relaxing
the accuracy gate.
