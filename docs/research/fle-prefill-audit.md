# FastLocalExperts prefill lending: source audit

Source inspected: [sirfyyn/FastLocalExperts, commit
`e6909b7550c9917194cd351998f3770717b34ce2`](https://github.com/sirfyyn/FastLocalExperts/tree/e6909b7550c9917194cd351998f3770717b34ce2),
2026-11-06. This is a source review, **not a local implementation or speed
measurement**. The upstream project is MIT-licensed.

## Mechanism and constraints

[`FLE_UBATCH_LEND`](https://github.com/sirfyyn/FastLocalExperts/blob/e6909b7550c9917194cd351998f3770717b34ce2/docs/fle/ENV.md)
is experimental and off by default. When a large prefill call crosses the
default 2,048-token threshold, FLE synchronizes execution, pauses its
cache-upload worker, frees complete per-layer GPU expert-cache allocations,
and reserves a larger compute buffer. Lent layers execute experts on CPU.
When short calls resume it restores the small compute-buffer configuration,
allocates the cache layers and refills previous resident slots from host
memory. A failed layer allocation leaves that layer on CPU; insufficient
lending falls back to the small ubatch. An epoch change prevents reuse of
graphs referencing the old cache tensors. The worker synchronization,
allocation, refill and graph rebuild are real costs, not free overlap.

`FLE_UBATCH_PREFETCH` is a separate, optional staging ring during lending.
Upstream defaults to three slots, each as large as the largest host expert
tensor; its example is 450 MiB per slot, plus a default 512 MiB lend reserve
and the increased compute-buffer requirement. Those example sizes are not
measurements of SC117. On a 16 GiB GPU, KV cache, ring, buffers, refill and
the displaced expert cache must be budgeted together. Refill and subsequent
decode must be included in any end-to-end comparison.

FLE's [`llama-moecache.cpp`](https://github.com/sirfyyn/FastLocalExperts/blob/e6909b7550c9917194cd351998f3770717b34ce2/src/llama-moecache.cpp)
owns device buffers per layer. Our b11669 Hybrid cache instead places
compatible layers in shared bank allocations with layer views and a group
LRU. Freeing a single view does not release its backing bank; a direct port
needs an allocation/lifetime redesign and safe in-flight transfer and graph
invalidation. Treat this as a feasibility question, not a patch-ready win.

FLE's `FLE_SLOT_EARLY_SAVE` saves a released slot into host prompt cache
earlier. It is not the periodic mid-message checkpoint mechanism studied
[here](prompt-reuse-checkpoints.md). `FLE_PREFILL_LEAN` targets the MTP
context and a QSA indexer; it is not evidence of a generic SC117 benefit.

Upstream's documented benchmark configuration also includes nonzero
`FLE_MOE_CACHE_BIAS`, which can change expert selection, Q8 KV and a separate
quantized MTP head. Its reported throughput uses a different model and GPU;
it is not comparable to our F16/unchanged-routing measurements. Do not copy
those settings as an accuracy-preserving optimization.

## Decision

Study a memory-lifetime design only after measuring the actual cost and
reclaimable bytes for our shared banks. A viable experiment must retain the
same weights, F16 KV and routing, and account for synchronized return,
reallocation, refill, prompt latency and post-prefill decode. Require
full-vocabulary raw-logit checks and repeated warm-up-excluded comparisons
across Coder, SC117 Qwen and GPT-OSS before considering a production variant.
No FLE code or settings are being installed or promoted.

Pinned source SHA-256: `src/llama-context.cpp`
`3a0e3d13e836c028f346c6db7ab41e0842f8188cb42c95b7ed470e64c45f214c`;
`src/llama-moecache.cpp`
`4ea69c4a8eb702f1b5e5b4c4c8f646157c3e45c46d16ab8d28e4866667d5f5c0`;
`docs/fle/ENV.md`
`5d411af870d0acbd1f2df9bf226074db7ee605ce5473e5bdf98e7808b024a31d`.
