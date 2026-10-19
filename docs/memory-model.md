# Memory Model

VRAMBridge models memory as explicit tiers with distinct costs.

| Tier | Intended role | Key constraint |
| --- | --- | --- |
| VRAM | active weights, hot KV, compute buffers | smallest capacity; required by CUDA kernels |
| pinned RAM | bounded asynchronous staging | consumes unswappable host pages; must be capped |
| pageable RAM | model backing store | DMA requires driver staging or registration |
| NVMe | optional cold storage | block-I/O latency; not memory-like |

An allocation's capacity cost and transfer cost are separate. Placement is only useful when transfer time can be amortized or hidden by useful compute.

## File Cache Versus RAM-disk Staging

On Linux, ordinary file-backed model weights can already reside in RAM through
the filesystem page cache. Mapped backend RSS and cached file pages overlap;
do not add them or interpret a low non-cache-used value as an unused model
working set. OS-reported availability is not a guarantee that those pages can
be reclaimed safely during registered/pinned inference.

A tmpfs RAM disk consumes the same host RAM, is volatile, and can use swap.
Copying a model there may duplicate file-cache pages or displace useful cache;
explicit host weight copies, KV allocations and pinned buffers must also fit.
It does not increase memory bandwidth or VRAM capacity.

Distinguish cold model loading, prompt processing, and continued generation
before staging. Include copy/setup time in a cold-start comparison, and measure
physical backend reads rather than cached read-call traffic or GPU transfers.
The [warm-model investigation](../benchmarks/results/ram-staging-io-investigation-summary.json)
found fully cached GPT weights and zero physical reads during six sampled
generation windows; it did not establish a RAM-disk speedup.

## Persistent Fast-disk Staging

An archive and a disposable model working set are distinct roles. A fast-disk
staging directory can reduce cold-load latency for an archived model without
changing warm RAM-backed decode. Preserve archive originals, stage every shard,
verify hashes, and record source/copy ownership before using the staged path.
Eviction must target only known staged copies that are no longer in use.

Choose a device from measured behavior rather than its NVMe label. The
[current read validation](../benchmarks/results/nvme-staging-read-validation-summary.json)
favors the Samsung 990 PRO over an unexpectedly slow WD result, but the Samsung
is still read-only and not accessible for user-owned staging. Safe writable
mounting, free-space headroom and filesystem copy performance remain prerequisites;
no filesystem conversion or data relocation is implied by that recommendation.

For a tensor of size $S$ bytes crossing a measured link bandwidth $B$, the optimistic transfer floor is:

$$
t_{copy} = \frac{S}{B}
$$

Prefetch can hide that copy only when the available independent compute window is at least $t_{copy}$, with additional allowance for launch, synchronization, and contention. The measured bandwidth is therefore an input to policy, not merely a machine specification.

Pinned memory is a staging resource, not a second cache with unbounded capacity. Initial experiments should use a configurable 1-8 GiB pool and test larger values only when data demonstrates a benefit.

CUDA Unified Memory remains a baseline tiering mechanism controlled by the CUDA driver. VRAMBridge's explicit design differs by using model execution order to decide what to move and when. No superiority is assumed before measurement.