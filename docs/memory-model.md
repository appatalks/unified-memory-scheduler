# Memory Model

VRAMBridge models memory as explicit tiers with distinct costs.

| Tier | Intended role | Key constraint |
| --- | --- | --- |
| VRAM | active weights, hot KV, compute buffers | smallest capacity; required by CUDA kernels |
| pinned RAM | bounded asynchronous staging | consumes unswappable host pages; must be capped |
| pageable RAM | model backing store | DMA requires driver staging or registration |
| NVMe | optional cold storage | block-I/O latency; not memory-like |

An allocation's capacity cost and transfer cost are separate. Placement is only useful when transfer time can be amortized or hidden by useful compute.

For a tensor of size $S$ bytes crossing a measured link bandwidth $B$, the optimistic transfer floor is:

$$
t_{copy} = \frac{S}{B}
$$

Prefetch can hide that copy only when the available independent compute window is at least $t_{copy}$, with additional allowance for launch, synchronization, and contention. The measured bandwidth is therefore an input to policy, not merely a machine specification.

Pinned memory is a staging resource, not a second cache with unbounded capacity. Initial experiments should use a configurable 1-8 GiB pool and test larger values only when data demonstrates a benefit.

CUDA Unified Memory remains a baseline tiering mechanism controlled by the CUDA driver. VRAMBridge's explicit design differs by using model execution order to decide what to move and when. No superiority is assumed before measurement.