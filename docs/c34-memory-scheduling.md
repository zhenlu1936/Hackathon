# C3.4 — Memory planning and scheduling

## Release contract

This 10-point section is code-reviewed. Five features are worth 2 points each, and each must be implemented through the actual execution-plan path. Interfaces, comments, logs, or unused classes do not score.

The remote H200 is the designated AEC device. Full C3.4 integration requires
the released plan to drive the corresponding CuPy/H200 allocation, transfer,
kernel, stream, and event operations. Do not submit cached plans or precomputed
schedules keyed to public or hidden model identity.

## A: device pool and weight preload

Provide real device allocation/free abstractions and connect initializers/constants to device buffers referenced by kernel steps.

The release represents weight residency with:

```text
ALLOC(weight_slot) -> H2D(initializer, weight_slot) -> KERNEL(..., weight_slot)
```

Cache immutable weights for the model lifetime. Do not recreate/upload them per input batch unless implementing intentional prefetch.

## B: intermediate lifetime reuse

Compute `first_use` and `last_use` over the final kernel schedule, including C3.2 intermediate tensors and C3.3 fused outputs. Use interval allocation:

1. Sort tensors by first use.
2. Expire slots whose last use precedes the next tensor's first use.
3. Reuse a compatible free slot by device, alignment, dtype constraints, and capacity.
4. Rewrite execution-plan tensor bindings to physical slots.

Views/aliases need special handling so their base allocation cannot be reused while an alias remains live.

## C: pool reuse and fragmentation management

A free-list alone earns only partial credit. Add and expose at least one concrete policy:

- best-fit blocks;
- size classes;
- adjacent-block coalescing;
- segmented arenas.

Track requested bytes, reserved bytes, active bytes, internal fragmentation, reuse hits, and peak reserved bytes. These counters make both code review and C3.5 tuning defensible.

## D: weight prefetch

Uploading all weights before the first kernel is not prefetch. The plan must show overlap semantics:

```text
copy_stream:    H2D weights for layer k+1
compute_stream: execute layer k
dependency:     layer k+1 waits for its weight-ready event
```

Keep frequently reused or small weights resident. Prefetch is most useful when startup memory pressure motivates staged residency; measure whether extra synchronization hurts total process time.

## E: stream-level parallelism

Assign independent work to different compute streams based on DAG dependencies, not round-robin assignment. Insert events for cross-stream producer/consumer edges. A valid schedule must be deterministic and race-free.

Parallel opportunities may be limited in a mostly sequential MLP/ResNet. Transformer attention projections or independent branches offer better candidates. If no safe concurrency exists for a graph region, a single stream is correct.

## Integration requirements

The released `ExecutionPlan` exposes:

- allocation ID and size for every tensor;
- initializer upload and residency steps;
- kernel input/output allocation bindings;
- copy and compute stream IDs;
- event dependencies;
- lifetime intervals and reuse decisions;
- peak planned memory.

## Released planning path

The released scheduler creates explicit allocations and weight uploads,
model-lifetime weight residency, intermediate lifetime reuse, a managed pool,
copy-stream readiness events, and dependency-aware compute-stream assignments.
These are connected plan artifacts, but they do not yet directly drive the
CuPy operations executed on the AEC H200.

## Validation evidence

- An execution plan proving each of A–E is connected.
- Allocation trace showing two non-overlapping tensors reuse one slot.
- Pool trace showing a freed block serves a later allocation and the selected fragmentation policy.
- Timeline showing at least one prefetch overlapping earlier compute.
- Timeline showing independent kernels on different compute streams with dependencies respected.
- Numerical output unchanged from the single-stream, no-reuse FP32 baseline.
