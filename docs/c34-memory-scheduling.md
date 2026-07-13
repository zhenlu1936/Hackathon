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

The released scheduler creates explicit arena ranges and weight uploads,
model-lifetime weight residency, intermediate lifetime reuse, a managed pool,
copy-stream readiness events, and dependency-aware compute-stream assignments.
Tensor byte sizes are concretized from both dynamic shape metadata and ONNX
dtype width; in particular, dynamic INT64 graph inputs reserve eight bytes per
element rather than inheriting the four-byte FP32 default.
It lowers these records into one ordered `ALLOC`/`H2D`/`EVENT_WAIT`/`KERNEL`/
`EVENT_RECORD`/`FREE`/`D2H` timeline. `PlannedGraphExecutor` consumes that
timeline using one CuPy byte arena, typed allocation views, non-blocking CuPy
streams, CuPy events, and pinned asynchronous copies.

Device arenas are cached per execution plan. The first planned upload also
becomes the executor-lifetime resident weight/constant view; later invocations
bind it directly or copy it device-to-device into a different batch plan's
arena, then record fresh readiness events without another host upload. Inputs,
intermediates, outputs, and synchronization remain per run.

Physical CuPy stream objects are cached by logical stream ID for the executor
lifetime instead of being recreated for every input batch. Plan events are
also cached and reused only after all participating streams synchronize. This
preserves the plan's logical stream/event contract while allowing CuPy's
stream-specific memory-pool arenas to reuse temporary blocks across batches.
The executor records a bounded per-batch trace of planned arena size plus CuPy
pool used/reserved bytes and reports the number of physical stream/event
objects created.

Cross-stream dataflow edges and reused physical byte ranges receive explicit
happens-before events. Weight transfers are staged before their first use so a
next-layer copy can overlap earlier compute. The runtime also records every
consumed action, allowing target validation to compare the trace with the
submitted plan.

The current working revision makes each C3.2 kernel reference an executable
timeline unit and dispatches it through `c32.kernel_registry` using the C3.4
arena bindings. Report 8 qualifies the direct path across all three released
models on H200, but most registry kernels do not physically consume their
tuning parameters. Some CuPy operations still produce temporary
arrays before copying into planned arena views. Target-side numerical,
dependency-trace, overlap, and actual-peak measurement remain required.

## Validation evidence

- An execution plan proving each of A–E is connected.
- Allocation trace showing two non-overlapping tensors reuse one slot.
- Pool trace showing a freed block serves a later allocation and the selected fragmentation policy.
- Timeline showing at least one prefetch overlapping earlier compute.
- Timeline showing independent kernels on different compute streams with dependencies respected.
- Numerical output unchanged from the single-stream, no-reuse FP32 baseline.

Local structural evidence is `511/511` C3.4 checks plus `6/6` executable-plan
regressions. The H200 run passes `56/56` combined C3.5/cross-stage/scoring
regressions and all three public models. Physical overlap and process-accounted
peak-memory evidence remain open.
