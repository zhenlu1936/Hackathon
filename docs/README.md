# C3 framework documentation

This directory contains the release documentation for the C3 operator
scheduling and model deployment framework.

## Component documentation

| Component | Documentation | Released implementation |
|---|---|---|
| C3.1 | [Graph parsing and representation](c31-graph-parsing.md) | ONNX import, graph validation, deterministic DAG JSON |
| C3.2 | [Decomposition and kernel selection](c32-decomposition.md) | Precision profiles, hardware capabilities, kernel sequences, tuning |
| C3.3 | [Fusion and graph optimization](c33-fusion.md) | Five guarded fusion patterns and transactional pass pipeline |
| C3.4 | [Memory planning and scheduling](c34-memory-scheduling.md) | Lifetimes, pool reuse, bindings, transfers, streams, and events |
| C3.5 | [Model deployment](c35-deployment.md) | Batched CuPy execution on the H200 AEC device |
| C3.5 runner | [Black-box validation runner](c35-standard-runner.md) | Golden checks, accuracy, cold timing, and GPU-memory evidence |

## Release and maintenance documents

- [Release implementation summary](fix-summary.md) describes the major
  behavior currently present in source.
- [Remaining problems](remaining-problems.md) separates verified functionality
  from AEC, evaluator, performance, and submission gaps.
- [Validation checklist](validation-checklist.md) defines the checks required
  before a release or competition submission.

## Governing contracts

The organizer documents under `.specification/` remain authoritative:

1. `.specification/general_requirements.md` — integrity, originality,
   dependency, offline, and submission rules.
2. `.specification/environments.txt` — native evaluation-server environment.
3. `.specification/spec.md` — C3 interfaces and functional requirements.
4. `.specification/scoring.md` — scoring details.

Written requirements take precedence over this documentation. Known conflicts
are recorded in [remaining problems](remaining-problems.md).

## Architecture

```text
ONNX
  -> Graph / Node / Tensor IR                       C3.1
  -> PrecisionProfile / KernelSpecRef               C3.2
  -> validated optimized Graph                      C3.3
  -> ExecutionPlan                                  C3.4
  -> ordered float32 logits and output manifest     C3.5
```

All stages share the same graph representation. The DAG JSON is an export view;
it is not used as a substitute for internal tensor metadata, attributes,
producer/consumer maps, or initializer state.

## Release boundary

The framework includes a connected CuPy/CUDA path that executes on the remote
H200 AEC device. It provides graph correctness, batching, output generation,
timing, and device-memory evidence for the released models. C3.4 plans remain
reviewable metadata rather than the objects driving allocations and launches,
and C3.2 kernel references remain structural until the H200 execution path
consumes those decomposed kernels directly.

The framework introduces no automatic dependency installation. Only standard
library modules and third-party packages natively available on the evaluation
server may be used.
