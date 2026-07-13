# Release implementation summary

This document summarizes the functionality present in the released framework.
It is descriptive rather than an official score report. Structural checks are
identified separately from execution evidence on the designated H200 AEC device.

## Current release gate

Last synchronized review: 2026-07-13.

The implementation is **not submission-ready**. Structural validation was
passing before the CuPy-only conversion, and earlier H200 runs passed MLP plus
ResNet, but all three models now require a fresh server run against the revised
runtime. Direct C3.2-kernel and C3.4-operation execution remains incomplete.
Dependency, AI-assistance, and archive disclosures also remain open release
blockers.

## Shared graph foundation

- ONNX models are imported into a shared `Graph`, `Node`, and `Tensor` IR.
- Initializers are excluded from runtime graph inputs.
- Symbolic shapes, optional empty inputs, constants, fan-out, multiple outputs,
  and duplicate or empty node names are represented.
- Producer/consumer indexes and topological order are rebuilt and validated
  after graph transformations.
- C3.1 JSON is generated as a deterministic export view of the richer IR.

## C3.2 precision and decomposition

- Precision selection is deterministic for a fixed graph, execution mode, and
  hardware profile.
- `FULL_FP32` provides the correctness baseline.
- Softmax, normalization, BatchNorm, and reduction operators remain FP32.
- The mixed policy exposes FP16, FP8, and W4A16 choices under shape and hardware
  guards, with deterministic fallback to safer supported types.
- All released operators lower to non-empty kernel sequences.
- Gemm, Softmax, LayerNormalization, and Conv lowerings expose named
  intermediates and required operator parameters.
- Tuning returns populated block, grid, and shared-memory fields subject to the
  configured hardware limits.

The released C3.2 checks report `14.17/15` structurally. This is not proof that
FP8/FP4 kernels execute on AEC or meet end-to-end numerical thresholds.

## C3.3 fusion

The pass pipeline implements:

- `FusedMatMulBias`
- `FusedConv2dBatchNorm`
- `FusedComputeActivation` (Gemm/MatMul/Conv + Relu/Erf)
- `FusedEWChain`
- `FusedSoftmaxDropout`
- `FusedResidualNorm`

Passes apply safety guards, preserve external operands and attributes, snapshot
the graph before mutation, restore it after failure, rebuild indexes, and emit
machine-readable fusion logs. The array executor implements numerical
semantics for each fused node.

Released-model launch reductions after the 2026-07-13 ComputeActivation
addition (still below the rubric's 60% full-credit target):

| Model | Launch reduction | Buffer reduction |
|---|---:|---:|
| MLP | 44.4% | 40.0% |
| ResNet-18 | 38.7% | 36.2% |
| Transformer | 18.2% | 27.9% |

The released
ResNet contains no BatchNorm nodes because BN was folded during export.

## C3.4 execution planning

- Kernel steps retain logical and physical input/output bindings.
- Graph inputs, weights, constants, intermediates, and outputs receive planned
  allocations.
- Weight and input transfers signal the readiness events consumed by kernels.
- Lifetime analysis maps non-overlapping tensors to reusable pool slots.
- The memory pool exposes reuse and fragmentation statistics.
- Plans include copy and compute streams plus cross-stream dependencies.
- Validation rejects missing allocations, bindings, events, and invalid producer
  order.

All released plan configurations validate structurally. Allocations, transfers,
streams, and events remain Python plan objects rather than the primitives
driving the CuPy execution path on the AEC H200.

## C3.5 H200 deployment

- The deployment CLI reads manifests, validates tensor names, dtypes, ranks,
  shapes, and sample counts, and supports arbitrary positive batch sizes.
- C3.3 optimization and C3.4 planning operate on the same imported graph.
- The planned executor rejects graph/plan mismatches before execution.
- CuPy 14.1.1 is the only numerical backend.
- Weights and batch tensors move to CuPy once, intermediate computation stays on
  the device, and final float32 output returns to the host for NPY generation.
- All 17 published ONNX operators and the six fused operator types are
  dispatched by the CuPy engine.
- CuPy-specific Split handling uses Python integer boundaries, matching the
  native server's CuPy 14.1.1 behavior.
- The CLI emits structured device and CuPy-pool evidence. The black-box runner
  prefers process accounting and uses the pool reservation as a labeled MIG
  fallback when process memory is hidden.

H200 MIG validation has confirmed the numerical gates for MLP and ResNet:

| Model | Accuracy | Maximum absolute difference | Cold wall time | CuPy-pool proxy |
|---|---:|---:|---:|---:|
| MLP | `0.9835` | `1.53e-05` | `0.768 s` | `3.96 MiB` |
| ResNet-18 | `0.9351` | `8.58e-06` | `7.597 s` | `1492.25 MiB` |

Transformer requires one final server rerun after the CuPy Split portability
fix. CuPy execution on the remote H200 is the designated AEC device path.

## Validation assets

- `c31/test_c31.py`
- `c32/test_c32.py` and `c32/test_precision_policy.py`
- `c33/test_c33.py`
- `c34/test_c34.py`
- `c35/test_c35.py` and `c35/test_cross_stage.py`
- `c3common/test_scoring_regressions.py`
- `run_c35.sh` for the release-facing C3.5 black-box workflow

See [validation checklist](validation-checklist.md) for the complete release
gate and [remaining problems](remaining-problems.md) for unclosed items.

## Changelog (2026-07-13)

### Resolved

| # | Change | Impact |
|---|--------|--------|
| 11, 29, 35 | Capped C3.3 and C3.4 self-scores to their maximums | C3.3: `8.60/8.6`, C3.4: `10.00/10.0` |
| 28 | Added `FusedComputeActivation` (Gemm/MatMul/Conv + Relu/Erf) | MLP: 0%→44%, ResNet: 11%→39% launch reduction; issue 5 remains open below 60% |
| 37 | Replaced scalar FP32 zero placeholder with `None` for omitted optional inputs | Type-agnostic optional input handling |
| — | Added the current academic-attribution draft to `docs/SUBMISSION.md` | Documentation progress only; issue 43 remains open pending final provenance review |
| — | Removed the unavailable Python NVML binding and disclosed the server-native `nvidia-smi` call boundary | Native-server dependency compliance |
| — | Multi-level nvidia-smi fallback for MIG GPU process tracking | `--query-compute-apps`, per-GPU, and `--query-accounted-apps` probes |
| — | Surfaced stderr/stdout tails on test failure for remote diagnostics | Faster crash triage |

### Still open (see `remaining-problems.md`)

| # | Area |
|---|------|
| 1–2 | AEC compiler/runtime/device backend |
| 3–4 | C3.2 FP8/FP4 qualification and hardware query |
| 5 | C3.3 launch and buffer reductions remain below the 60% full-credit target |
| 6 | BN weight folding and fused AEC kernels |
| 7–8 | C3.4 AEC runtime integration and stream concurrency |
| 9, 36 | C3.2 kernel-step execution on H200 |
| 12 | Evaluator benchmark API |
| 13–24 | AEC execution evidence and kernel qualification |
| 25–27 | Conv+BN weight folding and fused AEC kernel lowering |
| 30–34 | C3.4 AEC device operations |
| 38 | Non-default-attribute operator tests |
| 39–40 | Cold timing, NVML memory, and kernel/runtime H200 path |
| 10, 41–42 | Complete native dependency verification/disclosure, including direct `google.protobuf` use |
| 43 | Final academic and source-provenance review |
| 44 | Complete AI-assistance disclosure, including OpenAI Codex |
| 45 | Build and inspect the actual submission archive; `.gitignore` alone is not cleanliness evidence |
| 46 | Run all three models through the revised CuPy-only CLI and runner on the H200 |
| Q1–Q5 | Organizer questions |

## Review record

### 2026-07-13 consistency and integrity audit

- Reproduced the local structural evidence: C3.1 `7/7`, C3.2 `14.17/15`,
  C3.3 `51/51`, C3.4 `505/505`, scoring regressions `4/4`, and cross-stage
  tests `2/2`.
- Reopened submission items because direct `google.protobuf` use is not yet
  completely disclosed or independently verified on the remote server.
- Reopened the AI-assistance item because the existing disclosure names GitHub
  Copilot but not OpenAI Codex.
- Reopened archive cleanliness because ignored workspace content is not proof
  of the contents of a built submission archive.
- Integrity gate: no definite plagiarism, hardcoding, evaluator-bypass,
  precomputation, or hidden-case targeting violation was found in this audit;
  the undisclosed-dependency/assistance rule remains unresolved and blocks
  release.

### 2026-07-13 native-dependency cleanup

- Removed the unavailable optional Python NVML binding and its sampler code.
- The black-box runner now uses only the server-native `nvidia-smi` command for
  process-memory sampling, with labeled CuPy-pool evidence when MIG hides PID
  accounting.
- Focused runner-evidence regressions pass; remote H200 behavior still requires
  the next server run. Integrity gate: no new third-party dependency, hardcoding,
  evaluator bypass, precomputed artifact, or hidden-case behavior was introduced.

### 2026-07-13 CuPy-only conversion

- Removed the legacy CPU-array imports and CPU numerical backend from the executor,
  deployment CLI, black-box runner, regression fixtures, and release docs.
- Removed the non-specification `--backend`, `--labels`, and
  `--qualify-optimizations` options from the C3.5 CLI. The public interface now
  contains exactly `--onnx`, `--input`, `--output`, and optional `--batch-size`;
  accuracy remains the black-box runner's responsibility.
- Source-level parser inspection confirms both C3.1 entrypoints expose exactly
  `--onnx` and `--output`, while C3.5 exposes exactly the four arguments above.
  The runner and `run_c35.sh` command templates match the written examples.
- ONNX tensor decoding remains an external host-storage boundary through the
  ONNX API and is immediately converted to CuPy before framework execution.
- Local compilation plus unaffected structural checks pass: C3.1 `7/7`, C3.2
  `14.17/15`, C3.3 `51/51`, and C3.4 `505/505`. Converted C3.5 and shared
  numerical regressions require the remote CuPy/H200 server. A full three-model
  server rerun is required before closing correctness or performance items.
- Integrity gate: the conversion is general, introduces no case-specific logic,
  evaluator changes, generated answers, or new dependency. CuPy remains a
  disclosed server-native dependency.

[remaining-problems](remaining-problems.md) · [SUBMISSION](SUBMISSION.md)
