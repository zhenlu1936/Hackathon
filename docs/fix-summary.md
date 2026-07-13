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
Direct dependency verification and post-commit archive inspection also remain
open release blockers. Public-source and AI-assistance disclosures are current.

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

The released C3.2 checks report `14.17/15` structurally. FULL_FP32 numerical
correctness is a hard D1 gate, not a sixteenth point. On the remote H200 after
the bounded-region revision, the connected C3.1→C3.5 CuPy comparison passed
all three two-sample qualification batches with `top1_match=1.0`: MLP
`max_abs_diff=2.86e-06`, ResNet-18 `1.67e-06`, and Transformer `8.88e-06`.
This does not prove direct execution of individual C3.2 kernel steps or qualify
FP8/FP4 H200 correctness.

## C3.3 fusion

The pass pipeline implements:

- `FusedMatMulBias`
- `FusedConv2dBatchNorm`
- `FusedComputeActivation` (Gemm/MatMul/Conv + Relu/Erf)
- `FusedEWChain`
- `FusedSoftmaxDropout`
- `FusedResidualNorm`
- `FusedExecutionRegion` (bounded, topology-driven region program)

Passes apply safety guards, preserve external operands and attributes, snapshot
the graph before mutation, restore it after failure, rebuild indexes, and emit
machine-readable fusion logs. The array executor implements numerical
semantics for each fused node.

Released-model structural reductions after bounded execution-region formation:

| Model | Launch reduction | Buffer reduction |
|---|---:|---:|
| MLP | 88.9% | 100.0% |
| ResNet-18 | 84.0% | 76.6% |
| Transformer | 74.3% | 73.5% |

These exceed the published graph-level 60% thresholds. The region reference
executor still issues the contained CuPy operations separately, so the table
does not claim equivalent physical H200 kernel-launch reductions.

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
- All 17 published ONNX operators and the seven fused operator types are
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
| 2 | Corrected and ran the FULL_FP32 golden-output gate on H200 | All three released models passed `allclose` and `top1_match=1.0`; direct decomposed-kernel execution remains #1/#36 |
| 11, 29, 35 | Capped C3.3 and C3.4 self-scores to their maximums | C3.3: `8.60/8.6`, C3.4: `10.00/10.0` |
| 28 | Added `FusedComputeActivation` plus bounded, topology-driven `FusedExecutionRegion` formation | Released-graph structural launch/logical-buffer reductions now exceed 60% for all three models; physical fused H200 lowering remains open in #27 |
| 37 | Replaced scalar FP32 zero placeholder with `None` for omitted optional inputs | Type-agnostic optional input handling |
| 43–44 | Reconciled academic/public-source attribution and AI-assistance disclosure with the implementation | TVM, DNNFusion, MLIR Linalg, CUDA Graphs, GitHub Copilot, and OpenAI Codex are disclosed; keep current after later revisions |
| — | Removed the unavailable Python NVML binding and disclosed the server-native `nvidia-smi` call boundary | Native-server dependency compliance |
| — | Multi-level nvidia-smi fallback for MIG GPU process tracking | `--query-compute-apps`, per-GPU, and `--query-accounted-apps` probes |
| — | Surfaced stderr/stdout tails on test failure for remote diagnostics | Faster crash triage |

### Still open (see `remaining-problems.md`)

| # | Area |
|---|------|
| 1–2 | AEC compiler/runtime/device backend |
| 3–4 | C3.2 FP8/FP4 qualification and hardware query |
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

### 2026-07-13 C3.3 bounded execution-region revision and cleanup audit

- Added a deterministic region pass over supported single-output operations.
  Regions contain at most six nodes, cross only single-consumer internal edges,
  stop at graph outputs, exclude multi-output `Split`, and expose all external
  operands explicitly. The pass does not inspect model/test names, hashes,
  weights, or fixed input values.
- Added a CuPy reference executor for the retained ordered region program. This
  preserves a qualification path but currently executes internal operations as
  a sequence; it is not evidence of one physical H200 kernel per region.
- Local released-graph structural evidence: MLP 88.9% launch / 100.0% logical
  buffer reduction; ResNet-18 84.0% / 76.6%; Transformer 74.3% / 73.5%. All
  three optimized graphs validate.
- Corrected the C3.2 self-test scale back to the specified 15 points. The
  FULL_FP32 numerical check is now a hard D1 gate and tests golden top-1
  agreement rather than task-label accuracy.
- Removed the obsolete local virtual environment, Python bytecode caches, and
  generated submission archive. The archive builder remains source-controlled,
  while its generated `submission.tar.gz` is ignored.
- Validation: C3.1 `7/7`, C3.2 structural `14.17/15.0` with H200 numerical test
  skipped locally, C3.3 `64/64`, and C3.4 `505/505`. The C3.5 cross-stage and
  shared scoring suites could not import CuPy on this local machine; remote
  H200 numerical validation remains #46.
- Public influences were disclosed in `docs/c33-fusion.md` and
  `docs/SUBMISSION.md`: TVM, DNNFusion, MLIR Linalg, and the NVIDIA CUDA Graphs
  guide. No source code or prose was copied.
- Integrity gate: no plagiarism, testcase/model hardcoding, evaluator bypass,
  precomputed artifact, hidden-case targeting, or new dependency was
  introduced. Public references and OpenAI Codex assistance are disclosed.

### 2026-07-13 H200 FULL_FP32 gate correction

- Removed an invalid self-test heuristic that inferred precision from helper
  kernel names. FULL_FP32 precision is defined by the selected
  `PrecisionProfile`; decomposition completeness is checked independently.
- Remote H200 evidence supplied after the bounded-region revision: all 219
  nodes selected FP32, all 219 decomposed, and MLP, ResNet-18, and Transformer
  passed `cupy.allclose(rtol=1e-3, atol=1e-3)` with `top1_match=1.0` on the
  two-sample qualification batches. Maximum absolute differences were
  `2.86e-06`, `1.67e-06`, and `8.88e-06`, respectively.
- This closes the C3.2 FULL_FP32 graph-path gate (#2), but not direct execution
  of emitted kernel steps (#1/#36), full-dataset C3.5 validation (#46), or
  FP8/FP4 qualification (#3).
- Integrity gate: the change corrects validation logic only; it adds no runtime
  dependency, model-specific execution branch, evaluator bypass, or generated
  result artifact.

[remaining-problems](remaining-problems.md) · [SUBMISSION](SUBMISSION.md)
