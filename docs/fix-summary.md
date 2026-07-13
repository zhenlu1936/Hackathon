# Release implementation summary

This document summarizes the functionality present in the released framework.
It is descriptive rather than an official score report. Structural checks are
identified separately from execution evidence on the designated H200 AEC device.

## Current release gate

Last synchronized review: 2026-07-13.

The implementation is **not submission-ready**. The revised CuPy-only runtime
passed full-dataset golden and accuracy gates for all three released models in
a user-supplied pre-fix H200 transcript, but its CuPy pool reservation grew to
roughly 14.5 GiB for ResNet-18 and 9.7 GiB for Transformer. The source now
reuses physical streams/events across batches and removes several temporary
output allocations; this exact revision still needs an H200 run. Direct
C3.2-kernel execution remains incomplete. The current truthful fusion pipeline
clears the published 60% launch/buffer anchors locally, but target execution is
not yet qualified.
Process-accounted NVML peak memory, direct dependency verification, and
post-commit archive inspection also remain open release blockers. The new
generated C3.3 kernels clear every local released-model 60% launch/buffer
anchor, but have not compiled or passed numerical/profiler validation on H200.
Public-source and AI-assistance disclosures are current.

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
- Linear/Conv2d/LayerNorm aliases lower through the same semantic paths.
- Gemm, Softmax, multi-output LayerNormalization, and Conv lowerings expose
  connected named intermediates and required operator parameters.
- Every returned lowering is checked for unresolved inputs, duplicate
  producers, and missing node outputs; each kernel carries its selected
  precision profile.
- Tuning enforces threads-per-block, block-x, grid-x, and shared-memory limits.
- A validated CuPy device-query path records capability provenance; the static
  four-precision coverage profile remains explicitly unverified.

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
- `FusedEWChain`
- `FusedSoftmaxDropout`
- `FusedResidualNorm`

Passes apply safety guards, preserve external operands and attributes, snapshot
the graph before mutation, restore it after failure, rebuild indexes, and emit
machine-readable fusion logs. Observable intermediate graph outputs,
incompatible bias/BN shapes, Dropout with an unproven explicit training input,
and auxiliary BatchNorm/LayerNorm outputs block fusion.

The default pipeline now connects generated single-kernel Gemm/MatMul bias and
activation epilogues, direct Conv activation/residual epilogues, scaled/masked
attention scores, single-output LayerNormalization, elementwise chains,
residual normalization, Softmax+Dropout, and Transpose+Reshape. Each writes
directly into its C3.4 planned output view. Explicit Conv+BN graphs fold
parameters from the CuPy initializer store before C3.4 planning.
The obsolete sequential `FusedComputeActivation` and `FusedExecutionRegion`
matchers, runtimes, dispatch entries, and test-only ABI have been removed.

The current local same-lowering results are MLP `66.7%/75.0%`, ResNet-18
`62.7%/63.5%`, and Transformer `63.6%/61.6%` launch/logical-buffer reduction.
Problem #5 remains open only for exact-revision H200 compilation, numerical
qualification, and observed-launch evidence.
The released ResNet contains no BatchNorm nodes because BN was folded during
export and its original parameters are not recoverable.

## C3.4 execution planning

- Kernel steps retain logical and physical input/output bindings.
- Graph inputs, weights, constants, intermediates, and outputs receive planned
  allocations.
- Weight and input transfers signal the readiness events consumed by kernels.
- Lifetime analysis maps non-overlapping tensors to reusable pool slots.
- The memory pool exposes reuse and fragmentation statistics.
- Plans include copy and compute streams plus cross-stream dependencies.
- Physical arena-range reuse across streams receives explicit happens-before
  events.
- Plans include one ordered allocation/transfer/wait/kernel/record/free/readback
  timeline and retain the original C3.2 decomposition as review metadata.
- Validation rejects missing allocations, bindings, event producers, invalid
  producer order, unsafe physical overlap, and incomplete timelines.

All released plan configurations validate structurally. The CuPy executor now
consumes the plan with one byte arena, typed allocation views, non-blocking copy
and compute streams, CuPy events, pinned asynchronous H2D/D2H transfers, and an
action trace. The revision has not yet run on H200, and high-level CuPy
operators can allocate temporary outputs before copying into the arena.

## C3.5 H200 deployment

- The deployment CLI reads manifests, validates tensor names, dtypes, ranks,
  shapes, and sample counts, and supports arbitrary positive batch sizes.
- C3.3 optimization and C3.4 planning operate on the same imported graph.
- The planned executor rejects graph/plan mismatches before execution and
  consumes every action in the C3.4 timeline.
- CuPy 14.1.1 is the only numerical backend.
- Weights and batch tensors move to CuPy once, intermediate computation stays on
  the device, and final float32 output returns to the host for NPY generation.
- All 17 published ONNX operators and the seven fused operator types are
  dispatched by the CuPy engine.
- CuPy-specific Split handling uses Python integer boundaries, matching the
  native server's CuPy 14.1.1 behavior.
- Runtime evidence reports the executed timeline length and action counts.
- The CLI emits structured device and CuPy-pool evidence. The black-box runner
  prefers process accounting and uses the pool reservation as a labeled MIG
  fallback when process memory is hidden.

The revised CuPy-only black-box run confirms the full-dataset numerical gates
for all three released models:

| Model | Accuracy | Maximum absolute difference | Cold wall time | CuPy-pool proxy |
|---|---:|---:|---:|---:|
| MLP | `0.9835` | `1.53e-05` | `1.093 s` | `32.44 MiB` |
| ResNet-18 | `0.9351` | `8.58e-06` | `7.877 s` | `1542.43 MiB` |
| Transformer | n/a | `3.15e-05` | `1.743 s` | `247.35 MiB` |

The runner did not observe the child processes through `nvidia-smi`, so these
memory values are labeled CuPy-pool reservations and are not official NVML peak
memory evidence. CuPy execution on the remote H200 is the designated AEC device
path.

## Validation assets

- `c31/test_c31.py`
- `c32/test_c32.py` and `c32/test_precision_policy.py`
- `c33/test_c33.py`
- `c34/test_c34.py` and `c34/test_executable_plan.py`
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
| 6 | Reran the revised CuPy-only optimized graph on all three full released datasets | Current released-graph C3.3/C3.5 numerical gates pass; physical fused-kernel lowering remains #27 |
| 11, 29, 35 | Capped C3.3 and C3.4 self-scores to their maximums | C3.3: `8.60/8.6`, C3.4: `10.00/10.0` |
| 28 | Added `FusedComputeActivation` plus bounded, topology-driven `FusedExecutionRegion` formation | Historical only: both sequential implementations were later removed and never established F2/F3 reduction |
| 37 | Replaced scalar FP32 zero placeholder with `None` for omitted optional inputs | Type-agnostic optional input handling |
| 43–44 | Reconciled academic/public-source attribution and AI-assistance disclosure with the implementation | TVM, DNNFusion, MLIR Linalg, CUDA Graphs, GitHub Copilot, and OpenAI Codex are disclosed; keep current after later revisions |
| 46 | Ran the exact bounded-region revision's standard workflow on the remote H200 | All three released models pass; process-accounted NVML memory remains #39 |
| 47 | Replaced the unsupported CuPy random-generator fixture and reran it on H200 | All four scoring regressions pass on CuPy 14.1.1, including executable Conv+BN fusion |
| — | Removed the unavailable Python NVML binding and disclosed the server-native `nvidia-smi` call boundary | Native-server dependency compliance |
| — | Multi-level nvidia-smi fallback for MIG GPU process tracking | `--query-compute-apps`, per-GPU, and `--query-accounted-apps` probes |
| — | Surfaced stderr/stdout tails on test failure for remote diagnostics | Faster crash triage |

### Still open (see `remaining-problems.md`)

| # | Area |
|---|------|
| 1 | AEC compiler/runtime/device backend |
| 3–4 | C3.2 FP8/FP4 qualification and hardware query |
| 5 | Local generated-kernel reductions exceed 60%; H200 numerical and profiler qualification remains open |
| 7–8 | C3.4 AEC runtime integration and stream concurrency |
| 9, 36 | C3.2 kernel-step execution on H200 |
| 12 | Evaluator benchmark API |
| 13–24 | AEC execution evidence and kernel qualification |
| 25–27 | Conv+BN weight folding and fused AEC kernel lowering |
| 30–34 | C3.4 AEC device operations |
| 38 | Non-default-attribute operator tests |
| 39–40 | Process-accounted NVML memory and kernel/runtime H200 path |
| 10, 41–42 | Complete native dependency verification/disclosure, including direct `google.protobuf` use |
| 45 | Build and inspect the actual submission archive; `.gitignore` alone is not cleanliness evidence |
| Q1–Q5 | Organizer questions |

## Review record

### 2026-07-13 executable C3.3 60% reduction revision

- Added topology- and semantics-driven fusions for Gemm/MatMul epilogues,
  Conv activation/residual epilogues, rank-four scaled/masked attention scores,
  single-output LayerNormalization, and Transpose+Reshape layout copies.
  Unsupported shapes and axes remain unfused.
- Added generated CuPy/CUDA kernels for every new fused node and connected each
  to the C3.2 one-reference decomposition and C3.4 planned output allocation.
  Constant metadata references count as zero launches because both executors
  preload them and perform no Constant device launch.
- Launch/buffer evidence from `python3 -m c33.test_c33`: MLP `66.7%/75.0%`,
  ResNet-18 `62.7%/63.5%`, Transformer `63.6%/61.6%`; `PASS=68, FAIL=0`,
  written-rubric structural total `15.00/15.0` pending the numerical gate.
- Focused executable-fusion regressions pass `8/8`; C3.1 passes `7/7`, C3.2
  remains `14.17/15`, C3.2 contract/policy tests pass `12/12`, C3.4 passes `387/387`, and its
  executable-plan tests pass `6/6`. Optimized graphs also build valid C3.4
  plans. Python compilation and diff whitespace checks pass.
- Backend: local macOS structural validation with released ONNX graphs. CuPy is
  not installed locally, so the expanded generated-kernel tests, cross-stage
  numerical comparison, full model runs, cold performance, and launch
  profiling remain H200 gates.
- Integrity gate: the revision is operator-, topology-, shape-, attribute-,
  and capability-driven. It introduces no model/test names, known-weight or
  input-value branches, evaluator changes, precomputed artifacts, new
  dependency, or copied third-party implementation. Existing public design
  references and AI-assistance disclosure remain applicable.

### 2026-07-13 truthful F2/F3 launch-accounting diagnostic

- `python3 -m c33.test_c33` completes with `PASS=62, FAIL=6` and exit code
  `1`; the only failed assertions are the launch and buffer 60% targets for
  MLP, ResNet-18, and Transformer. The written-rubric structural total is
  `9.69/15.0`, with F1 `5/5` and structural F4 `4/4` pending the numerical
  gate.
- Reproduced the current released-model C3.3 structural result locally:
  MLP `9 -> 9` launches and `5 -> 5` logical graph buffers, ResNet-18
  `75 -> 67` launches and `47 -> 39` buffers, and Transformer `253 -> 224`
  launches and `136 -> 98` buffers. These counts agree with the reported
  `0.0%/0.0%`, `10.7%/17.0%`, and `11.5%/27.9%` reductions.
- Confirmed that the default pipeline does not enable
  `FusedExecutionRegion` or `FusedComputeActivation`: both reference
  implementations execute retained operations sequentially and therefore
  cannot truthfully count as one physical launch.
- The next general, executable opportunities are GEMM bias/activation
  epilogues, direct Conv/Conv+activation kernels, and standalone single-kernel
  LayerNormalization/Softmax lowerings. View and constant operations may count
  as zero launches only where the consumed runtime path proves that behavior.
- Evidence is local structural inspection with the released ONNX graphs, not
  CuPy 14.1.1/H200 profiling or official-evaluator evidence. Integrity gate:
  no plagiarism, model-identity hardcoding, evaluator bypass, precomputed
  result, hidden-case targeting, or new dependency was introduced; no source
  implementation changed in this diagnostic.

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
  H200 numerical validation was still #46 at this revision point; the later
  three-model black-box report resolves it for the released models.
- Public influences were disclosed in `docs/c33-fusion.md` and the root
  `README.md`: TVM, DNNFusion, MLIR Linalg, and the NVIDIA CUDA Graphs
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
  of emitted kernel steps (#1/#36) or FP8/FP4 qualification (#3).
  Full-dataset C3.5 validation (#46) was resolved by the later black-box report.
- Integrity gate: the change corrects validation logic only; it adds no runtime
  dependency, model-specific execution branch, evaluator bypass, or generated
  result artifact.

### 2026-07-13 three-model H200 black-box report review

- Examined a captured CuPy 14.1.1 report from an NVIDIA H200
  NVL MIG 1g.18gb. All three full released datasets passed output-contract and
  golden `allclose` gates; MLP accuracy was `0.9835`, ResNet-18 accuracy was
  `0.9351`, and Transformer required no classification threshold.
- Recorded cold wall times of `1.093 s`, `7.877 s`, and `1.743 s` for MLP,
  ResNet-18, and Transformer. These are valid per-run measurements but runtime
  points remain unknowable until submissions are ranked.
- The report's process sampler completed but did not observe a GPU process.
  Its `32.44 MiB`, `1542.43 MiB`, and `247.35 MiB` memory figures came from
  CuPy-pool reservations, not process-accounted NVML peaks; memory ranking
  evidence remains open.
- Confirmed the top-level ONNX files were byte-identical to the organizer copies,
  removed the redundant `models/` directory, and redirected documentation and
  tests to `.specification/testcases/release_to_competitors/models/`.
- Post-relocation local validation passed: Python compilation, C3.1 plus
  precision-policy unit tests (`12/12`), C3.2 structural scoring (`14.17/15`;
  H200 numerical gate skipped locally), C3.3 (`64/64`), and C3.4 (`505/505`).
- Integrity gate: the report shows no evidence of filename/hash branching,
  evaluator bypass, or precomputed outputs. This review does not close the
  existing direct-kernel, dependency-disclosure, or submission-archive blockers.

### 2026-07-13 root README and submission-contract consolidation

- Consolidated `docs/README.md` and `docs/SUBMISSION.md` into the root
  `README.md`, then removed the redundant files and updated the archive
  required-entry check.
- Added the exact evaluator-substitution templates for C3.1 and C3.5, including
  the specification's fixed registration value `--batch-size 256`, plus the
  required three-public-model golden and label checks.
- Preserved dependency, academic-source, originality, LLM-assistance, and
  archive-policy disclosures in the root document. Corrected archive language
  so cleanliness is not claimed before a post-commit artifact is built and
  inspected.
- Documented the unresolved conflict between the general third-party packaging
  rule and the C3 native-environment contract; direct remote
  `google.protobuf` verification and organizer confirmation remain open.
- Documentation-only validation: both command templates match
  `.specification/spec.md`; live links and build requirements no longer target
  the removed documents; local Markdown targets, archive required-entry names,
  and archive script syntax were checked. C3.1 `--help` passed. C3.5 parser
  arguments match by source inspection; its local `--help` cannot import the
  intentionally required CuPy dependency, so executable CLI validation remains
  H200-only.
- Integrity gate: no runtime behavior, evaluator component, dependency, or
  generated result changed. Required attribution and AI-assistance disclosure
  remain present in the submission root.

### 2026-07-13 historical problem 5 H200 evidence audit (superseded)

- Rechecked problem #5 against the written C3.3 F2/F3 formulas, the current
  `GraphPassPipeline` counters, the CuPy executor, and the available H200 report.
- The released graphs meet the 60% threshold in C3.2-lowering launch counts and
  logical intermediate-buffer counts. Running `python3 -m c33.test_c33` on the
  H200 can reproduce those structural results, but the metrics themselves are
  graph/lowering properties rather than device measurements.
- At that revision, the `FusedExecutionRegion` reference lowering iterated through its
  retained operator program and dispatches each CuPy operation separately.
  Therefore neither the H200 correctness report nor the structural C3.3 test
  proves a reduction in physical CUDA/AEC launches or allocations.
- H200 qualification for #5 should pair the structural test with
  `c35.test_cross_stage`, `c3common.test_scoring_regressions`, and
  `./run_c35.sh`. A physical-launch claim additionally requires an unfused
  versus fused execution switch and a target-side CUDA/AEC profiler or the
  organizer benchmark; no such profiler is declared in `environments.txt`.
- At that revision the audit treated #5 as resolved for the structural metric.
  The later truthful executable-lowering revision supersedes that conclusion
  and reopens #5. Physical lowering remains open as #27. No runtime or
  dependency change was made. The integrity gate found no new plagiarism,
  hardcoding, evaluator bypass, precomputation, hidden-case targeting, or
  disclosure issue.

### 2026-07-13 supplied H200 problem 5 run

- The supplied CuPy 14.1.1 / NVIDIA H200 NVL MIG 1g.18gb transcript reports
  C3.3 `PASS=51, FAIL=0`, not the current workspace's bounded-region
  `PASS=64, FAIL=0`. Its real-model structural reductions were MLP
  `44.4%/40.0%`, ResNet-18 `38.7%/36.2%`, and Transformer `18.2%/27.9%` for
  launch/logical-buffer counts, all below the 60% target.
- The output shape (`6 -> 4`, `48 -> 31`, `165 -> 127`) and 51-check total
  identify a revision-parity failure: the H200 checkout predates or does not
  execute that revision's `FusedExecutionRegion` pass (`6 -> 1`, `48 -> 12`,
  `165 -> 65` locally). Problem #5 is reopened until the exact current revision
  is synchronized and rerun on the target.
- Both cross-stage tests passed. Five of six combined tests passed; the
  Conv+BN numerical regression errored before numerical comparison because
  CuPy 14.1.1's `Generator` lacks `.normal`. This is tracked as #47 and is not
  a Conv+BN numerical failure.
- The full C3.5 run passed all three released models: MLP accuracy `0.9835`,
  ResNet-18 accuracy `0.9351`, and maximum absolute differences
  `1.53e-05`, `8.58e-06`, and `3.15e-05`. This confirms the deployed optimized
  high-level path remains numerically correct, but does not satisfy problem
  #5's missing structural thresholds or physical-launch evidence.
- Integrity gate: the supplied run exposes validation and revision-parity
  defects only; it provides no evidence of plagiarism, hardcoding, evaluator
  bypass, precomputation, hidden-case targeting, or a new undisclosed dependency.

### 2026-07-13 historical bounded-region H200 run (later superseded)

- The superseding CuPy 14.1.1 / NVIDIA H200 NVL MIG 1g.18gb transcript runs the
  bounded-region revision and reports C3.3 `PASS=64, FAIL=0`. Structural
  launch/logical-buffer reductions are MLP `88.9%/100.0%`, ResNet-18
  `84.0%/76.6%`, and Transformer `74.3%/73.5%`. That revision treated problem
  #5 as closed for the structural metric; the later executable-lowering audit
  invalidates the sequential-region launch count and reopens #5. Physical
  device-launch evidence remains #27.
- Both cross-stage tests passed. After replacing the unsupported
  `Generator.normal` fixture with deterministic `cupy.linspace` ramps, the
  H200/CuPy 14.1.1 rerun passes all four scoring regressions in `1.162 s`,
  including executable Conv+BN fusion. #47 is closed. The emitted
  CuPy/NumPy padding deprecation warning is non-fatal and does not alter the
  numerical result.
- After executable permission was restored, `./run_c35.sh` passed all three
  released models on the exact bounded-region revision: MLP accuracy `0.9835`,
  max diff `1.53e-05`, wall `0.993 s`; ResNet-18 accuracy `0.9351`, max diff
  `8.58e-06`, wall `7.699 s`; Transformer max diff `3.15e-05`, wall `1.570 s`.
  The report is `/tmp/c35-problem5.json`; #46 is closed. CuPy-pool memory
  remains proxy evidence, so process-accounted NVML memory stays open as #39.
- No runner or runner-instruction change is needed; the temporary documentation
  edits made in response to the permission error were reverted.
- Local validation is limited to Python compilation, shell syntax, and source
  review because CuPy/CUDA is unavailable here. No dependency was added and no
  runtime/evaluator path changed.
- Integrity gate: deterministic synthetic test tensors are general regression
  data, not model/testcase targeting. No plagiarism, evaluator bypass,
  precomputed output, hidden-case behavior, or new disclosure issue was found.

### 2026-07-13 problem 27 physical-lowering audit

- At that revision, `FusedExecutionRegion` admitted nearly every single-output operator,
  but its CuPy lowering loops over the retained program and calls each original
  operator separately. `PlannedGraphExecutor` similarly deduplicates C3.4 steps
  by node ID and then executes the high-level graph node. The current physical
  launch path therefore does not consume the advertised fused-kernel plan.
- The C3.3 launch counter uses C3.2 decomposition. An opaque fused-region node
  has no dedicated decomposer and is counted as one fallback kernel even when
  its reference execution performs multiple CuPy operations. Structural F2/F3
  results must remain distinct from physical H200 evidence.
- The implementation route is to partition only code-generation-compatible
  regions, generate CuPy/CUDA kernels from semantic expression/shape metadata at
  evaluation time, and lower unsupported regions to explicit kernel sequences
  with their real launch counts. Elementwise chains should come first, followed
  by GEMM/Conv bias-activation epilogues, residual LayerNorm, and row Softmax.
- Closure requires the C3.4 kernel steps and allocation bindings to drive C3.5,
  an unfused-versus-fused execution switch, numerical qualification at
  `rtol=atol=1e-3`, and target-side profiler evidence that reported and observed
  launches agree. CUDA source may be generated and JIT-compiled at evaluation
  time; no precompiled artifact may be submitted.
- Integrity gate: this design is operator-, shape-, and capability-driven. It
  introduces no model-name, filename, weight, sample, or hidden-case branch and
  requires no evaluator modification. Any new public implementation reference
  or dependency must be disclosed before use.

### 2026-07-13 active-problem register cleanup

- Removed confirmed solved items 2, 5, 6, 28, 29, 35, 37, 43, 44, 46, and 47
  from `remaining-problems.md`. Their evidence remains in this changelog and
  review history rather than appearing as active work.
- Corrected the still-open summary from `1–2` to `1`; the direct kernel/runtime
  integration limitation remains open independently of the completed FP32
  graph-path correctness gate.
- No implementation, runtime, evaluator, dependency, or disclosure changed.
  Integrity review found no plagiarism, hardcoding, precomputation, hidden-case
  targeting, evaluator bypass, or new third-party boundary.

### 2026-07-13 C3.5 opset-17 hidden-shape revision

- Corrected Flatten to always produce the two-dimensional ONNX result for
  axes 0 through rank, including negative axes.
- Corrected Conv dilation window extraction and added `VALID`, `SAME_UPPER`,
  and `SAME_LOWER` auto-padding, asymmetric padding, grouped-channel checks,
  and attribute validation. The policy is driven only by tensor shapes and
  ONNX attributes.
- Added opset-17 Split size-input handling and validation, optional Mean and
  InvStdDev outputs for LayerNormalization, dtype-preserving Gather and layout
  operations, Reshape/Transpose/axis guards, and numeric Constant scalar/list
  decoding with ONNX-specified dtypes.
- Added 16 non-default-attribute regression methods covering the released
  17-operator union. They pass on a test-only host array shim; this is semantic
  evidence only and leaves problem #38 open until the same tests and all three
  released models pass on CuPy 14.1.1/H200.
- Local validation: Python compilation and `git diff --check` pass; C3.1 plus
  precision/contract tests pass 19/19; C3.2 remains 14.17/15 with the numerical
  gate correctly skipped without CUDA; C3.3 earns 8.60/8.60 with 62 checks
  passing and six honest released-model reduction-target failures; C3.4 passes
  387/387 and all 12 plan configurations validate; the dedicated
  executable-timeline contract suite passes 5/5.
- The same workspace revision now contains a source-connected C3.4 timeline
  executor with one device arena, planned allocation views, pinned transfers,
  CuPy streams/events, and an execution trace. Local source/plan checks cover
  it, but no CuPy installation or CUDA device is available here; H200 numerical,
  ordering, overlap, and actual-memory validation remain #30/#31/#34/#40.
- Updated the C3.5 deployment guide and active-problem register to distinguish
  this high-level-node timeline from unresolved direct C3.2 kernel execution.
- The fused elementwise JIT uses SHA-256 only to derive a deterministic CUDA
  symbol name from the semantic operator program. It does not inspect or branch
  on model, filename, input, weight, evaluator, or test identifiers; `hashlib`
  is recorded as a standard-library boundary in the root disclosure.
- Integrity gate: the revision adds no model/test identifiers, filename or
  input/weight hashes, precomputed outputs, evaluator hooks, network access,
  third-party dependency, or hidden-case branch. Existing ONNX attribution and
  OpenAI Codex disclosure cover the semantic implementation and assistance.

### 2026-07-13 C3.2 contract hardening

- Added explicit CuPy/CUDA device discovery with validated resource limits,
  compute-capability/source metadata, and conservative Hopper FP4 handling.
  The target H200 query has not been run, so problem #4 remains open for target
  evidence rather than implementation plumbing.
- Added connected-decomposition validation and attached the selected
  `PrecisionProfile` to every kernel reference. Invalid sensitive-op precision,
  unresolved intermediate inputs, duplicate producers, and missing outputs now
  fail before planning.
- Added `Linear`, `Conv2d`, and `LayerNorm` aliases; preserved omitted Transpose
  permutation semantics; completed Split output metadata and three-output
  LayerNorm lowering; and restricted Winograd selection to eligible group-one,
  unit-stride, unit-dilation 3x3 Conv.
- Reworked tuning to size shared memory by the resident tile and enforce every
  declared block/grid/shared-memory limit, including very large problem sizes.
- Local evidence: seven new dependency-light C3.2 contract regressions pass, a
  synthetic C3.4 plan validates with no issues, Python compilation passes, and
  `git diff --check` passes. ONNX-based C3.2 scoring and H200 numerical checks
  cannot run in the local environment; the last verified structural score
  remains `14.17/15`.
- The root development-validation commands now include
  `python3 -m unittest -v c32.test_contract`.
- Final post-synchronization validation: C3.2 contract tests pass `7/7`; C3.3
  earns `8.60/8.60` with 62 passing checks and six open released-model
  reduction-target checks; the synthetic C3.4 plan has complete
  tuning/precision fields and no validation issues; repository Python
  compilation and `git diff --check` pass. The final C3.2 consistency search
  across both release records and the implementation guide also passes.
- A transient MatMul+bias matcher regression was superseded by the
  executable-fusion revision below. The current C3.3 functional score is
  `8.60/8.60`; six released-model 60% reduction targets remain visibly open.
- The D3 score is not inflated with redundant copy kernels just to create an
  intermediate for every single-kernel operator. Such copies would worsen the
  executable schedule and conflict with truthful launch-count reporting.
- Integrity gate: the changes are operator-, attribute-, shape-, and hardware-
  driven. They add no test/model identifiers, hashes, evaluator hooks,
  precomputed artifacts, network access, or third-party dependency. Existing
  OpenAI Codex disclosure applies.

### 2026-07-13 C3.3 executable-fusion revision

- Tightened all required matchers so they preserve observable intermediate
  graph outputs, reject incompatible MatMul bias and Conv/BN channel shapes,
  reject auxiliary BN/LayerNorm outputs, and handle opset-17 Dropout's explicit
  training input conservatively. Elementwise chains now take bounded five-node
  regions instead of rejecting a longer legal chain outright.
- Added runtime-generated single-launch implementations for elementwise chains,
  inference Softmax+Dropout, and residual Add+LayerNorm. CUDA source is generated
  and JIT-compiled from submitted code at runtime; no binary or cache is part of
  the submission.
- Added executor-side FP32 Conv+BN parameter folding from the live CuPy
  initializer store before C3.4 plan creation. The integrated planned executor
  also performs the fold with CuPy and feeds folded tensors through its device
  copy path; no host-array numerical fold remains. The unfurled reference
  operator remains available for numerical comparison.
- Added explicit C3.2 lowerings for the five fused operators. MatMul+bias remains
  a truthful two-launch sequence until a real GEMM epilogue is implemented.
- Removed sequential compute-activation and bounded execution-region utilities
  from the default pipeline. This reopens problem #5 and supersedes their prior
  60%-plus structural reduction figures; those labels did not represent one
  physical kernel.
- Final local evidence after correcting the residual-guard placement: Python
  compilation and `git diff --check` pass, the dependency-light C3.2 contract
  suite passes `7/7`, and C3.3 earns `8.60/8.60` with `62` passing checks and
  six explicit failures for the unreached 60% released-model reduction target.
  The new
  `c33.test_fused_kernels` tests and Conv+BN materialization require CuPy
  14.1.1/H200 and were not executed locally.
- Integrity gate: no model/test identifier, filename, input/weight hash,
  evaluator hook, precomputed artifact, network access, or new dependency was
  introduced. Runtime JIT uses the already-disclosed native CuPy/CUDA boundary;
  existing OpenAI Codex assistance disclosure applies.

### 2026-07-13 C3.4 executable-timeline revision

- Added concrete byte offsets/capacities to allocations and a single ordered
  timeline covering allocation, H2D, event waits/records, kernels, logical
  frees, and D2H.
- Connected the free-list plan to one real CuPy device arena. Tensor bindings
  are typed views over planned ranges; weights/constants stay on the host until
  their planned pinned asynchronous uploads execute.
- Cached arenas per plan and retained immutable weight/constant views for the
  executor lifetime, so model data uploads once even when a later partial batch
  uses a different plan. Each inference still performs input transfer,
  compute, synchronization, and output readback. Batched deployment snapshots
  arena-backed outputs before the next reuse.
- Added real non-blocking copy/compute streams and CuPy events. Cross-stream
  producer/consumer edges, output readback, and aliased-arena reuse are ordered
  by recorded events. Future weights are staged before their first-use kernel
  so copy submission is interleaved with preceding compute.
- Kept the plan honest: the executable kernel unit is one optimized graph node,
  while its C3.2 decomposition is retained as metadata until direct decomposed
  kernel execution is implemented under #1/#36.
- Added trace-to-plan and one-time model-upload assertions to the H200
  cross-stage tests and report action/residency counts from the deployment path.
- Local validation passes Python compilation, `git diff --check`, `19/19`
  dependency-light C3.1/C3.2 unit tests, `5/5` executable-plan tests, and
  `387/387` C3.4 checks across all 12 public configurations. The broader C3.3
  run still has the six already-tracked released-model reduction failures under
  #5/#27. CuPy/CUDA is not installed locally, so C3.4 numerical, ordering,
  overlap, and actual-memory proof remains open under #7/#8/#30–#34/#40 until
  the exact revision runs on H200.
- Integrity gate: scheduling decisions depend only on graph topology, tensor
  metadata, batch size, and hardware-neutral plan policy. No model/test name,
  filename, input/weight hash, evaluator hook, precomputed output, network
  access, or new third-party dependency was introduced.

### 2026-07-13 report 3 dynamic-INT64 allocation fix

- Report 3 exposed one general C3.4 sizing defect in both the two-sample
  FULL_FP32 gate and batch-256 C3.5: dynamic shapes used an implicit four-byte
  element width, so Transformer `input_ids` (INT64) received half the required
  capacity (`192` versus `288` bytes and `18432` versus `36864` bytes).
- Corrected lifetime sizing to derive element width from the tensor's ONNX
  dtype for concrete and dynamic shapes. The same path now resolves documented
  symbolic batch names and integer `-1`, rejects other negative dimensions,
  and preserves dtype-aware scalar-constant sizing.
- Added an executable-plan regression requiring a batch-2 `[N, 18]` INT64
  input to request `288` bytes and receive at least that physical capacity.
- Local validation: the changed files compile with a writable bytecode cache;
  dependency-light checks pass for both symbolic `N` and `-1` at `288` bytes.
  The model-backed unittest cannot run on this Mac because native `onnx` is not
  installed, and CuPy/H200 validation remains open under problem #40.
- Corrected the interpretation of the real-model C3.3 section: its local
  `0.60/0.60` diagnostic subtotal does not award the official F2/F3 points.
  The six 60% failures expose real optimization deficits; the current figures
  are MLP `0.0%/0.0%`, ResNet-18 `10.7%/17.0%`, and Transformer
  `11.5%/27.9%` for launch/logical-buffer reduction.
- Integrity gate: the fix is driven only by tensor dtype, shape, and requested
  batch size. It introduces no filename/model/test-ID branch, input or weight
  hash, evaluator modification, precomputed artifact, network access, or new
  dependency. Existing ONNX and OpenAI Codex disclosures remain sufficient.

### 2026-07-13 C3.3 written-rubric scoring alignment

- Audited the repository self-test against `.specification/spec.md` and
  `.specification/scoring.md`. F2 and F3 are each worth 3 points using
  `min(reduction * 5, 3)`, with 60% required for full credit; they are not a
  0.6-point bonus.
- Kept every per-model 60% assertion unchanged. The self-test now computes F2
  and F3 from MLP and ResNet-18, the two models named by the published C3.2/C3.3
  benchmark, and retains Transformer as additional non-scoring validation.
  Because the referenced organizer benchmark is absent and the documents do
  not define cross-model aggregation, the self-test prints each model's exact
  formula result and uses their conservative mean only for its local summary.
- Replaced the misleading 8.6-point summary with a 15-point written-rubric
  structural summary: F1 5, F2 3, F3 3, and F4 structural 4. The output states
  explicitly that F4 becomes zero if the required FP32 numerical gate fails;
  auxiliary pipeline/counting checks remain diagnostics rather than points.
- This alignment does not claim the current implementation reaches 60%. It
  makes the shortfall visible in the score while problem #5/#27 remains open.
- Local validation is dependency-limited because `onnx` and CuPy are absent.
  The revised C3.3 self-test passes 56 dependency-light diagnostics and reports
  `5.00/15` because it skips model-backed F2/F3/F4 scoring; Python compilation,
  seven C3.2 contract tests, and diff checks also pass. The exact model score
  must be rerun on the H200 server.
- Integrity gate: no evaluator or organizer-owned component was changed.
  `c33/test_c33.py` is a repository self-test; its revised score is derived
  only from graph statistics and the published formulas. No hardcoding,
  precomputation, hidden-case targeting, new dependency, or bypass was added.

### 2026-07-13 persistent-stream memory-regression fix

- A user-supplied pre-fix CuPy 14.1.1/H200 MIG transcript passed all numerical
  and accuracy gates but exposed stable pool reservations of roughly
  `60.64–64.54 MiB` for MLP, `14423.62–14535.62 MiB` for ResNet-18, and
  `9707.09–9716.04 MiB` for Transformer. These are labeled CuPy-pool values,
  not process-accounted NVML peaks.
- Traced the accumulation to `PlannedGraphExecutor` constructing fresh
  non-blocking streams and events on every batch. With 10,000 samples and
  batch size 256, 40 physical stream sets prevented CuPy's stream-specific
  free-block arenas from reaching a steady reusable state.
- Cached one physical CuPy stream per logical plan stream for the executor
  lifetime and cached plan events for reuse after every participating stream
  synchronizes. Different full/partial-batch plans share the same physical
  logical streams.
- Added bounded per-batch telemetry for planned arena bytes and CuPy pool
  used/reserved bytes. Backend evidence now reports batch count, physical
  stream/event object counts, maximum planned arena bytes, and first/last pool
  reservation values.
- Removed the generic executor's unconditional contiguous-result materialization
  before its planned-arena copy. The executable fused Softmax+Dropout and
  Residual+LayerNorm kernels now accept validated runtime-only planned outputs
  and write directly into their arena views, avoiding a temporary and D2D copy.
- Replaced the list of per-batch output allocations plus final concatenate with
  one preallocated full output populated in sample order after each batch.
- Added H200 tests for persistent resource identity/telemetry and direct fused
  output writes. They compile locally but require CuPy/CUDA to execute.
- Local evidence: Python compilation and `git diff --check` pass; C3.1 passes
  `7/7`, the C3.2 contract passes `7/7`, C3.2 structural scoring remains
  `14.17/15`, C3.4 passes `387/387`, and executable-plan tests pass `6/6`.
  C3.3 retains its six explicit released-model 60% reduction failures. The
  CuPy-dependent fused-kernel, cross-stage, and scoring regressions cannot run
  on this Mac, so post-fix correctness, wall time, and pool behavior remain
  open under #7/#8/#30–#34/#40 until `./run_c35.sh` is rerun on H200.
- Integrity gate: the revision is driven only by plan identity, logical stream
  IDs, tensor metadata, and batch boundaries. It introduces no model/test name,
  filename, input/weight hash, evaluator modification, precomputed artifact,
  hidden-case branch, network access, or new dependency. Existing CuPy and
  OpenAI Codex disclosures remain sufficient.

### 2026-07-13 framework cleanup

- Removed the unused sequential `FusedComputeActivation` and
  `FusedExecutionRegion` stacks across graph matching, runtime execution,
  dispatch registration, and test-only region ABI. The active generated-kernel
  fusion pipeline and all five required F1 patterns are unchanged.
- Replaced the duplicate 434-line `c31.ir.graph` implementation with a
  compatibility re-export of the shared `c3common` IR. Consolidated DAG JSON
  generation in `c31.export_dag`; the evaluator-required root command is now a
  thin wrapper over that implementation.
- Removed unused imports, the dead elementwise dispatch copy, and the misspelled
  internal `TUMABLE_OPS` name. Added the already-implemented Sub, Exp, and Sqrt
  operators to the live executor dispatch rather than retaining a dead private
  table.
- Preserved all seven tracked validation reports under `docs/` as durable
  release evidence. Added non-destructive `export-ignore` rules so repository
  evidence remains available while competition archives omit generated reports.
  Updated `run_all.sh` to fail fast and include the focused contract, fusion,
  plan, cross-stage, and scoring suites.
- Fixed the archive scanner's leading-hyphen regex handling and validated the
  exact staged tree: 65 archive entries, all 11 required entries present, and
  zero forbidden artifacts.
- Local evidence: C3.1 passes `7/7`; C3.2 policy/contract passes `12/12` and
  structural scoring remains `14.17/15`; C3.3 passes `61/61`, reports
  `15.00/15.0` structurally, and its focused executable-fusion suite passes
  `8/8`; C3.4 passes `387/387` and its executable-plan suite passes `6/6`.
  CuPy-dependent C3.3/C3.5/scoring suites remain H200-only and were not run on
  this machine.
- Integrity gate: cleanup decisions were based on import/call reachability,
  active pipeline registration, and packaging policy. No model/test identifier,
  input/weight hash, evaluator hook, precomputed output, network access, or new
  dependency was introduced. Existing ONNX, CuPy, public-reference, and OpenAI
  Codex disclosures remain sufficient.

### 2026-07-13 cleanup evidence-preservation correction

- Restored every tracked report under `docs/c35_reports/` and
  `docs/c3_reports/`; none remains staged for deletion.
- Replaced destructive report cleanup with repository-preserving packaging:
  `.gitattributes` excludes the report directories only from `git archive`,
  while the reports remain versioned and readable in the source repository.
- Added the durable rule to `.agents/skills/c3-track/SKILL.md`: general cleanup
  must preserve documentation and validation evidence by default, and any
  packaging exclusion must be non-destructive and explicitly auditable.
- Integrity gate: this correction restores evidence and changes packaging
  metadata only. It adds no runtime behavior, evaluator bypass, dependency,
  precomputed inference output, model-specific branch, or network access.

[remaining-problems](remaining-problems.md) · [root submission disclosure](../README.md#submission-disclosure)
