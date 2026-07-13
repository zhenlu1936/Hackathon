# C3 remaining problems

Updated: 2026-07-13.

This is the canonical list of unresolved C3 work. Solved and superseded items
belong in `fix-summary.md`, not here. Structural self-tests, H200 execution, and
official evaluator evidence are kept separate.

## Current evidence boundary

- Current repeated H200 evidence: `docs/c35_reports/c35_mean_report.json`; all
  three independent standard runs pass golden output and applicable accuracy
  gates through direct C3.2 dispatch.
- Current H200 environment and targeted-fix evidence: `docs/c3_reports/old_5.txt`.
- Prior ResNet attribution evidence: `docs/c35_reports/old_7.json`.
- Current local C3.3 structural result: `12.00/15.0`, with direct Conv lowering
  producing ResNet reductions of `-22.2%` launches and `-22.5%` buffers; the
  two 60% anchors intentionally remain failing.
- Current three-run mean direct-path C3.5 result: ResNet external wall
  `8.005168 s`, accuracy
  `0.9351`, and `max_abs_diff=1.53e-05`.
- Current memory result: ResNet's CuPy pool reserves `3,358,545,408` bytes
  versus a `1,418,311,232`-byte planned peak. This is proxy evidence, not
  process-accounted NVML measurement.
- Current registry evidence: `10/10` contract/registry regressions, the targeted
  ResNet test, and `56/56` C3.5/cross-stage/scoring regressions pass on H200;
  unsupported names fail closed.

## Priority register

| ID | Priority | Area | Unresolved problem | Completion evidence |
|---:|:---:|---|---|---|
| 1 | P0 | Architecture | Individual C3.2 steps dispatch through submitted CuPy source and C3.4 bindings and pass all three models on H200, but most tuning parameters do not yet control physical launches | Every emitted step resolves to submitted source/AEC output, consumes its tuning decision, and executes through C3.4 bindings on H200 for all three models |
| 3 | P1 | Precision | FP8 and FP4/W4A16 choices are structural only and have no executable H200 numerical qualification | Implement selected kernels and pass per-profile H200 numerical gates |
| 5 | P1 | Fusion | Current direct-lowering accounting is `-22.2%` launches and `-22.5%` buffers, below the written 60% F2/F3 target | A source-connected fused contraction meets the target, beats the BLAS fallback, passes FP32 gates, and has physical launch evidence |
| 7 | P1 | Memory integration | High-level operators, especially im2col Conv, allocate temporaries outside the C3.4 arena; actual allocation/plan agreement is unproven | Allocation trace explains or removes external temporaries and reconciles live/peak memory with the plan |
| 8 | P1 | Concurrency | Multi-stream scheduling and events exist, but target ordering and useful copy/compute overlap have not been traced | H200 trace demonstrates dependency safety and intended overlap |
| 39 | P1 | Peak memory | MIG process accounting did not observe the child process; reported memory is a CuPy-pool proxy | Capture organizer-accepted process/device peak memory including startup and child processes |
| 12 | P2 | Evaluator contract | The referenced C3.2/C3.3 benchmark implementation and exact schemas are absent from the release | Obtain organizer benchmark/API definitions and run them unchanged |

## Issue details

### 1 — Direct decomposed-kernel execution

The current working revision expands C3.3 semantic nodes into individual C3.2
steps, binds their intermediates through the C3.4 arena, and dispatches every
released-model kernel name through `c32.kernel_registry`. Report 8 proves all
three optimized released plans pass their H200 gates, and unsupported names
fail closed. Most registry functions still ignore `tuning_params`, and the
unfused Winograd path lacks equivalent target qualification. This single issue
replaces historical aliases 9, 13–18, 25–27, and 36.

Required proof:

1. every emitted kernel name resolves to submitted source;
2. tuning parameters drive compilation and launch;
3. C3.4 physical slots, transfers, events, and streams bind those launches;
4. unknown/unavailable kernels fail closed;
5. all three released models pass the FP32 golden gate through that path.

### 3 — Precision

`FULL_FP32` is qualified on the current connected H200 path. Mixed-precision
coverage is not. FP8 and FP4/W4A16 choices are structural only and have no
executable H200 numerical qualification.

Do not enable low precision in deployment until the exact selected kernels,
target capability source, and numerical error are recorded.

### 5 — ResNet fusion reductions

The current direct-step BLAS path reports `8.275 s` and passes the ResNet
golden/accuracy gates. It truthfully expands each semantic
fused Conv node into im2col, contraction, reshape/bias, and epilogue stages.
The current direct-step accounting reports `-22.2%` launch and `-22.5%` buffer
reductions, so the structural result remains below the written 60% target.

A future tiled or implicit-GEMM fused contraction may close this issue only if
it is general, source-connected, faster in cold H200 testing, numerically
qualified, and reflected accurately in physical launch counts.

### 7 and 39 — Runtime memory

Report 8 records for ResNet batch 256:

- planned peak: `1,418,311,232` bytes;
- CuPy pool used after inference: `2,352,974,848` bytes (`1.66x` planned);
- CuPy pool reserved after inference: `3,358,545,408` bytes (`2.37x` planned).

The likely source is im2col/BLAS workspace outside the arena, but this remains
an inference until allocation tracing identifies owners and lifetimes. CuPy
pool reservation is not equivalent to organizer process peak memory. Reduce or
explicitly plan reusable workspaces, then obtain an accepted target peak.

### 8 — Stream and event qualification

The C3.4 self-test validates plan structure and the CuPy executor reuses bounded
stream/event objects. It does not prove useful overlap or actual target order.
Capture a target trace that maps plan actions to physical operations and shows
that all reuse and producer/consumer dependencies are respected.

### 12 — Missing organizer benchmark

The specification names `benchmarks/c32_c33/bench_c32_c33.py`, but the released
assets do not include it. Local score scripts are diagnostics only. Do not
freeze guessed module schemas or claim official C3.2/C3.3 scores without the
organizer implementation.

## Organizer questions

1. Provide the referenced C3.2/C3.3 benchmark and exact public API schemas.
2. Confirm the H200 MIG capability-query interface and limits used by scoring.
3. Confirm how Conv+BatchNorm fusion is scored for the already-folded released
   ResNet, whose original BN parameters are not recoverable.
4. Confirm whether C3.5 peak memory is 10 points as written or 15 points as
   shown in the conflicting image.
5. Confirm the accepted peak-memory measurement for MIG when process queries do
   not expose the child process.

## Delivery rule

Do not close an issue from comments, local structure, or a superseded report.
Close it only with the organizer-facing artifact/backend evidence named in the
completion column. Every revision must update this file and `fix-summary.md`
together and rerun the six-rule integrity gate.

The `MemorySampler` compatibility API was restored during the final cleanup
review. It is a development instrumentation helper and does not change any
active issue or organizer-facing runtime claim above.

The pre-cleanup alltest wrapper stopped after the known C3.3 structural-score
diagnostic because it used `set -e`; this was an orchestration defect, not a
new model-correctness failure. The runner now executes every stage, labels
rubric diagnostics separately from required tests, and emits a final summary.
The first complete run additionally found two stale C3.3 test expectations for
the corrected Conv layout/order and planned-output contiguity; both tests were
aligned with the actual C3.2/C3.4 contracts without changing runtime behavior.
This is validation-harness maintenance and does not close any active issue.
The corrected final `docs/c3_reports/alltest-report.txt` reports zero required
failures and exactly one diagnostic shortfall, the already-open C3.3 ResNet
60% reduction anchors tracked by issue 5. All 52 C3.5/cross-stage tests, 4
scoring regressions, and the three-model black-box gate pass on H200.
The final integrity/package audit found no new active issue: local and remote
78-file content manifests match exactly, the staged diff is whitespace-clean,
and the 71-member staged-tree archive contains all 11 required entries with
zero forbidden paths. Historical reports and the agent skill remain retained
in the repository but are excluded from the competition archive.

The 2026-07-13 post-commit synchronization audit removed one ignored,
reproducible root-level C3.5 report from the remote worktree and confirmed
matching 78-file SHA-256 manifests. This cleanup does not change or close any
active architecture, precision, fusion, memory, concurrency, peak-memory, or
evaluator-contract issue.
