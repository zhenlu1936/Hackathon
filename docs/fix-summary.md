# C3 implementation and validation summary

Updated: 2026-07-13.

This document records the current implementation, the latest applicable
evidence, and completed fixes. It is not an official score report. Historical
run transcripts and JSON reports remain under `docs/c3_reports/` and
`docs/c35_reports/`; superseded results are not presented as current evidence.

## Release verdict

The current direct-C3.2-kernel revision passes all three released models on the
AEC H200 through CuPy 14.1.1. The current repeated-run evidence is
`docs/c35_reports/c35_mean_report.json`, which records arithmetic means from
three independent standard black-box runs:

| Model | Golden result | Accuracy | External wall | CuPy-pool proxy |
|---|---:|---:|---:|---:|
| MLP | `max_abs_diff=1.91e-05` | `0.9835` | `0.800 s` | `33.64 MiB` |
| ResNet-18 | `max_abs_diff=1.53e-05` | `0.9351` | `8.005 s` | `3202.96 MiB` |
| Transformer | `max_abs_diff=3.40e-05` | n/a | `2.006 s` | `63.87 MiB` |

All three satisfy `allclose(rtol=1e-3, atol=1e-3)` and the applicable accuracy
thresholds. These timings are measured by the standard runner around a child
process and include process startup and exit. The memory values are explicitly
CuPy-pool proxies because the MIG process was not observed by `nvidia-smi`.

The mean report also records the three source values for every averaged
timing, correctness, plan, stream/event, and CuPy-pool metric. Historical
report 8 records a verified H200 capability snapshot sourced from
`cupy.cuda.runtime.getDeviceProperties(device=0)` and excludes unsupported FP4.
Tuning-driven physical launches, physical launch/overlap traces,
process-accounted peak memory, and the missing organizer C3.2/C3.3 benchmark
contract remain open. See `remaining-problems.md` for the canonical list.

## Current implementation

### C3.1 graph import and DAG export

- Imports the released ONNX graphs into a shared validated IR.
- Excludes initializers from runtime inputs and rebuilds producer/consumer and
  topological indexes after transformations.
- Handles symbolic dimensions, optional empty inputs, constants, fan-out,
  multiple outputs, and duplicate or empty node names.
- Exports deterministic DAG JSON through the required CLI.

Current evidence: the seven C3.1 tests pass in `docs/c3_reports/old_4.txt`.

### C3.2 precision, decomposition, and tuning

- Keeps an explicit `FULL_FP32` path and forces numerically sensitive operators
  to FP32.
- Returns non-empty, connected kernel sequences with deterministic intermediate
  names and bounded tuning parameters.
- Exposes structural FP32, FP16, BF16, FP8, and FP4 capability coverage without
  claiming that unexecuted low-precision kernels are H200-qualified.
- Lowers current fused ResNet Conv epilogues truthfully as im2col, BLAS-backed
  contraction, optional bias, and one generated epilogue.

Current evidence from `docs/c3_reports/old_4.txt`:

- structural self-score `14.17/15.0`;
- all 219 nodes select and decompose in `FULL_FP32`;
- two-sample connected-path comparisons pass for all three released models,
  with maximum differences from `3.81e-06` to `8.70e-06`.

The working revision now exposes each released-model C3.2 step through a
submitted CuPy registry and a C3.4 allocation binding. Report 8 proves this path
passes all three released models on H200. Most kernels still do not consume
their tuning parameters physically, so issue 1 remains open in narrowed form.

### C3.3 graph fusion

- Implements guarded MatMul+bias, Conv+BatchNorm, elementwise-chain,
  Softmax+Dropout, and residual+normalization patterns.
- Rebuilds graph indexes and validates transactional rewrites.
- Uses generated single-launch kernels only where the execution path actually
  has one; the former sequential execution-region claims were removed.
- Keeps `FusedConvActivation` and `FusedConvResidualActivation` as semantic
  graph nodes while executing their contraction through the faster BLAS-backed
  Conv path.

Retained baseline structural evidence from `docs/c3_reports/old_4.txt`:

| Model | Launch reduction | Buffer reduction |
|---|---:|---:|
| MLP | `66.7%` | `75.0%` |
| ResNet-18 | `-5.3%` | `-5.4%` |
| Transformer | `63.6%` | `61.6%` |

The working revision remains `12.00/15.0` and `PASS=59, FAIL=2`, but direct
lowering changes ResNet to `-22.2%` launches and `-22.5%` buffers. The two
visible failures remain intentional rather than hidden behind false one-launch
metadata. Report 8 qualifies the current revision numerically. Physical
CUDA/AEC launch counts remain unmeasured.

### C3.4 memory planning and scheduling

- Builds allocation, transfer, kernel, event, free, and readback actions in one
  reviewable plan.
- Implements lifetime-based slot reuse, a coalescing/size-class pool,
  weight-readiness events, persistent logical streams, and dependency-aware
  multi-stream assignment.
- Executes planned arena views, streams, and events through CuPy and keeps model
  tensors resident across batch plans.

Current local structural evidence is `511/511` C3.4 checks. On H200, all 56
C3.5/cross-stage/scoring regressions pass and report 8 consumes 99 planned
ResNet kernel steps through the direct timeline. Physical dependency/overlap
tracing remains open, and the direct path's memory footprint is materially
larger than the earlier high-level-node baseline.

### C3.5 deployment

- Accepts the required ONNX, input, output, and optional batch-size arguments.
- Binds inputs by manifest/model tensor name, preserves order across partial
  batches, and emits complete C-contiguous float32 logits plus a manifest.
- Uses CuPy exclusively for numerical work on the designated H200 path.
- Implements the published 17-operator union with opset-17 attribute handling.
- Runs C3.1 import, C3.3 optimization, C3.4 planning, and planned execution in
  one connected source path.

The previous one-thread-per-output fused Conv kernel was numerically correct but
slow. It was replaced by the general im2col plus BLAS-backed `cupy.dot` Conv and
one Relu or residual+Relu epilogue. Current H200 evidence shows:

- current direct-kernel standard workflow: ResNet `8.275 s`, `0.9351` accuracy,
  and `1.53e-05` maximum difference (`docs/c35_reports/old_8.json`);
- development profile: `6.7645 s` inference and `7.562 s` internal total,
  versus `61.4541 s` inference for the superseded direct kernel
  (`docs/c35_reports/old_7.json`);
- performance recovery: `9.09x` at the inference-stage boundary and `8.23x`
  for the profiler's internal total.

Report 7 records 880 planned-node calls, exactly 22 nodes across 40 batches.
Those are planned high-level node calls, not physical CUDA/AEC launches.

## Direct-kernel integration revision

The current working revision replaces one-high-level-node dispatch with one
planned step per C3.2 reference, registers intermediate tensor shapes for arena
planning, and fails closed when a kernel name is absent from
`c32.kernel_registry`. All emitted names for the optimized released MLP,
ResNet-18, and Transformer plans resolve locally.

The supplied H200 transcript initially reached `47/48` C3.5 tests. Its remaining
ResNet batch-64 failure came from interpreting a contraction ordered as
`(N,H,W,O)` directly as `(N,O,H,W)`. The reshape now recovers NHWO and transposes
to ONNX NCHW. Bias-free Conv lowering also keeps the contraction internal so the
reshape is the sole producer of the public output.

Current evidence:

- two kernel-layout regressions pass, including an im2col → contraction →
  reshape → bias comparison against a direct NCHW reference with rectangular
  output, asymmetric padding, and stride 2;
- C3.1 passes `7/7`, the C3.2 contract plus registry suite passes `10/10`, and C3.4
  passes `511/511`;
- all optimized released-model plan names resolve after unsupported identity
  fallbacks were removed, unsupported names fail closed, and `git diff --check`
  passes;
- C3.2 remains `14.17/15.0` structurally and skips its numerical section because
  CuPy/CUDA is absent on the local macOS host;
- C3.3 remains `12.00/15.0` with its two intentional ResNet anchors failing;
  direct lowering now reports `-22.2%` launches and `-22.5%` buffers;
- H200 registry/contract tests pass `10/10`, the formerly failing ResNet
  batch-64 test passes, and the combined C3.5/cross-stage/scoring suite passes
  `56/56` in `122.459 s`;
- report 8 passes the three-model black-box workflow, and
  `docs/c35_reports/old_8.txt` records `16/16` opset-17 attribute tests passing.

This closes the direct path's H200 correctness gap. Issue 1 remains open only
for tuning-driven physical launches and related launch evidence.

## Memory evidence and interpretation

For ResNet batch 256, current report 8 records:

| Quantity | Bytes | Approximate size |
|---|---:|---:|
| C3.4 planned peak | `1,418,311,232` | `1352.61 MiB` |
| CuPy pool used after inference | `2,352,974,848` | `2243.97 MiB` |
| CuPy pool reserved after inference | `3,358,545,408` | `3202.96 MiB` |

Live used memory is `1.66x` the planned peak and retained pool reservation is
`2.37x`. This is consistent with im2col/BLAS temporaries outside the arena, but
an allocation trace is still required before attributing each byte. The pool
reservation is not an NVML-equivalent process peak.

## Development profiler cleanup

- Added scoped stage, planned-node, and CuPy-pool instrumentation in
  `c35/instrument.py` and `c35/profiler.py`.
- Removed global monkey-patch state so every profiler invocation restores its
  own executor method even when another profiler object exists.
- Restored the public `MemorySampler` compatibility API after the cleanup audit
  identified its removal as an unnecessary breaking change. Its documentation
  now states the actual contract: callers take explicit `snap()` samples while
  `start()`/`stop()` preserve the fluent interface; no background sampling is
  claimed.
- Removed the advertised but unimplemented `--qualify` and `--no-memory`
  switches. Correctness qualification belongs to `./run_c35.sh`; the profiler
  remains an attribution tool and does not claim evaluator-grade correctness.
- Preserved H200 profile report 7 as repository evidence.

## Resolved and superseded items

The following historical items are no longer open:

- FULL_FP32 and three-model C3.5 correctness gates now pass on the H200.
- Issue 4 is closed by report 8's verified H200 capability block: compute
  capability `[9,0]`, FP32/FP16/BF16/FP8 enabled, and FP4 excluded.
- Issue 10 is closed by the native-server probe in `docs/c3_reports/old_5.txt`,
  which records protobuf `7.35.1` together with its disclosed license, purpose,
  and call boundary in the root README.
- Issue 38 is closed by `docs/c35_reports/old_8.txt`: all 16 non-default opset-17
  attribute tests pass on CuPy 14.1.1/H200.
- The dynamic-INT64 final-batch allocation failure is fixed.
- Persistent stream/event objects no longer grow once per batch.
- The unsupported Python NVML binding was removed; the native `nvidia-smi`
  boundary is disclosed.
- Conv+BatchNorm parameter folding is implemented when explicit parameters are
  present; the released ResNet's already-folded weights remain a specification
  limitation rather than an implementation failure.
- Sequential `FusedExecutionRegion` and `FusedComputeActivation` paths and their
  misleading launch-reduction claims were removed.
- The slow direct fused Conv path is removed, and its H200 performance
  regression is resolved by the BLAS fallback.
- Historical issue 48 is closed by the current three-model standard run and
  report 8. Its memory follow-up is tracked only under active issues 7 and 39.
- Packaging issue 45 is closed: the exact staged cleanup tree produced a
  68-member archive with all 11 required entries and zero forbidden paths;
  report and agent directories were excluded by `export-ignore`.
- Historical duplicate aliases 9, 13–18, 25–27, and 36 are consolidated into
  active architecture issue 1; aliases 30–34 and 40 are consolidated into
  active runtime issues 7 and 8; aliases 41–42 were closed with issue 10.

Historical reports remain preserved for traceability. In particular,
`docs/c3_reports/old_3.txt` and `docs/c35_reports/old_5.json`/`old_6.json` describe
superseded revisions and must not be cited as current results.

## Current H200 validation

Validation on the NVIDIA H200 NVL MIG 1g.18gb with CuPy 14.1.1 produced:

- `10/10` C3.2 contract/registry tests passing;
- the targeted ResNet batch-64 regression passing in `9.182 s`;
- `56/56` C3.5, cross-stage, and scoring regressions passing in `122.459 s`;
- `16/16` opset-17 attribute tests passing in `0.442 s`;
- three-model black-box correctness and accuracy passing in report 8;
- verified device capability evidence and native versions ONNX `1.22.0` and
  protobuf `7.35.1`.

On 2026-07-13, three additional standard C3.5 runs all passed. Their arithmetic
means are MLP `0.799740 s`, ResNet `8.005168 s`, and Transformer `2.006208 s`
external wall time; accuracy and maximum-difference values were identical
across all three runs. The pre-cleanup alltest transcript also exposed that the
then-current `set -e` wrapper stopped at the documented C3.3 structural
diagnostic shortfall instead of running the remaining required stages. The
runner was changed to continue all stages and report required failures
separately from score diagnostics; the post-cleanup transcript supersedes that
incomplete baseline.

The first post-cleanup alltest then exposed two stale C3.3 regression-test
assumptions introduced by the Conv NHWO-to-NCHW correction: the structural
test still expected bias addition before the required reshape, and the fused
kernel test used `empty_like` on a non-contiguous transpose view even though
the C3.4 arena contract supplies contiguous planned outputs. The tests now
assert the executable `im2col -> matmul -> conv_reshape -> add_bias ->
epilogue` order and allocate the same contiguous output contract as C3.4. No
runtime fallback or correctness threshold was relaxed.

The corrected final transcript is `docs/c3_reports/alltest-report.txt`. It
completed every stage with zero required failures and one documented
diagnostic shortfall: the unchanged C3.3 ResNet 60% launch/buffer anchors. The
H200 run passed 15 C3.2 contract/registry tests, 19 executable C3.3 tests, 511
C3.4 checks, 6 executable-plan tests, 52 C3.5/cross-stage tests, 4 scoring
regressions, and the final three-model black-box gate. The final black-box wall
times were MLP `0.822808 s`, ResNet `8.067361 s`, and Transformer `1.975348 s`;
all correctness and applicable accuracy gates passed.

This is H200/CuPy public-model evidence, not the unreleased official C3.2/C3.3
evaluator or process-accounted NVML peak-memory evidence.

## Prior cleanup-baseline validation

Validation completed locally on macOS:

- 34 dependency-light unit tests pass;
- C3.2 structural diagnostics remain `14.17/15.0`; the H200 numerical section
  correctly skips because CuPy/CUDA is unavailable locally;
- C3.3 reports the expected `PASS=59, FAIL=2` and exits nonzero only for the two
  visible ResNet 60% structural anchors;
- C3.4 passes `387/387` checks;
- all Python files compile, report 6/7 JSON parses, shell scripts pass syntax
  checking, and `git diff --check` passes.
- AST/API validation confirms `MemorySampler` retains `start`, `snap`, `stop`,
  and `summary` without introducing a dependency or runtime thread.
- `scripts/build_submission.sh` passes against an exact synthetic commit made
  from the complete staged tree: 11 required entries, zero failures, and no
  forbidden path or generated report in the archive.

H200 results are taken only from preserved reports 4 and 7 and were not rerun
on macOS.

## Integrity and disclosure gate

The current diff and execution paths were checked against all six competition
rules:

1. no copied team implementation or undisclosed source was introduced;
2. runtime decisions do not depend on test IDs, filenames, hashes, weights, or
   fixed public answers;
3. no evaluator or organizer-owned component is modified or bypassed;
4. reports are retained only as documentation evidence and excluded from
   submission archives, never consumed by runtime;
5. no hidden-case identifier selects an optimization;
6. CuPy, ONNX/protobuf, public design references, GitHub Copilot, and OpenAI
   Codex assistance remain disclosed in the root README.

No new dependency was introduced. Native-server protobuf `7.35.1` is recorded
and disclosed. The NumPy-backed Conv regression is test-only shape/numerical
validation; runtime continues to import and execute only CuPy for numerical
work. Issue 1 remains open for tuning-driven physical launches, while issues 4,
10, and 38 are closed by retained H200 artifacts.

Final cleanup verification found no cache, bytecode, generated root report,
empty file, broken symlink, forbidden binary, or packaged report/agent path.
All 44 Python sources compile from source, all four shell scripts pass syntax
checking, and the complete staged diff passes `git diff --cached --check`.
Local and remote SHA-256 manifests each contain 78 framework files with zero
content differences. A submission archive built from the exact staged tree
contains 71 members, all 11 required entries, and zero forbidden paths. The
only runtime SHA-256 use names a generated elementwise kernel from its general
semantic operation sequence; it does not inspect a model, input, test ID,
filename, weight, or expected output.

## Post-commit synchronization audit

The local and remote framework trees were compared again on 2026-07-13 using
SHA-256 content manifests rather than their different Git baselines. The only
drift was an ignored, reproducible root-level `c35-standard-report.json` on the
remote server; it was not tracked or consumed by runtime and was removed. The
resulting manifests each contain 78 framework files with zero content
differences. No code, dependency, evaluator component, correctness claim, or
active issue changed. The six-rule integrity and disclosure gate remains clear.

[Remaining problems](remaining-problems.md) ·
[Validation checklist](validation-checklist.md) ·
[Submission disclosure](../README.md#submission-disclosure)
