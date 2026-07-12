# Fix summary and scoring audit

Updated: 2026-07-13

This corrects the earlier fix summary. A change is marked verified only when an independent check exercises the claimed behavior. Internal totals are not official scores when a harness awards more than the rubric maximum.

## Verified revisions

### C3.2 deterministic routing and dataflow

- Removed call-order-dependent precision rotation. Repeated selection is deterministic and `FULL_FP32` always returns FP32.
- Normalized emitted GEMM names and fixed bias-free Gemm to produce the declared output.
- Added operator parameters to kernel references, including Constant metadata and GlobalAveragePool axes.
- Added the documented `ProblemSize` input to tuning and tightened shared-memory validity.
- Changed `set_hardware()` to update the shared capability in place. Direct calls through `c32.hardware` now update public API and existing default-strategy references.
- Added deterministic engineering routing: sensitive operators FP32, general Conv/small GEMM FP16, aligned 1x1 Conv/GEMM FP8, and large aligned constant-weight GEMM as W4A16 with FP32 accumulation/output.
- Added hardware-supported fallback chains and kept C3.4/deployment explicitly on `FULL_FP32`.

Evidence: C3.2 structural testing completes at `14.17/15`, including D1 `3.0/3.0`. Five policy regressions prove sensitive-op FP32, FULL_FP32 preservation, four-way mixed coverage, determinism/emitted-kernel agreement, and safe fallback. The constrained-target test uses 32 threads/1024-byte shared memory with `ProblemSize(m=17,n=19,k=23)` and returns valid tuning.

Scoring caveat: fp32/fp16/fp8/fp4 are now selected and emitted structurally, but advertising/routing them is not evidence that FP8/FP4 AEC kernels execute or meet accuracy.

### C3.3 reference-executable fusion

- `FusedEWChain` includes every external operand and stores an ordered per-op dataflow program with attributes.
- `FusedConv2dBatchNorm` preserves scale, bias, mean, and variance; the NumPy oracle evaluates Conv then inference BatchNorm.
- Programmatically added/fused nodes now register their tensor-consumer edges, preserving topological dependencies.
- Fusion passes snapshot and restore the graph after a pass or validation failure.
- Launch counts use actual C3.2 decomposition instead of fixed estimates.

Evidence: independent EW-chain and Conv+BN numerical checks pass; the C3.3 structural script passes 51 checks.

Scoring caveats:

- The IR has no initializer payloads, so Conv weights are not actually folded.
- Actual public-model launch reductions are MLP 0.0%, ResNet 10.7%, Transformer 18.2%; buffer reductions are 0.0%, 17.0%, 27.9%. None reaches the 60% full-credit target.
- The connected CPU reference qualifies optimized against original FP32 and passes complete golden runs, but no AEC numerical gate exists. The self-test total `9.40/8.6` remains invalid.

### C3.4 binding and event integrity

- Kernel steps retain declared logical inputs/outputs; validation requires all declared bindings while allowing zero-input Constant kernels.
- Graph inputs receive allocations, H2D transfers, readiness events, and waits.
- Weight transfers signal the exact event IDs their consumers wait on; duplicate unsignalled wait events were removed.
- Scalar constants and released Transformer `unk__*` batch symbols receive nonzero plan sizes.

Evidence: MLP, ResNet, and Transformer plans validate; all 12 prefetch/multi-stream configurations pass; C3.4 reports 505 checks and zero failures.

Scoring caveat: `10.85/10` is not a valid score. The pool, streams, transfers, and events remain Python metadata rather than AEC runtime operations.

### C3.5 fused reference dispatch

The NumPy reference executor recognizes all five fused operator names, and Conv+BN/EW-chain now consume the metadata emitted by C3.3.

Scoring caveat: fused dispatch is a NumPy oracle; the connected reference path below consumes C3.3/C3.4 artifacts but still does not execute on AEC hardware.

### Cross-stage reference execution

- C3.5 now runs C3.3 optimization on the shared C3.1 IR.
- C3.4 builds its plan from the optimized graph through actual C3.2 decomposition.
- The planned reference executor rejects invalid plans, batch-size mismatches, and any plan that omits or invents optimized graph nodes.
- Node execution order comes from the validated C3.4 plan.
- The first optimized batch is compared with the original unfused FP32 graph using `rtol=atol=1e-3`.
- CLI diagnostics report fusion counts, maximum qualification difference, and plan kernel/allocation counts while labelling the backend `connected C3 CPU reference (not AEC)`.
- Manifest validation now rejects duplicate, missing, unexpected, dtype-mismatched, shape-mismatched, rank-mismatched, and unequal-sample inputs.

This also exposed and fixed two graph/fusion defects: same-name consumer rerouting deleted dependency indexes, and `FusedResidualNorm` omitted LayerNorm scale/bias inputs and standard axis/epsilon attributes.

Evidence:

- Cross-stage tests pass for all three models and reject a deliberately incomplete plan.
- MLP optimized/reference difference: `0`.
- ResNet executes 48→40 nodes with difference `0`.
- Transformer executes 165→127 nodes with maximum difference `7.361174e-06`.
- One complete golden CLI run per released model passes; combined local runtime was 187.041 seconds.

Scoring caveat: this closes the disconnected software reference path, not the AEC deployment requirement. C3.4 kernel steps authorize and validate execution order/bindings, but NumPy still evaluates high-level nodes rather than executing the individual AEC kernels.

## Incorrect statements removed

- “Conv weights folded at fusion time” was false.
- C3.3 did not previously emit the `_ops` program expected by `FusedEWChain` and omitted later external inputs.
- Requiring every kernel to have an input incorrectly rejected Constant.
- Weight transfers and kernel waits previously used different event IDs.
- Preserving `_alloc_map` alone did not bind graph inputs or zero-sized tensors.
- Advertised hardware capability was presented as executed precision/kernel diversity.
- Fixed launch estimates did not require AEC hardware to replace; C3.2 decomposition provides truthful structural counts.
- Cleanup claimed every `.DS_Store` was deleted, but a local ignored file remains. Check the final archive instead.

## Files changed

```text
c32/decompositions.py
c32/api.py
c32/hardware.py
c32/kernel_spec.py
c32/strategy.py
c32/test_precision_policy.py
c33/fusion.py
c33/pipeline.py
c34/execution_plan.py
c34/lifetime.py
c34/scheduler.py
c35/deploy.py
c35/engine.py
c35/executor.py
c35/test_cross_stage.py
c3common/ir/graph.py
c3common/test_scoring_regressions.py
guidance/fix-summary.md
guidance/remaining-problems.md
```

## Test commands

```bash
.venv/bin/python -m unittest -q c31.test_c31
.venv/bin/python -m c32.test_c32
.venv/bin/python -m c33.test_c33
.venv/bin/python -m c34.test_c34
.venv/bin/python -m unittest -v c3common.test_scoring_regressions
.venv/bin/python -m unittest -q c35.test_c35
```
