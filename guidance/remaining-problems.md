# Remaining problems and completion gates

Updated: 2026-07-13

This register follows `.specification/general_requirements.md` first and `.specification/scoring.md` second. A local pass closes a scoring item only when it exercises the same artifact the evaluator scores.

## Current priority list

| Priority | Area | Remaining problem | Scoring consequence |
|---|---|---|---|
| P0 | Architecture | No executable AEC compiler/runtime/device backend | C3.4 device behavior and C3.5 deployment/time/memory are unsatisfied |
| P0 | Correctness | No FULL_FP32 execution of C3.2 kernels against golden outputs | C3.2 hard numerical condition is unproven |
| P1 | C3.2 precision | Mixed routing now covers four precisions structurally, but FP8/FP4 AEC execution is not numerically qualified | D1 routing signals pass locally; required backend correctness remains unproven |
| P1 | C3.2 hardware | Default target is a hardcoded H100, not an AEC query | Capability and D5 claims may be false |
| P1 | C3.3 reduction | Public launch/buffer reductions are below 60% | F2/F3 cannot receive full credit |
| P1 | C3.3 correctness | Public reference qualification passes, but BN values cannot be folded and fused AEC kernels do not exist | F4 is unproven on the required backend |
| P1 | C3.4 runtime | Allocations, copies, streams, and events are metadata, not AEC calls | Code-review complete-chain condition is unmet |
| P1 | C3.4 concurrency | Linear lifetime reuse ignores stream happens-before; plan has separate transfer/kernel lists | Reuse safety and prefetch overlap are unproven |
| P1 | C3.5 deployment | Default numerical execution now uses CuPy/CUDA, but it still evaluates high-level nodes instead of AEC kernels | CUDA timing/memory are measurable, but this is not AEC compliance |
| P1 | Submission | Offline dependencies and third-party/LLM/originality disclosures are incomplete | General-requirements compliance is incomplete |
| P2 | Tests | C3.3/C3.4 self-scores exceed their maxima | Printed totals are not score evidence |
| P2 | Evaluator API | Referenced C3.2/C3.3 benchmark is absent | Hidden API compatibility is unknown |

## P0: build one executable AEC inference spine

```text
ONNX -> C3.1 IR -> C3.2 kernels -> C3.3 optimized graph
     -> C3.4 allocations/transfers/streams/events -> AEC runtime -> outputs
```

Completion evidence:

1. Every emitted kernel resolves to submitted source and an AEC artifact generated during evaluation.
2. Unknown kernels are rejected before execution.
3. Device alloc/free, H2D, launch, event wait/record, D2H, and errors use the AEC runtime.
4. All three models execute through this path.
5. `numpy.allclose(rtol=1e-3, atol=1e-3)` passes; MLP top-1 is at least 98% and ResNet top-1 at least 85%.
6. CPU/NumPy/ONNX Runtime paths are clearly reference-only.

Current reference status: C3.5 optimizes the C3.1 graph with C3.3, builds a C3.2-decomposed C3.4 plan for each batch size, rejects invalid or graph-incomplete plans, and uses the plan's node order for CuPy execution by default. This closes the disconnected software-stage defect for CUDA correctness/performance testing, but not the required AEC backend.

## P1: satisfy C3.2 without score gaming

Verified positives: deterministic routing, canonical names, connected Gemm outputs, retained parameters, constrained tuning, and live hardware references.

Remaining work:

- Query/load the real AEC capability instead of claiming H100 FP8/FP4/Winograd support.
- Implement every claimed f32/f16/f8/f4 kernel.
- Implement and numerically qualify the selected FP8 and W4A16 kernels on AEC; the current routing is shape/semantics based and deterministic.
- Demonstrate every selected profile within the numerical thresholds.
- Add executable non-default-attribute tests for the 17 operators.
- Validate block, grid, and shared memory against the target.

The C3.2 smoke script now prints `14.17/15`: D1 is 3.0/3.0 with fp32/fp16/fp8/fp4 coverage, while D3 remains 2.17/3.0. This is structural evidence and still partly depends on the unverified capability profile.

## P1: close C3.3 F2/F3/F4

| Model | Launch reduction | Buffer reduction | 60% target met? |
|---|---:|---:|---|
| MLP | 0.0% | 0.0% | No |
| ResNet-18 | 10.7% | 17.0% | No |
| Transformer | 18.2% | 27.9% | No |

Required work:

- Retain initializer payloads or add an initializer store so Conv+BN actually folds weights/bias.
- Extend the current first-batch original-versus-optimized qualification to the AEC backend; any violation makes F4 zero.
- Lower every fused node to one real AEC kernel or a truthful executable sequence. A fused label alone is not a saved launch.
- Improve general fusion/lowering toward the 60% F2/F3 targets.
- Cap and repair self-test arithmetic; `9.40/8.6` is invalid.

## P1: make C3.4 runtime-backed

Corrected plans now have complete bindings and consistent transfer/wait events. This closes the earlier empty-binding and unsignalled-event defects, but not the five code-review features.

Required work:

- Connect pool slots to AEC device allocations/frees.
- Put alloc, H2D, kernel, event, and D2H operations in one executable timeline.
- Place next-layer weight copies near preceding compute and capture a trace proving overlap.
- Base reuse on happens-before across streams, not only linear indices.
- Execute actual streams/events.
- Repair the self-score cap; `10.85/10` is invalid despite 505 structural checks.

## P1: complete C3.5 evaluator behavior

The CuPy path is now the default connected reference: weights, constants, batches, intermediates, and operator computation stay on the CUDA device until final output collection. NumPy is available only through explicit `--backend numpy` development mode. Neither path is an AEC deployment, and both still evaluate high-level nodes rather than executing individual decomposed kernels.

Remaining work:

- Replace high-level planned array-operator execution with the individual C3.2 kernel steps and physical bindings on AEC.
- Preserve required intermediate dtypes and represent omitted optional inputs explicitly instead of scalar FP32 zero.
- Test non-default attributes and hidden valid shapes for all operators.
- Measure cold time and NVML per-process peak GPU memory on the target.
- Replace CuPy high-level operator calls with submitted AEC kernel/runtime calls; GPU observation alone does not prove AEC execution.

Written C3.5 scoring is correctness/accuracy 15, time 25, memory 10. Use the written rubric until the conflicting image is clarified.

## P1: offline and integrity compliance

Before submission, provide:

- offline dependencies or confirmation that they are evaluator-provided;
- versions, licenses, purpose, and call boundaries;
- academic attribution;
- originality and LLM-assistance disclosure;
- a clean archive check excluding caches, outputs, development downloads, and `.DS_Store`.

The local macOS ARM environment is not evidence of parity with the specified Linux x86_64/CUDA environment.

## Verified in this audit

- C3.1: 7/7 tests pass.
- C3.2: structural script completes at 14.17/15; D1 is 3.0/3.0; five precision-policy and four independent scoring regressions pass.
- C3.3: 51 structural checks pass; independent EW-chain and Conv+BN numerical regressions pass.
- C3.4: 505 structural checks pass; all 12 public plan configurations validate; readiness waits use signalled transfer events.
- Independent scoring regressions: 4/4 pass.
- Cross-stage tests: 2/2 pass, covering all public models plus rejection of a plan/graph mismatch.
- Complete golden CLI tests for MLP, ResNet, and Transformer pass through the connected reference path (187.041 seconds locally).
- The revised standards-oriented runner passes all released models in explicit NumPy reference mode: MLP 0.185s/0.9835 accuracy, ResNet 107.247s/0.9351 accuracy, Transformer 82.338s; all precision gates pass. CuPy 14.1.1, GPU-memory sampling, and AEC execution still require target-server validation.
- Default CuPy mode fails closed when CuPy or a CUDA device is unavailable; there is no silent NumPy fallback.

These positives do not override unresolved AEC, numerical-gate, reduction, runtime, or compliance items.

## Organizer questions

1. Provide `benchmarks/c32_c33/bench_c32_c33.py` and exact API schemas.
2. Confirm the AEC device profile and capability query.
3. Confirm how the already-folded released ResNet is used for Conv+BN scoring.
4. Confirm whether C3.5 peak memory is 10 points or the conflicting image value.
5. Confirm whether build/JIT time and reusable caches count in cold timing without violating the no-precomputed-artifact rule.
