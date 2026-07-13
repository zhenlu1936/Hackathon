# Known limitations and completion gates

Updated: 2026-07-13 (CuPy-only synchronization) — resolved #11, #28, #29,
#35, #37; submission items #10 and #41–45 remain open; added #46 for mandatory
H200 validation of the revised runtime

This release register follows `.specification/general_requirements.md` first and
`.specification/scoring.md` second. A check closes a scoring item only when it
exercises the same artifact and backend evaluated by the organizer.

## Release priority list

| # | Priority | Area | Remaining problem | Scoring consequence |
|---|----------|------|-------------------|--------------------|
| 1 | P0 | Architecture | C3.2 kernel steps and C3.4 plan operations do not directly drive H200 execution | C3.2/C3.4 implementation claims remain partly structural |
| 2 | P0 | Correctness | No FULL_FP32 execution of the decomposed C3.2 kernel sequence against golden outputs | C3.2 hard numerical condition is unproven |
| 3 | P1 | C3.2 precision | Mixed routing covers four precisions structurally, but FP8/FP4 H200 kernels are not numerically qualified | D1 routing signals pass; low-precision correctness remains unproven |
| 4 | P1 | C3.2 hardware | Default target is an unverified capability profile rather than an AEC query | Capability and D5 claims may be false |
| 5 | P1 | C3.3 reduction | Public launch/buffer reductions are below 60% | F2/F3 cannot receive full credit |
| 6 | P1 | C3.3 correctness | Earlier H200 runs passed MLP and ResNet, but the CuPy-only revision has not been rerun; BN values cannot be reconstructed and fused nodes are not single fused H200 kernels | F4/backend launch evidence is incomplete |
| 7 | P1 | C3.4 runtime | Allocations, copies, streams, and events are metadata rather than the operations driving CuPy | Code-review complete-chain condition is unmet |
| 8 | P1 | C3.4 concurrency | Linear lifetime reuse ignores stream happens-before; plan has separate transfer/kernel lists | Reuse safety and prefetch overlap are unproven |
| 9 | P1 | C3.5 integration | CuPy executes on the AEC H200 but evaluates high-level nodes instead of C3.2 kernel steps | End-to-end device execution works; compiler-plan integration remains incomplete |
| 10 | P0 | Submission | Direct `google.protobuf` and OpenAI Codex assistance are not completely verified and disclosed; archive cleanliness is not proven | Integrity Rule 6 and submission reproducibility remain release blockers |
| 12 | P2 | Evaluator API | Referenced C3.2/C3.3 benchmark is absent | Hidden API compatibility is unknown |
| 46 | P0 | C3.5 validation | The revised CuPy-only CLI, runner, serialization, and tests have not run on the remote H200 | Current end-to-end correctness and performance are unproven |

## H200 execution integration

```text
ONNX -> C3.1 IR -> C3.2 kernels -> C3.3 optimized graph
     -> C3.4 allocations/transfers/streams/events -> CuPy on AEC H200 -> outputs
```

Completion evidence:

13. Every emitted kernel resolves to submitted source and an AEC artifact generated during evaluation.
14. Unknown kernels are rejected before execution.
15. Device alloc/free, H2D, launch, event wait/record, D2H, and errors are driven by the C3.4 plan on the H200.
16. All three models execute through this path.
17. `cupy.allclose(rtol=1e-3, atol=1e-3)` passes; MLP top-1 is at least 98% and ResNet top-1 at least 85%.
18. Confirm by source audit that no non-CuPy numerical fallback remains.

Release status: C3.5 optimizes the C3.1 graph with C3.3, builds a
C3.2-decomposed C3.4 plan for each batch size, rejects invalid or
graph-incomplete plans, and uses the plan's node order for CuPy execution by
default on the designated remote H200 AEC device. The unresolved boundary is
direct execution of C3.2 kernel steps and C3.4 plan operations.

## C3.2 backend qualification

Verified positives: deterministic routing, canonical names, connected Gemm outputs, retained parameters, constrained tuning, and live hardware references.

Remaining work:

19. Query the actual H200/MIG capability instead of relying on an unverified profile.
20. Implement every claimed f32/f16/f8/f4 kernel.
21. Implement and numerically qualify the selected FP8 and W4A16 kernels on the H200; the current routing is shape/semantics based and deterministic.
22. Demonstrate every selected profile within the numerical thresholds.
23. Add executable non-default-attribute tests for the 17 operators.
24. Validate block, grid, and shared memory against the target.

The C3.2 smoke script now prints `14.17/15`: D1 is 3.0/3.0 with fp32/fp16/fp8/fp4 coverage, while D3 remains 2.17/3.0. This is structural evidence and still partly depends on the unverified capability profile.

## C3.3 reduction and backend correctness

| Model | Launch reduction | Buffer reduction | 60% target met? |
|---|---:|---:|---|
| MLP | 44.4% (+44pp) | 40.0% | No |
| ResNet-18 | 38.7% (+28pp) | 36.2% | No |
| Transformer | 18.2% | 27.9% | No |

Required work:

25. Retain initializer payloads or add an initializer store so Conv+BN actually folds weights/bias.
26. Extend first-batch original-versus-optimized qualification to the decomposed H200 kernel path; any violation makes F4 zero.
27. Lower every fused node to one real H200 kernel or a truthful executable sequence. A fused label alone is not a saved launch.
28. ~~Improve general fusion/lowering toward the 60% F2/F3 targets~~ → Added `FusedComputeActivation`; MLP now 44%, ResNet 39%. Transformer Erf is inside EWChains so was not improved.
29. ~~Cap and repair self-test arithmetic; `9.40/8.6` is invalid~~ → Resolved.

## C3.4 runtime integration

Corrected plans now have complete bindings and consistent transfer/wait events. This closes the earlier empty-binding and unsignalled-event defects, but not the five code-review features.

Required work:

30. Make C3.4 pool slots drive the CuPy/H200 device allocations and frees.
31. Put alloc, H2D, kernel, event, and D2H operations in one executable timeline.
32. Place next-layer weight copies near preceding compute and capture a trace proving overlap.
33. Base reuse on happens-before across streams, not only linear indices.
34. Execute actual streams/events.
35. ~~Repair the self-score cap; `10.85/10` is invalid despite 505 structural checks~~ → Resolved (capped at `10.00/10.0`).

## C3.5 evaluator behavior

The CuPy path is the only AEC H200 execution path: weights, constants,
batches, intermediates, and operator computation stay on the device until final
output collection. The H200 path still evaluates high-level nodes rather than
executing the individual decomposed C3.2 kernels.

Remaining work:

36. Replace high-level array-operator execution with individual C3.2 kernel steps and C3.4 physical bindings on the H200.
37. ~~Preserve required intermediate dtypes and represent omitted optional inputs explicitly instead of scalar FP32 zero~~ → Resolved (passes `None` instead of FP32 zero).
38. Test non-default attributes and hidden valid shapes for all operators.
39. Measure cold time and NVML per-process peak GPU memory on the target.
40. Make the submitted kernel, allocation, stream, and event plans drive the existing H200 execution path.
46. Run MLP, ResNet, and Transformer through `./run_c35.sh` on the remote H200,
    record golden/accuracy results, confirm the CuPy-only report schema, and
    verify the registered command and CLI help expose only the required C3.5
    arguments.

Written C3.5 scoring is correctness/accuracy 15, time 25, memory 10. Use the written rubric until the conflicting image is clarified.

## Submission and integrity

Before submission, provide:

41. Verify every direct and optional dependency against the native server and
    record the direct `google.protobuf` dependency. The unavailable optional
    Python NVML binding has been removed from the framework.
42. Complete exact versions, licenses, purposes, and call boundaries for all
    used modules. The server-native `nvidia-smi` boundary is now recorded; do
    not infer an exact CuPy distribution name from the importable module alone.
43. Review the academic-attribution draft in `docs/SUBMISSION.md` against the
    actual implementation and disclose every public source that influenced it.
44. Expand the LLM-assistance disclosure to include OpenAI Codex as well as
    GitHub Copilot, and keep it current after later assisted revisions.
45. Build the actual submission archive and inspect its file list for virtual
    environments, caches, bytecode, reports, generated plans/outputs,
    `.agents`, `.specification`, and development-only assets. `.gitignore` is
    packaging policy, not evidence that the archive is clean.

The local macOS ARM environment is not evidence of parity with the specified Linux x86_64/CUDA environment.

## Verified release evidence

- C3.1: 7/7 tests pass.
- C3.2: structural script completes at 14.17/15; D1 is 3.0/3.0; five precision-policy and four independent scoring regressions pass.
- C3.3: 51 structural checks pass; independent EW-chain and Conv+BN numerical regressions pass.
- C3.4: 505 structural checks pass; all 12 public plan configurations validate; readiness waits use signalled transfer events.
- The earlier independent scoring and cross-stage regressions are historical;
  their CuPy conversions require issue 46's H200 validation.
- The former CPU reference evidence is historical and no longer exercises a
  supported framework path after the CuPy-only conversion.
- The CuPy-only framework fails closed when CuPy or a CUDA device is unavailable.
- Pre-conversion H200 MIG runs confirmed CuPy 14.1.1, CuPy-pool evidence, and numerical gates for MLP (`0.9835`, max diff `1.53e-05`) and ResNet (`0.9351`, max diff `8.58e-06`). They exposed CuPy 14.1.1's asymmetric `Split` contract. These results are historical evidence only; issue 46 requires a fresh three-model run of the revised framework.

These positives do not override unresolved kernel-plan integration,
low-precision qualification, reduction, Transformer H200 validation, or
submission-compliance items.

## Organizer questions

Q1. Provide `benchmarks/c32_c33/bench_c32_c33.py` and exact API schemas.
Q2. Confirm the exact H200 MIG slice limits and capability-query interface used by the evaluator.
Q3. Confirm how the already-folded released ResNet is used for Conv+BN scoring.
Q4. Confirm whether C3.5 peak memory is 10 points or the conflicting image value.
Q5. Confirm whether build/JIT time and reusable caches count in cold timing without violating the no-precomputed-artifact rule.
