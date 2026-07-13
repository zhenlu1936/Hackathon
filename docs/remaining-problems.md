# Known limitations and completion gates

Updated: 2026-07-13 (executable C3.3 fusion revision) — generated Gemm, Conv,
attention, LayerNormalization, elementwise, residual-normalization, and layout
kernels write directly into planned output views. All released graphs now clear
the local 60% launch/buffer anchors, but the exact revision still needs H200
compilation, numerical comparison, and observed-launch evidence. The
superseding H200 rerun remains #40; direct general C3.2 kernel-step execution
remains #1/#36.

This release register follows `.specification/general_requirements.md` first and
`.specification/scoring.md` second. A check closes a scoring item only when it
exercises the same artifact and backend evaluated by the organizer.

## Release priority list

| # | Priority | Area | Remaining problem | Scoring consequence |
|---|----------|------|-------------------|--------------------|
| 1 | P0 | Architecture | C3.4 now drives a high-level-node CuPy timeline, but individual C3.2 kernel references still do not drive H200 execution | The direct decomposed-kernel chain remains incomplete |
| 3 | P1 | C3.2 precision | Mixed routing covers four precisions structurally, but FP8/FP4 H200 kernels are not numerically qualified | D1 routing signals pass; low-precision correctness remains unproven |
| 4 | P1 | C3.2 hardware | A CuPy device-query path exists, but it has not been activated and recorded on the organizer H200; the default coverage profile remains unverified | Capability and D5 claims remain target-unproven |
| 5 | P1 | C3.3 reduction | Connected generated kernels clear all local 60% anchors, but this revision has not compiled or run on H200 | F2/F3 source path is fixed; target numerical/profiler evidence remains required |
| 7 | P1 | C3.4 runtime | Persistent streams/events and direct planned writes for generated fused kernels are source-connected, but other high-level operators retain temporaries and the revision has not run on H200 | Complete-chain and actual-peak claims remain unproven |
| 8 | P1 | C3.4 concurrency | Reuse dependencies, persistent physical streams, reusable plan events, and a unified timeline are implemented, but target ordering/overlap has not been traced | Reuse safety and prefetch overlap remain target-unproven |
| 9 | P1 | C3.5 integration | CuPy executes on the AEC H200 but evaluates high-level nodes instead of C3.2 kernel steps | End-to-end device execution works; compiler-plan integration remains incomplete |
| 10 | P0 | Submission | Direct `google.protobuf` native-server verification is not proven | Dependency verification under Integrity Rule 6 remains a release blocker; the staged archive is clean and source/AI disclosures are current |
| 12 | P2 | Evaluator API | Referenced C3.2/C3.3 benchmark is absent | Hidden API compatibility is unknown |

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

Release status: C3.5 optimizes the C3.1 graph with C3.3, records the C3.2
decomposition as review metadata, and builds a C3.4 high-level-node timeline
for each batch size. The new CuPy executor consumes that timeline, one device
arena, allocation views, copy/compute streams, and events in source, but the
revision has not run on H200. Direct C3.2 kernel-step execution remains the
architectural boundary.

## C3.2 backend qualification

Verified positives: deterministic routing, canonical names, connected Gemm outputs, retained parameters, constrained tuning, and live hardware references.

Remaining work:

19. Run `c32.api.activate_cupy_hardware()` on the actual H200/MIG target,
    record its source, limits, and compute capability, and leave FP4 disabled
    unless an executable qualified AEC FP4/W4A16 path is present.
20. Implement every claimed f32/f16/f8/f4 kernel.
21. Implement and numerically qualify the selected FP8 and W4A16 kernels on the H200; the current routing is shape/semantics based and deterministic.
22. Demonstrate every selected profile within the numerical thresholds.
23. Run the executable non-default-attribute suite on CuPy 14.1.1/H200. The
    host semantic shim passes 16 regression methods covering the published
    17-operator union, but host execution is not target-backend evidence.
24. Validate block, grid, and shared memory against the target. Local contract
    tests now enforce threads-per-block, block-x, grid-x, and shared-memory
    limits, but the discovered H200 values still need the target run in item 19.

The C3.2 smoke script now prints `14.17/15`: D1 is 3.0/3.0 with fp32/fp16/fp8/fp4 coverage, while D3 remains 2.17/3.0. The final local rerun passes all seven dependency-light C3.2 contract regressions; the synthetic downstream plan also validates with complete tuning and precision metadata. Repository compilation, diff checks, and the final cross-record C3.2 consistency search pass. The implementation intentionally does not insert redundant copy kernels solely to make every single-kernel node report an intermediate; that would inflate launch counts and misrepresent executable work. This remains structural evidence and still partly depends on the unverified capability profile.

## C3.3 reduction and backend correctness

Current local evidence: Python compilation and diff checks pass, C3.2 remains
`14.17/15`, `python3 -m c33.test_c33` passes `61/61` with a written-rubric
structural total of `15.00/15.0`, and the focused executable-fusion suite passes
`8/8`. The current same-lowering counts are MLP `9 -> 3` launches and `8 -> 2`
buffers, ResNet-18 `75 -> 28` and `74 -> 27`, and Transformer `217 -> 79` and
`224 -> 86`. Corresponding reductions are `66.7%/75.0%`, `62.7%/63.5%`, and
`63.6%/61.6%`.

The implementation uses generated single-launch kernels. The obsolete
sequential region and compute-activation utilities have been removed. Buffer
counting now includes named
C3.2 lowering intermediates, and Constant metadata references count as zero
launches because the runtime preloads them. Optimized graphs build complete
C3.4 plans locally. CuPy is unavailable on this machine, so the expanded device
numerical suite and all three optimized models still require H200 execution.

The self-test applies the published F2/F3 formula per MLP and ResNet-18 and
prints both results. Since the referenced organizer benchmark is absent and
cross-model aggregation is unspecified, its mean is explicitly a local
self-score convention rather than a claim about the unreleased evaluator.

The earlier MLP `88.9%/100.0%`, ResNet-18 `84.0%/76.6%`, and Transformer
`74.3%/73.5%` launch/logical-buffer figures are historical and no longer
current release evidence. They counted `FusedExecutionRegion` as one launch
while its executor ran the retained operations sequentially. Those deprecated
paths and their test-only ABI are no longer present.

Current implementation state:

- `FusedEWChain` uses one runtime-generated CuPy `ElementwiseKernel`.
- `FusedSoftmaxDropout` uses one runtime-generated CUDA reduction kernel for
  arbitrary valid axes.
- `FusedResidualNorm` uses one runtime-generated CUDA reduction/normalization
  kernel for equal-shaped residual inputs.
- `FusedMatMulBias` and `FusedGemmEpilogue` use one generated FP32 GEMM kernel
  with broadcast bias and optional absorbed Flatten/Relu semantics.
- `FusedConvActivation` and `FusedConvResidualActivation` use one direct NCHW
  convolution kernel with bias, optional residual, and Relu epilogues.
- `FusedAttentionScores` performs rank-four MatMul, scalar division, broadcast
  mask Add, and last-axis Softmax in one generated kernel.
- `FusedLayerNormalization` and `FusedTransposeReshape` have bounded generated
  kernel ABIs and direct planned-output writes.
- explicit Conv+BN parameters are folded in the executor's CuPy initializer
  store before C3.4 planning, including the planned executor's device-copy
  path; the public ResNet still has no recoverable BN.

Run the expanded device numerical tests and all three released graphs on H200,
then compare unfused and fused executions with a launch profiler and confirm
F2/F3. No profiler is declared in `environments.txt`, so observed-launch
evidence remains blocked under #27.

Required work:

25. Run the executor-side Conv+BN initializer fold on the H200 and confirm
    `max_abs_diff <= 1e-3`; implementation is present, but this revision has
    only local source and structural evidence.
26. Extend first-batch original-versus-optimized qualification to the decomposed H200 kernel path; any violation makes F4 zero.
27. Compile and validate the generated Gemm/MatMul, Conv/residual, attention,
    normalization, elementwise, and layout kernels on H200. Unsupported general
    regions remain disabled. Closure requires baseline-versus-fused numerical
    and profiler evidence whose observed launch counts agree with the reported
    lowerings.

## C3.4 runtime integration

Corrected plans now have complete bindings, a unified action timeline, explicit
reuse happens-before dependencies, and consistent transfer/wait events. The
CuPy source path consumes those artifacts, caches each plan's arena, retains
immutable model tensors across batch plans, reuses physical streams by logical
stream ID, and reuses plan events after synchronization. The local
12-configuration plan validation passes. Bounded per-batch pool telemetry is
connected to the C3.5 backend evidence.
Target execution is still unverified, input arrays enter the CLI through
`cupy.load` before the planned copy, and high-level operators may allocate
temporary device outputs outside the arena before copying results into their
planned views. Generated C3.3 kernels write directly into planned views, and
the generic path avoids a redundant contiguous copy.

Required work:

30. Run the arena/view allocation path on H200 and measure actual allocations;
    verify persistent-stream pool reuse and remove or account for the remaining
    high-level operator temporaries that escape the arena.
31. Qualify the new alloc, H2D/D2D, kernel, event, D2H, and free timeline on
    H200, including failure propagation and trace-to-plan agreement.
32. Place next-layer weight copies near preceding compute and capture a trace proving overlap.
33. Qualify the new reuse happens-before dependencies on real H200 streams; the
    plan-level overlap and event checks pass locally.
34. Capture H200 evidence that the actual CuPy streams/events execute in the
    planned order and that intended copy/compute overlap occurs.

## C3.5 evaluator behavior

The CuPy path is the only AEC H200 execution path: weights, constants,
batches, intermediates, and operator computation stay on the device until final
output collection. The H200 path still evaluates high-level nodes rather than
executing the individual decomposed C3.2 kernels.

Remaining work:

36. Replace high-level array-operator execution with individual C3.2 kernel steps and C3.4 physical bindings on the H200.
38. Run `C35Opset17AttributeTests` on CuPy 14.1.1/H200 and then rerun all three
    released models. The implemented suite already covers all 17 operators,
    including Flatten axes, Conv dilation/`auto_pad`/groups, opset-17 Split
    sizes, multi-output LayerNormalization, dtype-preserving Gather, Constant
    scalar/vector forms, broadcasting, and layout attributes; all 16 methods
    pass with the host semantic shim.
39. Obtain process-accounted NVML peak GPU memory on the target. Cold wall time
    is now measured, but MIG process accounting did not observe the child and
    the report therefore used labeled CuPy-pool reservation proxies.
40. Rerun all three models through the corrected plan-driven
    arena/stream/event path on H200 and confirm the execution trace matches the
    submitted plan. Confirm that `runtime_stream_objects` stays bounded while
    `batch_count` reaches 40 at batch size 256 and that CuPy-pool reservation no
    longer grows once per batch. Report 3 exposed that dynamic tensor sizing used FP32's
    four-byte width for the INT64 Transformer input, producing capacities of
    `192` bytes for a required `288` and `18432` for a required `36864`. The
    planner now derives byte width from ONNX dtype and handles both symbolic
    and `-1` batch dimensions; local dependency-light checks prove `288` bytes
    for `[2, 18]` INT64, but only the H200 rerun closes this item. This does not
    by itself close direct C3.2 kernel execution under #36.

Written C3.5 scoring is correctness/accuracy 15, time 25, memory 10. Use the written rubric until the conflicting image is clarified.

## Submission and integrity

Before submission, provide:

41. Verify every direct and optional dependency against the native server and
    record the direct `google.protobuf` dependency. The unavailable optional
    Python NVML binding has been removed from the framework.
42. Complete exact versions, licenses, purposes, and call boundaries for all
    used modules. The server-native `nvidia-smi` boundary is now recorded; do
    not infer an exact CuPy distribution name from the importable module alone.

Archive gate #45 uses non-destructive packaging: tracked reports remain in the
repository and `.gitattributes` excludes their directories from `git archive`.
Rebuild from the final staged tree and final commit before submission so later
changes cannot bypass that evidence.

The local macOS ARM environment is not evidence of parity with the specified Linux x86_64/CUDA environment.

## Verified release evidence

- C3.1: 7/7 tests pass.
- C3.2: structural script completes at 14.17/15; D1 is 3.0/3.0; five precision-policy and four independent scoring regressions pass.
- C3.2 FULL_FP32 report 3: all 219 nodes select FP32 and decompose. MLP and
  ResNet-18 pass `allclose` with `top1_match=1.0` and maximum absolute
  differences `2.86e-06` and `1.67e-06`; Transformer did not reach numerical
  comparison because its dynamic INT64 input allocation was half-sized. The
  source correction is locally checked, but the earlier three-model H200 pass
  is historical until report 3 is superseded by a rerun. This is not direct
  C3.2 kernel-step execution evidence.
- C3.3 current local evidence: the self-test passes `61/61` and reports
  `15.00/15.0` structurally; the focused executable-fusion suite passes `8/8`,
  and all released-model launch/buffer reductions exceed 60%. The historical
  bounded-region H200 figures remain superseded. The generated kernels and
  exact current graphs still require H200 compilation, numerical comparison,
  and observed-launch evidence.
- C3.4: the final local rerun passes 387/387; all 12 public plan configurations
  validate, timeline coverage is complete, and readiness/reuse waits use
  signalled events. The dedicated executable-timeline contract suite also
  passes 6/6, with Python compilation and diff checks clean. This remains
  structural rather than H200 runtime evidence.
- C3.5 operator semantics: 16 methods covering the full 17-operator union pass
  with a test-only host array shim. This validates control/shape semantics but
  does not replace the required CuPy 14.1.1/H200 rerun.
- Standard CuPy-only H200 report 3 passes MLP (accuracy `0.9835`, max diff
  `1.53e-05`) and ResNet-18 (accuracy `0.9351`, max diff `8.58e-06`), but its
  Transformer process exits on the same dynamic-INT64 allocation error. Older
  three-model reports remain historical rather than current-revision evidence.
- A later user-supplied pre-fix H200 transcript passes all three standard runs
  after the INT64 correction: MLP accuracy `0.9835`, ResNet-18 accuracy
  `0.9351`, and maximum absolute differences `1.53e-05`, `8.58e-06`, and
  `3.24e-05`. It also exposes the CuPy-pool reservation regression addressed by
  the new persistent-stream source change. Because that transcript predates the
  source fix, it is correctness evidence for the input-sizing revision rather
  than post-fix performance or memory evidence.
- The report's `nvidia-smi` sampler did not observe the GPU child process;
  memory evidence is from labeled CuPy-pool reservations, so official
  process-accounted NVML peak memory remains open under #39.
- The superseding H200 transcript passes both cross-stage tests. After the
  deterministic fixture replacement, all four scoring regressions pass on
  CuPy 14.1.1 in `1.162 s`, including executable Conv+BN fusion.
- After executable permission was restored, the exact bounded-region revision's
  full C3.5 workflow passed: MLP accuracy `0.9835`, max diff `1.53e-05`, wall
  `0.993 s`; ResNet-18 accuracy `0.9351`, max diff `8.58e-06`, wall `7.699 s`;
  Transformer max diff `3.15e-05`, wall `1.570 s`. The report used CuPy-pool
  memory figures, so process-accounted NVML evidence remains #39.
- After moving all repository references to the organizer-owned model directory,
  the historical revision passed C3.1, C3.2, C3.3, and C3.4 structural checks.
  Current per-revision evidence is recorded separately above.
- Submission instructions and disclosures are consolidated in the root
  `README.md`. Its C3.1/C3.5 templates and three-model pre-submission checks
  match `.specification/spec.md`; this documentation revision does not close
  native dependency verification (#41/#42) or archive inspection (#45). C3.1
  help and archive-script syntax validate locally; C3.5 help remains H200-only
  because importing its CLI intentionally requires CuPy.
- The former CPU reference evidence is historical and no longer exercises a
  supported framework path after the CuPy-only conversion.
- The CuPy-only framework fails closed when CuPy or a CUDA device is unavailable.
- Pre-conversion H200 MIG runs confirmed CuPy 14.1.1, CuPy-pool evidence, and
  numerical gates for MLP and ResNet, and exposed CuPy 14.1.1's asymmetric
  `Split` contract. These results are historical; the later revised three-model
  report is the current evidence recorded above.

These positives do not override unresolved kernel-plan integration,
low-precision qualification, physical H200 fusion, process-accounted NVML
memory, or submission-compliance items.

## Organizer questions

Q1. Provide `benchmarks/c32_c33/bench_c32_c33.py` and exact API schemas.
Q2. Confirm the exact H200 MIG slice limits and capability-query interface used by the evaluator.
Q3. Confirm how the already-folded released ResNet is used for Conv+BN scoring.
Q4. Confirm whether C3.5 peak memory is 10 points or the conflicting image value.
Q5. Confirm whether build/JIT time and reusable caches count in cold timing without violating the no-precomputed-artifact rule.
