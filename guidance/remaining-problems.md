# Remaining problems and completion gates

This file records unresolved problems found during the C3.1/C3.2 implementation reviews and the `general_requirements.md` compliance audit. Passing the current local tests does not close an item unless its completion evidence below also exists.

## Status summary

| Priority | Area | Remaining problem | Current consequence |
|---|---|---|---|
| P0 | Architecture | No executable AEC compiler/runtime/device path | C3.2 signals cannot become inference; C3.5 is impossible |
| P0 | Cross-stage | C3.3 fused ops, C3.4 plans, and C3.5 execution are disconnected | Optimized graphs and schedules cannot execute |
| P0 | Correctness | No FULL_FP32 kernel execution/numerical validation | C3.2 hard correctness condition is unproven |
| P0 | Integrity | Precision selection is stateful round-robin | Same input produces different decisions; resembles score targeting |
| P1 | C3.2 | Advertised and emitted GEMM kernel names disagree | Claimed D5 kernel diversity is false |
| P1 | C3.2 | Kernel references omit executable operator parameters | Many decompositions cannot preserve ONNX semantics |
| P1 | C3.2 | Gemm without bias does not produce the node output | Valid hidden/general input can generate a broken plan |
| P1 | Hardware | Hardcoded H100 profile and stale API singleton | Capability claims may not match the fixed AEC target |
| P1 | Tuning | Winograd/im2col ignore constrained hardware limits | Valid target profiles can receive invalid launches |
| P1 | Submission | Runtime packages are defined, but offline wheels and LLM/third-party disclosure are incomplete | Direct general-requirements violation if submitted |
| P1 | Submission | Bytecode/cache/OS artifacts are present | Repository is not submission-clean |
| P2 | Tests | C3.2 self-score contains circular/weak checks | `14.17/15` is not credible evidence |
| P2 | Specification | Official C3.2/C3.3 benchmark/API is absent | Exact evaluator compatibility remains unverified |
| P2 | Specification | C3.5 memory points conflict; Conv+BN test is unclear | Organizer clarification required |

## P0: establish a real AEC execution spine

The implementation currently ends at `KernelSpecRef` descriptions. There is no source path that compiles/encodes kernels for AEC, uploads them through the AEC runtime, launches them on an AEC device/CModel, and returns tensor results.

Required work:

1. Define a stable AEC kernel ABI and execution interface.
2. Map every emitted kernel name to submitted implementation source.
3. Carry all operator/kernel parameters required for execution.
4. Connect allocation, H2D, launch, synchronization, D2H, and error handling.
5. Execute FULL_FP32 decompositions and compare against the supplied golden outputs.
6. Label any CPU, PyTorch, ONNX Runtime, CUDA, or mock backend as reference-only.

Completion evidence:

- A kernel registry rejects unknown/unimplemented names.
- Every kernel emitted for the 17-operator public union resolves to executable source.
- All three public models run through the AEC path.
- `numpy.allclose(rtol=1e-3, atol=1e-3)` passes for every public model.
- MLP top-1 is at least 98% and ResNet top-1 is at least 85%.

## P0: replace score-oriented precision rotation

`c32/strategy.py` uses `coverage_mode=True` and `_precision_counter`, so identical calls can return different precisions depending on evaluator history. Remove all call-order and scoring state.

Required replacement:

- Add an explicit execution mode such as `FULL_FP32` versus an accuracy-qualified mixed-precision mode.
- Select precision deterministically from operator sensitivity, tensor dtype/shape, requested mode, and declared target capability.
- Keep all sensitive operators FP32.
- Use FP8/FP4 only when the target supports them and numerical qualification proves the model/operator remains within tolerance.

Completion evidence:

- Repeating `select_precision(node, graph)` produces the same profile.
- Reordering evaluator calls or processing another model first does not alter results.
- Selected precision is always supported by the active target.
- FULL_FP32 passes numerical and top-1 requirements.

## P1: make C3.2 decompositions truthful and complete

### Normalize actual kernel names

The hardware API advertises `matmul_f32`, `matmul_f16`, `matmul_f8`, and `matmul_f4`, while decomposition emits `matmul_fp32`, `matmul_fp16`, `matmul_fp8`, and `matmul_fp4`. Establish one canonical mapping and test actual emitted/executable names, not capability strings.

### Carry execution semantics

Extend kernel references or a companion parameter object to retain at least:

- Conv: pads, strides, dilations, groups, kernel shape, bias behavior.
- Gemm: `alpha`, `beta`, `transA`, `transB`.
- Softmax: axis and stable-reduction semantics.
- LayerNormalization: axis, epsilon, scale, bias.
- Transpose: permutation.
- Gather: axis and index dtype.
- Split: axis and split sizes.
- Reshape: target shape, `0`/`-1`, and `allowzero` semantics.

### Fix dataflow and semantics

- Gemm without bias must write the declared node output.
- Gemm intermediates must use the documented deterministic `__c3_inter_*` convention.
- GlobalAveragePool must preserve `[N,C,1,1]`; do not introduce an unconditional squeeze.
- Validate each kernel input against node inputs or earlier kernel outputs, and require every node output to be produced.

Completion evidence:

- Positive and negative decomposition tests for all 17 operators.
- Sequence-level dataflow validation for every public node.
- Operator-attribute tests covering non-default values.
- Numerical tests comparing decomposed and direct/reference execution.

## P1: make hardware and tuning configuration honest

The default `NVIDIA H100` profile is hardcoded and claims all scoring capabilities. `set_hardware()` also leaves the evaluator-facing `c32.api.hardware` and `strategy.hardware` objects stale.

Required work:

- Obtain the fixed AEC profile from an organizer-defined configuration or AEC device query.
- Validate positive thread/grid/shared-memory limits and supported precisions.
- Ensure public API objects use the same active immutable target snapshot.
- Make every tuning path clamp or reject invalid blocks/shared memory.
- Accept the documented `ProblemSize` representation consistently.
- Treat only `smem_bytes == -1` or `0 <= smem_bytes <= limit` as valid.

Completion evidence:

- Tests with small thread/shared-memory limits pass for every kernel family.
- Hardware switching or configuration creates no stale API state.
- Unsupported precisions/kernels are never selected or advertised as executable.

## P0: make C3.3, C3.4, and C3.5 executable together

The packages now contain structural implementations and self-tests, but they are disconnected simulations rather than one executable pipeline. Current self-tests report success even when their artifacts cannot be consumed by the next stage.

### C3.3 fusion

- Add executable implementations for all five fused operator types. C3.5 currently rejects every one as an unknown operator.
- Make elementwise-chain fusion include every external input used anywhere in the chain, not only the first node's inputs, and preserve each operation's attributes and order.
- Perform actual Conv+BN parameter folding. The current pass explicitly records metadata without changing weights, so it changes the graph without preserving computation.
- Apply passes transactionally. The pipeline currently catches pass exceptions and continues without restoring a possibly partially mutated graph.
- Run original and optimized graphs and enforce `max_abs_diff <= 1e-3`; structural validation alone is insufficient.
- Count launches and buffers from actual C3.2 decompositions/execution plans. Current fixed estimates give Conv=3, Softmax=5, and LayerNorm=8 regardless of selected lowering.
- Reach the documented 60% launch and buffer reductions on evaluator cases through real execution. Observed public-model reductions were: MLP 0%/0%, ResNet 9.1%/17.0%, Transformer 18.4%/27.9%.
- Do not claim Conv+BN reconstruction from already-folded weights without original BN data.

Observed test warning: `c33.test_c33` exits successfully but reports `9.40 / 8.6`; individual sections also exceed their maxima. Treat this harness only as structural smoke coverage.

### C3.4 memory and scheduling

- Connect `DeviceMemoryPool` to actual AEC device allocation/free. It currently manages Python metadata and integer slot IDs only.
- Preserve allocation bindings through schedule construction. The allocator removes entries from `_alloc_map` before kernel steps are created; observed empty output bindings were 9/9 MLP kernels, 76/76 ResNet kernels, and 253/253 Transformer kernels.
- Require every logical kernel input/output to have a physical binding. `ExecutionPlan.validate()` currently validates only bindings that happen to exist, so an entirely missing output map passes.
- Make weight transfers signal the events kernels wait on. Observed transfers with `event_id` were 0/7 for MLP, 0/43 for ResNet, and 0/56 for Transformer.
- Represent transfers and kernels in one ordered/timestamped plan. Separate lists do not prove that next-layer H2D overlaps current compute; all weights are currently created as bulk transfers before scheduling kernels.
- Base lifetime reuse on stream happens-before relationships, not only a linear step index. Linear non-overlap can be unsafe when kernels execute concurrently on different streams.
- Connect streams/events to actual AEC runtime calls and demonstrate overlap/concurrency with a trace.

Observed test warning: `c34.test_c34` exits successfully with 506 checks but reports `10.85 / 10.0`. Its checks prove metadata presence, not real device allocation, transfer, event, or execution behavior.

### C3.5 deployment

- Replace the NumPy CPU executor with the required AEC GPGPU execution path. `c35/engine.py` explicitly says NumPy is simulating AEC; this cannot satisfy GPU deployment, runtime ranking, peak GPU memory, or the general AEC-stack requirement.
- Consume the optimized C3.3 graph and C3.4 execution plan rather than re-walking the original ONNX graph directly.
- Add dispatch/execution for all fused operators or lower them back into supported executable kernels.
- Validate every input manifest entry against the graph and actual NPY dtype/shape, require all graph inputs, reject unexpected inputs, and require equal sample counts across inputs.
- Preserve intermediate/output dtypes rather than forcing every computed node output to float32; only the specified final output is necessarily float32.
- Handle optional inputs without replacing every empty slot with scalar float32 zero, which changes operator semantics.
- Test non-default attributes and hidden-shape cases for all 17 operators.
- Measure cold process time and NVML peak GPU memory on the target Linux/NVIDIA environment.

Observed functional results on the local NumPy reference path:

- 15/15 operator unit tests passed.
- MLP 10,000-sample golden/accuracy test passed in 0.203 seconds.
- ResNet 10,000-sample golden/accuracy test passed in 107.110 seconds on the local CPU path.
- Transformer 10,000-sample golden test passed in 4.902 seconds.
- Missing-model and invalid-batch CLI checks passed.

These results establish a useful FP32 reference implementation, not AEC/GPU completion.

## P1: satisfy originality, dependency, and offline rules

The repository now includes local and Linux GPU requirement files, a CUDA Docker definition, and an environment verifier. Portable local parity is confirmed for ONNX 1.22.0, ONNX Runtime 1.27.0, and Torch 2.13.0. The macOS ARM64 host cannot reproduce the target Linux/NVIDIA stack.

Remaining submission material:

- A successful `environment/verify_environment.py --target` run on Linux 6.8.0-110 x86_64 with Python 3.12.3, GCC/G++ 13.3.0, nvcc 12.8.61, driver 580.126.20, CUDA-enabled Torch 2.13.0+cu130, and CuPy 14.1.1.
- Downloaded offline wheels or documented evaluator-provided dependencies; the current requirement files still require network for first installation.
- A verified decision on whether the target CuPy module is supplied by `cupy-cuda12x` or another organizer image package.
- Third-party dependency declaration with license, purpose, and call boundary.
- Academic algorithm/source attribution.
- Originality declaration and explicit LLM-assistance disclosure.
- Explanation of which generated code the team reviewed and can maintain.

At minimum, disclose ONNX, ONNX Runtime, protobuf, NumPy, Torch, CuPy, and any future AEC/runtime/compiler libraries. Do not treat the local macOS environment or an unbuilt Dockerfile as proof of target parity.

## P1: make the submission clean

Development currently creates `.pyc`, `__pycache__`, and `.DS_Store` files. These may remain locally but must not enter the submission archive. Also exclude generated DAGs, inference outputs, cached plans, downloaded dependencies outside the declared offline package, and development-only binaries.

Add an ignore/packaging policy and validate the final archive contents. Never remove official evaluator inputs or required licensed source merely to make the tree smaller.

Completion evidence:

```text
find SUBMISSION -name '__pycache__' -o -name '*.pyc' -o -name '.DS_Store'
```

produces no results, and the clean archive builds and runs in a fresh network-disabled environment.

## P2: replace the C3.2 self-score with independent evidence

The current test suite inflates confidence because it:

- Includes Transformer although the named C3.2 microbenchmark models are MLP and ResNet.
- Checks advertised GEMM kernels rather than emitted/executable kernels.
- Defines the tuning denominator from kernels already classified as tunable.
- Accepts Softmax based mainly on sequence length.
- Does not execute FULL_FP32 kernels or check numerical output.

Keep useful structural checks, but add independent evaluator-shaped tests that call only the documented public API. A test must fail when capability strings and actual implementation disagree.

## Organizer questions that remain open

1. Obtain `benchmarks/c32_c33/bench_c32_c33.py` and the exact public API schemas.
2. Confirm the fixed AEC hardware/device profile and how capabilities are queried.
3. Confirm how `FusedConv2dBatchNorm` is evaluated when released ResNet contains no BN node or original BN parameters.
4. Confirm whether C3.5 peak memory is 10 points (written rules) or 15 points (`requirements.png`).
5. Confirm whether build/JIT time and reusable caches are included or allowed in timing, without conflicting with the no-precomputed-artifact rule.

## Current verified positives

- C3.1 passes seven specification-focused tests for all public models and adversarial graph cases.
- C3.1 uses general ONNX parsing rather than public filename/test-ID branches.
- No runtime network access, embedded golden output, evaluator interception, or input/model hash dispatch was found.
- The C3.2 public API imports and returns non-empty sequences for all released nodes.
- Default-profile C3.2 launch parameters pass the basic rubric assertions, although constrained-target handling remains broken.

These positives do not override any unresolved P0/P1 item above.
