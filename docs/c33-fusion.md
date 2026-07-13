# C3.3 — Operator fusion and graph optimization

## Release contract

Run `GraphPassPipeline(enable_fusion=True, ...)`, produce a validated optimized graph, and expose each matched transformation in `pass_results['Fusion']['stats']['fusion_log']`. Optimize without changing graph inputs, outputs, or FP32 results.

Match graph semantics, never public model identity. Fusion rules must not inspect filenames, test IDs, input hashes, or known weights to trigger special handling. A fusion log without a connected executable fused kernel is not an implementation. Generate optimized graphs and kernels from source during evaluation.

## F1: five target patterns (5 points)

| Required result | Match | Main guard conditions |
|---|---|---|
| `FusedMatMulBias` | MatMul -> bias Add | MatMul output has one consumer; Add operand is a compatible bias |
| `FusedConv2dBatchNorm` | Conv -> BatchNorm | compatible channel parameters; no observable intermediate |
| `FusedEWChain` | 2–5 adjacent elementwise nodes | single-use internal edges; compatible broadcast/dtype semantics |
| `FusedSoftmaxDropout` | Softmax -> Dropout | inference semantics; Dropout must not be training-active |
| `FusedResidualNorm` | residual Add -> LayerNorm | correct residual inputs, axis, epsilon, scale, and bias |

Each fusion log entry records the pattern, old node IDs, new node ID, removed
tensors, and a rejection reason when a near-match is unsafe. Log ordering and
generated IDs are deterministic.

## Critical BN issue

The public ResNet ONNX has BatchNorm already folded into Conv and therefore contains no BN node. A normal Conv->BN matcher cannot score that pattern.

The specification suggests a pre-fusion pass that reconstructs a merged Conv from BN parameters and Conv weights. However, if the exported graph contains only already-folded Conv weights and no original BN parameters or metadata, the original Conv/BN pair is not uniquely recoverable. Do not invent parameters. This item needs one of:

- evaluator-provided BN metadata/fixture;
- a synthetic microbenchmark graph containing Conv and BN;
- organizer acceptance through code review of a correct Conv+BN folding pass.

When explicit parameters exist, the executor folds Conv weight/bias with BN
scale, bias, mean, and variance from its live CuPy initializer store before
C3.4 plan creation. The release also documents the absence of recoverable BN
parameters in the public ResNet.

## F2/F3: launch and buffer reduction (6 points)

Both metrics reach full credit at 60% reduction. Count from the same lowering model before and after optimization:

```text
launch_reduction = (raw_launches - optimized_launches) / raw_launches
buffer_reduction = (raw_buffers - optimized_buffers) / raw_buffers
```

The default pipeline counts the C3.2 executable lowering for every node and
counts named lowering intermediates as logical buffers. Constant nodes retain
their required metadata `KernelSpecRef` for C3.2 coverage but count as zero
physical launches because both C3.5 executors preload them and return without a
device launch.

The connected single-launch lowerings are:

- MatMul+bias and Gemm+bias with optional absorbed Flatten/Relu epilogues;
- rank-four scaled/masked attention-score MatMul+Softmax;
- single-output LayerNormalization and residual LayerNormalization;
- elementwise chains and inference Softmax+Dropout;
- Transpose+Reshape layout copies.

Conv epilogue graph nodes are intentionally multi-stage in the current release:
im2col, BLAS-backed contraction, optional bias, and one generated Relu or
residual-Add+Relu epilogue. The previous one-thread-per-output direct kernel was
numerically correct but regressed H200 ResNet wall time from about `8.3 s` to
`62.7 s`. Its one-launch metadata and runtime path have been removed. The
generated epilogue still writes directly into the C3.4 planned output view.

The former sequential `FusedExecutionRegion` and `FusedComputeActivation`
reference paths have been removed. They had no single-kernel lowering and
therefore could not truthfully contribute to launch reduction.

The new passes are semantic and topology-driven. They inspect operator types,
shapes, attributes, fan-out, and broadcast compatibility, never graph/model
names or known weights. Unsupported batched-B MatMul epilogues, non-last-axis
attention Softmax, rank-greater-than-four layout regions, and incompatible
residual shapes remain unfused.

## F4: correctness (4 points plus hard numerical gate)

After every pass:

1. Preserve resolvable graph inputs.
2. Preserve resolvable graph outputs.
3. Rebuild producer/consumer indexes.
4. Run acyclic and tensor-reference validation.
5. Require optimized node count not to increase.
6. Execute original and optimized graphs in FP32 and require `max_abs_diff <= 1e-3` against the same reference.

Passes are individually switchable and run transactionally: match, verify
guards, rewrite, validate, then commit or restore the snapshot.

## Safe pass order

1. Constant and shape canonicalization without public-contract changes.
2. Conv+BN folding when both parameter sets are available.
3. MatMul/Gemm+bias fusion.
4. Residual Add+LayerNorm fusion.
5. Softmax+Dropout fusion under inference-only guards.
6. Elementwise-chain fusion.
7. Dead internal tensor/node cleanup.

Run validation after each pass, not only at pipeline completion.

## Validation evidence

- Per-pattern positive and negative tests.
- Before/after node, launch, and buffer counts.
- Complete `fusion_log` entries.
- Original versus optimized graph validation result.
- Original versus optimized versus golden numerical report.

The current dependency-light self-test reports `PASS=59, FAIL=2` and a local
written-rubric structural total of `12.00/15.0`. MLP remains
`66.7%/75.0%` and Transformer remains `63.6%/61.6%` for launches/logical
buffers. ResNet is now truthfully `-22.2%/-22.5%` because the direct timeline
exposes its im2col, contraction, reshape/bias, and epilogue steps. Nine focused
regressions pass,
including explicit checks that Conv epilogues no longer advertise a
single-launch direct kernel. The two self-test failures are the intentionally
visible ResNet 60% anchor deficits.

These are local structural results. Current report 8 passes all three optimized
graphs through direct dispatch on CuPy 14.1.1/H200; ResNet reaches
`max_abs_diff=1.53e-05`, accuracy `0.9351`, and `8.275 s` external wall time.
Physical observed-launch profiling is still absent, so the structural F2/F3
figures remain local diagnostics rather than official evaluator evidence.

## Public design references

The implementation is original repository code. Public material influenced
the design principles, not copied source code:

- TVM, *An Automated End-to-End Optimizing Compiler for Deep Learning*, for
  the separation of graph-level fusion from hardware-specific operator
  lowering: https://arxiv.org/abs/1802.04799
- DNNFusion, *Accelerating Deep Neural Networks Execution with Advanced
  Operator Fusion*, for operator classification, graph rewriting, and bounded
  fusion-plan formation: https://arxiv.org/abs/2108.13342
- MLIR Linalg documentation, for dependency-aware producer/consumer fusion and
  explicit temporary-buffer reasoning:
  https://mlir.llvm.org/docs/Dialects/Linalg/
- NVIDIA CUDA Programming Guide, CUDA Graphs, for the distinction between
  reducing host submission overhead and reducing the number of physical GPU
  kernels: https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html
