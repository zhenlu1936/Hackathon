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

The release implements standard Conv+BN behavior when parameters exist and
documents the absence of recoverable BN parameters in the public ResNet.

## F2/F3: launch and buffer reduction (6 points)

Both metrics reach full credit at 60% reduction. Count from the same lowering model before and after optimization:

```text
launch_reduction = (raw_launches - optimized_launches) / raw_launches
buffer_reduction = (raw_buffers - optimized_buffers) / raw_buffers
```

The structural pass removes internal materialization only from the optimized
graph. Launch reduction is counted through C3.2 decomposition rather than from
fused node names alone.

High-value opportunities in the released models include Gemm+bias, residual Add patterns, Transformer elementwise GELU chains, and residual Add+LayerNormalization. Softmax+Dropout may only appear in a training-style benchmark, because inference exports often omit or neutralize Dropout.

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
