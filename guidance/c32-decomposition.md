# C3.2 — Operator decomposition and kernel selection

## Objective and evaluator contract

For every imported node, select a supported precision, lower the high-level operator into a non-empty GPGPU kernel sequence, expose intermediate tensors, and provide valid launch parameters. The evaluator reads only public APIs:

- `import_onnx_graph(model.onnx)`
- `strategy.select_precision(node, graph)`
- `hardware.supported_precisions()`
- `strategy.decompose(node, graph, precision)`
- `strategy.tune_kernel(ref, precision, problem_size)`

Do not rely on hidden state that these calls cannot reproduce.

All decisions must be deterministic and input-general. Do not rotate precisions or emit unused kernel names merely to increase rubric diversity. Precision, kernel, and tuning choices must derive from operator semantics, tensor shapes, requested execution mode, and declared AEC hardware capabilities. Every claimed kernel must map to an executable AEC compiler/runtime path; a string-only `KernelSpecRef` is not proof of implementation.

## D1: precision routing (3 points)

Create an explicit rule table. Force `Softmax`, `LayerNorm`/`LayerNormalization`, `BatchNorm`, `ReduceMax`, `ReduceSum`, and `ReduceMean` to FP32. For `MatMul`, `Linear`/`Gemm`, and `Conv2d`/`Conv`, choose only from `hardware.supported_precisions()`.

The rubric rewards seeing fp32, fp16, fp8, and fp4, but the end-to-end tolerance is strict. Separate two concepts:

- capability/strategy coverage for the microbenchmark;
- an accuracy-qualified deployment policy for C3.5.

Always retain `FULL_FP32`, which must meet `max_abs_diff <= 1e-3` and `top1_match >= 0.99`.

### Implemented mixed-precision rule table

| Kernel/operator class | Preferred precision | Engineering guard |
|---|---|---|
| Softmax, LogSoftmax, LayerNorm, BatchNorm, reductions | FP32 | Numerically sensitive statistics and exponentials |
| General/spatial Conv | FP16 | FP32 accumulation/output; avoids aggressive quantization of 3x3 transforms |
| Aligned 1x1 Conv | FP8 | Input/output channels are concrete multiples of 16 |
| Small or unaligned MatMul/Gemm | FP16 | Safe lower-precision baseline |
| Aligned MatMul/Gemm | FP8 | `K,N >= 64` and both multiples of 16 |
| Large constant-weight MatMul/Gemm | W4A16 | `K,N >= 128`, multiples of 32, and `K*N >= 32768`; FP4 weight, FP16 activation, FP32 accumulation/output |
| Other operators | FP32 | Preserve semantics until a qualified lower-precision implementation exists |

If the preferred precision is unsupported, route deterministically toward safer available types (`fp4 -> fp8 -> fp16/bf16 -> fp32`, for example). No decision uses graph/model names, evaluator identity, hashes, call counts, or mutable coverage state. The public evaluator-facing strategy uses this mixed policy; C3.4 and deployment explicitly construct `FULL_FP32` strategies.

## D2: kernel sequence completeness (3 points)

Required recognizable sequences include:

| Operator | Sequence expectation |
|---|---|
| MatMul/Linear/Gemm | at least one `matmul_*` kernel |
| Softmax | `reduce_max` -> `exp` -> `reduce_sum` -> `div` |
| LayerNorm | `reduce_mean` -> `sub` -> `mul` -> `sqrt`, plus normalization/affine work as needed |
| 3x3 Conv | `winograd_forward_*` or `im2col_*` path |

Every node should return a non-empty sequence, including view-like or constant operators. If an operation requires no GPU work, represent it with an explicit metadata/alias kernel only if the evaluator accepts that convention; otherwise lower it to a concrete copy/index/update operation.

## D3: intermediate tensors (3 points)

Every multi-kernel lowering must name its internal results in `KernelSpecRef.outputs`. Use deterministic names such as `__c3_inter_<node_id>_<index>__`, and ensure at least one output differs from the original node outputs for key decompositions.

Track full tensor metadata: shape, dtype, byte size, producer kernel, consuming kernels, and last use. These tensors become the input to C3.4 memory planning.

## D4: tuning validity (3 points)

Target tuning coverage above 90%. Fill all required fields for every tunable kernel:

- `0 < block_x <= hardware.max_threads_per_block`
- `grid_x > 0`
- `smem_bytes <= hardware.smem_bytes`, or `-1` if the benchmark defines that sentinel as compliant

Use guarded integer formulas, for example `grid_x = max(1, ceil_div(work_items, block_x))`. Never allow an empty/dynamic problem size to produce zero.

## D5: hardware coverage (3 points)

Expose at least fp32 and fp16 GEMM kernels, and fp8/fp4 variants where hardware claims support. Make Conv selection observable:

- Winograd for suitable 3x3, stride-1 cases when supported and profitable.
- im2col for general shapes, strides, or unsupported Winograd conditions.

The hardware object must be the source of truth for precision, thread, shared-memory, and kernel capability decisions.

Do not hardcode a high-end GPU profile to claim capabilities. Load the fixed evaluator/AEC target profile through documented configuration or device query, validate it, and fail clearly when a requested precision or kernel is unavailable.

## Implementation order

1. Implement a safe non-empty lowering for all 17 public operators.
2. Implement complete MatMul/Gemm and Conv paths.
3. Implement the exact Softmax and LayerNorm decompositions with named intermediates.
4. Add generic, always-valid tuning defaults.
5. Add hardware-aware precision and specialized tuning.
6. Add fp8/fp4 coverage only after FP32 correctness is locked.

## Acceptance evidence

- Per-node report: selected precision, supported-precision intersection, kernel sequence, intermediates, and tuning.
- Coverage summary for D1–D5 using the rubric formulas.
- Explicit assertions for every tuning parameter.
- Numerical comparison of the FULL_FP32 decomposition against the reference.
- Evidence that every emitted kernel name resolves to submitted source and executes through the AEC path.
- Repeated-call determinism checks for precision, decomposition, and tuning.
- Offline dependency and originality disclosures for any kernel library or generated implementation.

Current structural evidence: all 13 released sensitive nodes select FP32; all 49 tunable nodes select a declared-supported precision; fp32/fp16/fp8/fp4 all appear; and five independent policy regressions pass. This is routing/decomposition evidence, not proof that FP8/FP4 AEC kernels meet end-to-end accuracy.

## Open dependency

The released package does not include `bench_c32_c33.py` or the concrete class definitions. Confirm exact imports, enum/string spellings (`LayerNorm` versus `LayerNormalization`, `Conv2d` versus `Conv`), and object schemas with the organizer before considering this interface complete.
