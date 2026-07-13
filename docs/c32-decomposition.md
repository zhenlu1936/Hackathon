# C3.2 — Operator decomposition and kernel selection

## Release contract

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

`Strategy.decompose` now validates each sequence before returning it: every
input must be a node input or a previously produced intermediate, no tensor may
be produced twice, and every non-empty node output must be produced. The
selected `PrecisionProfile` is attached to each `KernelSpecRef` so downstream
planning does not infer precision from kernel-name spelling.

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

`HardwareCapability.query_cupy_device()` and
`c32.api.activate_cupy_hardware()` provide the device-query path. CUDA resource
limits and compute capability are read from CuPy, the resulting profile is
marked with its source, and Hopper discovery conservatively leaves FP4 disabled
unless a separately executable and qualified AEC FP4/W4A16 path is explicitly
declared. The static four-precision object remains a microbenchmark coverage
profile and is intentionally marked `verified=False`; it is not H200 evidence.

## Implementation coverage

The release includes non-empty lowering for all 17 public operators, aliases
for `Linear`, `Conv2d`, and `LayerNorm`, connected multi-output LayerNorm,
complete MatMul/Gemm and Conv paths, named Softmax and normalization
intermediates, generic valid tuning defaults, and hardware-aware mixed-precision
selection. Conv selects Winograd only for group-one, unit-stride, unit-dilation
3x3 cases and otherwise uses im2col. FP8/FP4 coverage remains structural until
executable AEC kernels are connected and numerically qualified.

C3.3 optimized nodes now also have explicit one-reference decompositions for
Gemm epilogues, Conv activation/residual epilogues, attention scores,
single-output LayerNormalization, and Transpose+Reshape. These additions do not
alter the public unfused C3.2 sequences used for D2/D3 scoring; they let C3.3
launch and buffer accounting consume a connected lowering instead of an opaque
fallback label.

The current working revision dispatches released-model references through
`c32.kernel_registry` and C3.4 arena bindings. Report 8 proves the optimized
MLP, ResNet-18, and Transformer plans pass on H200, and unsupported names fail
closed. Most registry functions do not yet consume `tuning_params`, and the
unfused Winograd path lacks equivalent target qualification.

## Validation evidence

- Per-node report: selected precision, supported-precision intersection, kernel sequence, intermediates, and tuning.
- Coverage summary for D1–D5 using the rubric formulas.
- Explicit assertions for every tuning parameter.
- Numerical comparison of the FULL_FP32 decomposition against the reference.
- Evidence that every emitted kernel name resolves to submitted source and executes through the AEC path.
- Repeated-call determinism checks for precision, decomposition, and tuning.
- Native-server dependency and originality disclosures for any kernel library or generated implementation.

Current structural evidence: all 13 released sensitive nodes select FP32; all 49 tunable nodes select a declared-supported precision; fp32/fp16/fp8/fp4 all appear; and five independent policy regressions pass. Seven contract regressions, two Conv layout regressions, and one fail-closed registry regression pass. On the AEC H200, the targeted ResNet test and all three black-box models pass through direct dispatch in reports 5 and 8. This does not qualify FP8/FP4 or every unfused microbenchmark path.

## Open dependency

The released package does not include `bench_c32_c33.py` or the concrete class definitions. Confirm exact imports, enum/string spellings (`LayerNorm` versus `LayerNormalization`, `Conv2d` versus `Conv`), and object schemas with the organizer before considering this interface complete.
