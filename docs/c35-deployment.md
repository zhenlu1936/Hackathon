# C3.5 — Typical model deployment

## Release contract and scoring

C3.5 is half the contest score. Each model must pass strict numerical comparison and, for classifiers, accuracy before runtime or memory ranking matters. The written rules allocate 15 points to correctness/accuracy, 25 to total runtime, and 10 to peak GPU memory.

The evaluator times the whole process from startup to exit and samples per-process GPU memory through NVML, including child processes. Startup, imports, model parsing, compilation, file I/O, and teardown therefore all matter.

Inference runs exclusively through native CuPy/CUDA on the remote H200, which
is the designated AEC device for this release. No CPU array fallback is part of
the framework. The timed command performs no network access and generates
outputs from supplied inputs.

The released implementation uses CuPy/CUDA as its only connected H200
backend. It applies C3.3 to the shared graph, builds the C3.4 plan through C3.2
decomposition, validates bindings/events and plan/graph identity, and executes
nodes in planned order. Weights and batches move to CuPy once and final outputs
return to the host once. Numerical qualification remains an internal test path
and is not exposed through the evaluator-facing CLI.
The CLI exposes no backend selector; CuPy is the mandatory implementation.

## CLI and data contract

The deployment CLI is:

```text
<program> --onnx MODEL --input INPUT_DIR --output OUTPUT_DIR [--batch-size N]
```

Read `manifest.json`, load every named `.npy`, bind by tensor name, preserve sample order across batches, and write `manifest.json` plus output arrays. All three models output `logits` as `float32`.

The hidden model has the same structure as the public version but different weights. Never match behavior by filename, node name, or known weight value.

Also never select precomputed outputs by test ID, file hash, input hash, sample count, or hidden-case marker. Do not modify or bypass the evaluator/result script.

## Required operator coverage

| Model | Operators |
|---|---|
| MLP | Flatten, Gemm, Relu |
| ResNet | Conv, Relu, Add, GlobalAveragePool, Flatten, Gemm |
| Transformer | Gather, Add, LayerNormalization, MatMul, Constant, Split, Reshape, Transpose, Div, Softmax, Erf, Mul |

Implement ONNX opset-17 semantics, including broadcasting, axes, transposition, Gemm alpha/beta/trans flags, Conv padding/stride/group, LayerNormalization epsilon/axis, Split sizes/axis, and Reshape special values. Operator-name coverage without attribute correctness will fail hidden weights or shapes.

## Correctness gates

- All models: `cupy.allclose(out, golden, rtol=1e-3, atol=1e-3)`.
- MLP: top-1 accuracy at least 98%.
- ResNet: top-1 accuracy at least 85%.
- Transformer: no classification threshold, but the numerical gate still applies.

The release uses FP32 operator computation and disables unqualified deployment
precision changes. Optimizations must pass the same golden thresholds before
becoming part of the default path.

## Batch execution

Treat `--batch-size` as a maximum processing chunk:

1. Validate it is positive.
2. Slice all inputs on dimension 0 with identical boundaries.
3. Concretize dynamic `N` for that chunk.
4. Execute and collect outputs.
5. Concatenate in original order.
6. Verify the final leading dimension equals input `N`.

Do not require 10,000 samples or a particular batch size. Test `N` smaller than, equal to, and not divisible by the requested batch size.

## Performance characteristics

The implementation is organized around these performance constraints:

1. Eliminate repeated model parsing, compilation, allocation, and weight uploads inside the batch loop.
2. Use pinned host buffers and asynchronous copies where they improve end-to-end time.
3. Tune GPU kernels for the actual shapes and batch regime.
4. Fuse high-frequency compatible chains.
5. Reuse intermediate memory and release temporary compiler/runtime objects before timed steady work when the evaluation protocol permits.
6. Reduce Python/process startup and avoid child processes unless necessary.
7. Tune batch size against both throughput and peak memory.

Because scoring uses process wall time, a faster kernel that adds expensive just-in-time compilation can lose overall. Cache only if the evaluation environment and command lifecycle allow it; assume no network.

## Model-specific focus

### MLP

Gemm dominates. Fuse bias and activation where possible. Large batch chunks improve GEMM utilization, but initialization overhead can dominate this small model.

### ResNet

3x3 Conv dominates. Support both Winograd and im2col/direct alternatives, preserve residual Add semantics, and verify FP32 closely because accumulated low-precision error can exceed `1e-3`. BatchNorm is already folded into Conv weights.

### Transformer

Pay close attention to layout-changing Reshape/Transpose/Split, causal-mask broadcasting, stable FP32 Softmax, LayerNormalization, and int64 Gather indices. Fuse elementwise GELU and residual/norm regions only after exact shape and broadcast validation.

## Output validation

Before exit, validate:

- output directory exists or was created;
- output manifest names, file names, dtype, and shapes match actual arrays;
- every output is finite unless the golden output legitimately contains non-finite values;
- output is C-contiguous or saved correctly by the NPY writer;
- sample count and ordering match input.

## Main risks

- Precision optimizations pass accuracy but fail strict elementwise allclose.
- GPU context/library startup dominates small-model timing.
- Excessive batch size improves throughput but loses memory ranking.
- A hidden weight set exposes hardcoded shapes, names, or constant assumptions.
- Output file generation becomes a significant part of measured wall time, especially Transformer logits.
- An undeclared CPU fallback makes results accurate but bypasses the required H200 execution path.
- Build/runtime dependencies are unavailable because evaluation has no network access.

## Standards-oriented execution

Use `python3 -m c35.standard_runner` or `./run_c35.sh` for black-box validation
against released manifests, per-model thresholds, cold wall time, and GPU-memory
evidence. `c35/test_c35.py` is the local development regression suite and is not a
replica of the unreleased organizer evaluator. See
[C3.5 standard black-box runner](c35-standard-runner.md).

On the GPU server, the normal workflow is one command:

```bash
./run_c35.sh
```

It checks release-data presence, performs a real CuPy device smoke test, runs all three models in fresh subprocesses, samples memory using the server-native `nvidia-smi` command or a process-local CuPy-pool fallback on restrictive MIG systems, applies golden/accuracy gates, and writes `c35-standard-report.json`. Set `PYTHON`, `C35_REPORT`, or `COMMAND_TEMPLATE` only when overriding those defaults.
