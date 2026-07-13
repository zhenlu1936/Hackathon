# Release validation checklist

Use this checklist before tagging a release or preparing a competition
submission. A structural test does not substitute for execution on the artifact
and backend evaluated by the organizer.

## Environment and dependency gate

- [ ] Python and native package versions match `.specification/environments.txt`.
- [ ] `python3 -c 'import onnx, cupy; print(onnx.__version__, cupy.__version__)'` succeeds on the target server.
- [ ] CuPy reports at least one visible CUDA device.
- [ ] No runtime path installs, downloads, vendors, or bootstraps dependencies.
- [ ] Every imported third-party package is natively available on the server and
      records version, license, purpose, and call boundary.
- [ ] Runtime validation succeeds with networking disabled.

## Integrity gate

- [ ] No code, design, or documentation is copied from another team; every
      public academic or open-source influence is disclosed.
- [ ] Runtime behavior is independent of test IDs, filenames, graph names,
      model/input hashes, known weights, fixed public data, and sample-specific
      constants.
- [ ] No evaluator, result script, or official component is modified,
      intercepted, shadowed, replaced, or bypassed.
- [ ] Every evaluated result is generated from submitted source at evaluation
      time; no precomputed outputs, netlists, binaries, cached plans, bytecode,
      or generated answers are shipped.
- [ ] No optimization reads or infers hidden-case identifiers; every strategy is
      a general function of model semantics, shapes, documented configuration,
      and verified H200 capabilities.
- [ ] All third-party code, libraries, tools, academic references, and AI/LLM
      assistance are disclosed with version, license, purpose, and call boundary
      where applicable.
- [ ] CuPy execution is performed on the designated remote H200 AEC device and
      no CPU array fallback exists.

## C3.1 graph parsing

- [ ] `--onnx` and `--output` work with absolute paths.
- [ ] The command exits zero and writes valid JSON exactly to `--output`.
- [ ] Initializers are excluded from graph inputs.
- [ ] Symbolic shapes, fan-out, multiple outputs, constants, optional empty
      inputs, and duplicate/empty node names are handled.
- [ ] Every edge connects a producer output to a consumer input.
- [ ] Repeated export is deterministic.

## C3.2 decomposition and tuning

- [ ] Public API names and schemas match the organizer benchmark.
- [ ] Sensitive operations always select FP32.
- [ ] `FULL_FP32` satisfies the numerical and top-1 requirements.
- [ ] Every graph node returns a non-empty sequence.
- [ ] Required MatMul, Softmax, LayerNorm, and Conv sequence names are present.
- [ ] Multi-kernel operators expose deterministic intermediate tensors.
- [ ] Tuning coverage is at least 90%.
- [ ] Every block, grid, and shared-memory assertion passes.
- [ ] Repeated calls with identical inputs return identical decisions.
- [ ] Every advertised capability and emitted kernel is executable on the
      declared AEC target.

## C3.3 fusion

- [ ] Fusion can be enabled and disabled.
- [ ] All five target patterns have positive and guarded negative tests.
- [ ] Fusion logs use the evaluator-required path and stable pattern names.
- [ ] Producer/consumer indexes and topological order validate after each pass.
- [ ] Graph inputs and outputs remain resolvable.
- [ ] Original and optimized FP32 outputs satisfy `max_abs_diff <= 1e-3`.
- [ ] Launch and buffer reductions are counted from executable lowerings.
- [ ] The folded-BN release limitation is disclosed.

## C3.4 memory and scheduling

- [ ] Every logical kernel input/output has a physical allocation binding.
- [ ] Inputs and weights have H2D transfers and readiness events.
- [ ] Non-overlapping intermediate lifetimes reuse physical slots.
- [ ] The pool demonstrates a named fragmentation-management policy.
- [ ] Weight prefetch is ordered near earlier compute, not bulk uploaded only.
- [ ] Independent kernels use multiple streams with explicit dependencies where
      the graph permits concurrency.
- [ ] One reviewable plan contains allocations, transfers, kernels, events,
      lifetimes, reuse decisions, and peak planned memory.

## C3.5 deployment

- [ ] All 17 published operators implement required opset-17 semantics.
- [ ] Inputs are bound by manifest/model tensor name, not file order.
- [ ] Batch sizes `1`, a small prime, `256`, a non-divisor, and a value larger
      than the dataset complete without ordering or truncation errors.
- [ ] Output is complete, ordered, C-contiguous float32 `logits` with an accurate
      manifest.
- [ ] MLP, ResNet-18, and Transformer all satisfy
      `cupy.allclose(rtol=1e-3, atol=1e-3)`.
- [ ] MLP accuracy is at least `0.98`; ResNet-18 accuracy is at least `0.85`.
- [ ] Cold wall time includes startup, parsing, computation, and output I/O.
- [ ] GPU memory source is recorded; `cupy-pool` is identified as a proxy rather
      than an NVML-equivalent measurement.
- [ ] Hidden models with the published structure and different weights require
      no code changes.

## Commands

```bash
python3 -m unittest -q c31.test_c31
python3 -m c32.test_c32
python3 -m c33.test_c33
python3 -m c34.test_c34
python3 -m unittest -v c35.test_c35 c35.test_cross_stage
python3 -m unittest -v c3common.test_scoring_regressions
./run_c35.sh
```

## Package cleanliness

- [ ] Root `README.md` and every file under `docs/` match the released CLI and
      directory layout.
- [ ] The archive excludes `.venv`, `.ssh`, `.DS_Store`, `__pycache__`, `.pyc`,
      generated reports, local caches, and development-only binaries.
- [ ] C3.1 and C3.5 command templates are included in submission instructions.
- [ ] Remaining limitations and organizer questions are current.
