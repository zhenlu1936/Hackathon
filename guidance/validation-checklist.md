# Validation and submission checklist

## General compliance gate

- [ ] Every result is generated from submitted source during evaluation; no precomputed outputs, precompiled binaries, bytecode caches, generated plans, or result artifacts are submitted.
- [ ] Runtime behavior is independent of test ID, filename, graph name, input/model hash, known weight values, and public-data constants.
- [ ] No evaluator/result script or official component is modified, shadowed, intercepted, or bypassed.
- [ ] The complete build and run succeed with networking disabled on the exact `specification/environments.txt` Linux x86_64/CUDA/Python package stack.
- [ ] `python environment/verify_environment.py --target` passes on the evaluation-class NVIDIA host.
- [ ] Every third-party dependency records name, version, license, purpose, and integration boundary.
- [ ] Open-source code and academic algorithms are attributed.
- [ ] LLM assistance is disclosed in the originality declaration, and the team can explain and maintain the generated code.
- [ ] Every claimed C3 kernel/runtime feature is connected to the AEC GPGPU path; reference/mock backends are clearly labeled and never silently substituted.
- [ ] Source archives exclude `__pycache__`, `.pyc`, `.DS_Store`, generated outputs, local caches, and development-only binaries.

## Correctness matrix

Run every relevant check for all three public models and several batch sizes.

| Check | MLP | ResNet | Transformer |
|---|---|---|---|
| ONNX import and graph validation | required | required | required |
| deterministic DAG JSON | required | required | required |
| FP32 inference allclose | required | required | required |
| accuracy threshold | >= 0.98 | >= 0.85 | n/a |
| arbitrary/dynamic batch | required | required | required |
| output manifest and float32 logits | required | required | required |

Suggested batch cases: `1`, a small non-power-of-two value, `256`, a value larger than the dataset, and a value that does not divide the sample count.

## C3.1 checklist

- [ ] CLI placeholders `{onnx}` and `{output}` work from arbitrary absolute paths.
- [ ] Successful run exits 0 and writes valid JSON to exactly `--output`.
- [ ] Initializers are excluded from graph inputs.
- [ ] Fan-out, multiple outputs, constants, empty optional inputs, and symbolic shapes are handled.
- [ ] Edge tensor/source/destination consistency and acyclicity are checked.

## C3.2 checklist

- [ ] Public API names and schemas match the organizer benchmark.
- [ ] Sensitive operations route to FP32.
- [ ] FULL_FP32 meets numerical and top-1 requirements.
- [ ] Every node returns a non-empty sequence.
- [ ] Required kernel-name prefixes and sequences are present.
- [ ] Key multi-kernel operators expose named intermediates.
- [ ] Tuning coverage is at least 90%.
- [ ] Every block/grid/shared-memory assertion passes.
- [ ] Precision and Conv strategy choices are hardware-capability driven.
- [ ] Repeating the same API call with the same inputs returns the same decision.
- [ ] Capability reports match kernels actually emitted and executable on the declared AEC target.
- [ ] No scoring counter, round-robin state, or benchmark-call order affects precision or kernel selection.

## C3.3 checklist

- [ ] Fusion pipeline can be enabled/disabled.
- [ ] All five patterns have implemented matchers and safety guards.
- [ ] Released-model BN limitation is documented and organizer handling is confirmed.
- [ ] Fusion log uses the exact expected location and stable pattern names.
- [ ] Launch and buffer reductions are computed from executable plans.
- [ ] Inputs/outputs remain resolvable and graph validation passes after every pass.
- [ ] Original and optimized FP32 outputs both meet `max_abs_diff <= 1e-3`.

## C3.4 checklist

- [ ] Device allocations, weight H2D, and kernel consumption form one traceable chain.
- [ ] Lifetime analysis changes actual physical slot assignment.
- [ ] Free blocks are reused with a named fragmentation-management policy.
- [ ] Weight prefetch overlaps earlier compute rather than bulk-uploading first.
- [ ] Independent kernels use multiple compute streams with explicit dependencies.
- [ ] A human-readable plan/timeline proves the five features for code review.

## C3.5 checklist

- [ ] No preprocessing is applied to already-preprocessed input.
- [ ] All 17 ops implement opset-17 attributes and broadcasting semantics.
- [ ] Input is bound by manifest/model tensor name, not file order.
- [ ] Output is `logits`, float32, complete, ordered, and accurately described by its manifest.
- [ ] Low-precision or fusion changes are individually accuracy-qualified.
- [ ] Runtime measurement includes cold process startup and file output.
- [ ] Peak GPU memory is sampled per process and children during local profiling.
- [ ] Hidden models with identical structure but different weights require no code changes.

## Submission package

- [ ] Source code and reproducible offline build instructions.
- [ ] Runtime dependencies available without network access.
- [ ] C3.1 command template with `{onnx}` and `{output}`.
- [ ] C3.5 command template with `{onnx}`, `{input}`, and `{output}`; optional batch size works.
- [ ] One-command public-model self-test.
- [ ] Machine-readable correctness and performance report.
- [ ] No absolute development-machine paths, public-model filenames, or embedded golden outputs.
- [ ] Third-party dependency manifest and originality/LLM disclosure.
- [ ] Clean-source check proves no bytecode, cache, prebuilt, or generated-result files are included.

## Organizer questions to close early

1. Which C3.5 memory allocation is authoritative: 10 points in written rules or 15 in the image?
2. Where are the C3.2/C3.3 benchmark and exact API class definitions?
3. How is `FusedConv2dBatchNorm` tested when released ResNet contains no BN node or original BN parameters?
4. Are no-op/view kernels acceptable as non-empty decompositions?
5. What exact GPU model, driver, supported precision set, and shared-memory/thread limits are used?
6. Are build time, one-time compilation, and reusable on-disk caches included or permitted in the timed run?
7. Is each model launched in a fresh process, and is warm-up included in measured time?
