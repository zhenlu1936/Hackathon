# C3 mission guidance

## Governing requirements

Read `specification/general_requirements.md` before the C3-specific specification. The general requirements govern originality, evaluation integrity, dependencies, reproducibility, and the offline environment; `specification/spec.md` and `specification/scoring.md` govern the C3 interfaces and points. When they appear to conflict, do not silently choose the easier interpretation—record the conflict and obtain organizer clarification.

The submission must satisfy these non-negotiable constraints:

- Generate every evaluated artifact from submitted source during evaluation. Do not submit precomputed outputs, precompiled binaries, caches, or generated answers.
- Use general algorithms for all legal inputs. Never branch on test IDs, model filenames, input hashes, known weights, hidden-case markers, or fixed public data.
- Do not manipulate or bypass evaluator/result scripts or official infrastructure.
- Run without network access and package every dependency for the exact target in `specification/environments.txt`: Linux 6.8.0-110 x86_64, Python 3.12.3, GCC/G++ 13.3.0, nvcc 12.8.61, driver 580.126.20, ONNX 1.22.0, ONNX Runtime 1.27.0, Torch 2.13.0+cu130, and CuPy 14.1.1.
- Disclose every third-party library, open-source component, academic reference, and LLM-assisted contribution with version, license, purpose, and call boundary.
- Keep the implementation explainable and maintainable by the team.
- Target the AEC GPGPU software stack. A mock or generic-GPU abstraction may support development, but cannot be presented as completed AEC execution unless it is connected to the required AEC compiler/runtime/device path.

## 1. Mission in one sentence

Build an AEC GPGPU inference compiler/runtime that accepts an ONNX model, turns it into a validated internal DAG, lowers and optimizes that graph for the available AEC target, produces an executable memory-and-stream schedule, and runs all three target model families with FP32-reference accuracy.

This is not five independent programs. C3.1–C3.4 should be stages of the same pipeline used by C3.5:

```text
ONNX
  -> import and canonical graph IR                  (C3.1)
  -> precision policy and kernel decomposition      (C3.2)
  -> graph fusion and simplification                (C3.3)
  -> tensor lifetimes, allocation and stream plan   (C3.4)
  -> GPU execution, batching and output             (C3.5)
```

## 2. Score and priority

| Sub-mission | Points | Primary risk | Recommended priority |
|---|---:|---|---:|
| C3.1 Graph parsing | 10 | malformed or incomplete IR | 1 |
| C3.2 Lowering/kernel selection | 15 | evaluator-facing API mismatch | 3 |
| C3.3 Fusion/graph optimization | 15 | graph corruption or numerical drift | 4 |
| C3.4 Memory/scheduling | 10 | disconnected review-only stubs | 5 |
| C3.5 End-to-end deployment | 50 | correctness gate makes performance irrelevant | 2, then 6 |

The recommended sequence is intentional: first establish the common IR (C3.1), then obtain a correct FP32 end-to-end path (C3.5 baseline), and only then introduce lowering, fusion, memory reuse, and performance work.

## 3. Architecture boundary

Use a single internal contract shared by all stages:

- `Graph`: named inputs, outputs, initializers, nodes, tensors, producer/consumer maps, topological order.
- `Node`: stable ID, ONNX op type, attributes, input/output tensor names.
- `Tensor`: dtype, symbolic/concrete shape, initializer flag, size when known, producer, consumers.
- `PrecisionProfile`: selected compute, accumulator, input, and output precision.
- `KernelSpecRef`: kernel name, inputs, outputs, workspace needs, launch constraints.
- `ExecutionPlan`: allocations, copies, kernels, dependencies, stream IDs, output copies.

Never make the exported C3.1 JSON the only IR. It is a report format; the compiler needs richer shape, attribute, initializer, lifetime, and dependency information.

## 4. Cross-mission invariants

Every stage should preserve these invariants:

1. Every non-input/non-initializer tensor has exactly one producer.
2. Every node input resolves to a graph input, initializer, constant, or earlier node output.
3. The graph remains acyclic and topologically sortable.
4. Graph input/output names and ordering remain stable unless an explicit output rewrite updates the public contract.
5. Dynamic batch `N` remains dynamic through import and is concretized only for a run/batch.
6. Optimizations can always be disabled to recover the known-correct FP32 path.
7. Output is always `float32`, covers all samples, and preserves sample order.
8. Decisions are deterministic functions of model semantics, tensor shapes, requested mode, and declared hardware capabilities—not evaluator call order or scoring counters.
9. Every reported kernel, fusion, memory optimization, and overlap corresponds to an executable path; capability advertisements and log-only placeholders are not evidence of implementation.
10. Runtime code performs no network access and does not load undeclared artifacts outside the submission and evaluator-provided inputs.

## 5. Released evidence and limitations

The package contains three public ONNX models and C3.5 data for 10,000 samples per model:

- MLP: input `[10000, 1, 28, 28]`, output `[10000, 10]`.
- ResNet: input `[10000, 3, 32, 32]`, output `[10000, 10]`.
- Transformer: input IDs `[10000, 18]`, output `[10000, 18, 14]`.

The release is explicitly a C3.1/C3.5 competitor package. The referenced C3.2/C3.3 benchmark (`benchmarks/c32_c33/bench_c32_c33.py`) and its API definitions are not included. Therefore, exact Python signatures, class import paths, serialization details, and hidden microbenchmark cases cannot be confirmed from the released files. Treat the names in the specification as required interfaces, but obtain the benchmark or organizer clarification before freezing package layout.

## 6. Source conflicts requiring confirmation

- Written `spec.md` and `scoring.md`: C3.5 is accuracy/precision 15, runtime 25, peak memory 10.
- `requirements.png`: its table says peak memory 15, which would make the displayed C3.5 components sum to 55 rather than 50.

Planning should use the internally consistent written allocation (15 + 25 + 10 = 50) until organizers say otherwise.

## 7. Recommended delivery phases

### Phase A: correctness spine

- Parse all three public models into one validated IR.
- Implement all 17 required ONNX operators in FP32.
- Run arbitrary batch sizes and write exact manifest/NPY output.
- Pass `numpy.allclose(rtol=1e-3, atol=1e-3)` on all public outputs.

### Phase B: evaluator contracts

- Export DAG JSON for C3.1.
- Implement the public strategy and hardware APIs for C3.2.
- Add pass pipeline and machine-readable fusion logs for C3.3.
- Make memory and stream decisions visible in a reviewable execution plan.

### Phase C: safe optimization

- Add per-operator precision selection with FP32 fallbacks.
- Add decomposition and valid tuning parameters.
- Add fusion one pattern at a time with before/after numerical checks.
- Add lifetime reuse and pooling without changing tensor semantics.

### Phase D: ranked performance

- Measure process startup, model load, H2D, kernels, D2H, and output I/O separately.
- Tune batch size per model under the precision and memory gates.
- Add asynchronous prefetch and independent-stream scheduling only where dependency analysis proves safety.

## 8. Guidance files

- [Remaining problems and completion gates](remaining-problems.md)
- [C3.1 graph parsing](c31-graph-parsing.md)
- [C3.2 decomposition and kernel selection](c32-decomposition.md)
- [C3.3 fusion and graph optimization](c33-fusion.md)
- [C3.4 memory planning and scheduling](c34-memory-scheduling.md)
- [C3.5 end-to-end deployment](c35-deployment.md)
- [Validation and submission checklist](validation-checklist.md)
