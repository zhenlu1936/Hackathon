---
name: c3-track
description: Analyze, design, implement, optimize, validate, or compliance-audit the C3 AEC GPGPU operator scheduling and model deployment mission. Use for ONNX graph import, DAG export, precision selection, AEC kernel decomposition and tuning, graph fusion, memory planning, stream scheduling, end-to-end inference, offline reproducibility, originality, anti-hardcoding, evaluator-integrity, dependency disclosure, or submission-cleanliness work for the released MLP, ResNet-18, and decoder-only Transformer models.
---

# C3 Track

## Use repository sources in this order

1. Read `.specification/general_requirements.md` for integrity, reproducibility, dependency, disclosure, offline, and AEC-stack constraints.
2. Read `.specification/environments.txt` and verify native module versions directly on the remote server before making environment-parity claims.
3. Read `.specification/spec.md` and `.specification/scoring.md` before making C3 interface or scoring claims.
4. Read the relevant file under `docs/` for the current sub-mission.
5. Read `docs/remaining-problems.md` before claiming a feature, sub-mission, or submission complete.
6. Inspect `.specification/testcases/release_to_competitors/` for public models, manifests, thresholds, and interface examples.
7. Treat written requirements as authoritative over summaries or images. Explicitly report conflicts between general and C3-specific requirements.

Do not assume the missing C3.2/C3.3 benchmark API beyond the names stated in the specification. Ask for or flag missing evaluator definitions before freezing module paths, enums, or object schemas.

## Apply the integrity gate to every revision and audit

Before changing code, documentation, tests, runners, plans, or generated
artifacts—and again before delivering any review or audit—check all six rules:

1. **No plagiarism.** Do not copy another team's code, design, or documentation. Disclose every public academic or open-source source that influences the work.
2. **No hardcoding.** Do not derive answers from test IDs, filenames, input/model hashes, fixed public data, known weights, or sample-specific constants. Produce outputs through general algorithms for every legal input.
3. **No evaluator bypass.** Do not bypass, intercept, shadow, replace, or modify result scripts, evaluator components, or official infrastructure.
4. **No precomputed submissions.** Do not submit precomputed outputs, netlists, plans, caches, precompiled binaries, or other generated answers. Build every evaluated artifact from submitted source in the evaluation environment.
5. **No hidden-case targeting.** Do not read or infer hidden-test identifiers to select special behavior. Optimizations must follow general model semantics, shapes, hardware capabilities, and documented configuration.
6. **No undisclosed dependencies or assistance.** Disclose all third-party code and libraries, academic references, tools, and AI/LLM-assisted contributions. Record version, license, purpose, and call boundary where applicable.

Treat a violation or unresolved suspicion as a release blocker. During an audit,
inspect the actual diff and execution path rather than accepting comments or
self-reported scores. Before delivery, report the integrity-gate result and any
required disclosure updates.

## Synchronize release records after every revision check

After every implementation revision, documentation revision, validation run,
or audit, update both `docs/fix-summary.md` and
`docs/remaining-problems.md` before delivery. This applies even when the check
is read-only or finds no new fix.

- In `fix-summary.md`, record what was inspected or changed, the evidence that
  was actually obtained, the backend/device used, and the integrity-gate
  result. Never promote a structural self-score to official score evidence.
- In `remaining-problems.md`, close only items proven by the organizer-facing
  artifact and backend. Add newly discovered blockers and reopen stale items
  when evidence or disclosure is incomplete.
- Reconcile duplicated issue numbers and status statements between the two
  documents. A problem cannot be both open and resolved.
- Keep structural plan evidence, H200/CuPy execution, and official-evaluator
  evidence explicitly distinct.
- Run a final consistency search across both documents and relevant disclosure
  files before reporting the revision or audit complete.

## Enforce general competition rules

- Generate all evaluated outputs from submitted source during offline evaluation. Reject precomputed outputs, precompiled binaries, cached plans, bytecode, and generated answers from the submission package.
- Never branch on test IDs, filenames, graph names, model/input hashes, known weights, fixed public data, or hidden-case markers.
- Never modify, intercept, or bypass evaluator/result scripts or official infrastructure.
- Use only disclosed third-party components. Record version, license, purpose, and call boundary; disclose academic references and LLM-assisted code.
- Ensure the team can explain and maintain all generated code.
- Use only the standard library and third-party modules provided natively by the remote evaluation server. Do not rely on downloading, installing, vendoring, or offline-packaging additional dependencies.
- Treat the remote evaluation server's H200 GPU as the designated AEC device and CuPy/CUDA as the only numerical framework path. Do not add a CPU array fallback; distinguish real H200 execution from plan metadata that is not consumed by a launch path.

## Preserve documentation and validation evidence during cleanup

- Treat every tracked file under `docs/` as durable release, audit, or
  validation evidence. General cleanup, deduplication, and submission hygiene
  do not authorize deleting these files.
- Never delete reports under `docs/`, including generated JSON or text reports,
  unless the user explicitly identifies the file or directory to delete.
- Keep repository retention separate from submission packaging. When generated
  reports must not ship, preserve them in the repository and use a
  non-destructive, reviewable mechanism such as `.gitattributes` `export-ignore`
  or an explicit archive allowlist.
- Before removing any tracked artifact, inspect references, history, and current
  release records. Default cleanup removal to untracked caches, bytecode,
  temporary outputs, and reproducible build artifacts outside `docs/`.
- Preserve compatibility wrappers when an evaluator or external caller may rely
  on a historical import or command path; consolidate the implementation behind
  the wrapper instead of deleting the public path.
- List proposed tracked-file deletions separately in the cleanup report. If
  their evidence value or API status is ambiguous, preserve them and ask for
  explicit direction.

## Gate every dependency before implementation

Before adding an import, library call, build dependency, or requirements entry:

1. Prefer the standard library or an already-used repository dependency.
2. Check `.specification/environments.txt` for the exact module and version.
3. When server access is available, verify the import with a minimal read-only import/version probe on that server. Local availability is not evidence of remote availability.
4. If the dependency is neither declared nor verified as natively installed, do not introduce it. Rework the implementation using available modules, or stop and ask the user before changing dependency scope.
5. Never add `pip install`, network bootstrap, vendored packages, binary wheels, shared libraries, or fallback auto-install logic to make an unavailable dependency appear present.
6. Treat optional imports as dependencies too. A fallback is acceptable only when every branch uses modules already available on the server and the selected branch is explicit, testable, and disclosed.

For every accepted third-party dependency change, record the server evidence, exact version, license, purpose, call boundary, and why existing native modules were insufficient. Review dependency diffs before delivery with searches over imports, requirement files, build scripts, and subprocess invocations. If native availability remains unverified, report the work as blocked rather than claiming it server-ready.

## Build one pipeline

Implement the five sub-missions as stages of one compiler/runtime:

```text
ONNX
  -> validated graph IR and DAG export              (C3.1)
  -> precision selection and kernel decomposition   (C3.2)
  -> fusion and graph optimization                  (C3.3)
  -> lifetime, allocation, copy, and stream plan    (C3.4)
  -> batched GPU execution and NPY output           (C3.5)
```

Maintain a shared IR containing graph inputs/outputs, initializers, nodes, tensors, shapes, dtypes, attributes, producer/consumer maps, and topological order. Keep the C3.1 JSON as an export view, not the sole internal representation.

Preserve these invariants after every transformation:

- Exclude initializers from runtime graph inputs.
- Give each produced tensor exactly one producer.
- Resolve every node input and graph output.
- Preserve acyclicity, graph inputs, graph outputs, and dynamic batch `N`.
- Keep an individually testable FULL_FP32 path.
- Preserve sample order and emit float32 outputs.
- Make precision, decomposition, fusion, and scheduling deterministic for identical inputs and target configuration.
- Require every reported capability, kernel, fusion, and memory feature to have a connected executable source path.
- Keep runtime network-free and independent of evaluator identity.

## Execute by sub-mission

### C3.1: graph parsing, 10 points

Accept `--onnx` and `--output`; exit 0 after writing valid DAG JSON. Handle symbolic shapes, optional empty inputs, constants, fan-out, multiple outputs, empty/duplicate node names, and initializer/input overlap. Construct edges from tensor producer/consumer relationships and validate the graph.

Read `docs/c31-graph-parsing.md` for the released contract, validation, and limitations.

### C3.2: decomposition and kernel selection, 15 points

Implement the public calls named by the specification: graph import, precision selection, hardware capability reporting, decomposition, and kernel tuning.

- Force Softmax, LayerNorm/LayerNormalization, BatchNorm, and reductions to FP32.
- Return a non-empty kernel sequence for every node.
- Emit recognizable MatMul, Softmax, LayerNorm, and Conv sequences.
- Expose deterministic intermediate tensor names through kernel outputs.
- Fill `block_x`, `grid_x`, and `smem_bytes`; enforce all hardware limits.
- Drive precision and Winograd/im2col decisions from hardware capabilities.
- Keep deployment precision separate from microbenchmark capability coverage.
- Do not use round-robin/scoring counters or advertise kernels that are not actually emitted and executable.

Read `docs/c32-decomposition.md` before implementing evaluator-facing types.

### C3.3: fusion, 15 points

Support `FusedMatMulBias`, `FusedConv2dBatchNorm`, `FusedEWChain`, `FusedSoftmaxDropout`, and `FusedResidualNorm`. Log matches at the exact path required by the specification. Apply guarded, transactional rewrites; rebuild graph indexes and validate after every pass.

Count launch and buffer reductions from executable plans, not renamed nodes. Compare original and optimized FP32 results; any path exceeding `max_abs_diff > 1e-3` fails fusion correctness.

Do not invent BatchNorm parameters. The released ResNet has BN folded into Conv, so original Conv/BN factors are not uniquely reconstructable without extra evaluator data. Implement correct Conv+BN folding when parameters exist and report this limitation.

Read `docs/c33-fusion.md` for pass ordering and safety guards.

### C3.4: memory and scheduling, 10 points

Connect every reviewed feature to the real execution plan:

- device allocation/free plus weight H2D and kernel consumption;
- first/last-use analysis plus physical slot reuse;
- a reusable pool with best-fit, size classes, coalescing, or equivalent fragmentation policy;
- next-layer weight prefetch overlapping current compute;
- dependency-aware placement of independent kernels on multiple compute streams.

Expose allocation bindings, sizes, plan steps, streams, events, lifetimes, reuse decisions, and planned peak memory for code review. Stubs, comments, and logging-only behavior do not score.

Read `docs/c34-memory-scheduling.md` for required evidence.

### C3.5: end-to-end deployment, 50 points

Accept `--onnx`, `--input`, `--output`, and optional `--batch-size`. Read and bind tensors by manifest name. Support arbitrary positive batch sizes, including a final partial batch. Write `manifest.json` and complete, ordered float32 `logits`.

Implement ONNX opset-17 semantics for the published 17-operator union. Do not hardcode public filenames, node names, weights, sample count, or fixed batch size; hidden models retain structure but change weights.

Treat correctness as a gate:

- all models: `cupy.allclose(rtol=1e-3, atol=1e-3)` on device arrays;
- MLP top-1: at least 0.98;
- ResNet top-1: at least 0.85.

Optimize only after all three public models pass in FP32. Measure cold process wall time and NVML peak memory, including startup, parsing, compilation, I/O, and child processes. Qualify every low-precision, fusion, and scheduling change against golden outputs.

Read `docs/c35-deployment.md` before performance work.

## Validate before delivery

Run the checklist in `docs/validation-checklist.md`. At minimum:

1. Test all three public models.
2. Test batch sizes 1, a non-power-of-two value, 256, a non-divisor of sample count, and a value larger than the dataset.
3. Validate deterministic DAG export and all graph invariants.
4. Compare baseline and every optimized mode with golden output.
5. Produce per-node C3.2 evidence, fusion logs, and a reviewable C3.4 execution plan.
6. Verify offline build/run instructions and both required command templates.
7. Audit the submission for originality, hardcoding, evaluator bypass, precomputed/precompiled artifacts, undeclared dependencies, and any unintended non-CuPy fallback execution.

## Track specification risks

- Use the written C3.5 split of correctness 15, runtime 25, and peak memory 10. `requirements.png` says peak memory 15, which conflicts with the 50-point total; flag it for organizer confirmation.
- The release contains C3.1/C3.5 assets but not the referenced C3.2/C3.3 benchmark implementation.
- Confirm how Conv+BN fusion is evaluated when the released ONNX contains no BN node or original BN parameters.
