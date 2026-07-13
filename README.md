# C3 Operator Scheduling and Model Deployment

This repository implements the C3 competition framework for importing ONNX
models, selecting precision and kernel decompositions, applying graph fusion,
planning memory and streams, and running end-to-end inference.

The released implementation uses one shared graph IR across C3.1–C3.5. C3.5
provides a CuPy/CUDA execution path for the published MLP, ResNet-18, and
decoder-only Transformer models on the designated remote H200 AEC device.

## Pipeline

```text
ONNX model
  -> validated graph IR and DAG JSON                  C3.1
  -> precision policy and kernel decomposition        C3.2
  -> transactional fusion and graph optimization      C3.3
  -> lifetime, allocation, transfer, and stream plan  C3.4
  -> batched CuPy inference on AEC H200 and NPY output  C3.5
```

## Repository layout

| Path | Purpose |
|---|---|
| `c31/` | ONNX import, shared graph construction, and DAG export |
| `c32/` | Precision policy, hardware capability model, decomposition, and tuning |
| `c33/` | Fusion passes and graph-pass pipeline |
| `c34/` | Lifetime analysis, memory pool, and execution scheduling |
| `c35/` | CuPy-only H200 execution, deployment CLI, and black-box runner |
| `c3common/` | Shared IR and cross-stage regression checks |
| `.specification/testcases/release_to_competitors/models/` | Organizer-released public ONNX models |
| `docs/` | Release documentation, validation, and remaining limitations |
| `.specification/` | Organizer specification and released test assets |

## Requirements

Use the Python and third-party modules provided by the evaluation server. The
implementation does not bootstrap, download, or vendor dependencies. The
published server environment includes Python 3.12.3, ONNX 1.22.0,
CuPy 14.1.1, and the CUDA runtime described in
`.specification/environments.txt`.

Before adding a dependency, confirm that its exact version is natively present
on the target server. Local availability is not sufficient.

## Quick start

### Export a C3.1 DAG

```bash
python3 export_dag.py \
  --onnx .specification/testcases/release_to_competitors/models/mlp_v1.onnx \
  --output /tmp/mlp.dag.json
```

The output contains `format_version`, graph inputs and outputs, nodes, and
producer-to-consumer tensor edges.

### Run C3.5 inference

```bash
python3 -m c35.deploy \
  --onnx .specification/testcases/release_to_competitors/models/mlp_v1.onnx \
  --input .specification/testcases/release_to_competitors/testdata/c35/mlp_v1/input \
  --output /tmp/mlp-output \
  --batch-size 256
```

The command writes `manifest.json` and `logits.npy`. CuPy is the only supported
numerical backend.

### Validate all released C3.5 models

```bash
./run_c35.sh
```

The runner performs a CuPy/device preflight, launches each model in a fresh
process, checks output manifests and golden tolerances, computes classification
accuracy, records cold wall time, and writes `c35-standard-report.json`.

On MIG systems, memory evidence is collected with the server-native
`nvidia-smi` command. If per-process accounting is hidden, the runner reports the
child process's CuPy memory-pool high-water proxy and labels the source
`cupy-pool`.

## Development validation

```bash
python3 -m unittest -q c31.test_c31
python3 -m c32.test_c32
python3 -m c33.test_c33
python3 -m c34.test_c34
python3 -m unittest -v c35.test_c35 c35.test_cross_stage
python3 -m unittest -v c3common.test_scoring_regressions
```

The C3.5 and shared scoring-regression commands require the remote H200 with
CuPy 14.1.1; there is no local CPU numerical fallback.

See [the validation checklist](docs/validation-checklist.md) for release and
submission gates.

## Release status

- C3.1 imports and exports all three released models through the required CLI.
- C3.2 exposes deterministic public precision, decomposition, and tuning APIs.
- C3.3 implements the five requested fusion patterns with transactional graph
  validation and machine-readable logs.
- C3.4 emits reviewable allocation, lifetime, transfer, event, and stream plans.
- C3.5 runs the connected optimized graph through CuPy; the revised CuPy-only
  path passes the three-model H200 black-box correctness and accuracy gates.

The framework executes inference on the designated H200 AEC device. The primary
remaining integration boundary is that C3.2 kernel references and C3.4 planned
allocations, transfers, streams, and events do not yet directly drive the CuPy
launch path. Additional evaluator ambiguities and scoring limitations are
tracked in [remaining problems](docs/remaining-problems.md).

## Documentation and governing contracts

| Component | Documentation | Released implementation |
|---|---|---|
| C3.1 | [Graph parsing and representation](docs/c31-graph-parsing.md) | ONNX import, graph validation, deterministic DAG JSON |
| C3.2 | [Decomposition and kernel selection](docs/c32-decomposition.md) | Precision profiles, hardware capabilities, kernel sequences, tuning |
| C3.3 | [Fusion and graph optimization](docs/c33-fusion.md) | Five guarded fusion patterns and transactional pass pipeline |
| C3.4 | [Memory planning and scheduling](docs/c34-memory-scheduling.md) | Lifetimes, pool reuse, bindings, transfers, streams, and events |
| C3.5 | [Model deployment](docs/c35-deployment.md) | Batched CuPy execution on the H200 AEC device |
| C3.5 runner | [Black-box validation runner](docs/c35-standard-runner.md) | Golden checks, accuracy, cold timing, and GPU-memory evidence |

Release and maintenance records:

- [Release implementation summary](docs/fix-summary.md) records behavior and
  evidence actually present in the repository.
- [Remaining problems](docs/remaining-problems.md) separates verified behavior
  from evaluator, backend, performance, and submission gaps.
- [Validation checklist](docs/validation-checklist.md) defines the checks
  required before a release or competition submission.

The organizer documents under `.specification/` are authoritative, in this
order:

1. `.specification/general_requirements.md` — integrity, originality,
   dependency, offline, and submission rules.
2. `.specification/environments.txt` — native evaluation-server environment.
3. `.specification/spec.md` — C3 interfaces and functional requirements.
4. `.specification/scoring.md` — scoring details.

Written requirements take precedence over this README. Known conflicts and
unreleased evaluator details are tracked in
[remaining problems](docs/remaining-problems.md).

The DAG JSON is an export view of the shared IR, not a replacement for tensor
metadata, attributes, producer/consumer maps, or initializer state. The current
CuPy/CUDA path executes released models on the remote H200 AEC device, but C3.2
kernel references and C3.4 allocation/stream/event plans remain structural
until they directly drive device operations.

## Submission checklist and command templates

The submission must contain the program source plus these build and run
instructions. This Python implementation requires no compilation and must not
install or download packages during evaluation. Run it from the submission
root with the Python environment provided by the organizer.

Register these command templates exactly, retaining the braces for evaluator
substitution:

### C3.1 command template

```bash
python3 export_dag.py --onnx {onnx} --output {output}
```

### C3.5 command template

```bash
python3 -m c35.deploy --onnx {onnx} --input {input} --output {output} --batch-size 256
```

Before submission, run all three public models. For every C3.5 output:

1. Compare every output tensor with its `golden/` tensor using
   `numpy.allclose(out, golden, rtol=1e-3, atol=1e-3)` as required by
   `.specification/spec.md`. The supplied GPU runner applies the equivalent
   `cupy.allclose` check with the same tolerances.
2. For MLP and ResNet-18, compare `logits.argmax(...)` with `labels.npy` and
   require top-1 accuracy of at least `0.98` and `0.85`, respectively.
   Transformer has no classification-accuracy threshold but must pass the
   elementwise golden comparison.
3. Confirm `manifest.json` names, file names, dtypes, and shapes match the
   emitted `.npy` files; all outputs must cover the complete input batch in
   original sample order and use `float32`.

On the organizer H200 environment, the repository-wide public-model check is:

```bash
./run_c35.sh
```

The runner invokes each model in a fresh process, verifies the output contract,
golden tolerances, and applicable accuracy threshold, and records cold wall
time plus available GPU-memory evidence. CuPy-pool memory is labeled as a proxy
when MIG hides per-process NVML accounting.

## Submission disclosure

### Native dependencies (#41, #42)

The implementation introduces no `pip install`, network bootstrap, vendored
package, binary wheel, or auto-install path. It uses the standard library and
packages native to the published evaluation environment.

| Module | Verifier | Version source | License | Purpose | Call boundary |
|---|---|---|---|---|---|
| `onnx` | `import onnx; onnx.__version__` | `.specification/environments.txt` | Apache-2.0 | ONNX load, validation, shape inference, checker, and helpers | `c31.import_onnx`, `c35.executor` |
| `google.protobuf` (`protobuf`) | `import google.protobuf; google.protobuf.__version__` | transitive dependency of native `onnx`; exact remote version still needs recording | BSD-3-Clause | `MessageToDict` serialization and `Message` type checking | `c31.import_onnx` (`_json_safe`) |
| `cupy` | `import cupy; cupy.__version__` | `.specification/environments.txt` | MIT | Framework array computation, golden comparison, and `.npy` I/O | C3.5, shared scoring tests, `c35.engine`, `c35.executor`, `c35.deploy`, `c35.standard_runner` |
| `onnxruntime` | `import onnxruntime; onnxruntime.__version__` | `.specification/environments.txt` | MIT | Available but not used at runtime | none |
| `torch` | `import torch; torch.__version__` | `.specification/environments.txt` | BSD-3-Clause | Available but not used at runtime | none |

Standard-library modules used include `argparse`, `base64`, `copy`,
`dataclasses`, `json`, `math`, `os`, `pathlib`, `shlex`, `shutil`, `signal`,
`subprocess`, `sys`, `tempfile`, `threading`, `time`, `typing`, and `unittest`.

CuPy 14.1.1 was confirmed on the remote H200 MIG validation environment. ONNX
decoding may materialize host storage internally; tensors are immediately
converted to CuPy, and the framework exposes no CPU numerical backend.

External tool:

| Tool | Version source | License | Purpose | Call boundary |
|---|---|---|---|---|
| `nvidia-smi` | NVIDIA driver `580.126.20` in `.specification/environments.txt` | NVIDIA proprietary driver utility | Best-effort process GPU-memory sampling and device diagnostics | `c35.standard_runner.NvidiaSmiProcessTreeSampler` via `subprocess` |

The runner does not import an optional Python NVML binding. If MIG hides
per-process accounting, it labels the child-reported CuPy memory-pool
reservation as a proxy rather than an official peak-memory value.

The general requirements say third-party dependencies must accompany the
submission, while the C3 specification mandates CuPy and publishes a native
server environment. This repository relies on those native packages and does
not vendor them; organizer confirmation of that interpretation and direct
remote verification of `google.protobuf` remain submission blockers.

### Academic attribution (#43)

- ONNX operator semantics follow the
  [ONNX opset 17 specification](https://github.com/onnx/onnx).
- The error-function approximation uses the Abramowitz & Stegun 7.1.26
  formula.
- [TVM](https://arxiv.org/abs/1802.04799) informed the separation between
  graph-level fusion and hardware-specific lowering.
- [DNNFusion](https://arxiv.org/abs/2108.13342) informed semantic operator
  classification and bounded fusion regions.
- [MLIR Linalg](https://mlir.llvm.org/docs/Dialects/Linalg/) informed
  dependency/single-consumer guards and explicit external operands.
- The [NVIDIA CUDA Graphs guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)
  informed the distinction between host launch-submission overhead and
  physical kernel-count reduction.

No source code or documentation text was copied from these references. The
fusion algorithm and implementation are original, AI-assisted team work.

### Originality and LLM-assistance disclosure (#44)

This submission is original work by the team. AI-assisted tools used during
development are:

- **GitHub Copilot** — inline assistance for boilerplate reduction, test
  generation, documentation drafts, and refactoring suggestions.
- **OpenAI Codex** — exploratory prototyping, ONNX-semantics debugging, kernel
  decomposition work, documentation review, and integration support.

All assisted changes were reviewed, tested, and integrated by the team. The
team can explain and maintain every component. No AI tool generated
precomputed outputs, cached plans, or evaluation artifacts.

### Submission archive policy (#45)

Build the archive only after committing the exact intended submission state:

```bash
bash scripts/build_submission.sh submission.tar.gz
```

The script exports tracked files from `HEAD`, lists every archive member,
rejects forbidden paths or generated artifacts, and verifies required runtime
entries. Uncommitted changes are intentionally absent from the archive.

The repository excludes OS files, Python bytecode, virtual environments,
generated reports/outputs/plans/caches, editor state, wheels, logs, `.ssh/`,
`.agents/`, and `.specification/` from the submission workspace. The archive
must be inspected to confirm it contains none of those items, no `.git/`
directory, and no precomputed outputs or generated answers. Ignore rules alone
are policy, not evidence; archive cleanliness remains open until the actual
post-commit artifact passes the script.
