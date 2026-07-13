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
| `models/` | Released public ONNX models |
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
python3 export_dag.py --onnx models/mlp_v1.onnx --output /tmp/mlp.dag.json
```

The output contains `format_version`, graph inputs and outputs, nodes, and
producer-to-consumer tensor edges.

### Run C3.5 inference

```bash
python3 -m c35.deploy \
  --onnx models/mlp_v1.onnx \
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
  path awaits a fresh three-model H200 black-box run.

The framework executes inference on the designated H200 AEC device. The primary
remaining integration boundary is that C3.2 kernel references and C3.4 planned
allocations, transfers, streams, and events do not yet directly drive the CuPy
launch path. Additional evaluator ambiguities and scoring limitations are
tracked in [remaining problems](docs/remaining-problems.md).

## Documentation

- [Documentation index](docs/README.md)
- [C3.1 graph parsing](docs/c31-graph-parsing.md)
- [C3.2 decomposition and kernel selection](docs/c32-decomposition.md)
- [C3.3 fusion and graph optimization](docs/c33-fusion.md)
- [C3.4 memory planning and scheduling](docs/c34-memory-scheduling.md)
- [C3.5 model deployment](docs/c35-deployment.md)
- [C3.5 black-box runner](docs/c35-standard-runner.md)
- [Release implementation summary](docs/fix-summary.md)
- [Remaining problems](docs/remaining-problems.md)
- [Validation checklist](docs/validation-checklist.md)
