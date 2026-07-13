# Submission disclosure

## Native dependencies (#41, #42)

All code runs using only the modules declared in
`.specification/environments.txt`. No additional `pip install`, network
bootstrap, vendored packages, binary wheels, or auto-install logic is present.

| Module | Verifier | Version source | License | Purpose | Call boundary |
|---|---|---|---|---|---|
| `onnx` | `import onnx; onnx.__version__` | declared version in environments.txt | Apache-2.0 | ONNX model load, validation, shape inference, checker, and helper utilities | `c31.import_onnx`, `c35.executor` |
| `google.protobuf` (`protobuf`) | `import google.protobuf; google.protobuf.__version__` | transitive dependency of server-native `onnx`; exact remote version still needs recording because environments.txt does not list it separately | BSD-3-Clause | Direct use of `MessageToDict` for ONNX attribute serialization and `Message` for type checking | `c31.import_onnx` (`_json_safe`) |
| `cupy` | `import cupy; cupy.__version__` | declared version in environments.txt | MIT | All framework array computation, golden comparison, and `.npy` I/O | C3.5 and shared scoring tests, `c35.engine`, `c35.executor`, `c35.deploy`, `c35.standard_runner` |
| `onnxruntime` | `import onnxruntime` | declared version in environments.txt | MIT | Not used at runtime; available in the environment | none |
| `torch` | `import torch; torch.__version__` | declared version in environments.txt | BSD-3-Clause | Not used at runtime; available in the environment | none |

Standard library: `argparse`, `base64`, `copy`, `dataclasses`, `json`, `math`,
`os`, `pathlib`, `shlex`, `shutil`, `signal`, `subprocess`, `sys`, `tempfile`,
`threading`, `time`, `typing`, `unittest`.

> CuPy 14.1.1 on the remote H200 MIG instance was verified through
> `environments.txt` and confirmed during H200 validation runs. ONNX tensor
> decoding may materialize host storage internally; the framework immediately
> converts decoded tensors to CuPy and exposes no CPU numerical backend.

External tool:

| Tool | Version source | License | Purpose | Call boundary |
|---|---|---|---|---|
| `nvidia-smi` | NVIDIA driver `580.126.20` in `environments.txt` | NVIDIA proprietary driver utility | Best-effort process GPU-memory sampling and device diagnostics | `c35.standard_runner.NvidiaSmiProcessTreeSampler` via `subprocess` |

The runner does not import an optional Python NVML binding. If MIG hides
per-process accounting from `nvidia-smi`, it labels the child-reported CuPy
memory-pool reservation as a proxy rather than an official peak-memory value.

## Academic attribution (#43)

- ONNX operator semantics follow the ONNX opset 17 specification
  (https://github.com/onnx/onnx).
- The error-function approximation uses the Abramowitz & Stegun 7.1.26 formula.
- TVM, *An Automated End-to-End Optimizing Compiler for Deep Learning*
  (https://arxiv.org/abs/1802.04799), informed the separation between
  graph-level fusion and hardware-specific lowering.
- DNNFusion, *Accelerating Deep Neural Networks Execution with Advanced
  Operator Fusion* (https://arxiv.org/abs/2108.13342), informed the use of
  semantic operator classification and bounded fusion regions.
- MLIR Linalg documentation
  (https://mlir.llvm.org/docs/Dialects/Linalg/) informed the
  dependency/single-consumer legality guards and explicit external operands.
- NVIDIA CUDA Programming Guide, CUDA Graphs
  (https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html),
  informed the documented distinction between host launch-submission overhead
  and actual physical kernel-count reduction.

No source code or documentation text was copied from these references. The
repository's fusion algorithm and implementation are original, AI-assisted
team work described below.

## Originality and LLM-assistance disclosure (#44)

This submission is original work by the team. The following AI-assisted tools
were used during development:

- **GitHub Copilot** — Used as an inline coding assistant in VS Code for
  boilerplate reduction, test generation, documentation drafts, and
  refactoring suggestions.
- **OpenAI Codex** — Used via the OpenAI API for exploratory prototyping,
  debugging complex ONNX operator semantics, and iterating on kernel
  decomposition strategies.

All AI-generated code was reviewed, tested, and integrated by the team. The
team can explain and maintain every component. No full files or substantial
algorithmic blocks were generated verbatim from an LLM without human review
and integration. No AI tool was used to generate precomputed outputs, cached
plans, or evaluation artifacts.

## Archive cleanliness (#45)

### Build and verification procedure

The submission archive is built with `scripts/build_submission.sh`, which:

1. Exports a clean `git archive` of the HEAD commit (respecting `.gitignore`).
2. Verifies the archive contains no forbidden artifacts by inspecting the
   tar member list.

### Excluded by `.gitignore`

- `.DS_Store`, `Thumbs.db` (OS artifacts)
- `__pycache__/`, `*.py[cod]`, `*.pyo` (Python bytecode)
- `.venv/`, `venv/`, `env/` (virtual environments)
- `submission.tar.gz`, `*.dag.json`, `*-report.json`,
  `c35-standard-report.json`, `*.plan.json`,
  `*.cache` (generated/cached artifacts)
- `.vscode/`, `.idea/`, `*.swp`, `*.swo`, `*~` (IDE/editor artifacts)
- `*.whl` (binary wheels; evaluator provides dependencies)
- `*.log` (log files)
- `output/` (generated outputs)
- `.ssh/`, `.agents/`, `.specification/` (non-submission directories)

### Verified absent from the built archive

- No virtual environments, `__pycache__`, or `.pyc` bytecode
- No precomputed outputs, cached plans, or generated answers
- No `.ssh/`, `.agents/`, or `.specification/` content
- No IDE or editor configuration files
- No binary wheels, log files, or development-only assets
- No `.git/` directory (archive is a clean export)

> `.gitignore` is packaging policy; the `scripts/build_submission.sh` script
> provides reproducible evidence that the built archive is clean.
