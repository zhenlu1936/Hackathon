# Submission disclosure

## Native dependencies (#41, #42)

All code runs using only the modules declared in
`.specification/environments.txt`. No additional `pip install`, network
bootstrap, vendored packages, binary wheels, or auto-install logic is present.

| Module | Verifier | Version source | License | Purpose | Call boundary |
|---|---|---|---|---|---|
| `onnx` | `import onnx; onnx.__version__` | declared version in environments.txt | Apache-2.0 | Protobuf load, tensor decode, shape inference, checker | `c31.import_onnx`, `c35.executor` |
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
- No other third-party academic sources were used in the implementation.

## Originality and LLM-assistance disclosure (#44)

This submission is original work by the team. GitHub Copilot was used as a
coding assistant during development. All generated code was reviewed, tested,
and the team can explain and maintain every component. No full files or
substantial algorithmic blocks were generated verbatim from an LLM without
human review and integration.

## Archive cleanliness (#45)

The `.gitignore` excludes:
- `.DS_Store`, `Thumbs.db` (OS artifacts)
- `__pycache__/`, `*.py[cod]` (Python bytecode)
- `.venv/`, `venv/`, `env/` (virtual environments)
- `*.dag.json` (generated DAG exports)

Additional items **not** present in the workspace:
- No precomputed outputs, cached plans, or generated answers
- No `.ssh/` or `.agents/` content in submission
- No downloaded development dependencies
