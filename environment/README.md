# Environment setup

The authoritative target is `specification/environments.txt`:

- Linux 6.8.0-110 x86_64
- Python 3.12.3
- GCC/G++ 13.3.0
- nvcc 12.8.61 and NVIDIA driver 580.126.20
- ONNX 1.22.0, ONNX Runtime 1.27.0
- Torch 2.13.0+cu130 with CUDA available
- CuPy 14.1.1

## Local macOS development

The repository `.venv` uses the available Python 3.12 interpreter and exact portable package versions:

```bash
source .venv/bin/activate
python -m pip install -r environment/requirements-local.txt
python environment/verify_environment.py
```

macOS ARM64 cannot match Linux x86_64, GCC, NVIDIA driver, nvcc, CUDA Torch, or CuPy. Local results must not be presented as GPU environment parity.

## Linux GPU target

On an x86_64 NVIDIA host with driver 580.126.20:

```bash
docker build --platform linux/amd64 -f environment/Dockerfile -t c3-target .
docker run --rm --gpus all c3-target
```

The driver comes from the host through NVIDIA Container Toolkit. The Docker image cannot install or guarantee the host driver/kernel. Run the verifier on the actual evaluation-class server before relying on parity.

For offline competition use, download and disclose all wheels in advance, then install with `--no-index --find-links`; do not depend on network access during evaluation.
