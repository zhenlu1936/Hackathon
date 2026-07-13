"""C3.5 — end-to-end model deployment.

Implements the connected compiler-stage deployment pipeline:
- ONNX model loading with weight extraction
- CuPy CUDA compute exclusively on the designated H200 AEC device
- Graph-level execution with topological ordering
- Batch execution with --batch-size
- Output validation and manifest generation

Usage:
    python -m c35.deploy --onnx MODEL --input INPUT_DIR --output OUTPUT_DIR [--batch-size N]
"""

def main() -> None:
    """Load the CLI lazily so ``python -m c35.deploy`` has no runpy warning."""
    from c35.deploy import main as deploy_main

    deploy_main()

__all__ = ["main"]
