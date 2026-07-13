"""C3.5 — end-to-end model deployment.

Implements a connected compiler-stage reference pipeline:
- ONNX model loading with weight extraction
- CuPy CUDA compute by default, with explicit NumPy development mode
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
