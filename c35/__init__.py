"""C3.5 — End-to-end model deployment.

Implements the full AEC GPGPU inference pipeline:
- ONNX model loading with weight extraction
- Numpy-based AEC compute engine (all 17 required operators, opset-17)
- Graph-level execution with topological ordering
- Batch execution with --batch-size
- Output validation and manifest generation

Usage:
    python -m c35.deploy --onnx MODEL --input INPUT_DIR --output OUTPUT_DIR [--batch-size N]
"""

from c35.deploy import main

__all__ = ["main"]
