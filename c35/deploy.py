#!/usr/bin/env python3
"""C3.5: End-to-end model deployment CLI.

Reads an ONNX model and input data, runs inference using CuPy, and writes
outputs with validation. CuPy/CUDA on the designated
remote H200 server is the release AEC device path.

Usage:
    python -m c35.deploy --onnx MODEL --input INPUT_DIR --output OUTPUT_DIR [--batch-size N]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cupy as cp


def validate_batch_size(value: str) -> int:
    """Validate and parse the --batch-size argument.

    Must be a positive integer.
    """
    try:
        bs = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"batch-size must be an integer, got '{value}'"
        )
    if bs <= 0:
        raise argparse.ArgumentTypeError(
            f"batch-size must be positive, got {bs}"
        )
    return bs


def main() -> None:
    """Main CLI entry point for C3.5 model deployment."""
    parser = argparse.ArgumentParser(
        description="C3.5: End-to-end model deployment on AEC GPGPU"
    )
    parser.add_argument(
        "--onnx",
        required=True,
        help="Path to the input ONNX model file",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Directory containing manifest.json and .npy input files",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory to write output logits.npy and manifest.json",
    )
    parser.add_argument(
        "--batch-size",
        type=validate_batch_size,
        default=None,
        help="Maximum batch size for inference (default: process all at once)",
    )
    args = parser.parse_args()

    onnx_path = args.onnx
    input_dir = args.input
    output_dir = args.output
    batch_size = args.batch_size

    # Validate paths
    if not os.path.isfile(onnx_path):
        print(f"Error: ONNX file not found: {onnx_path}", file=sys.stderr)
        sys.exit(1)

    if not os.path.isdir(input_dir):
        print(f"Error: Input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    input_manifest_path = os.path.join(input_dir, "manifest.json")
    if not os.path.isfile(input_manifest_path):
        print(f"Error: Input manifest not found: {input_manifest_path}", file=sys.stderr)
        sys.exit(1)

    # Report configuration
    print(f"C3.5 Model Deployment", file=sys.stderr)
    print(f"  ONNX model:  {onnx_path}", file=sys.stderr)
    print(f"  Input dir:   {input_dir}", file=sys.stderr)
    print(f"  Output dir:  {output_dir}", file=sys.stderr)
    if batch_size is not None:
        print(f"  Batch size:  {batch_size}", file=sys.stderr)
    print(file=sys.stderr)

    # Run inference
    try:
        from c35.executor import load_and_infer

        info = load_and_infer(
            model_path=onnx_path,
            input_dir=input_dir,
            output_dir=output_dir,
            batch_size=batch_size,
        )
    except Exception as e:
        print(f"Error during inference: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    # Report performance
    print(f"Inference complete:", file=sys.stderr)
    print(f"  Samples:     {info['total_samples']}", file=sys.stderr)
    print(f"  Output shape:{info['output_shape']}", file=sys.stderr)
    print(f"  Parse time:  {info['parse_time_s']:.3f}s", file=sys.stderr)
    print(f"  Infer time:  {info['infer_time_s']:.3f}s", file=sys.stderr)
    print(f"  Total time:  {info['total_time_s']:.3f}s", file=sys.stderr)
    if info.get("cross_stage_reference"):
        fusion = info["fusion_stats"]
        plan = info.get("plan_summary") or {}
        print(
            f"  Backend:     connected C3 {info['backend']} on AEC H200",
            file=sys.stderr,
        )
        print(
            f"  Fusion:      {fusion['node_count_before']} -> "
            f"{fusion['node_count_after']} nodes",
            file=sys.stderr,
        )
        if info["qualification_max_abs_diff"] is not None:
            print(
                f"  FP32 check:  max_abs_diff="
                f"{info['qualification_max_abs_diff']:.6e}",
                file=sys.stderr,
            )
        print(
            f"  C3.4 plan:   {plan.get('total_kernels', 0)} kernels, "
            f"{plan.get('total_allocations', 0)} allocations",
            file=sys.stderr,
        )
    print(
        "C35_GPU_EVIDENCE_JSON="
        + json.dumps(info.get("backend_evidence", {}), sort_keys=True),
        file=sys.stderr,
    )

    # Validate output files exist
    manifest_out = os.path.join(output_dir, "manifest.json")
    logits_out = os.path.join(output_dir, "logits.npy")
    if not os.path.isfile(manifest_out):
        print(f"Error: Output manifest was not created: {manifest_out}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(logits_out):
        print(f"Error: Output logits.npy was not created: {logits_out}", file=sys.stderr)
        sys.exit(1)

    # Validate manifest content
    with open(manifest_out, "r") as f:
        manifest = json.load(f)
    logits_arr = cp.load(logits_out, allow_pickle=False)

    for entry in manifest.get("tensors", []):
        if entry["name"] != "logits":
            print(f"Error: Unexpected output name '{entry['name']}'", file=sys.stderr)
            sys.exit(1)
        if entry["dtype"] != "float32":
            print(f"Error: Output dtype is '{entry['dtype']}', expected 'float32'", file=sys.stderr)
            sys.exit(1)
        expected_shape = entry["shape"]
        actual_shape = list(logits_arr.shape)
        if expected_shape != actual_shape:
            print(
                f"Error: Output shape mismatch: manifest says {expected_shape}, "
                f"actual is {actual_shape}",
                file=sys.stderr,
            )
            sys.exit(1)
        if logits_arr.dtype != cp.float32:
            print(
                f"Error: Output dtype is {logits_arr.dtype}, expected float32",
                file=sys.stderr,
            )
            sys.exit(1)
        if not logits_arr.flags["C_CONTIGUOUS"]:
            print(f"Warning: Output is not C-contiguous", file=sys.stderr)

    sys.exit(0)


if __name__ == "__main__":
    main()
