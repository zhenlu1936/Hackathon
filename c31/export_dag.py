#!/usr/bin/env python3
"""C3.1: Computation graph parsing and representation.

Loads an ONNX model and exports its computation graph as a DAG JSON file.

Usage:
    python export_dag.py --onnx <model.onnx> --output <dag.json>
"""

from __future__ import annotations

import argparse
import json
import sys
import os

# Add parent directory to path for direct script execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from c31.import_onnx import import_onnx


def main() -> None:
    parser = argparse.ArgumentParser(
        description="C3.1: Export ONNX computation graph as DAG JSON"
    )
    parser.add_argument(
        "--onnx",
        required=True,
        help="Path to the input ONNX model file",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path for the output DAG JSON file",
    )
    args = parser.parse_args()

    onnx_path = args.onnx
    output_path = args.output

    if not os.path.isfile(onnx_path):
        print(f"Error: ONNX file not found: {onnx_path}", file=sys.stderr)
        sys.exit(1)

    try:
        graph = import_onnx(onnx_path)
    except Exception as e:
        print(f"Error loading ONNX model: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        dag_json = graph.to_dag_json()
    except Exception as e:
        print(f"Error building DAG JSON: {e}", file=sys.stderr)
        sys.exit(1)

    # Write JSON deterministically
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(dag_json, f, indent=2, ensure_ascii=False, sort_keys=False)
        f.write("\n")

    sys.exit(0)


if __name__ == "__main__":
    main()
