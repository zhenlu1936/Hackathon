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

    # Build JSON string:
    #   graph_inputs/graph_outputs → one compact line each
    #   nodes                     → one field per line per element, inline values
    #   edges                     → one compact line per element
    def _compact(obj: object) -> str:
        return json.dumps(obj, ensure_ascii=False)

    def _format_node(node: dict, base_indent: int = 2) -> str:
        """Format a node with each field on its own line, values compact inline."""
        outer = " " * (base_indent + 2)   # indent for {  }
        inner = " " * (base_indent + 4)   # indent for fields
        lines = []
        lines.append(outer + "{")
        fields = []
        for key in ("name", "op_type", "inputs", "outputs", "attributes", "original_name"):
            if key not in node:
                continue
            fields.append(f'{inner}"{key}": {_compact(node[key])}')
        lines.append(",\n".join(fields))
        lines.append(outer + "}")
        return "\n".join(lines)

    def _format_node_array(arr: list, indent: int = 2) -> str:
        """Format nodes array with multi-line elements."""
        if not arr:
            return "[]"
        pad = " " * indent
        items = ",\n".join(_format_node(item, indent) for item in arr)
        return "[\n" + items + "\n" + pad + "]"

    def _compact_array(arr: list, indent: int = 2) -> str:
        """Format an array with each element on its own compact line."""
        if not arr:
            return "[]"
        pad = " " * indent
        items = ",\n".join(pad + _compact(item) for item in arr)
        return "[\n" + items + "\n" + pad + "]"

    lines = ["{"]
    lines.append(f'  "format_version": "1.0",')
    lines.append(f'  "graph_inputs": {_compact(dag_json["graph_inputs"])},')
    lines.append(f'  "graph_outputs": {_compact(dag_json["graph_outputs"])},')
    lines.append(f'  "nodes": {_format_node_array(dag_json["nodes"])},')
    lines.append(f'  "edges": {_compact_array(dag_json["edges"])}')
    lines.append("}")
    json_str = "\n".join(lines) + "\n"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(json_str)

    sys.exit(0)


if __name__ == "__main__":
    main()
