"""C3.5 Graph Executor — walks the computation graph and executes nodes.

Manages the tensor value dictionary, loads weights from ONNX initializers,
and executes nodes in topological order using the AEC compute engine.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnx

from c3common.ir.graph import Graph, Node
from c35.engine import execute_op


def _extract_initializer_data(model_path: str) -> Dict[str, np.ndarray]:
    """Extract all initializer tensors from an ONNX model as numpy arrays.

    Args:
        model_path: Path to the .onnx file.

    Returns:
        Dict mapping initializer name -> numpy array.
    """
    model = onnx.load(model_path)
    weights: Dict[str, np.ndarray] = {}
    for init in model.graph.initializer:
        arr = onnx.numpy_helper.to_array(init)
        weights[init.name] = arr
    return weights


def _extract_constant_values(model_path: str) -> Dict[str, np.ndarray]:
    """Extract constant tensor values from ONNX Constant nodes.

    Constant nodes store their value as the 'value' attribute.
    We map each constant-node output name to its numpy array.

    Args:
        model_path: Path to the .onnx file.

    Returns:
        Dict mapping constant output tensor name -> numpy array.
    """
    model = onnx.load(model_path)
    constants: Dict[str, np.ndarray] = {}
    for node in model.graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value" and attr.t.data_type:
                    arr = onnx.numpy_helper.to_array(attr.t)
                    for out_name in node.output:
                        constants[out_name] = arr
                    break
            # Also handle value_float / value_int attributes
            if not any(out in constants for out in node.output):
                for attr in node.attribute:
                    if attr.name == "value_float":
                        for out_name in node.output:
                            constants[out_name] = np.array(attr.f, dtype=np.float32)
                    elif attr.name == "value_int":
                        for out_name in node.output:
                            constants[out_name] = np.array(attr.i, dtype=np.int64)
    return constants


class GraphExecutor:
    """Executes a computation graph node by node.

    Walks the graph in topological order, looking up tensor values
    and computing node outputs using the AEC compute engine.

    Attributes:
        graph: The parsed computation graph IR.
        weights: Dict of initializer name -> numpy array.
        constants: Dict of constant output name -> numpy array.
        values: Dict of tensor name -> numpy array (current inference state).
    """

    def __init__(self, graph: Graph, model_path: str):
        """Initialize the executor with a parsed graph and ONNX model.

        Args:
            graph: Parsed computation graph from import_onnx.
            model_path: Path to the .onnx file for weight extraction.
        """
        self.graph = graph
        self.weights = _extract_initializer_data(model_path)
        self.constants = _extract_constant_values(model_path)
        self.values: Dict[str, np.ndarray] = {}

    def run(self, feed_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Execute the full graph.

        Args:
            feed_dict: Dict mapping graph input name -> numpy array.
                       Keys must match graph.inputs tensor names.

        Returns:
            Dict mapping graph output name -> numpy array.
        """
        # Initialize with inputs and weights
        self.values.clear()
        self.values.update(self.weights)
        self.values.update(self.constants)
        self.values.update(feed_dict)

        # Execute nodes in topological order
        for node_id in self.graph.node_order:
            node = self.graph.nodes.get(node_id)
            if node is None:
                continue
            self._execute_node(node)

        # Collect outputs
        outputs: Dict[str, np.ndarray] = {}
        for out_tensor in self.graph.outputs:
            name = out_tensor.name
            if name in self.values:
                outputs[name] = self.values[name]
            else:
                raise KeyError(
                    f"Graph output '{name}' not found in computed values. "
                    f"Available tensors: {sorted(self.values.keys())}"
                )

        return outputs

    def _execute_node(self, node: Node) -> None:
        """Execute a single graph node.

        Resolves inputs from self.values, calls the compute engine,
        and stores outputs back to self.values.

        Args:
            node: The node to execute.
        """
        op_type = node.op_type

        # Skip nodes that produce already-known outputs (Constant handled via pre-load)
        if op_type == "Constant":
            # Already loaded in self.constants
            for out_name in node.outputs:
                if out_name and out_name not in self.values:
                    raise KeyError(
                        f"Constant node '{node.id}' output '{out_name}' "
                        f"not pre-loaded. Available constants: "
                        f"{sorted(self.constants.keys())}"
                    )
            return

        # Resolve inputs
        inputs: List[np.ndarray] = []
        for inp_name in node.inputs:
            if not inp_name:
                # Empty optional input — skip it for operators that can handle it
                inputs.append(np.float32(0.0))
                continue
            if inp_name not in self.values:
                raise KeyError(
                    f"Node '{node.id}' ({op_type}) requires input '{inp_name}' "
                    f"which is not available. Available tensors: "
                    f"{sorted(self.values.keys())}"
                )
            inputs.append(self.values[inp_name])

        # Inject node-level metadata into attributes for ops that need it
        attrs = dict(node.attributes)
        if op_type == "Split":
            attrs["_num_outputs"] = len(node.outputs)

        # Execute
        try:
            result = execute_op(op_type, inputs, attrs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to execute node '{node.id}' ({op_type}): {e}"
            ) from e

        # Store outputs
        if isinstance(result, list):
            # Multi-output ops like Split
            for i, out_name in enumerate(node.outputs):
                if out_name and i < len(result):
                    self.values[out_name] = np.ascontiguousarray(
                        np.asarray(result[i], dtype=np.float32)
                    )
        else:
            for out_name in node.outputs:
                if out_name:
                    self.values[out_name] = np.ascontiguousarray(
                        np.asarray(result, dtype=np.float32)
                    )


def load_and_infer(
    model_path: str,
    input_dir: str,
    output_dir: str,
    batch_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Load model, run inference, and write outputs.

    This is the high-level orchestrator used by the CLI entry point.

    Args:
        model_path: Path to the .onnx model file.
        input_dir: Directory containing manifest.json and .npy input files.
        output_dir: Directory to write outputs (manifest.json + logits.npy).
        batch_size: Maximum batch size for processing. None means process all at once.

    Returns:
        Dict with timing and metadata information.
    """
    import json
    import os

    from c31.import_onnx import import_onnx

    t_start = time.perf_counter()

    # 1. Load ONNX graph
    t_parse_start = time.perf_counter()
    graph = import_onnx(model_path)
    t_parse_end = time.perf_counter()

    # 2. Create executor
    executor = GraphExecutor(graph, model_path)

    # 3. Load inputs
    input_manifest_path = os.path.join(input_dir, "manifest.json")
    if not os.path.isfile(input_manifest_path):
        raise FileNotFoundError(f"Input manifest not found: {input_manifest_path}")

    with open(input_manifest_path, "r") as f:
        input_manifest = json.load(f)

    input_arrays: Dict[str, np.ndarray] = {}
    input_meta: Dict[str, Dict[str, Any]] = {}
    for entry in input_manifest.get("tensors", []):
        name = entry["name"]
        file_name = entry["file"]
        file_path = os.path.join(input_dir, file_name)
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Input file not found: {file_path}")
        arr = np.load(file_path)
        input_arrays[name] = arr
        input_meta[name] = {
            "dtype": entry.get("dtype", "float32"),
            "shape": entry.get("shape", list(arr.shape)),
        }

    # 4. Determine batch size and validate
    if not input_arrays:
        raise ValueError("No input tensors found in manifest")
    first_key = next(iter(input_arrays))
    total_samples = input_arrays[first_key].shape[0]

    if batch_size is None or batch_size <= 0:
        actual_batch = total_samples
    else:
        actual_batch = min(batch_size, total_samples)

    # 5. Run inference in batches
    output_name = graph.outputs[0].name if graph.outputs else "logits"
    all_outputs: List[np.ndarray] = []

    t_infer_start = time.perf_counter()

    for start in range(0, total_samples, actual_batch):
        end = min(start + actual_batch, total_samples)

        # Slice inputs
        batch_feed: Dict[str, np.ndarray] = {}
        for name, arr in input_arrays.items():
            batch_feed[name] = arr[start:end]

        # Execute
        batch_outputs = executor.run(batch_feed)
        all_outputs.append(batch_outputs[output_name])

    t_infer_end = time.perf_counter()

    # 6. Concatenate outputs
    final_output = np.concatenate(all_outputs, axis=0)
    final_output = np.ascontiguousarray(final_output.astype(np.float32))

    # 7. Validate output
    if final_output.shape[0] != total_samples:
        raise ValueError(
            f"Output sample count mismatch: got {final_output.shape[0]}, "
            f"expected {total_samples}"
        )
    if not np.all(np.isfinite(final_output)):
        raise ValueError("Output contains non-finite values (NaN or Inf)")

    # 8. Write outputs
    os.makedirs(output_dir, exist_ok=True)

    # Write logits.npy
    output_file = "logits.npy"
    output_path = os.path.join(output_dir, output_file)
    np.save(output_path, final_output)

    # Write manifest.json
    output_manifest = {
        "tensors": [
            {
                "name": output_name,
                "file": output_file,
                "dtype": "float32",
                "shape": list(final_output.shape),
            }
        ]
    }
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(output_manifest, f, indent=2)

    t_end = time.perf_counter()

    return {
        "model_path": model_path,
        "input_dir": input_dir,
        "output_dir": output_dir,
        "batch_size": actual_batch,
        "total_samples": total_samples,
        "output_shape": list(final_output.shape),
        "parse_time_s": t_parse_end - t_parse_start,
        "infer_time_s": t_infer_end - t_infer_start,
        "total_time_s": t_end - t_start,
    }
