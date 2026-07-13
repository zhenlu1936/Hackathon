"""C3.5 CuPy graph executor.

Manages the tensor value dictionary, loads weights from ONNX initializers,
and executes nodes in topological order using CuPy.
The remote H200 GPU is the designated AEC execution device.
"""

from __future__ import annotations

import copy
import time
from typing import Any, Dict, List, Optional, Tuple

import cupy as cp
import onnx

from c3common.ir.graph import Graph, Node
from c34.execution_plan import ExecutionPlan
from c35 import engine


def _extract_initializer_data(model_path: str) -> Dict[str, cp.ndarray]:
    """Extract ONNX initializer tensors and move them directly to CuPy.

    Args:
        model_path: Path to the .onnx file.

    Returns:
        Dict mapping initializer name to a CuPy array.
    """
    model = onnx.load(model_path)
    weights: Dict[str, cp.ndarray] = {}
    for init in model.graph.initializer:
        weights[init.name] = cp.asarray(onnx.numpy_helper.to_array(init))
    return weights


def _extract_constant_values(model_path: str) -> Dict[str, cp.ndarray]:
    """Extract constant tensor values from ONNX Constant nodes.

    Constant nodes store their value as the 'value' attribute.
    We map each constant-node output name to its CuPy array.

    Args:
        model_path: Path to the .onnx file.

    Returns:
        Dict mapping constant output tensor name to a CuPy array.
    """
    model = onnx.load(model_path)
    constants: Dict[str, cp.ndarray] = {}
    for node in model.graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value" and attr.t.data_type:
                    arr = cp.asarray(onnx.numpy_helper.to_array(attr.t))
                    for out_name in node.output:
                        constants[out_name] = arr
                    break
            # Also handle value_float / value_int attributes
            if not any(out in constants for out in node.output):
                for attr in node.attribute:
                    if attr.name == "value_float":
                        for out_name in node.output:
                            constants[out_name] = cp.array(attr.f, dtype=cp.float32)
                    elif attr.name == "value_int":
                        for out_name in node.output:
                            constants[out_name] = cp.array(attr.i, dtype=cp.int64)
    return constants


class GraphExecutor:
    """Executes a computation graph node by node.

    Walks the graph in topological order, looking up tensor values
    and computing node outputs using the configured array backend.

    Attributes:
        graph: The parsed computation graph IR.
        weights: Dict of initializer name -> backend array.
        constants: Dict of constant output name -> backend array.
        values: Dict of tensor name -> backend array (current inference state).
    """

    def __init__(self, graph: Graph, model_path: str):
        """Initialize the executor with a parsed graph and ONNX model.

        Args:
            graph: Parsed computation graph from import_onnx.
            model_path: Path to the .onnx file for weight extraction.
        """
        self.graph = graph
        self.weights = {
            name: engine.to_device(value)
            for name, value in _extract_initializer_data(model_path).items()
        }
        self.constants = {
            name: engine.to_device(value)
            for name, value in _extract_constant_values(model_path).items()
        }
        self.values: Dict[str, Any] = {}

    def run(self, feed_dict: Dict[str, cp.ndarray]) -> Dict[str, cp.ndarray]:
        """Execute the full graph.

        Args:
            feed_dict: Dict mapping graph input name to a CuPy array.
                       Keys must match graph.inputs tensor names.

        Returns:
            Dict mapping graph output name -> backend array.
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
        outputs: Dict[str, cp.ndarray] = {}
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
        inputs: List[Any] = []
        for inp_name in node.inputs:
            if not inp_name:
                # Optional input omitted — pass None so operators that check
                # for it (Gemm C, Conv B, LayerNorm B, etc.) can handle the
                # absence type-agnostically instead of receiving a scalar FP32
                # zero that may be the wrong dtype or shape.
                inputs.append(None)
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
            result = engine.execute_op(op_type, inputs, attrs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to execute node '{node.id}' ({op_type}): {e}"
            ) from e

        # Store outputs
        if isinstance(result, list):
            # Multi-output ops like Split
            for i, out_name in enumerate(node.outputs):
                if out_name and i < len(result):
                    self.values[out_name] = engine.ascontiguousarray(
                        result[i], dtype=engine.array_module().float32
                    )
        else:
            for out_name in node.outputs:
                if out_name:
                    self.values[out_name] = engine.ascontiguousarray(
                        result, dtype=engine.array_module().float32
                    )


class PlannedGraphExecutor(GraphExecutor):
    """Reference executor whose node order is authorized by a C3.4 plan.

    The plan contains decomposed C3.2 kernel steps. This array executor still
    evaluates each high-level node with CuPy, but it refuses to execute unless
    every planned kernel binding/event is valid and every optimized graph node
    is represented in the plan.  It therefore connects and tests the compiler
    stages while executing numerical operators on the designated AEC H200.
    """

    def run_planned(
        self,
        feed_dict: Dict[str, cp.ndarray],
        plan: ExecutionPlan,
    ) -> Dict[str, cp.ndarray]:
        issues = plan.validate()
        if issues:
            raise ValueError(
                "C3.4 execution plan is invalid: " + "; ".join(issues[:10])
            )

        sample_counts = {arr.shape[0] for arr in feed_dict.values() if arr.ndim > 0}
        if len(sample_counts) != 1:
            raise ValueError("Planned execution requires equal input batch sizes")
        if sample_counts and plan.batch_size != next(iter(sample_counts)):
            raise ValueError(
                f"Plan batch size {plan.batch_size} does not match feed batch "
                f"size {next(iter(sample_counts))}"
            )

        planned_node_order: List[str] = []
        seen = set()
        for step in plan.kernel_steps:
            if step.node_id not in seen:
                seen.add(step.node_id)
                planned_node_order.append(step.node_id)

        graph_nodes = set(self.graph.nodes)
        if seen != graph_nodes:
            missing = sorted(graph_nodes - seen)
            unexpected = sorted(seen - graph_nodes)
            raise ValueError(
                "C3.4 plan/optimized-graph mismatch: "
                f"missing_nodes={missing[:10]}, unexpected_nodes={unexpected[:10]}"
            )

        self.values.clear()
        self.values.update(self.weights)
        self.values.update(self.constants)
        self.values.update(feed_dict)
        for node_id in planned_node_order:
            self._execute_node(self.graph.nodes[node_id])

        outputs: Dict[str, cp.ndarray] = {}
        for out_tensor in self.graph.outputs:
            if out_tensor.name not in self.values:
                raise KeyError(f"Planned graph did not produce '{out_tensor.name}'")
            outputs[out_tensor.name] = self.values[out_tensor.name]
        return outputs


class CrossStageReferencePipeline:
    """Connected C3.1→C3.5 FP32 deployment pipeline.

    C3.3 mutates an imported C3.1 graph, C3.4 consumes its C3.2 lowering, and
    :class:`PlannedGraphExecutor` consumes the resulting plan.  The first
    optimized batch is checked against the original unfused FP32 graph.
    """

    def __init__(self, graph: Graph, model_path: str,
                 qualify_optimizations: bool = False) -> None:
        from c33.pipeline import GraphPassPipeline

        baseline_graph = copy.deepcopy(graph)
        self.graph = graph
        self.fusion_result = GraphPassPipeline().run(self.graph)
        stats = self.fusion_result["Fusion"]["stats"]
        if not stats["validation_passed"]:
            raise ValueError("C3.3 produced an invalid optimized graph")

        self.baseline_executor = (
            GraphExecutor(baseline_graph, model_path)
            if qualify_optimizations else None
        )
        self.optimized_executor = PlannedGraphExecutor(self.graph, model_path)
        self._plan_cache: Dict[int, ExecutionPlan] = {}
        self._qualified = not qualify_optimizations
        self.qualification_max_abs_diff: Optional[float] = None
        self.last_plan: Optional[ExecutionPlan] = None

    def _plan_for_batch(self, batch_size: int) -> ExecutionPlan:
        from c34.scheduler import ExecutionScheduler

        if batch_size not in self._plan_cache:
            plan = ExecutionScheduler(self.graph, batch_size=batch_size).build()
            issues = plan.validate()
            if issues:
                raise ValueError(
                    "C3.4 rejected before execution: " + "; ".join(issues[:10])
                )
            self._plan_cache[batch_size] = plan
        return self._plan_cache[batch_size]

    def run(self, feed_dict: Dict[str, cp.ndarray]) -> Dict[str, cp.ndarray]:
        batch_sizes = {arr.shape[0] for arr in feed_dict.values() if arr.ndim > 0}
        if len(batch_sizes) != 1:
            raise ValueError("All graph inputs must have the same batch dimension")
        batch_size = next(iter(batch_sizes))
        plan = self._plan_for_batch(batch_size)
        self.last_plan = plan
        optimized = self.optimized_executor.run_planned(feed_dict, plan)

        if not self._qualified:
            if self.baseline_executor is None:
                raise RuntimeError("Optimization qualification executor is unavailable")
            baseline = self.baseline_executor.run(feed_dict)
            max_diff = 0.0
            for name, expected in baseline.items():
                if name not in optimized:
                    raise KeyError(f"Optimized graph did not produce '{name}'")
                actual = optimized[name]
                if actual.shape != expected.shape:
                    raise ValueError(
                        f"Optimized output '{name}' shape {actual.shape} != "
                        f"baseline {expected.shape}"
                    )
                if actual.size:
                    max_diff = max(
                        max_diff,
                        float(engine.array_module().max(
                            engine.array_module().abs(
                                actual.astype(engine.array_module().float32)
                                - expected.astype(engine.array_module().float32)
                            )
                        ).item()),
                    )
                aligned = engine.array_module().allclose(
                    actual, expected, rtol=1e-3, atol=1e-3
                )
                if hasattr(aligned, "item"):
                    aligned = aligned.item()
                if not bool(aligned):
                    raise ValueError(
                        f"C3.3 numerical qualification failed for '{name}': "
                        f"max_abs_diff={max_diff:.6e}"
                    )
            self.qualification_max_abs_diff = max_diff
            self._qualified = True

        return optimized


def load_and_infer(
    model_path: str,
    input_dir: str,
    output_dir: str,
    batch_size: Optional[int] = None,
    qualify_optimizations: bool = False,
) -> Dict[str, Any]:
    """Load model, run inference, and write outputs.

    This is the high-level orchestrator used by the CLI entry point.

    Args:
        model_path: Path to the .onnx model file.
        input_dir: Directory containing manifest.json and .npy input files.
        output_dir: Directory to write outputs (manifest.json + logits.npy).
        batch_size: Maximum batch size for processing. None means process all at once.
        qualify_optimizations: Compare the first optimized batch with an unfused run.

    Returns:
        Dict with timing and metadata information.
    """
    import json
    import os

    from c31.import_onnx import import_onnx

    t_start = time.perf_counter()
    engine.require_device()

    # 1. Load ONNX graph
    t_parse_start = time.perf_counter()
    graph = import_onnx(model_path)
    t_parse_end = time.perf_counter()

    # 2. Load and validate inputs before doing optimization/planning work.
    input_manifest_path = os.path.join(input_dir, "manifest.json")
    if not os.path.isfile(input_manifest_path):
        raise FileNotFoundError(f"Input manifest not found: {input_manifest_path}")

    with open(input_manifest_path, "r") as f:
        input_manifest = json.load(f)

    input_arrays: Dict[str, cp.ndarray] = {}
    input_meta: Dict[str, Dict[str, Any]] = {}
    for entry in input_manifest.get("tensors", []):
        name = entry["name"]
        if name in input_arrays:
            raise ValueError(f"Duplicate input tensor in manifest: '{name}'")
        file_name = entry["file"]
        file_path = os.path.join(input_dir, file_name)
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Input file not found: {file_path}")
        arr = cp.load(file_path, allow_pickle=False)
        declared_dtype = entry.get("dtype")
        if declared_dtype is not None and cp.dtype(declared_dtype) != arr.dtype:
            raise ValueError(
                f"Input '{name}' dtype mismatch: manifest={declared_dtype}, "
                f"npy={arr.dtype}"
            )
        declared_shape = entry.get("shape")
        if declared_shape is not None and list(arr.shape) != list(declared_shape):
            raise ValueError(
                f"Input '{name}' shape mismatch: manifest={declared_shape}, "
                f"npy={list(arr.shape)}"
            )
        input_arrays[name] = arr
        input_meta[name] = {
            "dtype": entry.get("dtype", "float32"),
            "shape": entry.get("shape", list(arr.shape)),
        }

    # 4. Determine batch size and validate
    if not input_arrays:
        raise ValueError("No input tensors found in manifest")
    expected_inputs = {tensor.name: tensor for tensor in graph.inputs}
    provided_names = set(input_arrays)
    expected_names = set(expected_inputs)
    if provided_names != expected_names:
        raise ValueError(
            "Input manifest names do not match graph inputs: "
            f"missing={sorted(expected_names - provided_names)}, "
            f"unexpected={sorted(provided_names - expected_names)}"
        )

    dtype_map = {
        "FLOAT": cp.dtype("float32"),
        "FLOAT16": cp.dtype("float16"),
        "DOUBLE": cp.dtype("float64"),
        "INT32": cp.dtype("int32"),
        "INT64": cp.dtype("int64"),
        "BOOL": cp.dtype("bool"),
    }
    for name, tensor in expected_inputs.items():
        arr = input_arrays[name]
        expected_dtype = dtype_map.get(tensor.dtype.value)
        if expected_dtype is not None and arr.dtype != expected_dtype:
            raise ValueError(
                f"Input '{name}' dtype {arr.dtype} does not match graph "
                f"dtype {expected_dtype}"
            )
        if len(arr.shape) != len(tensor.shape):
            raise ValueError(
                f"Input '{name}' rank {arr.ndim} does not match graph rank "
                f"{len(tensor.shape)}"
            )
        for axis, (actual, declared) in enumerate(zip(arr.shape, tensor.shape)):
            try:
                expected_dim = int(declared)
            except (TypeError, ValueError):
                continue
            if actual != expected_dim:
                raise ValueError(
                    f"Input '{name}' dimension {axis} is {actual}, expected "
                    f"{expected_dim}"
                )

    sample_counts = {arr.shape[0] for arr in input_arrays.values()}
    if len(sample_counts) != 1:
        raise ValueError(
            f"All graph inputs must have equal sample counts, got {sorted(sample_counts)}"
        )
    first_key = next(iter(input_arrays))
    total_samples = input_arrays[first_key].shape[0]
    if total_samples <= 0:
        raise ValueError("Input tensors must contain at least one sample")

    if batch_size is None or batch_size <= 0:
        actual_batch = total_samples
    else:
        actual_batch = min(batch_size, total_samples)

    # 5. Compile the connected C3.3/C3.4 deployment pipeline and run batches.
    executor = CrossStageReferencePipeline(
        graph, model_path, qualify_optimizations=qualify_optimizations
    )
    output_name = graph.outputs[0].name if graph.outputs else "logits"
    all_outputs: List[cp.ndarray] = []

    t_infer_start = time.perf_counter()

    for start in range(0, total_samples, actual_batch):
        end = min(start + actual_batch, total_samples)

        # Slice inputs
        batch_feed: Dict[str, cp.ndarray] = {}
        for name, arr in input_arrays.items():
            batch_feed[name] = engine.to_device(arr[start:end])

        # Execute
        batch_outputs = executor.run(batch_feed)
        all_outputs.append(batch_outputs[output_name])

    # 6. Concatenate outputs
    final_device_output = engine.array_module().concatenate(all_outputs, axis=0)
    final_device_output = engine.ascontiguousarray(
        final_device_output, dtype=engine.array_module().float32
    )
    engine.synchronize()
    t_infer_end = time.perf_counter()
    # 7. Validate output
    if final_device_output.shape[0] != total_samples:
        raise ValueError(
            f"Output sample count mismatch: got {final_device_output.shape[0]}, "
            f"expected {total_samples}"
        )
    if not bool(cp.all(cp.isfinite(final_device_output)).item()):
        raise ValueError("Output contains non-finite values (NaN or Inf)")

    # 8. Write outputs
    os.makedirs(output_dir, exist_ok=True)

    # Write logits.npy
    output_file = "logits.npy"
    output_path = os.path.join(output_dir, output_file)
    cp.save(output_path, final_device_output)

    # Write manifest.json
    output_manifest = {
        "tensors": [
            {
                "name": output_name,
                "file": output_file,
                "dtype": "float32",
                "shape": list(final_device_output.shape),
            }
        ]
    }
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(output_manifest, f, indent=2)

    t_end = time.perf_counter()

    fusion_stats = executor.fusion_result["Fusion"]["stats"]
    return {
        "model_path": model_path,
        "input_dir": input_dir,
        "output_dir": output_dir,
        "batch_size": actual_batch,
        "backend": "cupy",
        "backend_evidence": engine.runtime_evidence(),
        "total_samples": total_samples,
        "output_shape": list(final_device_output.shape),
        "parse_time_s": t_parse_end - t_parse_start,
        "infer_time_s": t_infer_end - t_infer_start,
        "total_time_s": t_end - t_start,
        "cross_stage_reference": True,
        "optimized_graph_nodes": len(executor.graph.nodes),
        "fusion_stats": fusion_stats,
        "qualification_max_abs_diff": executor.qualification_max_abs_diff,
        "plan_summary": executor.last_plan.summary() if executor.last_plan else None,
    }
