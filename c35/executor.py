"""C3.5 CuPy graph executor.

Manages the tensor value dictionary, loads weights from ONNX initializers,
and executes nodes in topological order using CuPy.
The remote H200 GPU is the designated AEC execution device.
"""

from __future__ import annotations

import copy
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import cupy as cp
import onnx

from c3common.ir.graph import Graph, Node, ONNSType
from c34.execution_plan import ExecutionPlan
from c35 import engine


def _extract_initializer_data(model_path: str) -> Dict[str, Any]:
    """Extract ONNX initializer tensors as host arrays.

    Args:
        model_path: Path to the .onnx file.

    Returns:
        Dict mapping initializer name to a host array. Each executor decides
        whether upload is eager or driven by the C3.4 transfer timeline.
    """
    model = onnx.load(model_path)
    weights: Dict[str, Any] = {}
    for init in model.graph.initializer:
        weights[init.name] = onnx.numpy_helper.to_array(init)
    return weights


def _extract_constant_values(model_path: str) -> Dict[str, Any]:
    """Extract constant tensor values from ONNX Constant nodes.

    Constant nodes store their value as the 'value' attribute.
    We map each constant-node output name to its CuPy array.

    Args:
        model_path: Path to the .onnx file.

    Returns:
        Dict mapping constant output tensor name to its host value.
    """
    model = onnx.load(model_path)
    constants: Dict[str, Any] = {}
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        supported = {
            "value", "value_float", "value_floats", "value_int", "value_ints",
        }
        values = [attr for attr in node.attribute if attr.name in supported]
        if len(values) != 1:
            raise ValueError(
                f"Constant node {node.name!r} must contain exactly one supported "
                f"numeric value attribute, found {[attr.name for attr in values]}"
            )
        attr = values[0]
        if attr.name == "value":
            value = onnx.numpy_helper.to_array(attr.t)
        elif attr.name == "value_float":
            value = onnx.numpy_helper.to_array(onnx.helper.make_tensor(
                "", onnx.TensorProto.FLOAT, [], [attr.f]
            ))
        elif attr.name == "value_floats":
            value = onnx.numpy_helper.to_array(onnx.helper.make_tensor(
                "", onnx.TensorProto.FLOAT, [len(attr.floats)], attr.floats
            ))
        elif attr.name == "value_int":
            value = onnx.numpy_helper.to_array(onnx.helper.make_tensor(
                "", onnx.TensorProto.INT64, [], [attr.i]
            ))
        else:
            value = onnx.numpy_helper.to_array(onnx.helper.make_tensor(
                "", onnx.TensorProto.INT64, [len(attr.ints)], attr.ints
            ))
        for out_name in node.output:
            if out_name:
                constants[out_name] = value
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
        self._materialize_conv_batchnorm_folds()
        self.values: Dict[str, Any] = {}

    def _materialize_conv_batchnorm_folds(self) -> None:
        """Fold explicit Conv+BN parameters once on the device.

        The C3.1 graph intentionally stores tensor metadata rather than host
        numerical arrays.  The executor is the first stage with the model's
        CuPy initializer store, so it materializes the standard inference fold
        here before C3.4 builds its plan.  The resulting fused node consumes a
        folded weight and bias and executes only Conv semantics.
        """
        values: Dict[str, cp.ndarray] = dict(self.weights)
        values.update(self.constants)

        for node_id in list(self.graph.node_order):
            node = self.graph.nodes.get(node_id)
            if node is None or node.op_type != "FusedConv2dBatchNorm":
                continue
            if node.attributes.get("bn_folded"):
                continue

            offset = int(node.attributes.get("bn_parameter_offset", len(node.inputs)))
            conv_inputs = list(node.inputs[:offset])
            bn_inputs = list(node.inputs[offset:offset + 4])
            if len(conv_inputs) < 2 or len(bn_inputs) != 4:
                raise ValueError(
                    f"Fused Conv+BN node '{node_id}' has an invalid parameter ABI"
                )
            required = [conv_inputs[1], *bn_inputs]
            missing = [name for name in required if name not in values]
            if missing:
                raise ValueError(
                    f"Fused Conv+BN node '{node_id}' cannot resolve initializers: "
                    f"{missing}"
                )

            weight = cp.asarray(values[conv_inputs[1]], dtype=cp.float32)
            scale, bn_bias, mean, variance = (
                cp.asarray(values[name], dtype=cp.float32).reshape(-1)
                for name in bn_inputs
            )
            channels = int(weight.shape[0])
            if any(int(value.size) != channels
                   for value in (scale, bn_bias, mean, variance)):
                raise ValueError(
                    f"Fused Conv+BN node '{node_id}' parameter/channel mismatch"
                )

            if len(conv_inputs) >= 3 and conv_inputs[2]:
                if conv_inputs[2] not in values:
                    raise ValueError(
                        f"Fused Conv+BN node '{node_id}' cannot resolve Conv bias "
                        f"'{conv_inputs[2]}'"
                    )
                conv_bias = cp.asarray(
                    values[conv_inputs[2]], dtype=cp.float32
                ).reshape(-1)
            else:
                conv_bias = cp.zeros_like(mean, dtype=cp.float32)

            epsilon = float(node.attributes.get("bn_epsilon", 1e-5))
            factor = scale / cp.sqrt(variance + cp.float32(epsilon))
            folded_weight = weight * factor.reshape(
                (channels,) + (1,) * (weight.ndim - 1)
            )
            folded_bias = (conv_bias - mean) * factor + bn_bias

            folded_weight_name = f"__c3_folded_{node_id}_weight__"
            folded_bias_name = f"__c3_folded_{node_id}_bias__"
            self.weights[folded_weight_name] = cp.ascontiguousarray(
                folded_weight, dtype=cp.float32
            )
            self.weights[folded_bias_name] = cp.ascontiguousarray(
                folded_bias, dtype=cp.float32
            )
            for name, value in (
                (folded_weight_name, self.weights[folded_weight_name]),
                (folded_bias_name, self.weights[folded_bias_name]),
            ):
                tensor = self.graph.register_tensor(
                    name,
                    dtype=ONNSType.FLOAT,
                    shape=list(value.shape),
                    is_initializer=True,
                )
                self.graph.initializers[name] = tensor
                self.graph.tensor_producer[name] = "INITIALIZER"

            self.graph.replace_node_inputs(
                node_id,
                list(node.inputs),
                [conv_inputs[0], folded_weight_name, folded_bias_name],
            )
            node.attributes["bn_folded"] = True
            node.attributes["bn_parameter_offset"] = 3
            node.attributes["folded_weight"] = folded_weight_name
            node.attributes["folded_bias"] = folded_bias_name

        self.graph.validate()

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
        if op_type in {"Split", "LayerNormalization"}:
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
                    self.values[out_name] = engine.ascontiguousarray(result[i])
        else:
            for out_name in node.outputs:
                if out_name:
                    self.values[out_name] = engine.ascontiguousarray(result)


class PlannedGraphExecutor(GraphExecutor):
    """CuPy executor driven by the complete C3.4 action timeline."""

    def __init__(self, graph: Graph, model_path: str):
        # Avoid GraphExecutor.__init__: eager device conversion there would
        # upload all model data before the plan's H2D actions execute.
        self.graph = graph
        self.host_weights = _extract_initializer_data(model_path)
        self.host_constants = _extract_constant_values(model_path)
        self._preplanned_device_tensors: set[str] = set()
        self._materialize_device_conv_batchnorm_folds()
        self.weights: Dict[str, Any] = {}
        self.constants: Dict[str, Any] = {}
        self.values: Dict[str, Any] = {}
        self.last_execution_trace: List[Dict[str, Any]] = []
        self.last_host_outputs: Dict[str, Any] = {}
        self.memory_trace: List[Dict[str, Any]] = []
        self._temporaries: List[Any] = []
        self._pinned_staging: List[Any] = []
        self._arena_cache: Dict[int, cp.ndarray] = {}
        self._resident_model_tensors: Dict[str, cp.ndarray] = {}
        # CuPy's default pool keeps free blocks in per-stream arenas.  Creating
        # fresh streams for every input batch prevents those blocks from being
        # reused and makes pool reservation grow roughly once per batch.  Keep
        # one physical stream for each logical plan stream for the lifetime of
        # the executor instead.
        self._streams: Dict[int, cp.cuda.Stream] = {}
        self._events_by_plan: Dict[
            int, Tuple[ExecutionPlan, Dict[str, cp.cuda.Event]]
        ] = {}
        self._stream_objects_created = 0
        self._event_objects_created = 0
        self._planned_runs = 0
        self._memory_trace_dropped = 0

    def _materialize_device_conv_batchnorm_folds(self) -> None:
        """Fold Conv+BN with CuPy and feed the result through planned D2D copies."""
        values: Dict[str, Any] = dict(self.host_weights)
        values.update(self.host_constants)
        for node_id in list(self.graph.node_order):
            node = self.graph.nodes.get(node_id)
            if node is None or node.op_type != "FusedConv2dBatchNorm":
                continue
            if node.attributes.get("bn_folded"):
                continue
            offset = int(node.attributes.get("bn_parameter_offset", len(node.inputs)))
            conv_inputs = list(node.inputs[:offset])
            bn_inputs = list(node.inputs[offset:offset + 4])
            if len(conv_inputs) < 2 or len(bn_inputs) != 4:
                raise ValueError(
                    f"Fused Conv+BN node '{node_id}' has an invalid parameter ABI"
                )
            required = [conv_inputs[1], *bn_inputs]
            missing = [name for name in required if name not in values]
            if missing:
                raise ValueError(
                    f"Fused Conv+BN node '{node_id}' cannot resolve initializers: "
                    f"{missing}"
                )
            weight = cp.asarray(values[conv_inputs[1]], dtype=cp.float32)
            scale, bn_bias, mean, variance = (
                cp.asarray(values[name], dtype=cp.float32).reshape(-1)
                for name in bn_inputs
            )
            channels = int(weight.shape[0])
            if any(int(value.size) != channels
                   for value in (scale, bn_bias, mean, variance)):
                raise ValueError(
                    f"Fused Conv+BN node '{node_id}' parameter/channel mismatch"
                )
            if len(conv_inputs) >= 3 and conv_inputs[2]:
                conv_bias = cp.asarray(
                    values[conv_inputs[2]], dtype=cp.float32
                ).reshape(-1)
            else:
                conv_bias = cp.zeros_like(mean, dtype=cp.float32)
            epsilon = float(node.attributes.get("bn_epsilon", 1e-5))
            factor = scale / cp.sqrt(variance + cp.float32(epsilon))
            folded_weight = cp.ascontiguousarray(
                weight * factor.reshape(
                    (channels,) + (1,) * (weight.ndim - 1)
                ),
                dtype=cp.float32,
            )
            folded_bias = cp.ascontiguousarray(
                (conv_bias - mean) * factor + bn_bias,
                dtype=cp.float32,
            )
            folded_weight_name = f"__c3_folded_{node_id}_weight__"
            folded_bias_name = f"__c3_folded_{node_id}_bias__"
            self.host_weights[folded_weight_name] = folded_weight
            self.host_weights[folded_bias_name] = folded_bias
            self._preplanned_device_tensors.update({
                folded_weight_name, folded_bias_name,
            })
            values[folded_weight_name] = folded_weight
            values[folded_bias_name] = folded_bias
            for name, value in (
                (folded_weight_name, folded_weight),
                (folded_bias_name, folded_bias),
            ):
                tensor = self.graph.register_tensor(
                    name,
                    dtype=ONNSType.FLOAT,
                    shape=list(value.shape),
                    is_initializer=True,
                )
                self.graph.initializers[name] = tensor
                self.graph.tensor_producer[name] = "INITIALIZER"
            self.graph.replace_node_inputs(
                node_id,
                list(node.inputs),
                [conv_inputs[0], folded_weight_name, folded_bias_name],
            )
            node.attributes["bn_folded"] = True
            node.attributes["bn_parameter_offset"] = 3
            node.attributes["folded_weight"] = folded_weight_name
            node.attributes["folded_bias"] = folded_bias_name
        self.graph.validate()

    @staticmethod
    def _array_shape_dtype(value: Any) -> Tuple[Tuple[int, ...], Any]:
        if isinstance(value, cp.ndarray):
            return tuple(value.shape), value.dtype
        shape = tuple(getattr(value, "shape", ()))
        dtype = getattr(value, "dtype", None)
        if dtype is None:
            dtype = cp.asarray(value).dtype
        return shape, cp.dtype(dtype)

    @staticmethod
    def _allocation_view(
        arena: cp.ndarray,
        allocation: Any,
        shape: Tuple[int, ...],
        dtype: Any,
    ) -> cp.ndarray:
        dtype = cp.dtype(dtype)
        elements = 1
        for dimension in shape:
            elements *= int(dimension)
        required = elements * dtype.itemsize
        capacity = allocation.capacity_bytes or allocation.size_bytes
        if required > capacity:
            raise ValueError(
                f"Tensor '{allocation.tensor_name}' requires {required} bytes, "
                f"planned capacity is {capacity}"
            )
        byte_view = arena[
            allocation.offset_bytes:allocation.offset_bytes + capacity
        ]
        typed = byte_view.view(dtype)
        return typed[:elements].reshape(shape)

    def _copy_h2d(
        self,
        source: Any,
        destination: cp.ndarray,
        stream: cp.cuda.Stream,
    ) -> None:
        """Issue a plan-controlled H2D or input-device copy."""
        if isinstance(source, cp.ndarray):
            with stream:
                cp.copyto(destination, source.astype(destination.dtype, copy=False))
            return
        if all(hasattr(source, attr) for attr in ("tobytes", "nbytes")):
            payload = source.tobytes(order="C")
            if len(payload) != destination.nbytes:
                raise ValueError(
                    f"H2D byte size mismatch: host={len(payload)}, "
                    f"device={destination.nbytes}"
                )
            pinned = cp.cuda.alloc_pinned_memory(len(payload))
            memoryview(pinned).cast("B")[:len(payload)] = payload
            self._pinned_staging.append(pinned)
            cp.cuda.runtime.memcpyAsync(
                destination.data.ptr,
                pinned.ptr,
                len(payload),
                cp.cuda.runtime.memcpyHostToDevice,
                stream.ptr,
            )
            return
        with stream:
            scalar = cp.asarray(source, dtype=destination.dtype).reshape(
                destination.shape
            )
            cp.copyto(destination, scalar)
            self._temporaries.append(scalar)

    def _execute_planned_node(
        self,
        node: Node,
        kernel_step: Any,
        arena: cp.ndarray,
        allocations: Dict[str, Any],
    ) -> None:
        """Execute one kernel step from the C3.4 plan.

        Resolves input tensors from ``self.values``, creates output arena
        views from the plan allocations, dispatches via the C3.2 kernel
        registry, and stores output views back in ``self.values``.

        This is the unified execution path for both simple graph nodes
        and individual C3.2 sub-kernel steps.  Every emitted kernel name
        resolves to submitted source in ``c32.kernel_registry``.
        """
        from c32.kernel_registry import lookup as kernel_lookup

        # Constant values are pre-loaded via H2D transfers and already
        # resident in self.values.  The constant kernel is a no-op.
        if kernel_step.kernel_name == "constant":
            return

        # Resolve input tensors from the value store.
        inputs: List[Any] = []
        for inp_name in kernel_step.logical_inputs:
            if not inp_name:
                inputs.append(None)
            elif inp_name not in self.values:
                raise KeyError(
                    f"Kernel '{kernel_step.kernel_name}' (node "
                    f"'{kernel_step.node_id}') requires unavailable "
                    f"input '{inp_name}'"
                )
            else:
                inputs.append(self.values[inp_name])

        # Create output arena views.
        outputs: List[cp.ndarray] = []
        for out_name in kernel_step.logical_outputs:
            if not out_name:
                continue
            alloc_id = kernel_step.outputs.get(out_name)
            if alloc_id is None or alloc_id not in allocations:
                raise KeyError(
                    f"Kernel '{kernel_step.kernel_name}' output "
                    f"'{out_name}' has no allocation"
                )
            # Create a view that spans the full planned capacity.
            # _write_result (in the kernel registry) writes exactly
            # the result size regardless of the view shape, so
            # using the full capacity never truncates the result.
            allocation = allocations[alloc_id]
            capacity = allocation.capacity_bytes or allocation.size_bytes
            byte_view = arena[
                allocation.offset_bytes:allocation.offset_bytes + capacity
            ]
            destination = byte_view.view(cp.float32)
            outputs.append(destination)

        # Look up and invoke the kernel.
        kernel_fn = kernel_lookup(kernel_step.kernel_name)
        kernel_fn(
            inputs, outputs,
            dict(kernel_step.operator_params),
            kernel_step.tuning_params,
        )

        # Store output views in the value dict for downstream consumers.
        # Use the allocation's element count (correctly computed by the
        # scheduler) and the graph tensor shape to determine the right
        # multi-dimensional shape for broadcasting compatibility.
        out_idx = 0
        for out_name in kernel_step.logical_outputs:
            if out_name and out_idx < len(outputs):
                flat_view = outputs[out_idx]
                total_elements = flat_view.size
                graph_tensor = self.graph.tensors.get(out_name)
                if graph_tensor is not None and graph_tensor.shape:
                    resolved_shape = self._resolve_shape_from_graph(
                        graph_tensor.shape, total_elements
                    )
                    if (resolved_shape is not None
                            and math.prod(resolved_shape) == total_elements):
                        self.values[out_name] = flat_view[:total_elements].reshape(resolved_shape)
                    else:
                        self.values[out_name] = flat_view
                else:
                    self.values[out_name] = flat_view
                out_idx += 1

    @staticmethod
    def _resolve_shape_from_graph(
        shape: List[Any], total_elements: int,
    ) -> Optional[Tuple[int, ...]]:
        """Resolve a graph tensor shape using the known element count.

        Symbolic dims (``'batch'``, ``'N'``, ``-1``, ``None``) are
        computed as ``total_elements // product_of_known_dims``.
        Returns ``None`` when the shape cannot be determined.
        """
        known: int = 1
        resolved: List[int] = []
        for d in shape:
            if d is None:
                resolved.append(-1)
                continue
            try:
                v = int(d)
            except (ValueError, TypeError):
                if isinstance(d, str) and (
                    d.lower() in ("n", "batch", "b") or d.startswith("unk_")
                ):
                    resolved.append(-1)
                    continue
                return None
            if v > 0:
                known *= v
                resolved.append(v)
            elif v == -1:
                resolved.append(-1)
            else:
                return None

        if known <= 0:
            # All dims are symbolic.  Return a flat shape of total_elements.
            return (total_elements,)

        symbolic_count = sum(1 for d in resolved if d == -1)

        if symbolic_count == 0:
            return tuple(resolved)

        if total_elements % known != 0:
            # Allocation doesn't divide evenly — fall back to flat.
            return None

        remaining = total_elements // known

        if symbolic_count == 1:
            return tuple(remaining if d == -1 else d for d in resolved)

        # Multiple symbolic dims: assume equal split
        import math
        root = round(remaining ** (1.0 / symbolic_count))
        if root ** symbolic_count == remaining and root > 0:
            return tuple(root if d == -1 else d for d in resolved)

        # As a last resort for multiple symbolic dims, assign the
        # full remaining to the first symbolic dim and 1 to others.
        result = list(resolved)
        first = True
        for i, d in enumerate(result):
            if d == -1:
                result[i] = remaining if first else 1
                first = False
        return tuple(result)

    def _runtime_resources(
        self,
        plan: ExecutionPlan,
    ) -> Tuple[Dict[int, cp.cuda.Stream], Dict[str, cp.cuda.Event]]:
        """Return persistent CUDA streams and plan-specific reusable events.

        Every planned invocation synchronizes all participating streams before
        returning, so reusing these events on the next batch is deterministic
        and race-free.  Logical stream IDs are shared across batch-size plans;
        a final partial batch therefore does not create another set of physical
        CUDA streams.
        """
        stream_ids = sorted({
            action.stream_id for action in plan.timeline
            if action.kind not in {"ALLOC", "FREE"}
        })
        for stream_id in stream_ids:
            if stream_id not in self._streams:
                self._streams[stream_id] = cp.cuda.Stream(non_blocking=True)
                self._stream_objects_created += 1

        plan_key = id(plan)
        cached = self._events_by_plan.get(plan_key)
        if cached is None or cached[0] is not plan:
            plan_events = {
                event.event_id: cp.cuda.Event(disable_timing=True)
                for event in plan.events
            }
            self._events_by_plan[plan_key] = (plan, plan_events)
            self._event_objects_created += len(plan_events)
        else:
            plan_events = cached[1]

        return (
            {stream_id: self._streams[stream_id] for stream_id in stream_ids},
            plan_events,
        )

    def runtime_resource_stats(self) -> Dict[str, int]:
        """Expose persistent runtime state for validation and release evidence."""
        return {
            "planned_runs": self._planned_runs,
            "stream_objects_created": self._stream_objects_created,
            "stream_objects_active": len(self._streams),
            "event_objects_created": self._event_objects_created,
            "event_plan_count": len(self._events_by_plan),
            "memory_trace_records": len(self.memory_trace),
            "memory_trace_dropped": self._memory_trace_dropped,
        }

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

        seen = {step.node_id for step in plan.kernel_steps}

        graph_nodes = set(self.graph.nodes)
        if seen != graph_nodes:
            missing = sorted(graph_nodes - seen)
            unexpected = sorted(seen - graph_nodes)
            raise ValueError(
                "C3.4 plan/optimized-graph mismatch: "
                f"missing_nodes={missing[:10]}, unexpected_nodes={unexpected[:10]}"
            )

        if not plan.timeline:
            raise ValueError("C3.4 execution plan has no executable timeline")

        allocations = {
            allocation.alloc_id: allocation for allocation in plan.allocations
        }
        plan_key = id(plan)
        arena_bytes = max(1, plan.pool_stats.peak_reserved_bytes)
        arena = self._arena_cache.get(plan_key)
        if arena is None or int(arena.nbytes) < arena_bytes:
            arena = cp.empty(arena_bytes, dtype=cp.uint8)
            self._arena_cache[plan_key] = arena
        newly_resident_model_tensors: Dict[str, cp.ndarray] = {}
        streams, events = self._runtime_resources(plan)
        pool = cp.get_default_memory_pool()
        pool_before = {
            "used_bytes": int(pool.used_bytes()),
            "reserved_bytes": int(pool.total_bytes()),
        }

        self.values.clear()
        self.last_execution_trace.clear()
        self.last_host_outputs.clear()
        self._temporaries.clear()
        self._pinned_staging.clear()

        for action in plan.timeline:
            trace_entry = {
                "step_index": action.step_index,
                "kind": action.kind,
                "stream_id": action.stream_id,
                "tensor_name": action.tensor_name,
                "event_id": action.event_id,
                "status": "executed",
            }
            self.last_execution_trace.append(trace_entry)
            if action.kind == "ALLOC":
                continue
            if action.kind == "FREE":
                if action.tensor_name is not None:
                    self.values.pop(action.tensor_name, None)
                continue

            stream = streams[action.stream_id]
            if action.kind == "EVENT_WAIT":
                stream.wait_event(events[action.event_id])
            elif action.kind == "EVENT_RECORD":
                events[action.event_id].record(stream)
            elif action.kind == "H2D":
                transfer = plan.transfers[action.ref_index]
                allocation = allocations[transfer.alloc_id]
                is_model_tensor = transfer.tensor_name in plan.weight_slots
                resident = self._resident_model_tensors.get(transfer.tensor_name)
                if resident is None:
                    resident = newly_resident_model_tensors.get(
                        transfer.tensor_name
                    )
                if is_model_tensor and resident is not None:
                    destination = self._allocation_view(
                        arena, allocation, tuple(resident.shape), resident.dtype
                    )
                    trace_entry["status"] = "resident"
                    if resident.data.ptr != destination.data.ptr:
                        with stream:
                            cp.copyto(destination, resident)
                else:
                    if transfer.tensor_name in feed_dict:
                        source = feed_dict[transfer.tensor_name]
                    elif transfer.tensor_name in self.host_weights:
                        source = self.host_weights[transfer.tensor_name]
                    elif transfer.tensor_name in self.host_constants:
                        source = self.host_constants[transfer.tensor_name]
                    else:
                        raise KeyError(
                            f"No H2D source for '{transfer.tensor_name}'"
                        )
                    shape, dtype = self._array_shape_dtype(source)
                    destination = self._allocation_view(
                        arena, allocation, shape, dtype
                    )
                    self._copy_h2d(source, destination, stream)
                    if is_model_tensor:
                        newly_resident_model_tensors[transfer.tensor_name] = (
                            destination
                        )
                self.values[transfer.tensor_name] = destination
            elif action.kind == "KERNEL":
                kernel_step = plan.kernel_steps[action.ref_index]
                node = self.graph.nodes[kernel_step.node_id]
                with stream:
                    self._execute_planned_node(
                        node, kernel_step, arena, allocations
                    )
            elif action.kind == "D2H":
                transfer = plan.transfers[action.ref_index]
                if transfer.tensor_name not in self.values:
                    raise KeyError(
                        f"D2H source '{transfer.tensor_name}' is unavailable"
                    )
                source = self.values[transfer.tensor_name]
                pinned = cp.cuda.alloc_pinned_memory(source.nbytes)
                self._pinned_staging.append(pinned)
                cp.cuda.runtime.memcpyAsync(
                    pinned.ptr,
                    source.data.ptr,
                    source.nbytes,
                    cp.cuda.runtime.memcpyDeviceToHost,
                    stream.ptr,
                )
                self.last_host_outputs[transfer.tensor_name] = pinned
            else:
                raise ValueError(f"Unsupported timeline action: {action.kind}")

        for stream in streams.values():
            stream.synchronize()
        self._resident_model_tensors.update(newly_resident_model_tensors)
        for tensor_name in (
            self._preplanned_device_tensors & newly_resident_model_tensors.keys()
        ):
            self.host_weights.pop(tensor_name, None)
        self._preplanned_device_tensors.difference_update(
            newly_resident_model_tensors
        )
        self._temporaries.clear()
        self._pinned_staging.clear()
        self._planned_runs += 1
        memory_record = {
            "run_index": self._planned_runs,
            "batch_size": plan.batch_size,
            "plan_arena_bytes": arena_bytes,
            "logical_stream_count": len(streams),
            "pool_used_bytes_before": pool_before["used_bytes"],
            "pool_reserved_bytes_before": pool_before["reserved_bytes"],
            "pool_used_bytes_after": int(pool.used_bytes()),
            "pool_reserved_bytes_after": int(pool.total_bytes()),
        }
        # Keep instrumentation bounded for required batch-size-one validation
        # while preserving the initial baseline and most recent observation.
        if len(self.memory_trace) < 64:
            self.memory_trace.append(memory_record)
        else:
            self.memory_trace[-1] = memory_record
            self._memory_trace_dropped += 1

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
    final_device_output: Optional[cp.ndarray] = None

    t_infer_start = time.perf_counter()

    for start in range(0, total_samples, actual_batch):
        end = min(start + actual_batch, total_samples)

        # Slice inputs
        batch_feed: Dict[str, cp.ndarray] = {}
        for name, arr in input_arrays.items():
            batch_feed[name] = engine.to_device(arr[start:end])

        # Execute
        batch_outputs = executor.run(batch_feed)
        batch_output = batch_outputs[output_name]
        if batch_output.dtype != cp.float32:
            raise ValueError(
                f"Output '{output_name}' must be float32, got {batch_output.dtype}"
            )
        if final_device_output is None:
            final_shape = (total_samples,) + tuple(batch_output.shape[1:])
            final_device_output = cp.empty(final_shape, dtype=cp.float32)
        if tuple(batch_output.shape[1:]) != tuple(final_device_output.shape[1:]):
            raise ValueError(
                f"Output '{output_name}' changed non-batch shape from "
                f"{final_device_output.shape[1:]} to {batch_output.shape[1:]}"
            )
        # The planned output is a reusable arena view.  Copy directly into its
        # final position instead of retaining one allocation per batch and
        # allocating another full output for concatenate().
        cp.copyto(final_device_output[start:end], batch_output)
        cp.cuda.get_current_stream().synchronize()

    # 6. Final output was assembled in sample order during batched execution.
    if final_device_output is None:
        raise RuntimeError("Inference produced no output batches")
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
    trace = executor.optimized_executor.last_execution_trace
    trace_counts = {
        kind: sum(action["kind"] == kind for action in trace)
        for kind in (
            "ALLOC", "H2D", "EVENT_WAIT", "KERNEL",
            "EVENT_RECORD", "FREE", "D2H",
        )
    }
    resident_h2d = sum(
        action["kind"] == "H2D" and action["status"] == "resident"
        for action in trace
    )
    backend_evidence = engine.runtime_evidence()
    runtime_resources = executor.optimized_executor.runtime_resource_stats()
    memory_trace = executor.optimized_executor.memory_trace
    if memory_trace:
        backend_evidence.update({
            "batch_count": runtime_resources["planned_runs"],
            "runtime_stream_objects": runtime_resources[
                "stream_objects_active"
            ],
            "runtime_event_objects": runtime_resources[
                "event_objects_created"
            ],
            "max_plan_arena_bytes": max(
                record["plan_arena_bytes"] for record in memory_trace
            ),
            "pool_reserved_bytes_before_first_batch": memory_trace[0][
                "pool_reserved_bytes_before"
            ],
            "pool_reserved_bytes_after_last_batch": memory_trace[-1][
                "pool_reserved_bytes_after"
            ],
        })

    return {
        "model_path": model_path,
        "input_dir": input_dir,
        "output_dir": output_dir,
        "batch_size": actual_batch,
        "backend": "cupy",
        "backend_evidence": backend_evidence,
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
        "plan_runtime": {
            "timeline_actions_consumed": len(trace),
            "action_counts": trace_counts,
            "resident_h2d_actions": resident_h2d,
            "resources": runtime_resources,
            "memory_trace": memory_trace,
        },
    }
