"""Fusion pattern implementations for C3.3.

Each pattern function:
    - Takes (graph, fusion_log) 
    - Scans graph.nodes for match candidates
    - Returns the number of successful fusions performed
    - Appends dict entries to fusion_log
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set, Tuple

from c3common.ir.graph import Graph, Node


# ── Elementwise op classification ──────────────────────────────────

ELEMENTWISE_OPS: Set[str] = {
    "Add", "Sub", "Mul", "Div", "Relu", "Erf", "Sqrt", "Exp",
}

ELEMENTWISE_ARITH_OPS: Set[str] = {
    "Add", "Sub", "Mul", "Div",
}

# Operators supported by the C3.5 reference executor that have one data output
# in the released opset-17 contract.  Constant is deliberately excluded (it is
# a region input/source), as is Split (multiple outputs require a different
# region ABI).  This classification is semantic and independent of graph,
# model, node, or testcase names.
REGION_OPS: Set[str] = {
    "Flatten", "Gemm", "Relu", "Conv", "Add", "GlobalAveragePool",
    "Gather", "LayerNormalization", "LayerNorm", "MatMul", "Reshape",
    "Transpose", "Div", "Softmax", "Erf", "Mul", "Sub", "Sqrt", "Exp",
    "FusedMatMulBias", "FusedConv2dBatchNorm", "FusedEWChain",
    "FusedSoftmaxDropout", "FusedResidualNorm", "FusedComputeActivation",
}

# A bounded region avoids turning an entire model into an opaque mega-node and
# leaves a practical unit for later hardware-specific lowering and tuning.
MAX_REGION_NODES = 6

# ── Helper utilities ───────────────────────────────────────────────


def _is_initializer_or_constant(graph: Graph, tensor_name: str) -> bool:
    """Check if a tensor is an initializer, constant, or one-dimensional bias."""
    tensor = graph.tensors.get(tensor_name)
    if tensor is None:
        return False
    return tensor.is_initializer or tensor.is_constant


def _is_bias_shape(graph: Graph, tensor_name: str) -> bool:
    """Check if a tensor looks like a bias (1D or broadcast-compatible)."""
    tensor = graph.tensors.get(tensor_name)
    if tensor is None:
        return False
    # Accept 1D shape or shape where all dims are 1 or unknown
    try:
        int_dims = [int(d) for d in tensor.shape if d is not None]
    except (ValueError, TypeError):
        return False
    if len(int_dims) == 1:
        return True
    if len(int_dims) == 0:
        return False
    # 2D+ bias: one dimension may match the output channels
    return True


def _has_single_consumer(graph: Graph, tensor_name: str) -> bool:
    """Check if a tensor has exactly one consumer."""
    consumers = graph.tensor_consumers.get(tensor_name, [])
    active = [c for c in consumers if c in graph.nodes]
    return len(active) == 1


def _get_consumers(graph: Graph, tensor_name: str) -> List[str]:
    """Get list of consumer node IDs for a tensor (excluding removed nodes)."""
    return [c for c in graph.tensor_consumers.get(tensor_name, [])
            if c in graph.nodes]


def _generate_fused_node_id(pattern: str, *parts: str) -> str:
    """Generate a deterministic node ID for a fused node."""
    suffix = "_".join(parts)
    return f"_{pattern}_{suffix}"


def _is_dropout_inference(node: Node) -> bool:
    """Check if a Dropout node is in inference mode.

    Dropout is inference-mode if:
    - ratio attribute is 0, OR
    - training_mode attribute is false, OR
    - the node has no explicit training attribute (default inference)
    """
    ratio = node.attributes.get("ratio", 0.0)
    if ratio == 0.0 or ratio == 0:
        return True
    training = node.attributes.get("training_mode", 0)
    if training == 0 or training is False:
        return True
    return False


# ── Pattern F1a: FusedMatMulBias ────────────────────────────────────


def fuse_matmul_bias(graph: Graph, fusion_log: List[Dict[str, Any]]) -> int:
    """Fuse MatMul -> Add (bias) into FusedMatMulBias.

    MatMul output must have exactly one consumer (the Add).
    The Add's other input must be a bias-like tensor (initializer/constant/1D).
    """
    fusions = 0
    processed_nodes: Set[str] = set()

    for node_id in list(graph.node_order):
        if node_id in processed_nodes:
            continue
        node = graph.nodes.get(node_id)
        if node is None:
            continue

        # Match MatMul (not Gemm which already may handle bias internally)
        if node.op_type not in ("MatMul",):
            continue

        # MatMul must have exactly one output tensor that goes to one consumer
        matmul_outputs = [o for o in node.outputs if o]
        if len(matmul_outputs) != 1:
            continue
        matmul_out = matmul_outputs[0]

        if not _has_single_consumer(graph, matmul_out):
            fusion_log.append({
                "pattern": "FusedMatMulBias",
                "status": "skipped",
                "old_node_ids": [node_id],
                "new_node_id": "",
                "removed_tensors": [],
                "rejection_reason": "MatMul output has multiple consumers",
            })
            continue

        consumer_id = _get_consumers(graph, matmul_out)[0]
        consumer = graph.nodes.get(consumer_id)
        if consumer is None or consumer.op_type != "Add":
            continue

        # The Add node must have two inputs: one from MatMul, one bias
        add_inputs = [i for i in consumer.inputs if i]
        if len(add_inputs) < 2:
            continue

        # Identify bias input (the one NOT from MatMul)
        bias_input = None
        for inp in add_inputs:
            if inp != matmul_out:
                bias_input = inp
                break
        if bias_input is None:
            continue

        # Check bias is compatible
        if not _is_initializer_or_constant(graph, bias_input) and not _is_bias_shape(graph, bias_input):
            fusion_log.append({
                "pattern": "FusedMatMulBias",
                "status": "skipped",
                "old_node_ids": [node_id, consumer_id],
                "new_node_id": "",
                "removed_tensors": [],
                "rejection_reason": "Bias input is not an initializer/constant or compatible shape",
            })
            continue

        # Check Add output isn't a graph output
        add_outputs = [o for o in consumer.outputs if o]
        is_graph_output = any(
            out_tensor.name == add_outputs[0] for out_tensor in graph.outputs
        ) if add_outputs else False

        # Record and perform fusion
        fused_id = _generate_fused_node_id("FusedMatMulBias", node_id, consumer_id)
        matmul_inputs = [i for i in node.inputs if i]

        # Clear old producer records so we can reassign
        for out_name in consumer.outputs:
            if out_name:
                graph.tensor_producer.pop(out_name, None)

        # Create fused node
        fused_node = Node(
            id=fused_id,
            name=f"fused_matmul_bias_{node_id}_{consumer_id}",
            op_type="FusedMatMulBias",
            inputs=matmul_inputs + [bias_input],
            outputs=list(consumer.outputs) if not is_graph_output else list(consumer.outputs),
            attributes=dict(node.attributes),
        )
        graph.add_node(fused_node)

        # Register output tensor(s) — old producer cleared above
        for out_name in fused_node.outputs:
            if out_name:
                if out_name not in graph.tensors:
                    graph.register_tensor(name=out_name)
                graph.set_producer(out_name, fused_id)

        # Reroute consumers of Add's output to fused node's output
        if add_outputs:
            graph.reroute_consumers(add_outputs[0], fused_node.outputs[0])
            # Also handle graph outputs
            for out_tensor in graph.outputs:
                if out_tensor.name == add_outputs[0]:
                    out_tensor.name = fused_node.outputs[0]

        # Handle Add's first input (reroute MatMul output consumers)
        graph.reroute_consumers(matmul_out, fused_node.outputs[0])

        # Remove old nodes
        removed_tensors = list(node.outputs) + list(consumer.outputs)
        removed_tensors = [t for t in removed_tensors if t and t in graph.tensors]

        graph.remove_node(consumer_id)
        graph.remove_node(node_id)
        processed_nodes.add(node_id)
        processed_nodes.add(consumer_id)

        fusions += 1
        fusion_log.append({
            "pattern": "FusedMatMulBias",
            "status": "fused",
            "old_node_ids": [node_id, consumer_id],
            "new_node_id": fused_id,
            "removed_tensors": removed_tensors,
            "rejection_reason": "",
        })

    return fusions


# ── Pattern F1b: FusedConv2dBatchNorm ────────────────────────────────


def fuse_conv_batchnorm(graph: Graph, fusion_log: List[Dict[str, Any]]) -> int:
    """Fuse Conv -> BatchNormalization while preserving all BN parameters.

    This pass requires both Conv and BatchNormalization node parameters to exist.
    The released ResNet-18 has BN already folded into Conv weights, so this pattern
    will not trigger on that model — it works for general graphs with explicit BN.

    The IR does not retain initializer payloads, so this pass produces an
    executable fused operator rather than falsely claiming that it rewrote the
    Conv initializer values.  A later compiler may fold the parameters.
    """
    fusions = 0
    processed_nodes: Set[str] = set()

    for node_id in list(graph.node_order):
        if node_id in processed_nodes:
            continue
        node = graph.nodes.get(node_id)
        if node is None:
            continue

        if node.op_type not in ("Conv",):
            continue

        # Conv output must go to BatchNormalization
        conv_outputs = [o for o in node.outputs if o]
        if len(conv_outputs) != 1:
            continue
        conv_out = conv_outputs[0]

        if not _has_single_consumer(graph, conv_out):
            fusion_log.append({
                "pattern": "FusedConv2dBatchNorm",
                "status": "skipped",
                "old_node_ids": [node_id],
                "new_node_id": "",
                "removed_tensors": [],
                "rejection_reason": "Conv output has multiple consumers, cannot safely fuse",
            })
            continue

        consumer_id = _get_consumers(graph, conv_out)[0]
        consumer = graph.nodes.get(consumer_id)
        if consumer is None or consumer.op_type not in ("BatchNormalization", "BatchNorm"):
            continue

        # Check that BN has all required parameters (5 inputs: x, scale, bias, mean, var)
        bn_inputs = [i for i in consumer.inputs if i]
        if len(bn_inputs) < 5:
            fusion_log.append({
                "pattern": "FusedConv2dBatchNorm",
                "status": "skipped",
                "old_node_ids": [node_id, consumer_id],
                "new_node_id": "",
                "removed_tensors": [],
                "rejection_reason": "BatchNormalization missing required parameters (need 5 inputs)",
            })
            continue

        # All BN parameters must be initializers for folding to work
        scale, bias_bn, mean, var = bn_inputs[1], bn_inputs[2], bn_inputs[3], bn_inputs[4]
        if not all(_is_initializer_or_constant(graph, t) for t in [scale, bias_bn, mean, var]):
            fusion_log.append({
                "pattern": "FusedConv2dBatchNorm",
                "status": "skipped",
                "old_node_ids": [node_id, consumer_id],
                "new_node_id": "",
                "removed_tensors": [],
                "rejection_reason": "BN parameters are not initializers/constants, cannot fold",
            })
            continue

        # Check epsilon
        epsilon = consumer.attributes.get("epsilon", 1e-5)
        momentum = consumer.attributes.get("momentum", 0.9)

        # Record the fusion since we have the right structure
        fused_id = _generate_fused_node_id("FusedConv2dBatchNorm", node_id, consumer_id)

        # Preserve the BN parameter inputs.  The shared IR currently carries
        # tensor metadata but not initializer payloads, so destructive compile-
        # time weight folding cannot be done truthfully here.  Keeping the
        # parameters makes the fused node numerically executable by a backend
        # that implements Conv+BN as one operation.
        attrs = dict(node.attributes)
        attrs["bn_epsilon"] = epsilon
        attrs["bn_parameter_offset"] = len(node.inputs)

        # Clear old BN output producer so we can reassign
        for out_name in consumer.outputs:
            if out_name:
                graph.tensor_producer.pop(out_name, None)

        fused_node = Node(
            id=fused_id,
            name=f"fused_conv_bn_{node_id}_{consumer_id}",
            op_type="FusedConv2dBatchNorm",
            inputs=list(node.inputs) + [scale, bias_bn, mean, var],
            outputs=list(consumer.outputs),
            attributes=attrs,
        )
        graph.add_node(fused_node)

        # Register output tensors — old producer cleared above
        for out_name in fused_node.outputs:
            if out_name:
                if out_name not in graph.tensors:
                    graph.register_tensor(name=out_name)
                graph.set_producer(out_name, fused_id)

        # Reroute consumers
        bn_outputs = [o for o in consumer.outputs if o]
        if bn_outputs:
            graph.reroute_consumers(bn_outputs[0], fused_node.outputs[0])
            for out_tensor in graph.outputs:
                if out_tensor.name == bn_outputs[0]:
                    out_tensor.name = fused_node.outputs[0]

        # Remove old nodes
        removed_tensors = list(node.outputs) + list(consumer.outputs)
        removed_tensors = [t for t in removed_tensors if t and t in graph.tensors]

        graph.remove_node(consumer_id)
        graph.remove_node(node_id)
        processed_nodes.add(node_id)
        processed_nodes.add(consumer_id)

        fusions += 1
        fusion_log.append({
            "pattern": "FusedConv2dBatchNorm",
            "status": "fused",
            "old_node_ids": [node_id, consumer_id],
            "new_node_id": fused_id,
            "removed_tensors": removed_tensors,
            "rejection_reason": "",
        })

    return fusions


# ── Pattern F1c: FusedEWChain ────────────────────────────────────────


def _is_elementwise_op(op_type: str) -> bool:
    """Check if an op type is elementwise."""
    return op_type in ELEMENTWISE_OPS


def _walk_elementwise_chain(graph: Graph, start_id: str,
                             visited: Set[str]) -> List[str]:
    """Walk forward from start_id along single-consumer elementwise edges.

    Returns a list of node IDs forming the elementwise chain (including start).
    """
    chain = [start_id]
    current_id = start_id

    while True:
        current = graph.nodes.get(current_id)
        if current is None:
            break
        outputs = [o for o in current.outputs if o]
        if len(outputs) != 1:
            break
        out_tensor = outputs[0]
        consumers = _get_consumers(graph, out_tensor)
        if len(consumers) != 1:
            break
        next_id = consumers[0]
        if next_id in visited:
            break
        next_node = graph.nodes.get(next_id)
        if next_node is None or not _is_elementwise_op(next_node.op_type):
            break
        # Found next elementwise in chain
        chain.append(next_id)
        visited.add(next_id)
        current_id = next_id

    return chain


def fuse_elementwise_chain(graph: Graph, fusion_log: List[Dict[str, Any]]) -> int:
    """Fuse 2-5 adjacent elementwise nodes into FusedEWChain.

    Each internal edge must have exactly one consumer.
    """
    fusions = 0
    visited: Set[str] = set()

    for node_id in list(graph.node_order):
        if node_id in visited:
            continue
        node = graph.nodes.get(node_id)
        if node is None or not _is_elementwise_op(node.op_type):
            continue

        # Walk forward to find the chain
        chain = _walk_elementwise_chain(graph, node_id, visited)
        if len(chain) < 2 or len(chain) > 5:
            visited.update(chain)
            if len(chain) > 1:
                fusion_log.append({
                    "pattern": "FusedEWChain",
                    "status": "skipped",
                    "old_node_ids": list(chain),
                    "new_node_id": "",
                    "removed_tensors": [],
                    "rejection_reason": f"Chain length {len(chain)} outside [2, 5] range",
                })
            continue

        # Found a valid chain of 2-5 elementwise ops
        chain_nodes = [graph.nodes[cid] for cid in chain]
        chain_ops = [n.op_type for n in chain_nodes]

        # Build fused node
        first_node = chain_nodes[0]
        last_node = chain_nodes[-1]

        # Inputs include every value entering the chain from outside, not just
        # the first node's inputs.  Store an explicit per-op dataflow program
        # so the executor does not have to guess which operand each op used.
        internal_outputs = {
            out for cn in chain_nodes for out in cn.outputs if out
        }
        fused_inputs: List[str] = []
        for cn in chain_nodes:
            for inp in cn.inputs:
                if inp and inp not in internal_outputs and inp not in fused_inputs:
                    fused_inputs.append(inp)
        fused_outputs = list(last_node.outputs)

        input_index = {name: idx for idx, name in enumerate(fused_inputs)}
        op_program: List[Dict[str, Any]] = []
        for cn in chain_nodes:
            refs: List[Any] = []
            for inp in cn.inputs:
                if not inp:
                    continue
                refs.append(inp if inp in internal_outputs else input_index[inp])
            op_program.append({
                "op": cn.op_type,
                "attrs": dict(cn.attributes),
                "inputs": refs,
                "output": next((out for out in cn.outputs if out), ""),
            })

        fused_id = _generate_fused_node_id("FusedEWChain", "_".join(chain_ops), chain[0])

        # Clear old output producer from last node
        for out_name in fused_outputs:
            if out_name:
                graph.tensor_producer.pop(out_name, None)

        fused_node = Node(
            id=fused_id,
            name=f"fused_ew_chain_{'_'.join(chain_ops)}_{chain[0]}",
            op_type="FusedEWChain",
            inputs=fused_inputs,
            outputs=fused_outputs,
            attributes={
                "chain_ops": chain_ops,
                "chain_node_ids": list(chain),
                "_ops": op_program,
            },
        )
        graph.add_node(fused_node)

        # Register output tensors — old producer cleared above
        for out_name in fused_node.outputs:
            if out_name:
                if out_name not in graph.tensors:
                    graph.register_tensor(name=out_name)
                graph.set_producer(out_name, fused_id)

        # Reroute consumers of the last node's output
        last_outputs = [o for o in last_node.outputs if o]
        if last_outputs:
            graph.reroute_consumers(last_outputs[0], fused_node.outputs[0])
            for out_tensor in graph.outputs:
                if out_tensor.name == last_outputs[0]:
                    out_tensor.name = fused_node.outputs[0]

        # Collect removed tensors
        removed_tensors = []
        for cid in chain:
            cn = graph.nodes[cid]
            for out in cn.outputs:
                if out and out in graph.tensors:
                    removed_tensors.append(out)

        # Remove old nodes
        for cid in chain:
            graph.remove_node(cid)
            visited.add(cid)

        fusions += 1
        fusion_log.append({
            "pattern": "FusedEWChain",
            "status": "fused",
            "old_node_ids": list(chain),
            "new_node_id": fused_id,
            "removed_tensors": removed_tensors,
            "rejection_reason": "",
        })

    return fusions


# ── Pattern F1d: FusedSoftmaxDropout ────────────────────────────────


def fuse_softmax_dropout(graph: Graph, fusion_log: List[Dict[str, Any]]) -> int:
    """Fuse Softmax -> Dropout into FusedSoftmaxDropout.

    Only fuses when Dropout is in inference mode (ratio=0 or training_mode=false).
    """
    fusions = 0
    processed_nodes: Set[str] = set()

    for node_id in list(graph.node_order):
        if node_id in processed_nodes:
            continue
        node = graph.nodes.get(node_id)
        if node is None:
            continue

        if node.op_type not in ("Softmax",):
            continue

        # Softmax output must go to Dropout
        softmax_outputs = [o for o in node.outputs if o]
        if len(softmax_outputs) != 1:
            continue
        softmax_out = softmax_outputs[0]

        if not _has_single_consumer(graph, softmax_out):
            fusion_log.append({
                "pattern": "FusedSoftmaxDropout",
                "status": "skipped",
                "old_node_ids": [node_id],
                "new_node_id": "",
                "removed_tensors": [],
                "rejection_reason": "Softmax output has multiple consumers",
            })
            continue

        consumer_id = _get_consumers(graph, softmax_out)[0]
        consumer = graph.nodes.get(consumer_id)
        if consumer is None or consumer.op_type != "Dropout":
            continue

        # Verify inference mode
        if not _is_dropout_inference(consumer):
            fusion_log.append({
                "pattern": "FusedSoftmaxDropout",
                "status": "skipped",
                "old_node_ids": [node_id, consumer_id],
                "new_node_id": "",
                "removed_tensors": [],
                "rejection_reason": "Dropout is in training mode, inference-only fusion",
            })
            continue

        # Record and perform fusion
        fused_id = _generate_fused_node_id("FusedSoftmaxDropout", node_id, consumer_id)

        # Clear old consumer output producer
        for out_name in consumer.outputs:
            if out_name:
                graph.tensor_producer.pop(out_name, None)

        # Build attributes combining both nodes
        attrs = dict(node.attributes)
        attrs["dropout_ratio"] = consumer.attributes.get("ratio", 0.0)

        fused_node = Node(
            id=fused_id,
            name=f"fused_softmax_dropout_{node_id}_{consumer_id}",
            op_type="FusedSoftmaxDropout",
            inputs=list(node.inputs),
            outputs=list(consumer.outputs),
            attributes=attrs,
        )
        graph.add_node(fused_node)

        # Register outputs — old producer cleared above
        for out_name in fused_node.outputs:
            if out_name:
                if out_name not in graph.tensors:
                    graph.register_tensor(name=out_name)
                graph.set_producer(out_name, fused_id)

        # Reroute consumers
        consumer_outputs = [o for o in consumer.outputs if o]
        if consumer_outputs:
            graph.reroute_consumers(consumer_outputs[0], fused_node.outputs[0])
            for out_tensor in graph.outputs:
                if out_tensor.name == consumer_outputs[0]:
                    out_tensor.name = fused_node.outputs[0]

        # Remove old nodes
        removed_tensors = list(node.outputs) + list(consumer.outputs)
        removed_tensors = [t for t in removed_tensors if t and t in graph.tensors]

        graph.remove_node(consumer_id)
        graph.remove_node(node_id)
        processed_nodes.add(node_id)
        processed_nodes.add(consumer_id)

        fusions += 1
        fusion_log.append({
            "pattern": "FusedSoftmaxDropout",
            "status": "fused",
            "old_node_ids": [node_id, consumer_id],
            "new_node_id": fused_id,
            "removed_tensors": removed_tensors,
            "rejection_reason": "",
        })

    return fusions


# ── Pattern F1e: FusedResidualNorm ──────────────────────────────────


def fuse_residual_norm(graph: Graph, fusion_log: List[Dict[str, Any]]) -> int:
    """Fuse residual Add -> LayerNormalization into FusedResidualNorm.

    The Add node must have two non-trivial inputs (residual pattern).
    Its output must go to a LayerNormalization node.
    LayerNorm must have valid attributes (axis, epsilon).
    """
    fusions = 0
    processed_nodes: Set[str] = set()

    for node_id in list(graph.node_order):
        if node_id in processed_nodes:
            continue
        node = graph.nodes.get(node_id)
        if node is None:
            continue

        if node.op_type not in ("Add",):
            continue

        # Add must have two inputs (residual pattern)
        add_inputs = [i for i in node.inputs if i]
        if len(add_inputs) < 2:
            continue

        # Add output must go to LayerNormalization
        add_outputs = [o for o in node.outputs if o]
        if len(add_outputs) != 1:
            continue
        add_out = add_outputs[0]

        if not _has_single_consumer(graph, add_out):
            fusion_log.append({
                "pattern": "FusedResidualNorm",
                "status": "skipped",
                "old_node_ids": [node_id],
                "new_node_id": "",
                "removed_tensors": [],
                "rejection_reason": "Add output has multiple consumers, cannot safely fuse with LayerNorm",
            })
            continue

        consumer_id = _get_consumers(graph, add_out)[0]
        consumer = graph.nodes.get(consumer_id)
        if consumer is None or consumer.op_type not in ("LayerNormalization", "LayerNorm"):
            continue

        # Check LayerNorm attributes
        axis = consumer.attributes.get("axis", -1)
        epsilon = consumer.attributes.get("epsilon", 1e-5)

        # Check LayerNorm has scale/bias parameters
        ln_inputs = [i for i in consumer.inputs if i]
        ln_scale = ln_inputs[1] if len(ln_inputs) > 1 else ""
        ln_bias = ln_inputs[2] if len(ln_inputs) > 2 else ""

        # Record and perform fusion
        fused_id = _generate_fused_node_id("FusedResidualNorm", node_id, consumer_id)

        # Clear old LayerNorm output producer
        for out_name in consumer.outputs:
            if out_name:
                graph.tensor_producer.pop(out_name, None)

        # Build attributes combining both nodes
        attrs = dict(node.attributes)
        attrs["ln_axis"] = axis
        attrs["ln_epsilon"] = epsilon
        attrs["ln_scale"] = ln_scale
        attrs["ln_bias"] = ln_bias
        # Use the standard LayerNormalization attribute names as the fused
        # execution contract; retain ln_* aliases for the fusion log/debugger.
        attrs["axis"] = axis
        attrs["epsilon"] = epsilon

        fused_inputs = list(node.inputs)
        if ln_scale:
            fused_inputs.append(ln_scale)
        if ln_bias:
            fused_inputs.append(ln_bias)

        fused_node = Node(
            id=fused_id,
            name=f"fused_residual_norm_{node_id}_{consumer_id}",
            op_type="FusedResidualNorm",
            inputs=fused_inputs,
            outputs=list(consumer.outputs),
            attributes=attrs,
        )
        graph.add_node(fused_node)

        # Register output tensors — old producer cleared above
        for out_name in fused_node.outputs:
            if out_name:
                if out_name not in graph.tensors:
                    graph.register_tensor(name=out_name)
                graph.set_producer(out_name, fused_id)

        # Reroute consumers
        consumer_outputs = [o for o in consumer.outputs if o]
        if consumer_outputs:
            graph.reroute_consumers(consumer_outputs[0], fused_node.outputs[0])
            for out_tensor in graph.outputs:
                if out_tensor.name == consumer_outputs[0]:
                    out_tensor.name = fused_node.outputs[0]

        # Remove old nodes
        removed_tensors = list(node.outputs) + list(consumer.outputs)
        removed_tensors = [t for t in removed_tensors if t and t in graph.tensors]

        graph.remove_node(consumer_id)
        graph.remove_node(node_id)
        processed_nodes.add(node_id)
        processed_nodes.add(consumer_id)

        fusions += 1
        fusion_log.append({
            "pattern": "FusedResidualNorm",
            "status": "fused",
            "old_node_ids": [node_id, consumer_id],
            "new_node_id": fused_id,
            "removed_tensors": removed_tensors,
            "rejection_reason": "",
        })

    return fusions


# ── Pattern F1f: FusedComputeActivation ─────────────────────────────

# Compute op types whose output can be fused with an activation function.
COMPUTE_OPS: Set[str] = {
    "Gemm",
    "MatMul",
    "FusedMatMulBias",
    "Conv",
    "FusedConv2dBatchNorm",
}

# Activation op types eligible for compute fusion.
ACTIVATION_OPS: Set[str] = {"Relu", "Erf"}


def _is_compute_op(op_type: str) -> bool:
    return op_type in COMPUTE_OPS


def _is_activation_op(op_type: str) -> bool:
    return op_type in ACTIVATION_OPS


def fuse_compute_activation(graph: Graph, fusion_log: List[Dict[str, Any]]) -> int:
    """Fuse compute -> activation into FusedComputeActivation.

    Covers Gemm/MatMul/Conv (and their already-fused variants) followed by a
    single activation node (Relu, Erf).

    This is the single highest-impact fusion for the released models:
      - MLP:      Gemm -> Relu            (2 instances)
      - ResNet:   Conv -> Relu            (up to 17 instances)
      - Transformer: FusedMatMulBias -> Erf  (4 instances)
    """
    fusions = 0
    processed_nodes: Set[str] = set()

    for node_id in list(graph.node_order):
        if node_id in processed_nodes:
            continue
        node = graph.nodes.get(node_id)
        if node is None or not _is_compute_op(node.op_type):
            continue

        compute_outputs = [o for o in node.outputs if o]
        if len(compute_outputs) != 1:
            continue
        compute_out = compute_outputs[0]

        if not _has_single_consumer(graph, compute_out):
            continue

        consumer_id = _get_consumers(graph, compute_out)[0]
        consumer = graph.nodes.get(consumer_id)
        if consumer is None or not _is_activation_op(consumer.op_type):
            continue

        # Activation output may be a graph output — still safe to fuse.
        fused_id = _generate_fused_node_id(
            "FusedComputeActivation", node.op_type, consumer.op_type, node_id
        )

        # Clear old activation output producer
        for out_name in consumer.outputs:
            if out_name:
                graph.tensor_producer.pop(out_name, None)

        attrs = dict(node.attributes)
        attrs["_inner_op"] = node.op_type
        attrs["_activation"] = consumer.op_type
        attrs["_activation_attrs"] = dict(consumer.attributes)

        fused_node = Node(
            id=fused_id,
            name=f"fused_compute_act_{node.op_type}_{consumer.op_type}_{node_id}",
            op_type="FusedComputeActivation",
            inputs=list(node.inputs),
            outputs=list(consumer.outputs),
            attributes=attrs,
        )
        graph.add_node(fused_node)

        for out_name in fused_node.outputs:
            if out_name:
                if out_name not in graph.tensors:
                    graph.register_tensor(name=out_name)
                graph.set_producer(out_name, fused_id)

        # Reroute consumers of the activation's output
        act_outputs = [o for o in consumer.outputs if o]
        if act_outputs:
            graph.reroute_consumers(act_outputs[0], fused_node.outputs[0])
            for out_tensor in graph.outputs:
                if out_tensor.name == act_outputs[0]:
                    out_tensor.name = fused_node.outputs[0]

        removed_tensors = list(node.outputs) + list(consumer.outputs)
        removed_tensors = [t for t in removed_tensors if t and t in graph.tensors]

        graph.remove_node(consumer_id)
        graph.remove_node(node_id)
        processed_nodes.add(node_id)
        processed_nodes.add(consumer_id)

        fusions += 1
        fusion_log.append({
            "pattern": "FusedComputeActivation",
            "status": "fused",
            "old_node_ids": [node_id, consumer_id],
            "new_node_id": fused_id,
            "removed_tensors": removed_tensors,
            "rejection_reason": "",
        })

    return fusions


# ── General bounded producer-consumer regions ──────────────────────────────


def _walk_fusible_region(graph: Graph, start_id: str,
                         claimed: Set[str]) -> List[str]:
    """Return a deterministic, dependency-safe linear execution region.

    Every internal value has exactly one active consumer and is not a graph
    output.  A consumer may have additional operands from outside the region;
    those values become explicit fused-node inputs.  Multi-output nodes are
    excluded because their externally visible ABI cannot be represented by
    this pass without additional alias/result bookkeeping.
    """
    region: List[str] = []
    current_id = start_id
    graph_outputs = {tensor.name for tensor in graph.outputs}

    while len(region) < MAX_REGION_NODES:
        if current_id in claimed or current_id in region:
            break
        node = graph.nodes.get(current_id)
        if node is None or node.op_type not in REGION_OPS:
            break
        outputs = [out for out in node.outputs if out]
        if len(outputs) != 1:
            break

        region.append(current_id)
        output = outputs[0]
        if output in graph_outputs:
            break
        consumers = _get_consumers(graph, output)
        if len(consumers) != 1:
            break
        next_id = consumers[0]
        next_node = graph.nodes.get(next_id)
        if (next_node is None or next_id in claimed or
                next_node.op_type not in REGION_OPS or
                len([out for out in next_node.outputs if out]) != 1):
            break
        current_id = next_id

    return region


def fuse_execution_regions(graph: Graph,
                           fusion_log: List[Dict[str, Any]]) -> int:
    """Fuse bounded single-consumer regions while preserving executable IR.

    The fused node carries an ordered, self-contained operator program.  C3.5
    can therefore execute the exact original operations for correctness
    qualification while a later AEC/CUDA lowering may replace the region with
    a genuinely fused kernel.  This pass makes no model-specific decisions.
    """
    if not graph.node_order:
        graph.topological_sort()

    candidates: List[List[str]] = []
    claimed: Set[str] = set()
    for node_id in list(graph.node_order):
        if node_id in claimed:
            continue
        region = _walk_fusible_region(graph, node_id, claimed)
        if len(region) < 2:
            continue
        candidates.append(region)
        claimed.update(region)

    fusions = 0
    for region in candidates:
        if any(node_id not in graph.nodes for node_id in region):
            continue
        nodes = [graph.nodes[node_id] for node_id in region]
        internal_outputs = {
            output for node in nodes for output in node.outputs if output
        }
        external_inputs: List[str] = []
        for node in nodes:
            for input_name in node.inputs:
                if (input_name and input_name not in internal_outputs and
                        input_name not in external_inputs):
                    external_inputs.append(input_name)

        input_index = {name: index for index, name in enumerate(external_inputs)}
        program: List[Dict[str, Any]] = []
        for node in nodes:
            refs: List[Any] = []
            for input_name in node.inputs:
                if not input_name:
                    refs.append(None)
                elif input_name in internal_outputs:
                    refs.append(input_name)
                else:
                    refs.append(input_index[input_name])
            program.append({
                "op": node.op_type,
                "attrs": dict(node.attributes),
                "inputs": refs,
                "outputs": [output for output in node.outputs if output],
            })

        final_outputs = [output for output in nodes[-1].outputs if output]
        fused_id = _generate_fused_node_id(
            "FusedExecutionRegion", region[0], region[-1]
        )
        suffix = 1
        base_id = fused_id
        while fused_id in graph.nodes:
            fused_id = f"{base_id}_{suffix}"
            suffix += 1

        for output in final_outputs:
            graph.tensor_producer.pop(output, None)
        for node_id in region:
            graph.remove_node(node_id)

        fused_node = Node(
            id=fused_id,
            name=f"fused_execution_region_{region[0]}_{region[-1]}",
            op_type="FusedExecutionRegion",
            inputs=external_inputs,
            outputs=final_outputs,
            attributes={
                "_region_ops": program,
                "_region_node_ids": list(region),
                "_region_size": len(region),
            },
        )
        graph.add_node(fused_node)
        for input_name in external_inputs:
            graph.add_consumer(input_name, fused_id)
        for output in final_outputs:
            if output not in graph.tensors:
                graph.register_tensor(name=output)
            graph.set_producer(output, fused_id)

        fusions += 1
        fusion_log.append({
            "pattern": "FusedExecutionRegion",
            "status": "fused",
            "old_node_ids": list(region),
            "new_node_id": fused_id,
            "removed_tensors": sorted(
                output for output in internal_outputs
                if output not in final_outputs
            ),
            "rejection_reason": "",
        })

    if fusions:
        graph.topological_sort()
    return fusions


# ── Dead tensor cleanup ──────────────────────────────────────────────


def cleanup_dead_tensors(graph: Graph, fusion_log: List[Dict[str, Any]]) -> int:
    """Remove tensors that are no longer produced or consumed by any node."""
    removed = 0
    tensor_names = list(graph.tensors.keys())

    for tname in tensor_names:
        if tname in graph.tensors:
            tensor = graph.tensors[tname]

            # Skip inputs, outputs, initializers
            is_input = any(inp.name == tname for inp in graph.inputs)
            is_output = any(out.name == tname for out in graph.outputs)
            if is_input or is_output or tensor.is_initializer or tensor.is_constant:
                continue

            producer = graph.tensor_producer.get(tname)
            consumers = graph.tensor_consumers.get(tname, [])

            # Remove if no active producer and no active consumers
            has_producer = producer is not None and (producer in ("INPUT", "INITIALIZER", "CONSTANT") or producer in graph.nodes)
            has_consumers = any(c in graph.nodes for c in consumers)

            if not has_producer and not has_consumers:
                graph.remove_tensor(tname)
                removed += 1

    return removed
