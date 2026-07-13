"""Fusion pattern implementations for C3.3.

Each pattern function:
    - Takes (graph, fusion_log) 
    - Scans graph.nodes for match candidates
    - Returns the number of successful fusions performed
    - Appends dict entries to fusion_log
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set

from c3common.ir.graph import Graph, Node


# ── Elementwise op classification ──────────────────────────────────

ELEMENTWISE_OPS: Set[str] = {
    "Add", "Sub", "Mul", "Div", "Relu", "Erf", "Sqrt", "Exp",
}

ELEMENTWISE_ARITH_OPS: Set[str] = {
    "Add", "Sub", "Mul", "Div",
}

# ── Helper utilities ───────────────────────────────────────────────


def _is_initializer_or_constant(graph: Graph, tensor_name: str) -> bool:
    """Check if a tensor is an initializer, constant, or one-dimensional bias."""
    tensor = graph.tensors.get(tensor_name)
    if tensor is None:
        return False
    return tensor.is_initializer or tensor.is_constant


def _is_graph_output(graph: Graph, tensor_name: str) -> bool:
    """Return whether ``tensor_name`` is part of the public graph ABI."""
    return any(tensor.name == tensor_name for tensor in graph.outputs)


def _dim_compatible(lhs: Any, rhs: Any) -> bool:
    """Conservatively compare two possibly-symbolic dimensions."""
    if lhs is None or rhs is None:
        return True
    try:
        return int(lhs) == int(rhs)
    except (TypeError, ValueError):
        return str(lhs) == str(rhs)


def _is_bias_shape(graph: Graph, tensor_name: str,
                   output_name: str = "") -> bool:
    """Check for a channel bias compatible with the MatMul output.

    C3.3 asks for MatMul followed by *bias* Add, not arbitrary broadcast Add.
    Accept a one-dimensional channel vector or an equivalent broadcast shape
    whose non-channel leading dimensions are all one.  Unknown metadata is not
    enough to prove that a runtime tensor is a bias.
    """
    tensor = graph.tensors.get(tensor_name)
    if tensor is None or not tensor.shape:
        return False
    shape = list(tensor.shape)
    if len(shape) > 1:
        for dim in shape[:-1]:
            try:
                if int(dim) != 1:
                    return False
            except (TypeError, ValueError):
                return False
    if output_name:
        output = graph.tensors.get(output_name)
        if output is not None and output.shape:
            if not _dim_compatible(shape[-1], output.shape[-1]):
                return False
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


def _unique_fused_node_id(graph: Graph, pattern: str, *parts: str) -> str:
    """Return a deterministic fused ID that does not collide in ``graph``."""
    base = _generate_fused_node_id(pattern, *parts)
    fused_id = base
    suffix = 1
    while fused_id in graph.nodes:
        fused_id = f"{base}_{suffix}"
        suffix += 1
    return fused_id


def _replace_region(
    graph: Graph,
    old_node_ids: List[str],
    fused_node: Node,
) -> List[str]:
    """Replace ``old_node_ids`` with one node while preserving the final ABI."""
    final_outputs = {name for name in fused_node.outputs if name}
    removed_tensors: List[str] = []
    for node_id in old_node_ids:
        node = graph.nodes[node_id]
        removed_tensors.extend(
            name for name in node.outputs
            if name and name not in final_outputs
        )

    for node_id in reversed(old_node_ids):
        graph.remove_node(node_id)
    graph.add_node(fused_node)
    for output_name in fused_node.outputs:
        if not output_name:
            continue
        if output_name not in graph.tensors:
            graph.register_tensor(name=output_name)
        graph.set_producer(output_name, fused_node.id)
    return sorted(set(removed_tensors))


def _shapes_maybe_equal(lhs: List[Any], rhs: List[Any]) -> bool:
    """Require equal ranks and reject only provably unequal dimensions."""
    if not lhs or not rhs or len(lhs) != len(rhs):
        return False
    for left, right in zip(lhs, rhs):
        try:
            if int(left) != int(right):
                return False
        except (TypeError, ValueError):
            # Dynamic/symbolic dimensions are checked by the runtime kernel.
            continue
    return True


def _is_dropout_inference(node: Node) -> bool:
    """Check if a Dropout node is in inference mode.

    Dropout is inference-mode if:
    - ratio attribute is 0, OR
    - training_mode attribute is false, OR
    - the node has no explicit training attribute (default inference)
    """
    training = node.attributes.get("training_mode", 0)
    if training not in (0, False, None):
        return False
    # Since opset 12, training_mode is the third input.  Without retained
    # scalar payloads its value cannot be proven false, so reject conservatively.
    if len(node.inputs) >= 3 and node.inputs[2]:
        return False
    return True


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

        # The connected one-launch epilogue kernel supports arbitrary leading
        # dimensions on A with a shared rank-2 weight matrix B.  Batched B
        # remains on the unfused ONNX MatMul path instead of being mislabeled.
        if len(node.inputs) < 2:
            continue
        weight = graph.tensors.get(node.inputs[1])
        if weight is None or len(weight.shape) != 2:
            fusion_log.append({
                "pattern": "FusedMatMulBias",
                "status": "skipped",
                "old_node_ids": [node_id],
                "new_node_id": "",
                "removed_tensors": [],
                "rejection_reason": "Single-kernel epilogue requires rank-2 B",
            })
            continue

        # MatMul must have exactly one output tensor that goes to one consumer
        matmul_outputs = [o for o in node.outputs if o]
        if len(matmul_outputs) != 1:
            continue
        matmul_out = matmul_outputs[0]

        if _is_graph_output(graph, matmul_out):
            fusion_log.append({
                "pattern": "FusedMatMulBias",
                "status": "skipped",
                "old_node_ids": [node_id],
                "new_node_id": "",
                "removed_tensors": [],
                "rejection_reason": "MatMul output is an observable graph output",
            })
            continue

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
        if len(add_inputs) != 2:
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
        if (not _is_initializer_or_constant(graph, bias_input) or
                not _is_bias_shape(graph, bias_input, matmul_out)):
            fusion_log.append({
                "pattern": "FusedMatMulBias",
                "status": "skipped",
                "old_node_ids": [node_id, consumer_id],
                "new_node_id": "",
                "removed_tensors": [],
                "rejection_reason": "Bias input is not a compatible initializer/constant",
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

        if _is_graph_output(graph, conv_out):
            fusion_log.append({
                "pattern": "FusedConv2dBatchNorm",
                "status": "skipped",
                "old_node_ids": [node_id],
                "new_node_id": "",
                "removed_tensors": [],
                "rejection_reason": "Conv output is an observable graph output",
            })
            continue

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

        bn_outputs = [o for o in consumer.outputs if o]
        if len(bn_outputs) != 1:
            fusion_log.append({
                "pattern": "FusedConv2dBatchNorm",
                "status": "skipped",
                "old_node_ids": [node_id, consumer_id],
                "new_node_id": "",
                "removed_tensors": [],
                "rejection_reason": "BatchNormalization has observable auxiliary outputs",
            })
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

        weight = graph.tensors.get(node.inputs[1]) if len(node.inputs) > 1 else None
        channels = weight.shape[0] if weight is not None and weight.shape else None
        if channels is not None:
            incompatible = []
            for name in (scale, bias_bn, mean, var):
                parameter = graph.tensors.get(name)
                if (parameter is None or len(parameter.shape) != 1 or
                        not _dim_compatible(parameter.shape[0], channels)):
                    incompatible.append(name)
            if incompatible:
                fusion_log.append({
                    "pattern": "FusedConv2dBatchNorm",
                    "status": "skipped",
                    "old_node_ids": [node_id, consumer_id],
                    "new_node_id": "",
                    "removed_tensors": [],
                    "rejection_reason": "BN parameter shape does not match Conv channels",
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
    graph_outputs = {tensor.name for tensor in graph.outputs}

    while len(chain) < 5:
        current = graph.nodes.get(current_id)
        if current is None:
            break
        outputs = [o for o in current.outputs if o]
        if len(outputs) != 1:
            break
        out_tensor = outputs[0]
        if out_tensor in graph_outputs:
            break
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
        if len(chain) < 2:
            visited.update(chain)
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

        if _is_graph_output(graph, softmax_out):
            fusion_log.append({
                "pattern": "FusedSoftmaxDropout",
                "status": "skipped",
                "old_node_ids": [node_id],
                "new_node_id": "",
                "removed_tensors": [],
                "rejection_reason": "Softmax output is an observable graph output",
            })
            continue

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

        dropout_outputs = [o for o in consumer.outputs if o]
        if len(dropout_outputs) != 1:
            fusion_log.append({
                "pattern": "FusedSoftmaxDropout",
                "status": "skipped",
                "old_node_ids": [node_id, consumer_id],
                "new_node_id": "",
                "removed_tensors": [],
                "rejection_reason": "Dropout mask output is observable",
            })
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
        if len(add_inputs) != 2:
            continue
        if any(_is_initializer_or_constant(graph, name) for name in add_inputs):
            continue
        left = graph.tensors.get(add_inputs[0])
        right = graph.tensors.get(add_inputs[1])
        if (left is not None and right is not None and left.shape and right.shape and
                (len(left.shape) != len(right.shape) or
                 any(not _dim_compatible(a, b)
                     for a, b in zip(left.shape, right.shape)))):
            continue

        # Add output must go to LayerNormalization
        add_outputs = [o for o in node.outputs if o]
        if len(add_outputs) != 1:
            continue
        add_out = add_outputs[0]

        if _is_graph_output(graph, add_out):
            fusion_log.append({
                "pattern": "FusedResidualNorm",
                "status": "skipped",
                "old_node_ids": [node_id],
                "new_node_id": "",
                "removed_tensors": [],
                "rejection_reason": "Residual Add output is an observable graph output",
            })
            continue

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

        ln_outputs = [o for o in consumer.outputs if o]
        if len(ln_outputs) != 1:
            fusion_log.append({
                "pattern": "FusedResidualNorm",
                "status": "skipped",
                "old_node_ids": [node_id, consumer_id],
                "new_node_id": "",
                "removed_tensors": [],
                "rejection_reason": "LayerNormalization has observable auxiliary outputs",
            })
            continue

        # Check LayerNorm attributes
        axis = consumer.attributes.get("axis", -1)
        epsilon = consumer.attributes.get("epsilon", 1e-5)
        try:
            axis = int(axis)
            epsilon = float(epsilon)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            continue

        # Check LayerNorm has scale/bias parameters
        ln_inputs = [i for i in consumer.inputs if i]
        ln_scale = ln_inputs[1] if len(ln_inputs) > 1 else ""
        ln_bias = ln_inputs[2] if len(ln_inputs) > 2 else ""
        if not ln_scale:
            continue
        normalized = graph.tensors.get(add_out)
        if normalized is not None and normalized.shape:
            rank = len(normalized.shape)
            normalized_axis = axis + rank if axis < 0 else axis
            if normalized_axis < 0 or normalized_axis >= rank:
                continue
            expected_shape = list(normalized.shape[normalized_axis:])
            parameters_valid = True
            for parameter_name in (ln_scale, ln_bias):
                if not parameter_name:
                    continue
                parameter = graph.tensors.get(parameter_name)
                if (parameter is None or len(parameter.shape) != len(expected_shape) or
                        any(not _dim_compatible(a, b)
                            for a, b in zip(parameter.shape, expected_shape))):
                    parameters_valid = False
                    break
            if not parameters_valid:
                continue

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


# ── Executable released-model epilogues ─────────────────────────────


def fuse_gemm_epilogues(graph: Graph,
                        fusion_log: List[Dict[str, Any]]) -> int:
    """Lower Gemm, optional producer Flatten, and optional Relu as one kernel.

    ONNX Gemm already contains its bias operand, so a MatMul->Add matcher cannot
    optimize the released MLP.  This pass forms a semantic GEMM epilogue node
    whose C3.5 implementation is one generated CUDA kernel.
    """
    if not graph.node_order:
        graph.topological_sort()
    fusions = 0
    for node_id in list(graph.node_order):
        gemm = graph.nodes.get(node_id)
        if gemm is None or gemm.op_type != "Gemm":
            continue
        if len([name for name in gemm.outputs if name]) != 1:
            continue

        old_ids: List[str] = []
        fused_inputs = list(gemm.inputs)
        attrs = dict(gemm.attributes)

        # Flatten is a pure reshape.  Absorb it only when its value has no
        # other observer; the generated kernel reads the original contiguous
        # input with the exact ONNX Flatten axis.
        if fused_inputs:
            producer_id = graph.tensor_producer.get(fused_inputs[0])
            producer = graph.nodes.get(producer_id or "")
            if (
                producer is not None
                and producer.op_type == "Flatten"
                and len([name for name in producer.outputs if name]) == 1
                and _has_single_consumer(graph, producer.outputs[0])
                and not _is_graph_output(graph, producer.outputs[0])
            ):
                attrs["_flatten_axis"] = int(
                    producer.attributes.get("axis", 1)
                )
                fused_inputs[0] = producer.inputs[0]
                old_ids.append(producer.id)

        old_ids.append(gemm.id)
        final_node = gemm
        output_name = next(name for name in gemm.outputs if name)
        if not _is_graph_output(graph, output_name) and _has_single_consumer(
            graph, output_name
        ):
            consumer_id = _get_consumers(graph, output_name)[0]
            consumer = graph.nodes.get(consumer_id)
            if (
                consumer is not None
                and consumer.op_type == "Relu"
                and len([name for name in consumer.outputs if name]) == 1
            ):
                attrs["_activation"] = "Relu"
                final_node = consumer
                old_ids.append(consumer.id)

        fused_id = _unique_fused_node_id(
            graph, "FusedGemmEpilogue", old_ids[0], old_ids[-1]
        )
        fused_node = Node(
            id=fused_id,
            name=f"fused_gemm_epilogue_{gemm.id}",
            op_type="FusedGemmEpilogue",
            inputs=fused_inputs,
            outputs=list(final_node.outputs),
            attributes=attrs,
            domain=gemm.domain,
        )
        removed = _replace_region(graph, old_ids, fused_node)
        fusion_log.append({
            "pattern": "FusedGemmEpilogue",
            "status": "fused",
            "old_node_ids": old_ids,
            "new_node_id": fused_id,
            "removed_tensors": removed,
            "rejection_reason": "",
        })
        fusions += 1

    if fusions:
        graph.topological_sort()
    return fusions


def _preferred_conv_for_add(graph: Graph, add: Node,
                            order_index: Dict[str, int]) -> Optional[str]:
    """Choose the latest Conv producer feeding a residual Add.

    Transition blocks can feed Add from both a main-path Conv and a shortcut
    Conv.  Selecting the latest producer is topology-driven and keeps the
    longer main path in the fused epilogue without using node/model names.
    """
    candidates: List[str] = []
    for input_name in add.inputs:
        producer_id = graph.tensor_producer.get(input_name)
        producer = graph.nodes.get(producer_id or "")
        if producer is not None and producer.op_type == "Conv":
            candidates.append(producer.id)
    if not candidates:
        return None
    return max(candidates, key=lambda node_id: order_index.get(node_id, -1))


def fuse_conv_epilogues(graph: Graph,
                        fusion_log: List[Dict[str, Any]]) -> int:
    """Fuse Conv->Relu and Conv->residual Add->Relu into direct kernels."""
    if not graph.node_order:
        graph.topological_sort()
    order_index = {node_id: index for index, node_id in enumerate(graph.node_order)}
    fusions = 0
    for node_id in list(graph.node_order):
        conv = graph.nodes.get(node_id)
        if conv is None or conv.op_type != "Conv":
            continue
        outputs = [name for name in conv.outputs if name]
        if len(outputs) != 1:
            continue
        conv_output = outputs[0]
        if _is_graph_output(graph, conv_output) or not _has_single_consumer(
            graph, conv_output
        ):
            continue
        consumer_id = _get_consumers(graph, conv_output)[0]
        consumer = graph.nodes.get(consumer_id)
        if consumer is None:
            continue

        old_ids: List[str]
        fused_inputs = list(conv.inputs)
        attrs = dict(conv.attributes)
        final_node: Optional[Node] = None
        op_type = "FusedConvActivation"

        if consumer.op_type == "Relu":
            old_ids = [conv.id, consumer.id]
            final_node = consumer
            attrs["_activation"] = "Relu"
        elif consumer.op_type == "Add" and len(consumer.inputs) == 2:
            if _preferred_conv_for_add(graph, consumer, order_index) != conv.id:
                continue
            add_outputs = [name for name in consumer.outputs if name]
            if (
                len(add_outputs) != 1
                or _is_graph_output(graph, add_outputs[0])
                or not _has_single_consumer(graph, add_outputs[0])
            ):
                continue
            relu_id = _get_consumers(graph, add_outputs[0])[0]
            relu = graph.nodes.get(relu_id)
            if relu is None or relu.op_type != "Relu":
                continue
            residual_name = (
                consumer.inputs[1]
                if consumer.inputs[0] == conv_output
                else consumer.inputs[0]
            )
            conv_shape = list(graph.tensors.get(conv_output).shape)
            residual_tensor = graph.tensors.get(residual_name)
            if residual_tensor is None or not _shapes_maybe_equal(
                conv_shape, list(residual_tensor.shape)
            ):
                continue
            attrs["_activation"] = "Relu"
            attrs["_residual_input_index"] = len(fused_inputs)
            fused_inputs.append(residual_name)
            old_ids = [conv.id, consumer.id, relu.id]
            final_node = relu
            op_type = "FusedConvResidualActivation"
        else:
            continue

        fused_id = _unique_fused_node_id(
            graph, op_type, old_ids[0], old_ids[-1]
        )
        fused_node = Node(
            id=fused_id,
            name=f"fused_conv_epilogue_{conv.id}",
            op_type=op_type,
            inputs=fused_inputs,
            outputs=list(final_node.outputs),
            attributes=attrs,
            domain=conv.domain,
        )
        removed = _replace_region(graph, old_ids, fused_node)
        fusion_log.append({
            "pattern": op_type,
            "status": "fused",
            "old_node_ids": old_ids,
            "new_node_id": fused_id,
            "removed_tensors": removed,
            "rejection_reason": "",
        })
        fusions += 1

    if fusions:
        graph.topological_sort()
    return fusions


def fuse_attention_scores(graph: Graph,
                          fusion_log: List[Dict[str, Any]]) -> int:
    """Fuse rank-4 MatMul->Div->Add(mask)->Softmax attention scores.

    The generated kernel supports equal/broadcast batch-head dimensions,
    scalar division, a NumPy-broadcastable rank-four mask, and last-axis
    Softmax.  More general MatMul/Softmax forms remain unfused.
    """
    if not graph.node_order:
        graph.topological_sort()
    fusions = 0
    for node_id in list(graph.node_order):
        matmul = graph.nodes.get(node_id)
        if matmul is None or matmul.op_type != "MatMul" or len(matmul.inputs) != 2:
            continue
        outputs = [name for name in matmul.outputs if name]
        if len(outputs) != 1 or _is_graph_output(graph, outputs[0]):
            continue
        lhs = graph.tensors.get(matmul.inputs[0])
        rhs = graph.tensors.get(matmul.inputs[1])
        result_tensor = graph.tensors.get(outputs[0])
        if (
            lhs is None or rhs is None or result_tensor is None
            or len(lhs.shape) != 4 or len(rhs.shape) != 4
            or len(result_tensor.shape) != 4
            or not _has_single_consumer(graph, outputs[0])
        ):
            continue

        div = graph.nodes.get(_get_consumers(graph, outputs[0])[0])
        if (
            div is None or div.op_type != "Div" or len(div.inputs) != 2
            or div.inputs[0] != outputs[0]
            or len([name for name in div.outputs if name]) != 1
        ):
            continue
        divisor = graph.tensors.get(div.inputs[1])
        if divisor is None or len(divisor.shape) > 1:
            continue
        div_output = next(name for name in div.outputs if name)
        if _is_graph_output(graph, div_output) or not _has_single_consumer(
            graph, div_output
        ):
            continue

        add = graph.nodes.get(_get_consumers(graph, div_output)[0])
        if add is None or add.op_type != "Add" or len(add.inputs) != 2:
            continue
        mask_name = add.inputs[1] if add.inputs[0] == div_output else add.inputs[0]
        mask = graph.tensors.get(mask_name)
        add_outputs = [name for name in add.outputs if name]
        if (
            mask is None or len(mask.shape) > 4 or len(add_outputs) != 1
            or _is_graph_output(graph, add_outputs[0])
            or not _has_single_consumer(graph, add_outputs[0])
        ):
            continue

        softmax = graph.nodes.get(_get_consumers(graph, add_outputs[0])[0])
        if softmax is None or softmax.op_type != "Softmax":
            continue
        axis = int(softmax.attributes.get("axis", -1))
        if axis not in (-1, 3):
            continue

        fused_id = _unique_fused_node_id(
            graph, "FusedAttentionScores", matmul.id, softmax.id
        )
        fused_node = Node(
            id=fused_id,
            name=f"fused_attention_scores_{matmul.id}",
            op_type="FusedAttentionScores",
            inputs=[matmul.inputs[0], matmul.inputs[1], div.inputs[1], mask_name],
            outputs=list(softmax.outputs),
            attributes={"axis": -1},
            domain=matmul.domain,
        )
        old_ids = [matmul.id, div.id, add.id, softmax.id]
        removed = _replace_region(graph, old_ids, fused_node)
        fusion_log.append({
            "pattern": "FusedAttentionScores",
            "status": "fused",
            "old_node_ids": old_ids,
            "new_node_id": fused_id,
            "removed_tensors": removed,
            "rejection_reason": "",
        })
        fusions += 1

    if fusions:
        graph.topological_sort()
    return fusions


def fuse_transpose_reshape(graph: Graph,
                           fusion_log: List[Dict[str, Any]]) -> int:
    """Fuse Transpose->Reshape into one layout-copy kernel.

    The reshape does not change element order after the transpose; the fused
    kernel writes the transposed contiguous order directly using the final
    shape.  Rank is bounded to four because that is the executable kernel ABI.
    """
    if not graph.node_order:
        graph.topological_sort()
    fusions = 0
    for node_id in list(graph.node_order):
        transpose = graph.nodes.get(node_id)
        if transpose is None or transpose.op_type != "Transpose":
            continue
        outputs = [name for name in transpose.outputs if name]
        if (
            len(outputs) != 1 or _is_graph_output(graph, outputs[0])
            or not _has_single_consumer(graph, outputs[0])
        ):
            continue
        input_tensor = graph.tensors.get(transpose.inputs[0])
        if input_tensor is None or not 1 <= len(input_tensor.shape) <= 4:
            continue
        reshape = graph.nodes.get(_get_consumers(graph, outputs[0])[0])
        if (
            reshape is None or reshape.op_type != "Reshape"
            or len(reshape.inputs) < 2
            or len([name for name in reshape.outputs if name]) != 1
        ):
            continue
        perm_attr = transpose.attributes.get("perm")
        if perm_attr is None:
            perm = list(range(len(input_tensor.shape) - 1, -1, -1))
        else:
            perm = [int(value) for value in perm_attr]
        if sorted(perm) != list(range(len(input_tensor.shape))):
            continue
        fused_id = _unique_fused_node_id(
            graph, "FusedTransposeReshape", transpose.id, reshape.id
        )
        fused_node = Node(
            id=fused_id,
            name=f"fused_transpose_reshape_{transpose.id}",
            op_type="FusedTransposeReshape",
            inputs=[transpose.inputs[0], reshape.inputs[1]],
            outputs=list(reshape.outputs),
            attributes={
                "perm": perm,
                "allowzero": int(reshape.attributes.get("allowzero", 0)),
            },
            domain=transpose.domain,
        )
        old_ids = [transpose.id, reshape.id]
        removed = _replace_region(graph, old_ids, fused_node)
        fusion_log.append({
            "pattern": "FusedTransposeReshape",
            "status": "fused",
            "old_node_ids": old_ids,
            "new_node_id": fused_id,
            "removed_tensors": removed,
            "rejection_reason": "",
        })
        fusions += 1
    if fusions:
        graph.topological_sort()
    return fusions


def fuse_layer_normalization_kernels(
    graph: Graph, fusion_log: List[Dict[str, Any]]
) -> int:
    """Replace single-output LayerNormalization with one generated kernel."""
    fusions = 0
    for node_id in list(graph.node_order):
        node = graph.nodes.get(node_id)
        if (
            node is None
            or node.op_type not in {"LayerNormalization", "LayerNorm"}
            or len([name for name in node.outputs if name]) != 1
        ):
            continue
        fused_id = _unique_fused_node_id(
            graph, "FusedLayerNormalization", node.id
        )
        fused_node = Node(
            id=fused_id,
            name=f"fused_layer_normalization_{node.id}",
            op_type="FusedLayerNormalization",
            inputs=list(node.inputs),
            outputs=list(node.outputs),
            attributes=dict(node.attributes),
            domain=node.domain,
        )
        removed = _replace_region(graph, [node.id], fused_node)
        fusion_log.append({
            "pattern": "FusedLayerNormalization",
            "status": "fused",
            "old_node_ids": [node.id],
            "new_node_id": fused_id,
            "removed_tensors": removed,
            "rejection_reason": "",
        })
        fusions += 1
    if fusions:
        graph.topological_sort()
    return fusions


# ── Dead tensor cleanup ──────────────────────────────────────────────


def cleanup_dead_tensors(graph: Graph) -> int:
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
