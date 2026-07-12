"""ONNX model import and graph construction.

Loads an ONNX protobuf, builds the internal IR (Graph),
and handles edge cases like empty/duplicate node names,
optional inputs, Constant nodes, and initializer filtering.
"""

from __future__ import annotations

import base64
from typing import Any

import onnx
from google.protobuf.json_format import MessageToDict
from google.protobuf.message import Message
from onnx import ValueInfoProto

from c3common.ir.graph import Graph, Node, ONNSType, generate_node_id


def _json_safe(value: Any) -> Any:
    """Convert ONNX attribute values into lossless JSON-safe structures."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return {"base64": base64.b64encode(value).decode("ascii")}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Message):
        return MessageToDict(value, preserving_proto_field_name=True)
    return str(value)


def _extract_attribute(attr: onnx.AttributeProto) -> Any:
    """Extract an attribute with ONNX's enum-aware helper."""
    return _json_safe(onnx.helper.get_attribute_value(attr))


def _type_and_shape_from_value_info(
    v: ValueInfoProto,
) -> tuple[ONNSType, list[int | str | None]]:
    """Extract dtype and shape from a ValueInfoProto."""
    t = v.type.tensor_type
    dtype = ONNSType.from_onnx(t.elem_type)
    shape: list[int | str | None] = []
    for d in t.shape.dim:
        if d.dim_param:  # symbolic dimension (e.g., "batch")
            shape.append(d.dim_param)
        elif d.HasField("dim_value"):
            shape.append(d.dim_value)
        else:
            shape.append(None)
    return dtype, shape


def import_onnx(model_path: str) -> Graph:
    """Load an ONNX file and return the internal Graph IR.

    Args:
        model_path: Path to the .onnx file.

    Returns:
        A populated Graph instance.

    Raises:
        FileNotFoundError: If the model file does not exist.
        ValueError: If the graph is malformed or contains cycles.
        onnx.onnx_cpp2py_export.ONNXParserError: If the protobuf is invalid.
    """
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    try:
        model = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    except (onnx.shape_inference.InferenceError, RuntimeError, ValueError):
        # Valid custom-domain models may not have schemas available locally.
        pass
    graph_proto = model.graph

    graph = Graph()
    graph.name = graph_proto.name or ""
    graph.opsets = {
        opset.domain or "ai.onnx": opset.version for opset in model.opset_import
    }

    # ------------------------------------------------------------------
    # 1. Collect initializer names for quick lookup
    # ------------------------------------------------------------------
    init_names: set[str] = set()
    for init in graph_proto.initializer:
        init_names.add(init.name)

    # Also register initializers as tensors
    for init in graph_proto.initializer:
        dtype = ONNSType.from_onnx(init.data_type)
        shape = list(init.dims)
        tensor = graph.register_tensor(
            name=init.name,
            dtype=dtype,
            shape=shape,
            is_initializer=True,
        )
        graph.initializers[init.name] = tensor
        graph.tensor_producer[init.name] = "INITIALIZER"

    # ------------------------------------------------------------------
    # 2. Process graph inputs, excluding initializers
    # ------------------------------------------------------------------
    for inp in graph_proto.input:
        if inp.name in init_names:
            # Skip initializers that are also listed as inputs
            continue
        dtype, shape = _type_and_shape_from_value_info(inp)
        t = graph.register_tensor(
            name=inp.name,
            dtype=dtype,
            shape=shape,
            is_initializer=False,
        )
        graph.inputs.append(t)
        graph.tensor_producer[inp.name] = "INPUT"

    # ------------------------------------------------------------------
    # 3. Process graph outputs
    # ------------------------------------------------------------------
    for out in graph_proto.output:
        dtype, shape = _type_and_shape_from_value_info(out)
        t = graph.register_tensor(
            name=out.name,
            dtype=dtype,
            shape=shape,
            is_initializer=False,
        )
        graph.outputs.append(t)

    # Preserve declared and inferred metadata for intermediate tensors.
    for value_info in graph_proto.value_info:
        dtype, shape = _type_and_shape_from_value_info(value_info)
        graph.register_tensor(value_info.name, dtype=dtype, shape=shape)

    # ------------------------------------------------------------------
    # 4. Process nodes
    # ------------------------------------------------------------------
    used_ids: set[str] = set()

    for idx, node_proto in enumerate(graph_proto.node):
        # Generate deterministic internal ID
        node_id = generate_node_id(node_proto.name, node_proto.op_type, idx)

        # Handle duplicate IDs by appending _idx suffix
        if node_id in used_ids:
            base = node_id
            counter = 1
            while f"{base}_{counter}" in used_ids:
                counter += 1
            node_id = f"{base}_{counter}"
        used_ids.add(node_id)

        # Extract attributes
        attrs = {}
        for attr in node_proto.attribute:
            attrs[attr.name] = _extract_attribute(attr)

        # Preserve optional-input positions; empty names never become edges.
        inputs = list(node_proto.input)

        # Collect outputs (always present)
        outputs = list(node_proto.output)

        node = Node(
            id=node_id,
            name=node_proto.name or "",
            op_type=node_proto.op_type,
            inputs=inputs,
            outputs=outputs,
            attributes=attrs,
            domain=node_proto.domain or "",
        )
        graph.add_node(node)

        # Register output tensors
        for out_name in outputs:
            if out_name:  # skip empty optional outputs
                graph.register_tensor(name=out_name)
                graph.set_producer(out_name, node_id)

        # Handle Constant nodes: register their constant value tensor
        if node_proto.op_type == "Constant":
            for attr in node_proto.attribute:
                if attr.name == "value" and attr.type == 4:  # TENSOR
                    const_t = attr.t
                    const_dtype = ONNSType.from_onnx(const_t.data_type)
                    const_shape = list(const_t.dims)
                    for out_name in node.outputs:
                        if out_name:
                            t = graph.register_tensor(
                                name=out_name,
                                dtype=const_dtype,
                                shape=const_shape,
                                is_constant=True,
                            )
                            graph.tensor_producer[out_name] = node_id
                    break

    # ------------------------------------------------------------------
    # 5. Build consumer maps
    # ------------------------------------------------------------------
    for node_id, node in graph.nodes.items():
        for inp in node.inputs:
            if not inp:  # skip optional empty inputs
                continue
            # Register the tensor if not already known
            if inp not in graph.tensors:
                graph.register_tensor(name=inp)
            graph.add_consumer(inp, node_id)

    # ------------------------------------------------------------------
    # 6. Ensure graph output tensors have appropriate producer records
    # ------------------------------------------------------------------
    for out_tensor in graph.outputs:
        out_name = out_tensor.name
        if out_name not in graph.tensor_producer:
            # If the output is an initializer, it's already registered
            if out_name in init_names:
                continue
            # If it's directly produced by a node, it's already set
            # Otherwise, find which node produces it
            for node_id, node in graph.nodes.items():
                if out_name in node.outputs:
                    graph.tensor_producer[out_name] = node_id
                    break

    # ------------------------------------------------------------------
    # 7. Validate and topologically sort
    # ------------------------------------------------------------------
    graph.validate()

    return graph
