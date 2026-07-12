"""Internal intermediate representation for computation graphs."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ONNSType(str, Enum):
    """Mapping from ONNX TensorProto.DataType to readable names."""

    UNDEFINED = "UNDEFINED"
    FLOAT = "FLOAT"
    UINT8 = "UINT8"
    INT8 = "INT8"
    UINT16 = "UINT16"
    INT16 = "INT16"
    INT32 = "INT32"
    INT64 = "INT64"
    STRING = "STRING"
    BOOL = "BOOL"
    FLOAT16 = "FLOAT16"
    DOUBLE = "DOUBLE"
    UINT32 = "UINT32"
    UINT64 = "UINT64"
    BFLOAT16 = "BFLOAT16"

    @classmethod
    def from_onnx(cls, data_type: int) -> "ONNSType":
        mapping = {
            0: cls.UNDEFINED,
            1: cls.FLOAT,
            2: cls.UINT8,
            3: cls.INT8,
            4: cls.UINT16,
            5: cls.INT16,
            6: cls.INT32,
            7: cls.INT64,
            8: cls.STRING,
            9: cls.BOOL,
            10: cls.FLOAT16,
            11: cls.DOUBLE,
            12: cls.UINT32,
            13: cls.UINT64,
            16: cls.BFLOAT16,
        }
        return mapping.get(data_type, cls.UNDEFINED)


@dataclass
class Tensor:
    """A tensor in the computation graph."""

    name: str
    dtype: ONNSType = ONNSType.UNDEFINED
    shape: List[int | str | None] = field(default_factory=list)
    is_initializer: bool = False
    is_constant: bool = False

    def size_bytes(self) -> Optional[int]:
        """Compute the total byte size if shape is fully concrete."""
        if not self.shape:
            return None
        try:
            dims = [int(d) for d in self.shape]
        except (ValueError, TypeError):
            return None

        type_size = {
            ONNSType.FLOAT: 4,
            ONNSType.FLOAT16: 2,
            ONNSType.BFLOAT16: 2,
            ONNSType.DOUBLE: 8,
            ONNSType.INT8: 1,
            ONNSType.INT16: 2,
            ONNSType.INT32: 4,
            ONNSType.INT64: 8,
            ONNSType.UINT8: 1,
            ONNSType.UINT16: 2,
            ONNSType.UINT32: 4,
            ONNSType.UINT64: 8,
            ONNSType.BOOL: 1,
        }.get(self.dtype, 4)

        total = type_size
        for d in dims:
            total *= d
        return total


@dataclass
class Node:
    """A node (operator) in the computation graph."""

    id: str  # deterministic internal ID
    name: str  # original ONNX name (may be empty)
    op_type: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    domain: str = ""


@dataclass
class Edge:
    """A data dependency edge between two nodes."""

    src_node: str  # source node ID
    dst_node: str  # destination node ID
    tensor: str  # tensor name flowing on this edge


class Graph:
    """Internal computation graph IR.

    Maintains producer/consumer maps for efficient traversal,
    topological ordering, and validation.
    """

    def __init__(self) -> None:
        self.name: str = ""
        self.opsets: Dict[str, int] = {}
        self.nodes: Dict[str, Node] = {}  # node_id -> Node
        self.tensors: Dict[str, Tensor] = {}  # tensor_name -> Tensor
        self.inputs: List[Tensor] = []  # public graph inputs (excluding initializers)
        self.outputs: List[Tensor] = []  # public graph outputs
        self.initializers: Dict[str, Tensor] = {}  # tensor_name -> Tensor (weights)
        self.node_order: List[str] = []  # topological order of node IDs

        # Producer/consumer maps
        self.tensor_producer: Dict[str, str] = {}  # tensor_name -> node_id or "INPUT"
        self.tensor_consumers: Dict[str, List[str]] = {}  # tensor_name -> [node_id, ...]

    def add_node(self, node: Node) -> None:
        self.nodes[node.id] = node

    def get_tensor(self, name: str) -> Optional[Tensor]:
        return self.tensors.get(name)

    def register_tensor(
        self,
        name: str,
        dtype: ONNSType = ONNSType.UNDEFINED,
        shape: Optional[List[int | str | None]] = None,
        is_initializer: bool = False,
        is_constant: bool = False,
    ) -> Tensor:
        if name in self.tensors:
            t = self.tensors[name]
            # Update with new info if available
            if dtype != ONNSType.UNDEFINED:
                t.dtype = dtype
            if shape is not None:
                t.shape = shape
            if is_initializer:
                t.is_initializer = True
            if is_constant:
                t.is_constant = True
            return t
        t = Tensor(
            name=name,
            dtype=dtype,
            shape=[] if shape is None else shape,
            is_initializer=is_initializer,
            is_constant=is_constant,
        )
        self.tensors[name] = t
        return t

    def set_producer(self, tensor_name: str, node_id: str) -> None:
        """Record the producer of a tensor."""
        existing = self.tensor_producer.get(tensor_name)
        if existing is not None and existing != node_id:
            raise ValueError(
                f"Tensor '{tensor_name}' already has producer '{existing}', "
                f"cannot set to '{node_id}'"
            )
        self.tensor_producer[tensor_name] = node_id

    def add_consumer(self, tensor_name: str, node_id: str) -> None:
        """Record that a node consumes a tensor."""
        if tensor_name not in self.tensor_consumers:
            self.tensor_consumers[tensor_name] = []
        if node_id not in self.tensor_consumers[tensor_name]:
            self.tensor_consumers[tensor_name].append(node_id)

    def topological_sort(self) -> List[str]:
        """Kahn's algorithm for topological sort.

        Returns node IDs in topological order.
        Raises ValueError if a cycle is detected.
        """
        in_degree: Dict[str, int] = {}
        for node_id in self.nodes:
            in_degree[node_id] = 0

        # Build adjacency: for each node, count consumers of its outputs
        # Actually, we need: for each node, how many of its inputs are produced
        # by other nodes (i.e., edges from other nodes to this node)
        adj: Dict[str, List[str]] = {nid: [] for nid in self.nodes}

        for tensor_name, consumers in self.tensor_consumers.items():
            producer = self.tensor_producer.get(tensor_name)
            if producer is None or producer in ("INPUT", "INITIALIZER", "CONSTANT"):
                continue
            if producer not in self.nodes:
                continue
            for consumer in consumers:
                if consumer in self.nodes:
                    adj[producer].append(consumer)
                    in_degree[consumer] = in_degree.get(consumer, 0) + 1

        # Kahn's algorithm
        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        sorted_nodes = []

        while queue:
            # Sort for determinism
            queue.sort()
            nid = queue.pop(0)
            sorted_nodes.append(nid)
            for neighbor in adj[nid]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(sorted_nodes) != len(self.nodes):
            cycle_nodes = set(self.nodes.keys()) - set(sorted_nodes)
            raise ValueError(
                f"Graph contains a cycle involving {len(cycle_nodes)} nodes: "
                f"{sorted(cycle_nodes)[:10]}..."
            )

        self.node_order = sorted_nodes
        return sorted_nodes

    def build_edges(self) -> List[Edge]:
        """Build edges from producer/consumer maps.

        Each tensor with a producer node and at least one consumer node
        generates one edge per consumer.
        """
        edges: List[Edge] = []
        for tensor_name, consumers in self.tensor_consumers.items():
            producer = self.tensor_producer.get(tensor_name)
            if producer is None or producer in ("INPUT", "INITIALIZER", "CONSTANT"):
                continue
            if producer not in self.nodes:
                continue
            for consumer in consumers:
                if consumer not in self.nodes:
                    continue
                edges.append(Edge(
                    src_node=producer,
                    dst_node=consumer,
                    tensor=tensor_name,
                ))
        # Sort for determinism
        edges.sort(key=lambda e: (e.src_node, e.dst_node, e.tensor))
        return edges

    def validate(self) -> None:
        """Validate graph consistency.

        Raises ValueError on any violation.
        """
        # 1. Every node input resolves to something
        for node_id, node in self.nodes.items():
            for inp in node.inputs:
                if not inp:  # skip optional empty inputs
                    continue
                if inp not in self.tensors:
                    raise ValueError(
                        f"Node '{node_id}' input '{inp}' is not a registered tensor"
                    )
                producer = self.tensor_producer.get(inp)
                if producer is None:
                    raise ValueError(
                        f"Node '{node_id}' input '{inp}' has no producer"
                    )
                if producer not in ("INPUT", "INITIALIZER", "CONSTANT"):
                    producer_node = self.nodes.get(producer)
                    if producer_node is None or inp not in producer_node.outputs:
                        raise ValueError(
                            f"Node '{node_id}' input '{inp}' has invalid producer "
                            f"'{producer}'"
                        )

        # 2. Every edge tensor is in producer outputs and consumer inputs
        for tensor_name, consumers in self.tensor_consumers.items():
            producer = self.tensor_producer.get(tensor_name)
            if producer is None:
                raise ValueError(
                    f"Tensor '{tensor_name}' has consumers but no producer"
                )
            if producer not in ("INPUT", "INITIALIZER", "CONSTANT"):
                producer_node = self.nodes.get(producer)
                if producer_node is None:
                    raise ValueError(
                        f"Tensor '{tensor_name}' refers to missing producer "
                        f"'{producer}'"
                    )
                if tensor_name not in producer_node.outputs:
                    raise ValueError(
                        f"Tensor '{tensor_name}' claimed producer '{producer}' "
                        f"does not list it as output"
                    )
            for consumer in consumers:
                consumer_node = self.nodes.get(consumer)
                if consumer_node is None or tensor_name not in consumer_node.inputs:
                    raise ValueError(
                        f"Tensor '{tensor_name}' has invalid consumer '{consumer}'"
                    )

        # 3. Every graph output resolves
        for out in self.outputs:
            if out.name not in self.tensors:
                raise ValueError(
                    f"Graph output '{out.name}' is not a registered tensor"
                )
            producer = self.tensor_producer.get(out.name)
            if producer is None:
                raise ValueError(
                    f"Graph output '{out.name}' has no producer"
                )

        # 4. Acyclicity (topological sort)
        self.topological_sort()

    def _get_node_display_name(self, node_id: str) -> str:
        """Get the display name for a node in the JSON output.

        Uses the original name if it exists and is unique; otherwise uses the
        internal ID to avoid ambiguity from duplicate or empty names.
        """
        node = self.nodes[node_id]
        if not node.name:
            return node_id
        # Check if any other node shares the same original name
        for nid, other in self.nodes.items():
            if nid != node_id and other.name == node.name:
                return node_id  # duplicate, use internal ID
        return node.name

    def _should_include_original_name(self, node_id: str) -> bool:
        """Check if the node needs an original_name field."""
        node = self.nodes[node_id]
        if not node.name:
            return False  # empty original name, nothing to add
        # Include if the display name differs from the original name
        return self._get_node_display_name(node_id) != node.name

    @staticmethod
    def _shape_to_json(shape: list[str]) -> list:
        """Convert shape dimensions: numeric strings to ints, keep symbolic as strings."""
        result = []
        for d in shape:
            try:
                result.append(int(d))
            except (ValueError, TypeError):
                result.append(d)
        return result

    def to_dag_json(self) -> dict:
        """Serialize the graph to the C3.1 DAG JSON format."""
        # Ensure topological sort
        if not self.node_order:
            self.topological_sort()

        graph_inputs_json = [
            {
                "name": t.name,
                "dtype": t.dtype.value,
                "shape": self._shape_to_json(t.shape),
            }
            for t in self.inputs
        ]

        graph_outputs_json = [
            {
                "name": t.name,
                "dtype": t.dtype.value,
                "shape": self._shape_to_json(t.shape),
            }
            for t in self.outputs
        ]

        # Sort nodes by topological order
        nodes_json = []
        for nid in self.node_order:
            node = self.nodes[nid]
            display_name = self._get_node_display_name(nid)
            node_entry = {
                "name": display_name,
                "op_type": node.op_type,
                "inputs": list(node.inputs),
                "outputs": list(node.outputs),
            }
            if self._should_include_original_name(nid):
                node_entry["original_name"] = node.name
            if node.attributes:
                node_entry["attributes"] = dict(node.attributes)
            nodes_json.append(node_entry)

        edges = self.build_edges()
        edges_json = [
            {
                "src_node": self._get_node_display_name(e.src_node),
                "dst_node": self._get_node_display_name(e.dst_node),
                "tensor": e.tensor,
            }
            for e in edges
        ]

        result = {
            "format_version": "1.0",
            "graph_inputs": graph_inputs_json,
            "graph_outputs": graph_outputs_json,
            "nodes": nodes_json,
            "edges": edges_json,
        }
        return result


def generate_node_id(name: str, op_type: str, idx: int) -> str:
    """Generate a deterministic internal node ID.

    If the ONNX name is non-empty and unique, use it.
    Otherwise generate a deterministic ID from op_type and index.
    """
    if name:
        return name
    return f"_{op_type}_{idx}"
