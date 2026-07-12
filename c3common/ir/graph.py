"""Internal intermediate representation for computation graphs.

Shared across all C3 sub-missions.
"""

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
        # Keep dependency metadata synchronized for programmatically-created
        # nodes (notably C3.3 fused nodes).  Importers may call add_consumer
        # again; that method is intentionally idempotent.
        for tensor_name in node.inputs:
            if tensor_name:
                self.add_consumer(tensor_name, node.id)

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
        existing = self.tensor_producer.get(tensor_name)
        if existing is not None and existing != node_id:
            raise ValueError(
                f"Tensor '{tensor_name}' already has producer '{existing}', "
                f"cannot set to '{node_id}'"
            )
        self.tensor_producer[tensor_name] = node_id

    def add_consumer(self, tensor_name: str, node_id: str) -> None:
        if tensor_name not in self.tensor_consumers:
            self.tensor_consumers[tensor_name] = []
        if node_id not in self.tensor_consumers[tensor_name]:
            self.tensor_consumers[tensor_name].append(node_id)

    def topological_sort(self) -> List[str]:
        """Kahn's algorithm for topological sort."""
        in_degree: Dict[str, int] = {}
        for node_id in self.nodes:
            in_degree[node_id] = 0

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

        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        sorted_nodes = []

        while queue:
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
        edges.sort(key=lambda e: (e.src_node, e.dst_node, e.tensor))
        return edges

    def validate(self) -> None:
        for node_id, node in self.nodes.items():
            for inp in node.inputs:
                if not inp:
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
                if node_id not in self.tensor_consumers.get(inp, []):
                    raise ValueError(
                        f"Node '{node_id}' input '{inp}' is missing its consumer index"
                    )

        # Collect dead tensors (no producer, no active consumers) for cleanup
        dead_tensors = []
        for tensor_name, consumers in self.tensor_consumers.items():
            producer = self.tensor_producer.get(tensor_name)
            active_consumers = [c for c in consumers if c in self.nodes]
            if producer is None and not active_consumers:
                dead_tensors.append(tensor_name)
                continue
            if producer is None and active_consumers:
                raise ValueError(f"Tensor '{tensor_name}' has consumers but no producer")
            if producer not in ("INPUT", "INITIALIZER", "CONSTANT"):
                producer_node = self.nodes.get(producer)
                if producer_node is None:
                    raise ValueError(f"Tensor '{tensor_name}' refers to missing producer '{producer}'")
                if tensor_name not in producer_node.outputs:
                    raise ValueError(f"Tensor '{tensor_name}' claimed producer '{producer}' does not list it as output")
            for consumer in active_consumers:
                consumer_node = self.nodes.get(consumer)
                if consumer_node is None or tensor_name not in consumer_node.inputs:
                    raise ValueError(f"Tensor '{tensor_name}' has invalid consumer '{consumer}'")
        # Remove dead tensors
        for tname in dead_tensors:
            self.remove_tensor(tname)

        for out in self.outputs:
            if out.name not in self.tensors:
                raise ValueError(f"Graph output '{out.name}' is not a registered tensor")
            producer = self.tensor_producer.get(out.name)
            if producer is None:
                raise ValueError(f"Graph output '{out.name}' has no producer")

        self.topological_sort()

    def remove_node(self, node_id: str) -> None:
        """Remove a node from the graph and clean up producer/consumer maps."""
        if node_id not in self.nodes:
            return
        node = self.nodes[node_id]

        # Remove this node from consumer lists of its input tensors
        for inp in node.inputs:
            if inp and inp in self.tensor_consumers:
                if node_id in self.tensor_consumers[inp]:
                    self.tensor_consumers[inp].remove(node_id)

        # Remove producer records for outputs of this node
        for out in node.outputs:
            if out and self.tensor_producer.get(out) == node_id:
                del self.tensor_producer[out]

        # Remove from node_order
        if node_id in self.node_order:
            self.node_order.remove(node_id)

        del self.nodes[node_id]

    def remove_tensor(self, tensor_name: str) -> None:
        """Remove a tensor from the graph."""
        if tensor_name not in self.tensors:
            return
        del self.tensors[tensor_name]
        self.tensor_producer.pop(tensor_name, None)
        self.tensor_consumers.pop(tensor_name, None)
        # Also remove from initializers
        self.initializers.pop(tensor_name, None)

    def replace_node_outputs(self, node_id: str,
                              old_outputs: List[str], new_outputs: List[str]) -> None:
        """Replace output tensor names on a node, updating producer maps."""
        node = self.nodes[node_id]
        # Remove old producer mapping
        for old_out in node.outputs:
            if self.tensor_producer.get(old_out) == node_id:
                del self.tensor_producer[old_out]
        node.outputs = list(new_outputs)
        for new_out in new_outputs:
            if new_out not in self.tensors:
                self.register_tensor(name=new_out)
            self.tensor_producer[new_out] = node_id

    def replace_node_inputs(self, node_id: str,
                             old_inputs: List[str], new_inputs: List[str]) -> None:
        """Replace input tensor names on a node, updating consumer maps."""
        node = self.nodes[node_id]
        for old_inp in node.inputs:
            if old_inp and old_inp in self.tensor_consumers:
                if node_id in self.tensor_consumers[old_inp]:
                    self.tensor_consumers[old_inp].remove(node_id)
        node.inputs = list(new_inputs)
        for new_inp in new_inputs:
            if new_inp:
                if new_inp not in self.tensors:
                    self.register_tensor(name=new_inp)
                self.add_consumer(new_inp, node_id)

    def reroute_consumers(self, old_tensor: str, new_tensor: str) -> None:
        """Reroute all consumers of old_tensor to new_tensor."""
        # Fusion commonly preserves the final tensor name.  Treating that as a
        # normal reroute used to rebuild and then delete the same consumer list,
        # silently removing dependency edges and corrupting topological order.
        if old_tensor == new_tensor:
            return
        if old_tensor not in self.tensor_consumers:
            return
        consumers = list(self.tensor_consumers.get(old_tensor, []))
        for consumer_id in consumers:
            node = self.nodes.get(consumer_id)
            if node is None:
                continue
            # Replace old_tensor with new_tensor in node inputs
            new_inputs = [new_tensor if inp == old_tensor else inp for inp in node.inputs]
            self.replace_node_inputs(consumer_id, node.inputs, new_inputs)
            # Register consumer on new tensor
            if new_tensor not in self.tensor_consumers:
                self.tensor_consumers[new_tensor] = []
            if consumer_id not in self.tensor_consumers[new_tensor]:
                self.tensor_consumers[new_tensor].append(consumer_id)
        # Clean up old consumer entries
        del self.tensor_consumers[old_tensor]

    def copy_node_metadata(self, source_id: str, target_id: str) -> None:
        """Copy metadata (attributes, domain, name) from one node to another."""
        if source_id not in self.nodes or target_id not in self.nodes:
            return
        src = self.nodes[source_id]
        dst = self.nodes[target_id]
        dst.attributes = dict(src.attributes)
        dst.domain = src.domain
        if src.name and not dst.name:
            dst.name = src.name

    def _get_node_display_name(self, node_id: str) -> str:
        node = self.nodes[node_id]
        if not node.name:
            return node_id
        for nid, other in self.nodes.items():
            if nid != node_id and other.name == node.name:
                return node_id
        return node.name

    def _should_include_original_name(self, node_id: str) -> bool:
        node = self.nodes[node_id]
        if not node.name:
            return False
        return self._get_node_display_name(node_id) != node.name

    @staticmethod
    def _shape_to_json(shape: list[str]) -> list:
        result = []
        for d in shape:
            try:
                result.append(int(d))
            except (ValueError, TypeError):
                result.append(d)
        return result

    def to_dag_json(self) -> dict:
        if not self.node_order:
            self.topological_sort()

        graph_inputs_json = [
            {"name": t.name, "dtype": t.dtype.value, "shape": self._shape_to_json(t.shape)}
            for t in self.inputs
        ]
        graph_outputs_json = [
            {"name": t.name, "dtype": t.dtype.value, "shape": self._shape_to_json(t.shape)}
            for t in self.outputs
        ]

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
            {"src_node": self._get_node_display_name(e.src_node),
             "dst_node": self._get_node_display_name(e.dst_node),
             "tensor": e.tensor}
            for e in edges
        ]

        return {
            "format_version": "1.0",
            "graph_inputs": graph_inputs_json,
            "graph_outputs": graph_outputs_json,
            "nodes": nodes_json,
            "edges": edges_json,
        }


def generate_node_id(name: str, op_type: str, idx: int) -> str:
    if name:
        return name
    return f"_{op_type}_{idx}"
