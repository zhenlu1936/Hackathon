"""Compatibility package for the shared C3 graph IR."""

from c3common.ir import Edge, Graph, Node, ONNSType, Tensor, generate_node_id

__all__ = ["Graph", "Node", "Tensor", "Edge", "ONNSType", "generate_node_id"]
