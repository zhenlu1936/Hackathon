"""Compatibility import for the shared C3 graph IR.

New code should import from :mod:`c3common.ir.graph`.  This module contains no
second implementation; it preserves the historical ``c31.ir.graph`` path for
external callers.
"""

from c3common.ir.graph import Edge, Graph, Node, ONNSType, Tensor, generate_node_id

__all__ = ["Graph", "Node", "Tensor", "Edge", "ONNSType", "generate_node_id"]
