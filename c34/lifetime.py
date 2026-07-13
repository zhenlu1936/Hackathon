"""Tensor lifetime analysis for C3.4 memory reuse.

Computes first_use and last_use for every tensor across the kernel schedule.
These intervals drive slot reuse: two tensors with non-overlapping lifetimes
can share the same physical device buffer.

Feature B: Intermediate tensor lifetime memory reuse.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from c3common.ir.graph import Graph
from c32.kernel_spec import KernelSpecRef
from c34.execution_plan import LifetimeInterval


_DTYPE_SIZE_BYTES = {
    "FLOAT": 4,
    "FLOAT16": 2,
    "BFLOAT16": 2,
    "DOUBLE": 8,
    "INT8": 1,
    "UINT8": 1,
    "BOOL": 1,
    "INT16": 2,
    "UINT16": 2,
    "INT32": 4,
    "UINT32": 4,
    "INT64": 8,
    "UINT64": 8,
}


def _tensor_size_bytes(
    graph: Graph,
    tensor_name: str,
    batch_size: int = 1,
) -> int:
    """Get tensor size in bytes, defaulting to 0 if unknown.

    Resolves symbolic batch dimensions (``"N"``, ``"batch"``, ``-1``)
    to the concrete batch_size.
    """
    tensor = graph.tensors.get(tensor_name)
    if tensor is not None:
        # ONNX represents a scalar with an empty dimension list.  Constant
        # scalar outputs still occupy one element on device.
        if tensor.is_constant and not tensor.shape:
            return _DTYPE_SIZE_BYTES.get(tensor.dtype.value, 4)
        # Resolve all shapes here instead of calling Tensor.size_bytes() first:
        # that helper does not concretize dynamic dimensions, and an integer
        # ``-1`` would otherwise produce a negative allocation size.
        if tensor.shape:
            return _compute_size_with_batch(
                tensor.shape,
                batch_size,
                elem_bytes=_DTYPE_SIZE_BYTES.get(tensor.dtype.value, 4),
            )
    return 0


def _compute_size_with_batch(
    shape: list,
    batch_size: int = 1,
    elem_bytes: int = 4,
) -> int:
    """Compute tensor byte size, replacing symbolic batch dims with batch_size."""
    total = elem_bytes
    for d in shape:
        if d is None:
            return 0
        try:
            dimension = int(d)
        except (ValueError, TypeError):
            if isinstance(d, str) and (
                d.lower() in ("n", "batch", "b") or d.startswith("unk__")
            ):
                dimension = batch_size
            else:
                return 0  # unknown symbolic dim
        if dimension == -1:
            dimension = batch_size
        elif dimension < 0:
            return 0
        total *= dimension
    return total


def compute_lifetimes(
    graph: Graph,
    kernel_schedule: List[KernelSpecRef],
    batch_size: int = 1,
) -> Dict[str, LifetimeInterval]:
    """Compute lifetime intervals for all tensors across the kernel schedule.

    Args:
        graph: The computation graph IR.
        kernel_schedule: Flat list of kernel references in execution order.
        batch_size: Concrete batch size for resolving symbolic dims.

    Returns:
        Dict mapping tensor_name → LifetimeInterval with first_use and last_use
        indices into the kernel_schedule list.
    """
    # Collect input/output tensor names
    input_names = {t.name for t in graph.inputs}
    output_names = {t.name for t in graph.outputs}
    initializer_names = set(graph.initializers.keys())

    # Track tensor usage across kernel steps
    # Step 1: Scan all kernels and record tensor usage per step
    tensor_first: Dict[str, int] = {}
    tensor_last: Dict[str, int] = {}
    tensor_producers: Dict[str, List[str]] = {}   # tensor_name -> [node_id]
    tensor_consumers: Dict[str, List[int]] = {}    # tensor_name -> [step_index]

    for step_idx, krn in enumerate(kernel_schedule):
        # Kernel inputs are read at this step
        for inp in krn.inputs:
            if not inp:
                continue
            if inp not in tensor_first:
                tensor_first[inp] = step_idx
            tensor_last[inp] = step_idx
            if inp not in tensor_consumers:
                tensor_consumers[inp] = []
            tensor_consumers[inp].append(step_idx)

        # Kernel outputs are written at this step
        for out in krn.outputs:
            if not out:
                continue
            if out not in tensor_first:
                tensor_first[out] = step_idx
            tensor_last[out] = step_idx
            if out not in tensor_producers:
                tensor_producers[out] = []

            # Find the originating node ID
            node_id = _find_node_id_for_tensor(graph, out)
            if node_id and node_id not in tensor_producers[out]:
                tensor_producers[out].append(node_id)

    # Build lifetime intervals
    lifetimes: Dict[str, LifetimeInterval] = {}
    for tname in set(list(tensor_first.keys()) + list(tensor_last.keys())):
        first = tensor_first.get(tname, -1)
        last = tensor_last.get(tname, -1)
        if first < 0:
            continue

        size_bytes = _tensor_size_bytes(graph, tname, batch_size)
        tensor = graph.tensors.get(tname)
        is_weight = tname in initializer_names or bool(
            tensor is not None and tensor.is_constant
        )
        is_output = tname in output_names
        is_input = tname in input_names and tname not in initializer_names

        lifetimes[tname] = LifetimeInterval(
            tensor_name=tname,
            first_use=first,
            last_use=last,
            size_bytes=size_bytes,
            is_weight=is_weight,
            is_output=is_output,
            is_input=is_input,
            producers=tensor_producers.get(tname, []),
            consumers=[str(index) for index in tensor_consumers.get(tname, [])],
        )

    return lifetimes


def find_overlap_groups(
    lifetimes: Dict[str, LifetimeInterval],
) -> List[List[LifetimeInterval]]:
    """Group tensors with non-overlapping lifetimes that can share slots.

    Uses a greedy interval-scheduling approach:
    1. Sort tensors by first_use.
    2. Maintain a list of active groups (slots).
    3. For each tensor, check if any active group's last_use < tensor's first_use.
       If yes, assign to that group (reuse).
       If no, start a new group.

    Returns:
        List of groups, where each group is a list of LifetimeIntervals
        that share the same slot.
    """
    # Sort by first_use
    sorted_tensors = sorted(
        lifetimes.values(),
        key=lambda li: (li.first_use, li.last_use),
    )

    groups: List[List[LifetimeInterval]] = []

    for interval in sorted_tensors:
        # Skip weights (model-lifetime residency, not reused)
        if interval.is_weight:
            groups.append([interval])
            continue

        # Skip graph inputs (already allocated before execution)
        if interval.is_input:
            groups.append([interval])
            continue

        placed = False
        for group in groups:
            # Check if this tensor's lifetime doesn't overlap with any tensor in the group
            # Weight groups and input groups can't be reused
            group_has_weight = any(g.is_weight for g in group)
            group_has_input = any(g.is_input for g in group)
            if group_has_weight or group_has_input:
                continue

            # Find the max last_use in the group
            max_last = max(g.last_use for g in group)
            if max_last < interval.first_use:
                group.append(interval)
                placed = True
                break

        if not placed:
            groups.append([interval])

    return groups


def _find_node_id_for_tensor(graph: Graph, tensor_name: str) -> Optional[str]:
    """Find which node produces this tensor."""
    producer = graph.tensor_producer.get(tensor_name)
    if producer and producer not in ("INPUT", "INITIALIZER", "CONSTANT"):
        return producer
    return None
