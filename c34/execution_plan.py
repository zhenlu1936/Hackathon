"""Execution plan data structures for C3.4 memory planning and scheduling.

The ExecutionPlan is the central artifact produced by the scheduler.
It captures allocations, transfers, kernel steps, event dependencies,
lifetime intervals, reuse decisions, and pool statistics — all the
evidence needed for code review of features A–E.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── Allocation ─────────────────────────────────────────────────────

@dataclass
class Allocation:
    """A device memory allocation binding a logical tensor to a physical slot."""

    alloc_id: str          # unique allocation ID
    tensor_name: str       # logical tensor name from the IR
    slot_id: int           # physical device slot ID
    size_bytes: int        # requested allocation size
    alignment: int = 64    # alignment in bytes
    is_weight: bool = False   # model-lifetime residency
    is_output: bool = False   # final graph output (D2H needed)
    offset_bytes: int = 0     # byte offset in the physical device arena
    capacity_bytes: int = 0   # aligned physical capacity at this offset


# ── Transfer ───────────────────────────────────────────────────────

@dataclass
class Transfer:
    """H2D or D2H data transfer step."""

    kind: str               # "H2D" or "D2H"
    tensor_name: str        # logical tensor name
    alloc_id: str           # target allocation
    size_bytes: int
    stream_id: int          # copy stream ID
    event_id: Optional[str] = None  # signal event after completion
    depends_on: List[str] = field(default_factory=list)


# ── Kernel step ────────────────────────────────────────────────────

@dataclass
class KernelStep:
    """A single kernel launch in the execution schedule."""

    step_index: int               # position in the global schedule
    kernel_name: str              # e.g. "matmul_fp32", "relu_fp32"
    node_id: str                  # originating graph node
    logical_inputs: List[str] = field(default_factory=list)
    logical_outputs: List[str] = field(default_factory=list)
    inputs: Dict[str, str] = field(default_factory=dict)   # tensor_name -> alloc_id
    outputs: Dict[str, str] = field(default_factory=dict)  # tensor_name -> alloc_id
    stream_id: int = 1            # compute stream ID (0 = copy stream)
    depends_on: List[str] = field(default_factory=list)    # event IDs to wait on
    signals: List[str] = field(default_factory=list)       # event IDs to signal
    tuning_params: Optional[Dict[str, int]] = None         # block_x, grid_x, smem_bytes
    operator_params: Dict[str, Any] = field(default_factory=dict)
    precision_profile: Optional[Dict[str, Optional[str]]] = None
    lowered_kernels: List[str] = field(default_factory=list)  # C3.2 review metadata


# ── Event dependency ───────────────────────────────────────────────

@dataclass
class EventDep:
    """Cross-stream event dependency."""

    event_id: str
    src_stream: int       # stream that signals the event
    dst_stream: int       # stream that waits on the event
    description: str = ""


# ── Executable timeline ───────────────────────────────────────────

@dataclass
class TimelineStep:
    """One ordered action consumed by the C3.5 CuPy plan runtime.

    ``ref_index`` addresses ``transfers`` for H2D/D2H actions and
    ``kernel_steps`` for KERNEL actions.  EVENT_WAIT/EVENT_RECORD use
    ``event_id``; ALLOC/FREE use ``alloc_id``.
    """

    step_index: int
    kind: str
    stream_id: int = 0
    ref_index: Optional[int] = None
    alloc_id: Optional[str] = None
    event_id: Optional[str] = None
    tensor_name: Optional[str] = None


# ── Lifetime record ────────────────────────────────────────────────

@dataclass
class LifetimeInterval:
    """First and last use of a tensor across the kernel schedule."""

    tensor_name: str
    first_use: int        # step index of first use (read or write)
    last_use: int         # step index of last use (read)
    size_bytes: int
    is_weight: bool = False
    is_output: bool = False
    is_input: bool = False
    producers: List[str] = field(default_factory=list)   # kernel step indices
    consumers: List[str] = field(default_factory=list)   # kernel step indices


# ── Pool statistics ────────────────────────────────────────────────

@dataclass
class PoolStats:
    """Memory pool tracking counters for C3.5 tuning and code review."""

    requested_bytes: int = 0         # total bytes requested by all allocs
    reserved_bytes: int = 0          # total device memory reserved
    active_bytes: int = 0            # currently in-use bytes
    peak_reserved_bytes: int = 0     # high-water mark
    internal_fragmentation: int = 0  # wasted bytes inside allocated blocks
    reuse_hits: int = 0              # number of times a freed block was reused
    total_allocs: int = 0
    total_frees: int = 0
    free_list_blocks: int = 0        # current number of free blocks


# ── Execution plan ─────────────────────────────────────────────────

@dataclass
class ExecutionPlan:
    """The complete memory-and-stream execution plan.

    This is the main deliverable for C3.4. It should be fully reviewable
    without reading the runtime — allocation IDs, sizes, kernel bindings,
    stream IDs, events, lifetimes, and reuse decisions are all visible.
    """

    model_name: str = ""
    batch_size: int = 1

    # Ordered steps
    allocations: List[Allocation] = field(default_factory=list)
    transfers: List[Transfer] = field(default_factory=list)
    kernel_steps: List[KernelStep] = field(default_factory=list)
    events: List[EventDep] = field(default_factory=list)
    timeline: List[TimelineStep] = field(default_factory=list)

    # Lifetime analysis
    lifetimes: Dict[str, LifetimeInterval] = field(default_factory=dict)

    # Pool state
    pool_stats: PoolStats = field(default_factory=PoolStats)

    # Model-lifetime weight residency
    weight_slots: Dict[str, str] = field(default_factory=dict)  # tensor_name -> alloc_id

    # Stream assignments
    num_compute_streams: int = 1
    copy_stream_id: int = 0

    @property
    def peak_memory_bytes(self) -> int:
        return self.pool_stats.peak_reserved_bytes

    def summary(self) -> Dict[str, Any]:
        """Return a human-readable summary for logging and review."""
        return {
            "model": self.model_name,
            "batch_size": self.batch_size,
            "total_kernels": len(self.kernel_steps),
            "total_allocations": len(self.allocations),
            "total_transfers": len(self.transfers),
            "total_events": len(self.events),
            "timeline_steps": len(self.timeline),
            "weight_slots": len(self.weight_slots),
            "peak_reserved_mb": self.pool_stats.peak_reserved_bytes / (1024 * 1024),
            "reuse_hits": self.pool_stats.reuse_hits,
            "internal_fragmentation_mb": self.pool_stats.internal_fragmentation / (1024 * 1024),
            "num_compute_streams": self.num_compute_streams,
            "free_list_blocks": self.pool_stats.free_list_blocks,
        }

    def validate(self) -> List[str]:
        """Run structural checks on the execution plan. Returns list of issues.

        Every logical kernel input and output must have a physical binding.
        Missing bindings are reported as errors.
        """
        issues: List[str] = []

        alloc_ids = {a.alloc_id for a in self.allocations}
        event_ids = {e.event_id for e in self.events}
        producer_steps: Dict[str, int] = {}
        for k in self.kernel_steps:
            for tname in k.logical_outputs:
                if tname:
                    producer_steps[tname] = k.step_index

        for k in self.kernel_steps:
            # Every declared logical input/output must have a binding.  A
            # Constant kernel legitimately has no inputs, so checking whether
            # the binding dict itself is empty would reject valid plans.
            for tname in k.logical_inputs:
                if tname and tname not in k.inputs:
                    issues.append(
                        f"Kernel step {k.step_index} '{k.kernel_name}' "
                        f"is missing input binding for '{tname}'"
                    )
                producer_step = producer_steps.get(tname)
                if producer_step is not None and producer_step >= k.step_index:
                    issues.append(
                        f"Kernel step {k.step_index} '{k.kernel_name}' consumes "
                        f"'{tname}' before producer step {producer_step}"
                    )
            for tname, aid in k.inputs.items():
                if aid not in alloc_ids:
                    issues.append(
                        f"Kernel step {k.step_index} '{k.kernel_name}' "
                        f"input '{tname}' references unknown alloc_id '{aid}'"
                    )
            for tname in k.logical_outputs:
                if tname and tname not in k.outputs:
                    issues.append(
                        f"Kernel step {k.step_index} '{k.kernel_name}' "
                        f"is missing output binding for '{tname}'"
                    )
            for tname, aid in k.outputs.items():
                if aid not in alloc_ids:
                    issues.append(
                        f"Kernel step {k.step_index} '{k.kernel_name}' "
                        f"output '{tname}' references unknown alloc_id '{aid}'"
                    )
            # Every dependency event must exist
            for evt in k.depends_on:
                if evt not in event_ids:
                    issues.append(
                        f"Kernel step {k.step_index} depends on unknown event '{evt}'"
                    )
            for evt in k.signals:
                if evt not in event_ids:
                    issues.append(
                        f"Kernel step {k.step_index} signals unknown event '{evt}'"
                    )
            if k.tuning_params is None:
                issues.append(
                    f"Kernel step {k.step_index} '{k.kernel_name}' has no tuning parameters"
                )
            else:
                block_x = int(k.tuning_params.get("block_x", 0))
                grid_x = int(k.tuning_params.get("grid_x", 0))
                smem_bytes = int(k.tuning_params.get("smem_bytes", -2))
                if block_x <= 0:
                    issues.append(
                        f"Kernel step {k.step_index} has invalid block_x={block_x}"
                    )
                if grid_x <= 0:
                    issues.append(
                        f"Kernel step {k.step_index} has invalid grid_x={grid_x}"
                    )
                if smem_bytes < -1:
                    issues.append(
                        f"Kernel step {k.step_index} has invalid smem_bytes={smem_bytes}"
                    )
            if k.precision_profile is None:
                issues.append(
                    f"Kernel step {k.step_index} '{k.kernel_name}' has no precision profile"
                )

        # Every transfer must reference a known allocation
        for t in self.transfers:
            if t.alloc_id not in alloc_ids:
                issues.append(
                    f"Transfer '{t.kind}' for '{t.tensor_name}' "
                    f"references unknown alloc_id '{t.alloc_id}'"
                )

        # Weight slots should be allocations
        for tname, aid in self.weight_slots.items():
            if aid not in alloc_ids:
                issues.append(
                    f"Weight slot '{tname}' references unknown alloc_id '{aid}'"
                )

        # Every waited event must have one real producer.  A dependency record
        # without a transfer/kernel record operation would deadlock at runtime.
        transfer_signals = {
            t.event_id for t in self.transfers if t.event_id is not None
        }
        kernel_signals = {
            event_id for kernel in self.kernel_steps for event_id in kernel.signals
        }
        signalled_events = transfer_signals | kernel_signals
        waited_events = {
            event_id for kernel in self.kernel_steps for event_id in kernel.depends_on
        } | {
            event_id for transfer in self.transfers
            for event_id in transfer.depends_on
        }
        for event_id in sorted(waited_events - signalled_events):
            issues.append(f"Event '{event_id}' is waited on but never signalled")

        # Physical arena ranges may be reused only when logical lifetimes do
        # not overlap.  Checking alloc_id or slot_id alone misses coalesced and
        # split free-list blocks that receive a new slot ID.
        for index, left in enumerate(self.allocations):
            left_capacity = left.capacity_bytes or left.size_bytes
            left_range = (left.offset_bytes, left.offset_bytes + left_capacity)
            left_lifetime = self.lifetimes.get(left.tensor_name)
            if left_lifetime is None:
                continue
            for right in self.allocations[index + 1:]:
                right_capacity = right.capacity_bytes or right.size_bytes
                right_range = (
                    right.offset_bytes,
                    right.offset_bytes + right_capacity,
                )
                shares_bytes = (
                    left_range[0] < right_range[1]
                    and right_range[0] < left_range[1]
                )
                if not shares_bytes:
                    continue
                right_lifetime = self.lifetimes.get(right.tensor_name)
                if right_lifetime is None:
                    continue
                lifetimes_overlap = not (
                    left_lifetime.last_use < right_lifetime.first_use
                    or right_lifetime.last_use < left_lifetime.first_use
                )
                if lifetimes_overlap:
                    issues.append(
                        "Physical allocation overlap for live tensors "
                        f"'{left.tensor_name}' and '{right.tensor_name}'"
                    )

        # The unified timeline is the runtime contract.  Validate references,
        # exact coverage, ordering, and that every wait follows an enqueued
        # record operation in host submission order.
        valid_kinds = {
            "ALLOC", "FREE", "H2D", "D2H", "EVENT_WAIT",
            "EVENT_RECORD", "KERNEL",
        }
        transfer_refs: List[int] = []
        kernel_refs: List[int] = []
        recorded_at: Dict[str, int] = {}
        for expected_index, action in enumerate(self.timeline):
            if action.step_index != expected_index:
                issues.append(
                    f"Timeline index {action.step_index} is not contiguous at "
                    f"position {expected_index}"
                )
            if action.kind not in valid_kinds:
                issues.append(f"Unknown timeline action '{action.kind}'")
                continue
            if action.kind in {"ALLOC", "FREE"}:
                if action.alloc_id not in alloc_ids:
                    issues.append(
                        f"Timeline {action.kind} references unknown allocation "
                        f"'{action.alloc_id}'"
                    )
            elif action.kind in {"H2D", "D2H"}:
                if action.ref_index is None or not (
                    0 <= action.ref_index < len(self.transfers)
                ):
                    issues.append(
                        f"Timeline {action.kind} has invalid transfer index "
                        f"{action.ref_index}"
                    )
                else:
                    transfer_refs.append(action.ref_index)
                    transfer = self.transfers[action.ref_index]
                    if transfer.kind != action.kind:
                        issues.append(
                            f"Timeline {action.kind} references {transfer.kind} "
                            f"transfer {action.ref_index}"
                        )
            elif action.kind == "KERNEL":
                if action.ref_index is None or not (
                    0 <= action.ref_index < len(self.kernel_steps)
                ):
                    issues.append(
                        f"Timeline KERNEL has invalid kernel index {action.ref_index}"
                    )
                else:
                    kernel_refs.append(action.ref_index)
            elif action.kind in {"EVENT_WAIT", "EVENT_RECORD"}:
                if action.event_id not in event_ids:
                    issues.append(
                        f"Timeline {action.kind} references unknown event "
                        f"'{action.event_id}'"
                    )
                elif action.kind == "EVENT_RECORD":
                    recorded_at.setdefault(action.event_id, expected_index)
                elif action.event_id not in recorded_at:
                    issues.append(
                        f"Timeline waits on event '{action.event_id}' before "
                        "its record operation is enqueued"
                    )

        if self.timeline:
            if sorted(transfer_refs) != list(range(len(self.transfers))):
                issues.append("Timeline does not execute every transfer exactly once")
            if sorted(kernel_refs) != list(range(len(self.kernel_steps))):
                issues.append("Timeline does not execute every kernel exactly once")

        return issues
