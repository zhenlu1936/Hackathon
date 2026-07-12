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


# ── Kernel step ────────────────────────────────────────────────────

@dataclass
class KernelStep:
    """A single kernel launch in the execution schedule."""

    step_index: int               # position in the global schedule
    kernel_name: str              # e.g. "matmul_fp32", "relu_fp32"
    node_id: str                  # originating graph node
    inputs: Dict[str, str] = field(default_factory=dict)   # tensor_name -> alloc_id
    outputs: Dict[str, str] = field(default_factory=dict)  # tensor_name -> alloc_id
    stream_id: int = 1            # compute stream ID (0 = copy stream)
    depends_on: List[str] = field(default_factory=list)    # event IDs to wait on
    signals: List[str] = field(default_factory=list)       # event IDs to signal
    tuning_params: Optional[Dict[str, int]] = None         # block_x, grid_x, smem_bytes


# ── Event dependency ───────────────────────────────────────────────

@dataclass
class EventDep:
    """Cross-stream event dependency."""

    event_id: str
    src_stream: int       # stream that signals the event
    dst_stream: int       # stream that waits on the event
    description: str = ""


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

        for k in self.kernel_steps:
            # Every kernel input must have a binding
            if not k.inputs:
                issues.append(
                    f"Kernel step {k.step_index} '{k.kernel_name}' "
                    f"has no input bindings"
                )
            for tname, aid in k.inputs.items():
                if aid not in alloc_ids:
                    issues.append(
                        f"Kernel step {k.step_index} '{k.kernel_name}' "
                        f"input '{tname}' references unknown alloc_id '{aid}'"
                    )
            # Every kernel output must have a binding
            if not k.outputs:
                issues.append(
                    f"Kernel step {k.step_index} '{k.kernel_name}' "
                    f"has no output bindings"
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

        return issues
