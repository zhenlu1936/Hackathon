"""C3.4 — Memory planning and scheduling.

Provides the execution scheduler that builds a complete memory-and-stream
execution plan from a computation graph. Integrates all five features A–E
as required by the specification.

Public API:

    from c34.scheduler import ExecutionScheduler
    from c34.execution_plan import ExecutionPlan, PoolStats

    scheduler = ExecutionScheduler(graph, batch_size=1)
    plan: ExecutionPlan = scheduler.build()

    # Review the plan
    print(plan.summary())
    issues = plan.validate()
"""

from __future__ import annotations

from c34.scheduler import ExecutionScheduler
from c34.execution_plan import (
    Allocation,
    Transfer,
    KernelStep,
    EventDep,
    LifetimeInterval,
    PoolStats,
    ExecutionPlan,
)
from c34.memory_pool import DeviceMemoryPool, FitPolicy
from c34.lifetime import compute_lifetimes, find_overlap_groups

__all__ = [
    "ExecutionScheduler",
    "ExecutionPlan",
    "Allocation",
    "Transfer",
    "KernelStep",
    "EventDep",
    "LifetimeInterval",
    "PoolStats",
    "DeviceMemoryPool",
    "FitPolicy",
    "compute_lifetimes",
    "find_overlap_groups",
]
