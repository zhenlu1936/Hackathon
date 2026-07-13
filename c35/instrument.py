"""C3.5 instrumentation — lightweight timing and profiling utilities.

Non-invasive helpers that wrap the existing pipeline without changing its
behaviour.  Every tool can be toggled off so production runs are unchanged.

Usage:
    from c35.instrument import StageTimer, profile_block, Timer

    timer = StageTimer("parse")
    with timer:
        graph = import_onnx(model_path)
    print(timer.summary())
"""

from __future__ import annotations

import functools
import gc
import time
import tracemalloc
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional

import cupy as cp


# ── Constants ──────────────────────────────────────────────────────

_ENABLE_TRACEMALLOC = False   # set True to track Python heap per stage
_TRACEMALLOC_FRAMES = 1


# ── Low-level timing ───────────────────────────────────────────────

def _now_ns() -> int:
    """Monotonic nanoseconds (wall clock)."""
    return time.perf_counter_ns()


def _gpu_time_ns() -> int:
    """Nanoseconds since CUDA epoch (GPU-side).  Synchronises the device first
    so the timestamp is meaningful relative to kernel launches."""
    try:
        cp.cuda.Device(0).synchronize()
        import cupy.cuda.runtime
        return cupy.cuda.runtime.getCUDARTimestamp()
    except Exception:
        return 0


# ── Timer — single stopwatch ───────────────────────────────────────

class Timer:
    """A single named stopwatch.  Call .start() / .stop() or use as a
    context manager.

    Attributes:
        name: Human-readable label.
        elapsed_ns: Accumulated wall-clock nanoseconds.
        gpu_elapsed_ns: Accumulated GPU-side nanoseconds.
        starts: How many times the timer was started (for avg).
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.elapsed_ns: int = 0
        self.gpu_elapsed_ns: int = 0
        self.starts: int = 0
        self._t0_ns: int = 0
        self._gpu_t0_ns: int = 0
        self._running: bool = False

    def start(self) -> Timer:
        if self._running:
            return self
        self._t0_ns = _now_ns()
        self._gpu_t0_ns = _gpu_time_ns()
        self._running = True
        return self

    def stop(self) -> Timer:
        if not self._running:
            return self
        wall = _now_ns() - self._t0_ns
        gpu = _gpu_time_ns() - self._gpu_t0_ns
        self.elapsed_ns += wall
        self.gpu_elapsed_ns += max(0, gpu) if gpu > 0 else 0
        self.starts += 1
        self._running = False
        return self

    @property
    def elapsed_s(self) -> float:
        return self.elapsed_ns / 1e9

    @property
    def gpu_elapsed_s(self) -> float:
        return self.gpu_elapsed_ns / 1e9

    @property
    def avg_s(self) -> float:
        return (self.elapsed_ns / max(1, self.starts)) / 1e9

    def __enter__(self) -> Timer:
        return self.start()

    def __exit__(self, *_: Any) -> None:
        self.stop()

    def summary(self) -> str:
        base = f"{self.name:40s}  {self.elapsed_s:10.4f}s"
        if self.starts > 1:
            base += f"  ({self.starts} calls, avg {self.avg_s:.4f}s)"
        return base


# ── StageTimer — grouped timers with traces ─────────────────────────

class StageTimer:
    """A named pipeline stage that owns sub-timers.

    Use as a context manager for the stage itself and call .sub(name) for
    child timers:

        timer = StageTimer("inference")
        with timer:
            h2d = timer.sub("h2d")
            with h2d:
                upload_weights()
            kernel = timer.sub("kernel")
            with kernel:
                run_kernels()
        print(timer.summary())
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._wall: Timer = Timer(name)
        self._subs: Dict[str, Timer] = {}
        self._order: List[str] = []
        self._tracemalloc_start: Optional[Any] = None
        self._tracemalloc_end: Optional[Any] = None
        self._pool_before: Dict[str, int] = {}
        self._pool_after: Dict[str, int] = {}

    def sub(self, name: str) -> Timer:
        if name not in self._subs:
            self._subs[name] = Timer(name)
            self._order.append(name)
        return self._subs[name]

    def _snap_pool(self) -> Dict[str, int]:
        try:
            pool = cp.get_default_memory_pool()
            return {
                "used_bytes": int(pool.used_bytes()),
                "total_bytes": int(pool.total_bytes()),
            }
        except Exception:
            return {}

    # -- context manager --------------------------------------------------
    def __enter__(self) -> StageTimer:
        self._pool_before = self._snap_pool()
        if _ENABLE_TRACEMALLOC:
            tracemalloc.start(_TRACEMALLOC_FRAMES)
            self._tracemalloc_start = tracemalloc.take_snapshot()
        gc.collect()
        self._wall.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self._wall.stop()
        gc.collect()
        if _ENABLE_TRACEMALLOC:
            self._tracemalloc_end = tracemalloc.take_snapshot()
            tracemalloc.stop()
        self._pool_after = self._snap_pool()

    # -- reporting --------------------------------------------------------
    @property
    def elapsed_s(self) -> float:
        return self._wall.elapsed_s

    @property
    def sub_times(self) -> Dict[str, float]:
        return {k: self._subs[k].elapsed_s for k in self._order}

    def summary(self, indent: int = 0) -> str:
        prefix = "  " * indent
        lines: List[str] = [f"{prefix}{self._wall.summary()}"]
        for key in self._order:
            lines.append(f"{prefix}  └─ {self._subs[key].summary()}")
        if self._pool_before and self._pool_after:
            delta = self._pool_after["total_bytes"] - self._pool_before["total_bytes"]
            lines.append(
                f"{prefix}  pool:  {self._pool_before['total_bytes']/1e6:.1f} MB  →  "
                f"{self._pool_after['total_bytes']/1e6:.1f} MB  "
                f"(Δ{delta/1e6:+.1f} MB)"
            )
        if _ENABLE_TRACEMALLOC and self._tracemalloc_start and self._tracemalloc_end:
            stats = self._tracemalloc_end.compare_to(
                self._tracemalloc_start, "lineno"
            )
            lines.append(f"{prefix}  python heap top-3:")
            for stat in stats[:3]:
                lines.append(f"{prefix}    {stat}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.name,
            "elapsed_s": self.elapsed_s,
            "sub_times": self.sub_times,
            "pool_before_bytes": self._pool_before.get("total_bytes", 0),
            "pool_after_bytes": self._pool_after.get("total_bytes", 0),
        }


# ── profile_block — decorator for one-off timing ───────────────────

def profile_block(name: Optional[str] = None) -> Callable:
    """Decorator / context-manager factory that times a block of code and
    prints the result when done.

    As decorator:
        @profile_block("my_func")
        def my_func(): ...

    As context manager:
        with profile_block("upload"):
            do_work()
    """
    class _BlockTimer:
        def __init__(self, label: str) -> None:
            self.label = label
            self._t0: int = 0

        def __enter__(self) -> _BlockTimer:
            self._t0 = _now_ns()
            return self

        def __exit__(self, *_: Any) -> None:
            elapsed = (_now_ns() - self._t0) / 1e9
            print(f"[profile] {self.label:40s}  {elapsed:10.4f}s", flush=True)

        def __call__(self, fn: Callable) -> Callable:
            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                with _BlockTimer(self.label):
                    return fn(*args, **kwargs)
            return wrapper

    if callable(name):
        return _BlockTimer(name.__name__)(name)
    return _BlockTimer(name or "block")


# ── KernelTrace — per-node execution trace with timing and shapes ───

class KernelTrace:
    """Collect per-kernel timing, input/output shapes, and op type during a
    single execution run.

    Usage:
        trace = KernelTrace()
        with trace.record("conv_0", "Conv"):
            output = execute_op(...)
        print(trace.table())
    """

    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []

    @contextmanager
    def record(self, node_id: str, op_type: str, **meta: Any):
        t0 = _now_ns()
        gpu_t0 = _gpu_time_ns()
        record: Dict[str, Any] = {
            "node_id": node_id,
            "op_type": op_type,
            **meta,
        }
        try:
            yield record
        finally:
            cp.cuda.Device(0).synchronize()
            record["wall_ns"] = _now_ns() - t0
            record["gpu_ns"] = max(0, _gpu_time_ns() - gpu_t0)
            record["wall_s"] = record["wall_ns"] / 1e9
            self.records.append(record)

    def table(self, top_n: int = 30) -> str:
        """Return a sorted text table of the most expensive kernels."""
        sorted_recs = sorted(self.records, key=lambda r: -r["wall_ns"])
        header = f"{'node_id':40s} {'op_type':25s} {'wall_s':>10s}  {'shapes'}"
        lines = [header, "-" * len(header)]
        for rec in sorted_recs[:top_n]:
            shapes = rec.get("shapes", "")
            lines.append(
                f"{rec['node_id']:40s} {rec['op_type']:25s} "
                f"{rec['wall_s']:10.4f}  {shapes}"
            )
        return "\n".join(lines)

    def summary_dict(self) -> Dict[str, Any]:
        if not self.records:
            return {}
        total = sum(r["wall_ns"] for r in self.records)
        by_op: Dict[str, float] = {}
        for r in self.records:
            by_op[r["op_type"]] = by_op.get(r["op_type"], 0.0) + r["wall_s"]
        return {
            "total_wall_s": total / 1e9,
            "kernel_count": len(self.records),
            "by_op_type": dict(sorted(by_op.items(), key=lambda x: -x[1])),
            "top_5": [
                {"node_id": r["node_id"], "op_type": r["op_type"], "wall_s": r["wall_s"]}
                for r in sorted(self.records, key=lambda x: -x["wall_ns"])[:5]
            ],
        }


# ── MemorySampler — explicit GPU memory snapshots ──────────────────

class MemorySampler:
    """Collect explicit CuPy-pool snapshots through the compatibility API.

    ``start()`` and ``stop()`` retain the original fluent interface. Sampling
    is intentionally explicit through :meth:`snap`; the class does not create
    a background thread or claim a periodic trace that it did not capture.
    ``interval_s`` is retained for callers that schedule their own snapshots.
    """

    def __init__(self, interval_s: float = 0.1) -> None:
        self.interval_s = interval_s
        self.snapshots: List[Dict[str, Any]] = []
        self._running = False

    def _snap(self) -> Dict[str, Any]:
        snap: Dict[str, Any] = {"wall_s": _now_ns() / 1e9}
        try:
            pool = cp.get_default_memory_pool()
            snap["pool_used_bytes"] = int(pool.used_bytes())
            snap["pool_total_bytes"] = int(pool.total_bytes())
        except Exception:
            pass
        return snap

    def start(self) -> MemorySampler:
        self._running = True
        return self

    def snap(self) -> Dict[str, Any]:
        rec = self._snap()
        self.snapshots.append(rec)
        return rec

    def stop(self) -> MemorySampler:
        self._running = False
        return self

    def summary(self) -> str:
        if not self.snapshots:
            return "No memory snapshots."
        peak = max(s.get("pool_total_bytes", 0) for s in self.snapshots)
        return (
            f"Memory snapshots: {len(self.snapshots)} samples, "
            f"peak pool reserved = {peak/1e6:.1f} MB"
        )


# ── ProfileReport — aggregate all instrumentation ──────────────────

class ProfileReport:
    """Collects stage timers, kernel traces, and memory samples into one JSON-
    serialisable report.

    Usage:
        report = ProfileReport()
        report.add_stage(parse_timer)
        report.add_kernel_trace(trace)
        report.add_pool_sample(before, after)
        print(report.json(indent=2))
    """

    def __init__(self, model_name: str = "", batch_size: int = 0) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.stages: Dict[str, Dict[str, Any]] = {}
        self.kernel_traces: List[Dict[str, Any]] = []
        self.pool_samples: List[Dict[str, Any]] = []
        self.extra: Dict[str, Any] = {}

    def add_stage(self, timer: StageTimer) -> None:
        self.stages[timer.name] = timer.to_dict()

    def add_kernel_trace(self, trace: KernelTrace) -> None:
        self.kernel_traces.append(trace.summary_dict())

    def add_pool_sample(self, label: str, used: int, total: int) -> None:
        self.pool_samples.append({
            "label": label,
            "used_bytes": used,
            "total_bytes": total,
        })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "batch_size": self.batch_size,
            "stages": self.stages,
            "kernel_traces": self.kernel_traces,
            "pool_samples": self.pool_samples,
            "extra": self.extra,
        }

    def json(self, indent: int = 2) -> str:
        import json
        return json.dumps(self.to_dict(), indent=indent)

    def print_summary(self) -> None:
        """Print a human-readable summary to stderr."""
        import sys

        print(f"\n{'='*70}", file=sys.stderr)
        print(f" PROFILE REPORT — {self.model_name}  (batch={self.batch_size})", file=sys.stderr)
        print(f"{'='*70}", file=sys.stderr)

        for name, stage in self.stages.items():
            print(f"\n[{name}]  {stage['elapsed_s']:.4f}s", file=sys.stderr)
            for sub_name, sub_time in stage.get("sub_times", {}).items():
                pct = (sub_time / max(stage["elapsed_s"], 1e-9)) * 100
                print(f"  {sub_name:35s} {sub_time:10.4f}s  ({pct:5.1f}%)", file=sys.stderr)

        for trace in self.kernel_traces:
            by_op = trace.get("by_op_type", {})
            if by_op:
                print(f"\n[Kernel time by op type]", file=sys.stderr)
                total = trace.get("total_wall_s", 1.0)
                for op, t in list(by_op.items())[:10]:
                    print(f"  {op:35s} {t:10.4f}s  ({t/max(total,1e-9)*100:5.1f}%)", file=sys.stderr)

        print(f"\n{'='*70}\n", file=sys.stderr)


# ── Utility: snapshot pool ─────────────────────────────────────────

def pool_snapshot() -> Dict[str, int]:
    """Return the current CuPy memory pool state."""
    try:
        pool = cp.get_default_memory_pool()
        return {
            "used_bytes": int(pool.used_bytes()),
            "total_bytes": int(pool.total_bytes()),
        }
    except Exception:
        return {}
