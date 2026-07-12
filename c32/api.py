"""C3.2 public API — matches the evaluator's expected calling conventions.

The evaluator imports from this module:

    from c32.api import import_onnx_graph, strategy, hardware
    graph = import_onnx_graph("model.onnx")
    prec = strategy.select_precision(node, graph)
    hw_precs = hardware.supported_precisions()
    kernels = strategy.decompose(node, graph, prec)
    params = strategy.tune_kernel(krn, prec, problem_size)
"""

from __future__ import annotations

from c3common.ir.graph import Graph
from c31.import_onnx import import_onnx as _import_onnx
from c32.strategy import Strategy, ExecutionMode
from c32.hardware import HardwareCapability, get_hardware, set_hardware
from c32.kernel_spec import KernelSpecRef, KernelTuningParams, PrecisionProfile


def import_onnx_graph(model_path: str) -> Graph:
    """Load an ONNX model and return the internal Graph IR."""
    return _import_onnx(model_path)


# Default strategy: FULL_FP32 is the safe baseline.
# Evaluators may set a different mode before calling select/decompose/tune.
strategy: Strategy = Strategy(mode=ExecutionMode.FULL_FP32)
hardware: HardwareCapability = get_hardware()


def _refresh_globals() -> None:
    """Keep module-level ``strategy`` and ``hardware`` in sync after
    ``set_hardware()`` changes the global capability snapshot."""
    global hardware, strategy
    hardware = get_hardware()
    strategy = Strategy(hardware=hardware, mode=strategy.mode)


# Patch set_hardware so it refreshes the public API singletons automatically.
_original_set_hw = set_hardware


def _set_hardware_with_refresh(hw: HardwareCapability) -> None:
    _original_set_hw(hw)
    _refresh_globals()


set_hardware = _set_hardware_with_refresh  # type: ignore[assignment]


__all__ = [
    "import_onnx_graph", "strategy", "hardware",
    "Strategy", "ExecutionMode", "HardwareCapability",
    "KernelSpecRef", "KernelTuningParams", "PrecisionProfile",
    "set_hardware", "get_hardware",
]
