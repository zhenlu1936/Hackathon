"""Strategy module — precision selection, decomposition, and kernel tuning."""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from c3common.ir.graph import Graph, Node, ONNSType
from c32.hardware import HardwareCapability
from c32.kernel_spec import KernelSpecRef, KernelTuningParams, PrecisionProfile
from c32.decompositions import DECOMPOSE_DISPATCH


class ExecutionMode(str, Enum):
    """Explicit execution mode that governs precision selection.

    FULL_FP32: Every operator runs in fp32. The safe, correct baseline.
    MIXED_PRECISION: Allow lower-precision compute for qualified operators
        whose numerical sensitivity and the target hardware permit it.
    """
    FULL_FP32 = "FULL_FP32"
    MIXED_PRECISION = "MIXED_PRECISION"


SENSITIVE_OPS: Set[str] = {
    "Softmax", "LayerNormalization", "BatchNormalization",
    "ReduceMax", "ReduceSum", "ReduceMean", "LogSoftmax",
}

TUMABLE_OPS: Set[str] = {"MatMul", "Gemm", "Conv"}

# Priority order for precision selection within MIXED_PRECISION mode.
# Earlier entries are preferred when the hardware supports them.
PRECISION_PRIORITY: List[str] = ["fp32", "fp16", "bf16", "fp8", "fp4"]

TUMABLE_KERNEL_PREFIXES: Set[str] = {
    "matmul_", "winograd_forward_", "im2col_", "add_bias_",
    "add_", "mul_", "div_", "reduce_max", "reduce_sum", "reduce_mean",
    "exp", "sub_", "sqrt", "erf_", "relu_",
}


class Strategy:
    """Precision, decomposition, and tuning strategy.

    Precision selection is purely deterministic: given the same node, graph,
    execution mode, and hardware capabilities, the result is always identical.
    """

    def __init__(self, hardware: Optional[HardwareCapability] = None,
                 mode: ExecutionMode = ExecutionMode.FULL_FP32) -> None:
        from c32.hardware import get_hardware
        self._hardware_arg = hardware  # keep for refresh
        self.hardware = hardware or get_hardware()
        self.mode = ExecutionMode(mode)

    def refresh_hardware(self) -> None:
        """Re-read the global hardware snapshot.

        Call after ``set_hardware()`` to keep this strategy in sync.
        """
        from c32.hardware import get_hardware
        if self._hardware_arg is None:
            self.hardware = get_hardware()
        # If a specific hardware was passed at construction, keep it.

    # ── D1: Precision selection (deterministic) ─────────────────

    def select_precision(self, node: Node, graph: Graph) -> PrecisionProfile:
        """Select compute precision for a node deterministically.

        Rules (in order):
        1. FULL_FP32 mode → fp32 for everything.
        2. Sensitive ops → always fp32 regardless of mode.
        3. Tunable ops in MIXED_PRECISION → highest-priority precision
           that the hardware supports.
        4. Everything else → fp32.
        """
        op = node.op_type

        # Rule 1: FULL_FP32 forces fp32 for everything
        if self.mode == ExecutionMode.FULL_FP32:
            return PrecisionProfile(compute_dtype="fp32", accumulator_dtype="fp32",
                                    input_dtype="fp32", output_dtype="fp32")

        # Rule 2: Sensitive ops always fp32
        if op in SENSITIVE_OPS:
            return PrecisionProfile(compute_dtype="fp32", accumulator_dtype="fp32",
                                    input_dtype="fp32", output_dtype="fp32")

        # Rule 3: Tunable ops may use lower precision
        if op in TUMABLE_OPS:
            chosen = self._select_tunable_precision()
            return PrecisionProfile(compute_dtype=chosen, accumulator_dtype="fp32",
                                    input_dtype=chosen, output_dtype="fp32")

        # Rule 4: Default fp32
        return PrecisionProfile(compute_dtype="fp32", accumulator_dtype="fp32",
                                input_dtype="fp32", output_dtype="fp32")

    def _select_tunable_precision(self) -> str:
        """Deterministic: highest-priority supported precision wins."""
        supported = set(self.hardware.supported_precisions())
        for prec in PRECISION_PRIORITY:
            if prec in supported:
                return prec
        return "fp32"

    def get_precision_intersection(self, node: Node, graph: Graph) -> dict:
        profile = self.select_precision(node, graph)
        supported = self.hardware.supported_precisions()
        return {
            "node_id": node.id, "op_type": node.op_type,
            "selected": profile.compute_dtype,
            "supported_by_hardware": supported,
            "in_supported_set": profile.compute_dtype in supported,
            "profile": profile,
        }

    # ── D2 + D3: Decomposition ─────────────────

    def decompose(self, node: Node, graph: Graph,
                  precision: Optional[PrecisionProfile] = None) -> List[KernelSpecRef]:
        if precision is None:
            precision = self.select_precision(node, graph)
        op = node.op_type
        decomposer = DECOMPOSE_DISPATCH.get(op)
        if decomposer is None:
            return [KernelSpecRef(
                kernel_name=f"{op.lower()}_{precision.compute_dtype}",
                inputs=list(node.inputs), outputs=list(node.outputs),
                operator_params=dict(node.attributes),
            )]
        if op == "Conv":
            use_winograd = "winograd" in self.hardware.conv_strategies_available()
            return decomposer(node, graph, precision, use_winograd=use_winograd)
        return decomposer(node, graph, precision)

    # ── D4: Kernel tuning (with hardware-limit clamping) ──

    def tune_kernel(self, ref: KernelSpecRef, precision: PrecisionProfile,
                    problem_size: Optional[Dict[str, int]] = None) -> KernelTuningParams:
        ps = problem_size or {}
        max_threads = self.hardware.max_threads_per_block
        max_smem = self.hardware.smem_bytes
        name = ref.kernel_name

        if name.startswith("matmul_"):
            m, n, k = ps.get("m", 1024), ps.get("n", 1024), ps.get("k", 1024)
            bx = min(128, max_threads)
            by = min(8, max(1, max_threads // max(1, bx)))
            smem = k * 4 + 1024
            if smem > max_smem:
                smem = -1  # mark as infeasible
            return KernelTuningParams(
                block_x=bx, block_y=by,
                grid_x=max(1, _ceil_div(n, bx)),
                grid_y=max(1, _ceil_div(m, by)),
                smem_bytes=_clamp_smem(smem, max_smem),
            )

        if name.startswith("winograd_forward_"):
            bx = min(256, max_threads)
            smem = ps.get("smem", 16384)
            return KernelTuningParams(
                block_x=bx,
                grid_x=max(1, ps.get("out_channels", 64)),
                smem_bytes=_clamp_smem(smem, max_smem),
            )

        if name.startswith("im2col_"):
            bx = min(256, max_threads)
            return KernelTuningParams(
                block_x=bx,
                grid_x=max(1, _ceil_div(ps.get("num_tiles", 4096), bx)),
            )

        if name.startswith(("reduce_", "exp", "sqrt")):
            elements = ps.get("elements", 1024)
            bx = min(256, max_threads)
            return KernelTuningParams(
                block_x=bx,
                grid_x=max(1, _ceil_div(elements, bx)),
            )

        if any(name.startswith(p) for p in ("add_", "mul_", "div_", "sub_", "relu_", "erf_", "add_bias_")):
            elements = ps.get("elements", 1024)
            bx = min(256, max_threads)
            return KernelTuningParams(
                block_x=bx,
                grid_x=max(1, _ceil_div(elements, bx)),
            )

        if name in ("reshape", "transpose", "flatten", "split", "gather", "constant", "squeeze"):
            elements = ps.get("elements", 1024)
            bx = min(128, max_threads)
            return KernelTuningParams(block_x=bx, grid_x=max(1, _ceil_div(elements, bx)))

        return KernelTuningParams(block_x=min(128, max_threads), grid_x=1)

    def process_graph(self, graph: Graph) -> Dict[str, Any]:
        results = {}
        for nid in graph.node_order:
            node = graph.nodes[nid]
            precision = self.select_precision(node, graph)
            kernels = self.decompose(node, graph, precision)
            for krn in kernels:
                if krn.is_tunable or self._is_kernel_tunable(krn.kernel_name):
                    krn.tuning_params = self.tune_kernel(krn, precision)
            results[nid] = {"node": node, "precision": precision, "kernels": kernels}
        return results

    def _is_kernel_tunable(self, name: str) -> bool:
        return any(name.startswith(p) for p in TUMABLE_KERNEL_PREFIXES)


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _clamp_smem(smem_bytes: int, max_smem: int) -> int:
    """Clamp shared memory to the valid range for the target.

    Only ``smem_bytes == -1`` (dynamic) or ``0 <= smem_bytes <= max_smem``
    are valid.  Returns ``-1`` when the requested value exceeds the limit,
    signalling *infeasible* to the caller.
    """
    if smem_bytes == -1:
        return -1
    if smem_bytes < 0:
        return 0
    if smem_bytes > max_smem:
        return -1
    return smem_bytes
