"""Strategy module — precision selection, decomposition, and kernel tuning."""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from c3common.ir.graph import Graph, Node, ONNSType
from c32.hardware import HardwareCapability
from c32.kernel_spec import KernelSpecRef, KernelTuningParams, PrecisionProfile, ProblemSize
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
    "Softmax", "LogSoftmax", "LayerNorm", "LayerNormalization",
    "BatchNorm", "BatchNormalization",
    "ReduceMax", "ReduceSum", "ReduceMean",
}

TUMABLE_OPS: Set[str] = {"MatMul", "Linear", "Gemm", "Conv", "Conv2d"}

# Safe fallback order after an engineering rule chooses a preferred precision.
# Lower precision is never selected merely because it appears in this list.
SAFE_FALLBACKS: Dict[str, List[str]] = {
    "fp4": ["fp4", "fp8", "fp16", "bf16", "fp32"],
    "fp8": ["fp8", "fp16", "bf16", "fp32"],
    "fp16": ["fp16", "bf16", "fp32"],
    "bf16": ["bf16", "fp16", "fp32"],
    "fp32": ["fp32"],
}

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
        3. Tunable ops in MIXED_PRECISION → shape/semantics-based engineering
           rule followed by a safe hardware-supported fallback.
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
            preferred = self._preferred_tunable_precision(node, graph)
            chosen = self._supported_fallback(preferred)
            # FP4 is used only as W4A16: activations remain FP16 while constant
            # weights use FP4.  Accumulation/output stay FP32.
            activation_dtype = "fp16" if chosen == "fp4" else chosen
            return PrecisionProfile(compute_dtype=chosen, accumulator_dtype="fp32",
                                    input_dtype=activation_dtype,
                                    weight_dtype=chosen,
                                    output_dtype="fp32")

        # Rule 4: Default fp32
        return PrecisionProfile(compute_dtype="fp32", accumulator_dtype="fp32",
                                input_dtype="fp32", output_dtype="fp32")

    def _supported_fallback(self, preferred: str) -> str:
        """Return the safest available implementation near ``preferred``."""
        supported = set(self.hardware.supported_precisions())
        for prec in SAFE_FALLBACKS[preferred]:
            if prec == "fp4" and "fp16" not in supported:
                continue  # W4A16 requires an FP16 activation path.
            if prec in supported:
                return prec
        return "fp32"

    def _preferred_tunable_precision(self, node: Node, graph: Graph) -> str:
        """Choose a mixed precision from general DL engineering properties.

        * Convolution defaults to FP16.  Aligned 1x1 convolution is a common
          FP8 tensor-core case; spatial 3x3 kernels remain FP16 because their
          accumulation and transform ranges are less forgiving.
        * GEMM defaults to FP16.  Aligned medium/large GEMM can use FP8.
          FP4 is limited to large aligned constant-weight GEMM, corresponding
          to the common weight-only quantized linear-layer use case.

        Accumulation and outputs remain FP32 in all cases.
        """
        if node.op_type in {"Conv", "Conv2d"}:
            kernel_shape = list(node.attributes.get("kernel_shape", []))
            weight = graph.get_tensor(node.inputs[1]) if len(node.inputs) > 1 else None
            out_channels = _concrete_dim(weight.shape[0]) if weight and weight.shape else None
            in_channels = _concrete_dim(weight.shape[1]) if weight and len(weight.shape) > 1 else None
            aligned = (
                out_channels is not None and in_channels is not None
                and out_channels % 16 == 0 and in_channels % 16 == 0
            )
            return "fp8" if kernel_shape in ([1, 1], [1]) and aligned else "fp16"

        if node.op_type in {"MatMul", "Linear", "Gemm"}:
            k, n = _gemm_k_n(node, graph)
            weight = graph.get_tensor(node.inputs[1]) if len(node.inputs) > 1 else None
            constant_weight = bool(weight and (weight.is_initializer or weight.is_constant))
            aligned_16 = k is not None and n is not None and k % 16 == 0 and n % 16 == 0
            aligned_32 = k is not None and n is not None and k % 32 == 0 and n % 32 == 0
            large = k is not None and n is not None and k >= 128 and n >= 128
            if constant_weight and aligned_32 and large and k * n >= 32768:
                return "fp4"
            if aligned_16 and k is not None and n is not None and k >= 64 and n >= 64:
                return "fp8"
            return "fp16"

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
                    problem_size: Optional[Dict[str, int] | ProblemSize] = None) -> KernelTuningParams:
        if isinstance(problem_size, ProblemSize):
            ps: Dict[str, int] = {
                "m": problem_size.m,
                "n": problem_size.n,
                "k": problem_size.k,
                "batch": problem_size.batch,
                "num_heads": problem_size.num_heads,
                "seq_len": problem_size.seq_len,
                **problem_size.spatial,
            }
        else:
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


def _concrete_dim(value: Any) -> Optional[int]:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def _gemm_k_n(node: Node, graph: Graph) -> tuple[Optional[int], Optional[int]]:
    """Return logical reduction/output dimensions for MatMul-like operators."""
    if len(node.inputs) < 2:
        return None, None
    a = graph.get_tensor(node.inputs[0])
    b = graph.get_tensor(node.inputs[1])
    if a is None or b is None or len(a.shape) < 2 or len(b.shape) < 2:
        return None, None
    trans_a = int(node.attributes.get("transA", 0))
    trans_b = int(node.attributes.get("transB", 0))
    k = _concrete_dim(a.shape[-2] if trans_a else a.shape[-1])
    n = _concrete_dim(b.shape[-2] if trans_b else b.shape[-1])
    return k, n


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
