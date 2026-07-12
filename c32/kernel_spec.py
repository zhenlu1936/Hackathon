"""Data structures for kernel decomposition and tuning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PrecisionProfile:
    """Selected precision for a node's computation.

    The evaluator intersects the strategy's decision with
    ``hardware.supported_precisions()``.
    """

    compute_dtype: str = "fp32"
    accumulator_dtype: str = "fp32"
    input_dtype: str = "fp32"
    output_dtype: str = "fp32"

    def __post_init__(self) -> None:
        valid = {"fp32", "fp16", "bf16", "fp8", "fp4", "int8"}
        for field_name in ("compute_dtype", "accumulator_dtype", "input_dtype", "output_dtype"):
            val = getattr(self, field_name)
            if val not in valid:
                raise ValueError(f"Unknown precision '{val}' in {field_name}")


@dataclass
class KernelTuningParams:
    """GPU kernel launch parameters."""

    block_x: int = 128
    block_y: int = 1
    block_z: int = 1
    grid_x: int = 1
    grid_y: int = 1
    grid_z: int = 1
    smem_bytes: int = 0

    def is_valid(self, max_threads: int, max_smem: int) -> bool:
        threads = self.block_x * self.block_y * self.block_z
        checks = (
            0 < threads <= max_threads,
            self.block_x > 0,
            self.grid_x > 0,
            self.grid_y > 0,
            self.grid_z > 0,
            self.smem_bytes <= max_smem or self.smem_bytes == -1,
        )
        return all(checks)


@dataclass
class KernelSpecRef:
    """Reference to a single GPU kernel in a decomposition sequence.

    Carries all operator parameters required to preserve ONNX semantics
    during execution: Conv pads/strides/dilations, Gemm alpha/beta/trans,
    Softmax axis, LayerNorm epsilon/axis, etc.
    """

    kernel_name: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    workspace_bytes: int = 0
    tuning_params: Optional[KernelTuningParams] = None
    # Operator semantics required for execution correctness.
    # Populated by the decomposition functions with the originating
    # node's ONNX attributes plus any derived parameters.
    operator_params: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_tunable(self) -> bool:
        return self.tuning_params is not None


@dataclass
class ProblemSize:
    """Describes the computational problem size for a kernel."""

    m: int = 1
    n: int = 1
    k: int = 1
    batch: int = 1
    num_heads: int = 1
    seq_len: int = 1
    spatial: Dict[str, int] = field(default_factory=dict)
