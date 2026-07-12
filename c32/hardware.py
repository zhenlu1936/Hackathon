"""Hardware capability description for GPU kernel selection and tuning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set


@dataclass
class HardwareCapability:
    """Describes a GPU's compute capabilities relevant to kernel selection."""

    name: str = "Generic GPU"
    max_threads_per_block: int = 1024
    max_block_dim: int = 1024
    max_grid_dim: int = 65535
    smem_bytes: int = 49152
    smem_bytes_total: int = 166912
    supports_fp16: bool = True
    supports_bf16: bool = False
    supports_fp8: bool = False
    supports_fp4: bool = False
    supports_tf32: bool = True
    supports_tensor_core: bool = True
    supports_winograd: bool = True
    max_shared_memory_per_block: int = 49152

    def supported_precisions(self) -> List[str]:
        precisions: List[str] = ["fp32"]
        if self.supports_fp16:
            precisions.append("fp16")
        if self.supports_bf16:
            precisions.append("bf16")
        if self.supports_fp8:
            precisions.append("fp8")
        if self.supports_fp4:
            precisions.append("fp4")
        return precisions

    def gemm_kernels_available(self) -> Set[str]:
        kernels = {"matmul_f32"}
        if self.supports_fp16:
            kernels.add("matmul_f16")
        if self.supports_bf16:
            kernels.add("matmul_bf16")
        if self.supports_fp8:
            kernels.add("matmul_f8")
        if self.supports_fp4:
            kernels.add("matmul_f4")
        return kernels

    def conv_strategies_available(self) -> List[str]:
        strategies = ["im2col"]
        if self.supports_winograd:
            strategies.append("winograd")
        return strategies


_default_hardware = HardwareCapability(
    name="NVIDIA H100 (default)",
    max_threads_per_block=1024,
    max_block_dim=1024,
    max_grid_dim=65535,
    smem_bytes=49152,
    smem_bytes_total=228864,
    supports_fp16=True,
    supports_bf16=True,
    supports_fp8=True,
    supports_fp4=True,
    supports_tf32=True,
    supports_tensor_core=True,
    supports_winograd=True,
)


def get_hardware() -> HardwareCapability:
    return _default_hardware


def set_hardware(hw: HardwareCapability) -> None:
    global _default_hardware
    _default_hardware = hw
