"""Hardware capability description for GPU kernel selection and tuning.

The evaluator-facing object can be populated either from a documented AEC
profile or from the active CuPy/CUDA device.  Device discovery is deliberately
an explicit call: importing :mod:`c32` must remain deterministic and must not
silently replace an organizer-provided AEC profile with the host CUDA device.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Set, Tuple


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
    compute_capability: Optional[Tuple[int, int]] = None
    source: str = "static"
    verified: bool = False

    def __post_init__(self) -> None:
        positive_limits = {
            "max_threads_per_block": self.max_threads_per_block,
            "max_block_dim": self.max_block_dim,
            "max_grid_dim": self.max_grid_dim,
            "smem_bytes": self.smem_bytes,
            "smem_bytes_total": self.smem_bytes_total,
            "max_shared_memory_per_block": self.max_shared_memory_per_block,
        }
        invalid = [name for name, value in positive_limits.items() if int(value) <= 0]
        if invalid:
            raise ValueError(
                "Hardware limits must be positive: " + ", ".join(sorted(invalid))
            )
        if self.smem_bytes > self.smem_bytes_total:
            raise ValueError("Per-block shared memory cannot exceed total shared memory")
        if self.max_shared_memory_per_block > self.smem_bytes_total:
            raise ValueError(
                "max_shared_memory_per_block cannot exceed total shared memory"
            )
        if self.compute_capability is not None:
            major, minor = self.compute_capability
            if major < 0 or minor < 0:
                raise ValueError("compute_capability components must be non-negative")

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

    def supports_precision(self, precision: str) -> bool:
        """Return whether *precision* is declared by this capability snapshot."""
        return precision in self.supported_precisions()

    @classmethod
    def from_cuda_properties(
        cls,
        properties: Mapping[Any, Any],
        *,
        source: str = "cupy.cuda.runtime.getDeviceProperties",
        supports_fp4: Optional[bool] = None,
        supports_winograd: bool = True,
    ) -> "HardwareCapability":
        """Construct a validated profile from CUDA device properties.

        CUDA reports resource limits and compute capability directly.  Native
        low-precision availability is conservatively derived from the compute
        capability; callers may explicitly override FP4 when the submitted AEC
        software stack provides a qualified emulation/kernel path.
        """

        def _get(name: str, default: Any) -> Any:
            if name in properties:
                return properties[name]
            encoded = name.encode("utf-8")
            return properties.get(encoded, default)

        name = _get("name", "CUDA device")
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")

        major = int(_get("major", 0))
        minor = int(_get("minor", 0))
        per_block = int(
            _get("sharedMemPerBlockOptin", _get("sharedMemPerBlock", 49152))
        )
        total_smem = int(_get("sharedMemPerMultiprocessor", per_block))
        max_threads = int(_get("maxThreadsPerBlock", 1024))

        max_threads_dim = _get("maxThreadsDim", (max_threads, 1, 1))
        max_grid_size = _get("maxGridSize", (2147483647, 65535, 65535))
        max_block_dim = int(max_threads_dim[0]) if max_threads_dim else max_threads
        # The compact public schema exposes one grid bound, so use the most
        # restrictive active x/y limit.  z is always one in current tuning.
        max_grid_dim = (
            min(int(max_grid_size[0]), int(max_grid_size[1]))
            if len(max_grid_size) >= 2 else int(max_grid_size[0])
        ) if max_grid_size else 65535

        # Volta+ supports FP16 Tensor Core work, Ampere+ supports BF16, and
        # Hopper+ supports FP8.  FP4 remains opt-in unless a Blackwell-or-newer
        # device or a separately qualified AEC software implementation says so.
        native_fp4 = major >= 10
        return cls(
            name=str(name),
            max_threads_per_block=max_threads,
            max_block_dim=max_block_dim,
            max_grid_dim=max_grid_dim,
            smem_bytes=per_block,
            smem_bytes_total=max(total_smem, per_block),
            supports_fp16=major >= 7,
            supports_bf16=major >= 8,
            supports_fp8=major >= 9,
            supports_fp4=native_fp4 if supports_fp4 is None else bool(supports_fp4),
            supports_tf32=major >= 8,
            supports_tensor_core=major >= 7,
            supports_winograd=bool(supports_winograd),
            max_shared_memory_per_block=per_block,
            compute_capability=(major, minor),
            source=source,
            verified=True,
        )

    @classmethod
    def query_cupy_device(
        cls,
        device_id: Optional[int] = None,
        *,
        supports_fp4: Optional[bool] = None,
        supports_winograd: bool = True,
    ) -> "HardwareCapability":
        """Query the active CUDA device through the server-native CuPy API."""
        try:
            import cupy as cp
        except ImportError as exc:  # pragma: no cover - target-only dependency
            raise RuntimeError("CuPy is required for CUDA hardware discovery") from exc

        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("No CUDA device is visible to CuPy")
        selected = int(cp.cuda.Device().id if device_id is None else device_id)
        properties = cp.cuda.runtime.getDeviceProperties(selected)
        return cls.from_cuda_properties(
            properties,
            source=f"cupy.cuda.runtime.getDeviceProperties(device={selected})",
            supports_fp4=supports_fp4,
            supports_winograd=supports_winograd,
        )


_default_hardware = HardwareCapability(
    # Capability-oriented default for the released C3.2 microbenchmark.  It is
    # not a claim about H100 (which does not natively provide FP4); deployment
    # must replace this with the organizer/AEC device query.
    name="AEC mixed-precision profile (unverified)",
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
    source="competition C3.2 capability-coverage profile",
    verified=False,
)


def get_hardware() -> HardwareCapability:
    return _default_hardware


def set_hardware(hw: HardwareCapability) -> None:
    """Update the active capability snapshot without invalidating references.

    Evaluator modules commonly import ``hardware`` once.  Mutating the shared
    dataclass keeps those references and already-created default strategies in
    sync even when callers import this function directly from this module.
    """
    if not isinstance(hw, HardwareCapability):
        raise TypeError("hw must be a HardwareCapability")
    for field_name in HardwareCapability.__dataclass_fields__:
        setattr(_default_hardware, field_name, getattr(hw, field_name))
