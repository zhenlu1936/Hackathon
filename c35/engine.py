"""C3.5 array compute engine — ONNX operator implementations.

Implements all 17 required ONNX operators with opset-17 semantics. Numerical
work uses CuPy exclusively on the designated remote H200 AEC device.

Operators implemented:
    Flatten, Gemm, Relu, Conv, Add, GlobalAveragePool, Gather,
    LayerNormalization, MatMul, Constant, Split, Reshape, Transpose,
    Div, Softmax, Erf, Mul
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, List, Optional, Tuple

import cupy as cp

xp = cp

_ELEMENTWISE_FUSION_CACHE: Dict[str, Any] = {}
_SOFTMAX_FUSION_KERNEL: Optional[Any] = None
_RESIDUAL_NORM_KERNELS: Dict[bool, Any] = {}
_GEMM_EPILOGUE_KERNEL: Optional[Any] = None
_CONV_ACTIVATION_EPILOGUE: Optional[Any] = None
_CONV_RESIDUAL_ACTIVATION_EPILOGUE: Optional[Any] = None
_ATTENTION_SCORES_KERNEL: Optional[Any] = None
_LAYER_NORM_KERNEL: Optional[Any] = None
_TRANSPOSE_RESHAPE_KERNEL: Optional[Any] = None


def _planned_output(attrs: Dict[str, Any], reference: xp.ndarray) -> xp.ndarray:
    """Return a validated runtime-only planned output or allocate a new one."""
    out = attrs.get("_planned_output")
    if out is None:
        return xp.empty_like(reference)
    if not isinstance(out, cp.ndarray):
        raise TypeError("_planned_output must be a CuPy array")
    if out.shape != reference.shape:
        raise ValueError(
            f"Planned output shape {out.shape} does not match {reference.shape}"
        )
    if out.dtype != xp.float32:
        raise ValueError(f"Planned output must be float32, got {out.dtype}")
    if not out.flags["C_CONTIGUOUS"]:
        raise ValueError("Planned output must be C-contiguous")
    return out


def _planned_output_for_shape(
    attrs: Dict[str, Any], shape: Tuple[int, ...]
) -> xp.ndarray:
    """Return a contiguous FP32 output with an explicitly derived shape."""
    out = attrs.get("_planned_output")
    if out is None:
        return xp.empty(shape, dtype=xp.float32)
    if not isinstance(out, cp.ndarray):
        raise TypeError("_planned_output must be a CuPy array")
    if tuple(out.shape) != tuple(shape):
        raise ValueError(
            f"Planned output shape {out.shape} does not match {shape}"
        )
    if out.dtype != xp.float32:
        raise ValueError(f"Planned output must be float32, got {out.dtype}")
    if not out.flags["C_CONTIGUOUS"]:
        raise ValueError("Planned output must be C-contiguous")
    return out


def _resolve_reshape_shape(
    input_shape: Tuple[int, ...], shape_value: xp.ndarray, allowzero: int,
) -> Tuple[int, ...]:
    """Resolve the ONNX Reshape shape input without numerical array work."""
    target_shape_raw = to_host(shape_value).astype("int64", copy=False).tolist()
    if allowzero not in (0, 1):
        raise ValueError("Reshape allowzero must be 0 or 1")
    if allowzero and 0 in target_shape_raw and -1 in target_shape_raw:
        raise ValueError("Reshape cannot combine a literal zero and -1 when allowzero=1")
    target: List[int] = []
    infer_index: Optional[int] = None
    known_product = 1
    for index, raw_dimension in enumerate(target_shape_raw):
        dimension = int(raw_dimension)
        if dimension == 0 and not allowzero:
            if index >= len(input_shape):
                raise ValueError("Reshape zero index exceeds input rank")
            dimension = int(input_shape[index])
        if dimension == -1:
            if infer_index is not None:
                raise ValueError("Reshape permits at most one inferred dimension")
            infer_index = index
            target.append(-1)
        else:
            if dimension < 0:
                raise ValueError("Reshape dimensions must be non-negative or -1")
            target.append(dimension)
            known_product *= dimension
    input_elements = math.prod(int(dimension) for dimension in input_shape)
    if infer_index is not None:
        if known_product == 0 or input_elements % known_product != 0:
            raise ValueError("Reshape inferred dimension is not integral")
        target[infer_index] = input_elements // known_product
    elif math.prod(target) != input_elements:
        raise ValueError("Reshape target element count does not match input")
    return tuple(target)


def _broadcast_shape(*shapes: Tuple[int, ...]) -> Tuple[int, ...]:
    """Resolve NumPy broadcasting using shape metadata only."""
    rank = max((len(shape) for shape in shapes), default=0)
    result = [1] * rank
    for shape in shapes:
        padded = (1,) * (rank - len(shape)) + tuple(map(int, shape))
        for index, dimension in enumerate(padded):
            if result[index] == 1:
                result[index] = dimension
            elif dimension not in (1, result[index]):
                raise ValueError(f"Shapes {shapes} are not broadcastable")
    return tuple(result)


def require_device() -> None:
    """Fail closed when the mandatory CuPy CUDA device is unavailable."""
    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError("CuPy requires a visible CUDA device")


def to_device(value: Any) -> Any:
    return xp.asarray(value)


def to_host(value: Any) -> Any:
    """Copy device data to host only for serialization or control metadata."""
    return cp.asnumpy(value)


def array_module() -> Any:
    return xp


def ascontiguousarray(value: Any, dtype: Any = None) -> Any:
    return xp.ascontiguousarray(xp.asarray(value, dtype=dtype))


def synchronize() -> None:
    xp.cuda.get_current_stream().synchronize()


def runtime_evidence() -> Dict[str, Any]:
    """Return structured evidence from the active numerical backend.

    CuPy's default pool retains allocations, so ``total_bytes`` is a stable
    process-local high-water proxy even when MIG hides the process from
    ``nvidia-smi --query-compute-apps``.
    """
    evidence: Dict[str, Any] = {"backend": "cupy"}
    device = xp.cuda.Device()
    properties = xp.cuda.runtime.getDeviceProperties(device.id)
    name = properties.get("name", "unknown")
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="replace")
    pool = xp.get_default_memory_pool()
    evidence.update({
        "device_id": int(device.id),
        "device_name": str(name),
        "cupy_version": str(xp.__version__),
        "pool_used_bytes": int(pool.used_bytes()),
        "pool_reserved_bytes": int(pool.total_bytes()),
    })

    # Include the verified hardware capability snapshot for auditability.
    try:
        from c32.hardware import get_hardware
        hw = get_hardware()
        evidence["hardware_capability"] = {
            "name": hw.name,
            "source": hw.source,
            "verified": hw.verified,
            "compute_capability": (
                list(hw.compute_capability)
                if hw.compute_capability is not None else None
            ),
            "supported_precisions": hw.supported_precisions(),
            "gemm_kernels": sorted(hw.gemm_kernels_available()),
            "conv_strategies": hw.conv_strategies_available(),
            "max_threads_per_block": hw.max_threads_per_block,
            "max_shared_memory_per_block": hw.max_shared_memory_per_block,
            "smem_bytes_total": hw.smem_bytes_total,
        }
    except Exception:
        pass

    return evidence

# ── Erf implementation ──────────────────────────────────────────────
# Rational approximation for the error function.
# Accurate to ~1e-7, no scipy dependency required.

def _erf_approx(x: xp.ndarray) -> xp.ndarray:
    """Rational approximation of erf(x) for all real x.

    Uses the Abramowitz & Stegun 7.1.26 formula.
    """
    x = xp.asarray(x, dtype=xp.float32)
    sign = xp.sign(x)
    x = xp.abs(x)

    # Compute via: erf(x) ≈ 1 - 1/(1 + a1*x + a2*x^2 + a3*x^3 + a4*x^4)^4
    t = 1.0 / (1.0 + 0.3275911 * x)
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    t5 = t4 * t

    coeffs = xp.asarray(
        [0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429],
        dtype=xp.float32,
    )
    poly = coeffs[0]*t + coeffs[1]*t2 + coeffs[2]*t3 + coeffs[3]*t4 + coeffs[4]*t5

    return sign * (1.0 - poly * xp.exp(-x * x))


# ── Operator implementations ────────────────────────────────────────


def op_flatten(inputs: List[xp.ndarray], attrs: Dict[str, Any]) -> xp.ndarray:
    """Flatten an input to the two-dimensional ONNX result shape.

    ONNX defines the output as ``(prod(dims[:axis]), prod(dims[axis:]))``.
    In particular, ``axis=0`` and axes other than one must still produce a
    rank-two tensor.
    """
    x = inputs[0]
    axis = int(attrs.get("axis", 1))
    if axis < 0:
        axis = len(x.shape) + axis
    if axis < 0 or axis > len(x.shape):
        raise ValueError(f"Flatten axis {axis} is invalid for rank {len(x.shape)}")
    outer = math.prod(x.shape[:axis])
    inner = math.prod(x.shape[axis:])
    return x.reshape((outer, inner))


def op_gemm(inputs: List[xp.ndarray], attrs: Dict[str, Any]) -> xp.ndarray:
    """Gemm: Y = alpha * A' * B' + beta * C

    ONNX attributes:
        alpha (float, default 1.0)
        beta (float, default 1.0)
        transA (int, default 0)
        transB (int, default 0)
    """
    a = xp.asarray(inputs[0], dtype=xp.float32)
    b = xp.asarray(inputs[1], dtype=xp.float32)
    c = xp.asarray(inputs[2], dtype=xp.float32) if len(inputs) >= 3 and inputs[2] is not None else None

    alpha = float(attrs.get("alpha", 1.0))
    beta = float(attrs.get("beta", 1.0))
    transA = int(attrs.get("transA", 0))
    transB = int(attrs.get("transB", 0))

    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("Gemm inputs A and B must both have rank 2")
    if transA not in (0, 1) or transB not in (0, 1):
        raise ValueError("Gemm transA and transB must be 0 or 1")

    if transA:
        a = a.T
    if transB:
        b = b.T

    result = xp.asarray(alpha, dtype=xp.float32) * xp.dot(a, b)
    if c is not None:
        result += xp.asarray(beta, dtype=xp.float32) * c

    return result.astype(xp.float32, copy=False)


def op_relu(inputs: List[xp.ndarray], attrs: Dict[str, Any]) -> xp.ndarray:
    """Relu: elementwise max(x, 0)."""
    x = inputs[0]
    return xp.maximum(x, xp.float32(0))


def op_conv(inputs: List[xp.ndarray], attrs: Dict[str, Any]) -> xp.ndarray:
    """Conv: NCHW convolution using im2col + tensordot.

    ONNX attributes:
        kernel_shape: [kH, kW]
        auto_pad: NOTSET, VALID, SAME_UPPER, or SAME_LOWER
        pads: [padH_begin, padW_begin, padH_end, padW_end]
        strides: [sH, sW] (default [1, 1])
        dilations: [dH, dW] (default [1, 1])
        group: int (default 1)

    Keeps computation in float32 for memory efficiency and speed.
    Uses xp.tensordot for BLAS-optimized contraction.
    """
    x = xp.asarray(inputs[0], dtype=xp.float32)
    w = xp.asarray(inputs[1], dtype=xp.float32)
    b = xp.asarray(inputs[2], dtype=xp.float32) if len(inputs) >= 3 and inputs[2] is not None else None

    if x.ndim != 4 or w.ndim != 4:
        raise ValueError("This C3 Conv implementation requires rank-4 NCHW tensors")

    kernel_shape = attrs.get("kernel_shape", [w.shape[-2], w.shape[-1]])
    kH = int(kernel_shape[0])
    kW = int(kernel_shape[-1]) if len(kernel_shape) > 1 else int(kernel_shape[0])

    strides = list(attrs.get("strides", [1, 1]))
    if len(strides) != 2:
        raise ValueError("Conv strides must contain two values")
    sH = int(strides[0])
    sW = int(strides[1])

    dilations = list(attrs.get("dilations", [1, 1]))
    if len(dilations) != 2:
        raise ValueError("Conv dilations must contain two values")
    dH = int(dilations[0])
    dW = int(dilations[1])

    group = int(attrs.get("group", 1))

    N, C, H, W_in = x.shape
    O = w.shape[0]

    if min(kH, kW, sH, sW, dH, dW, group) <= 0:
        raise ValueError("Conv kernel, stride, dilation, and group values must be positive")
    if tuple(w.shape[-2:]) != (kH, kW):
        raise ValueError(
            f"Conv kernel_shape {(kH, kW)} does not match weights {w.shape[-2:]}"
        )
    if C % group != 0 or O % group != 0 or w.shape[1] != C // group:
        raise ValueError("Conv group is incompatible with input/output channels")
    if b is not None and (b.ndim != 1 or b.shape[0] != O):
        raise ValueError("Conv bias must be one-dimensional with one value per output channel")

    auto_pad = attrs.get("auto_pad", "NOTSET")
    if isinstance(auto_pad, bytes):
        auto_pad = auto_pad.decode("utf-8")
    auto_pad = str(auto_pad).upper()
    if auto_pad not in {"NOTSET", "VALID", "SAME_UPPER", "SAME_LOWER"}:
        raise ValueError(f"Unsupported Conv auto_pad value: {auto_pad}")

    effective_h = dH * (kH - 1) + 1
    effective_w = dW * (kW - 1) + 1
    if auto_pad in {"SAME_UPPER", "SAME_LOWER"}:
        out_h = math.ceil(H / sH)
        out_w = math.ceil(W_in / sW)
        total_h = max((out_h - 1) * sH + effective_h - H, 0)
        total_w = max((out_w - 1) * sW + effective_w - W_in, 0)
        if auto_pad == "SAME_UPPER":
            pHT, pWL = total_h // 2, total_w // 2
        else:
            pHT, pWL = (total_h + 1) // 2, (total_w + 1) // 2
        pHB, pWR = total_h - pHT, total_w - pWL
    elif auto_pad == "VALID":
        pHT = pWL = pHB = pWR = 0
    else:
        pads = list(attrs.get("pads", [0, 0, 0, 0]))
        if len(pads) != 4:
            raise ValueError("Conv pads must contain four values")
        pHT, pWL, pHB, pWR = [int(p) for p in pads]
        if min(pHT, pWL, pHB, pWR) < 0:
            raise ValueError("Conv pads must be non-negative")

    # Output spatial dims
    H_out = (H + pHT + pHB - effective_h) // sH + 1
    W_out = (W_in + pWL + pWR - effective_w) // sW + 1
    if H_out <= 0 or W_out <= 0:
        raise ValueError("Conv attributes produce a non-positive output dimension")

    # Pad input
    x_padded = xp.pad(x, ((0, 0), (0, 0), (pHT, pHB), (pWL, pWR)),
                      mode='constant', constant_values=0)

    # Use CuPy sliding_window_view for clean im2col.
    # Extract spatial windows: (N, C, H_out, W_out, kH, kW)
    try:
        patches = xp.lib.stride_tricks.sliding_window_view(
            x_padded, (effective_h, effective_w), axis=(2, 3)
        )[:, :, ::sH, ::sW, ::dH, ::dW]
        patches = patches[:, :, :H_out, :W_out, :, :]
    except Exception:
        # Fallback to manual as_strided
        col_shape = (N, C, H_out, W_out, kH, kW)
        col_strides = (
            x_padded.strides[0],
            x_padded.strides[1],
            sH * x_padded.strides[2],
            sW * x_padded.strides[3],
            dH * x_padded.strides[2],
            dW * x_padded.strides[3],
        )
        patches = xp.lib.stride_tricks.as_strided(
            x_padded, shape=col_shape, strides=col_strides
        )

    # patches: (N, C, H_out, W_out, kH, kW)
    # w: (O, C//group, kH, kW) for group=1: (O, C, kH, kW)

    if group == 1:
        # Reshape to use xp.dot (BLAS-optimized)
        # patches: (N, C, H_out, W_out, kH, kW) -> (N * H_out * W_out, C * kH * kW)
        patches_2d = patches.transpose(0, 2, 3, 1, 4, 5).reshape(-1, C * kH * kW)
        # w: (O, C, kH, kW) -> (C * kH * kW, O)
        w_2d = w.reshape(O, -1).T
        # matmul: (N*H_out*W_out, C*kH*kW) @ (C*kH*kW, O) -> (N*H_out*W_out, O)
        out_2d = xp.dot(patches_2d, w_2d)
        out = out_2d.reshape(N, H_out, W_out, O).transpose(0, 3, 1, 2)
    else:
        # Grouped convolution: process each group separately
        Cg_in = C // group
        Og_out = O // group
        out_parts = []

        for g in range(group):
            c_start = g * Cg_in
            c_end = c_start + Cg_in
            o_start = g * Og_out
            o_end = o_start + Og_out

            patches_g = patches[:, c_start:c_end, :, :, :, :]
            w_g = w[o_start:o_end, :, :, :]

            patches_2d = patches_g.transpose(0, 2, 3, 1, 4, 5).reshape(-1, Cg_in * kH * kW)
            w_2d = w_g.reshape(Og_out, -1).T
            out_g_2d = xp.dot(patches_2d, w_2d)
            out_g = out_g_2d.reshape(N, H_out, W_out, Og_out).transpose(0, 3, 1, 2)
            out_parts.append(out_g)

        out = xp.concatenate(out_parts, axis=1)

    if b is not None:
        out += b.reshape(1, O, 1, 1)

    return out.astype(xp.float32, copy=False)


def op_add(inputs: List[xp.ndarray], attrs: Dict[str, Any]) -> xp.ndarray:
    """Add: elementwise addition with broadcasting."""
    a = xp.asarray(inputs[0], dtype=xp.float32)
    b = xp.asarray(inputs[1], dtype=xp.float32)
    return xp.add(a, b, dtype=xp.float32)


def op_global_average_pool(inputs: List[xp.ndarray], attrs: Dict[str, Any]) -> xp.ndarray:
    """GlobalAveragePool: mean over spatial dimensions H and W.

    Reduces from NCHW to NC11 (keeps dims).
    """
    x = xp.asarray(inputs[0], dtype=xp.float32)
    # Average over all spatial dims (axes 2, 3)
    axes = tuple(range(2, len(x.shape)))
    result = x.mean(axis=axes, keepdims=True)
    return result.astype(xp.float32, copy=False)


def op_gather(inputs: List[xp.ndarray], attrs: Dict[str, Any]) -> xp.ndarray:
    """Gather: select entries from data along axis.

    ONNX opset 13+ semantics:
        For data of rank r >= 1 and indices of rank q >= 0,
        gather entries along axis and concatenate into output of rank q + r - 1.
        Output shape = data.shape[:axis] + indices.shape + data.shape[axis+1:]

        axis (default 0)
    """
    x = inputs[0]
    indices = xp.asarray(inputs[1], dtype=xp.int64)
    axis = int(attrs.get("axis", 0))
    if axis < 0:
        axis = len(x.shape) + axis
    if axis < 0 or axis >= len(x.shape):
        raise ValueError(f"Gather axis {axis} is invalid for rank {len(x.shape)}")

    # Use xp.take which preserves indices shape in output
    return xp.take(x, indices, axis=axis)


def op_layer_normalization(inputs: List[xp.ndarray], attrs: Dict[str, Any]) -> xp.ndarray:
    """LayerNormalization: normalize over specified axes.

    ONNX:
        axis (default -1): first normalization axis.
            Normalizes over axes [axis, ..., rank-1].
        epsilon (default 1e-5)
        Inputs: X, Scale, [Bias]
    """
    x = xp.asarray(inputs[0], dtype=xp.float32)
    scale = xp.asarray(inputs[1], dtype=xp.float32)
    bias = xp.asarray(inputs[2], dtype=xp.float32) if len(inputs) >= 3 and inputs[2] is not None else None

    axis = int(attrs.get("axis", -1))
    epsilon = xp.float32(attrs.get("epsilon", 1e-5))

    if axis < 0:
        axis = len(x.shape) + axis
    if axis < 0 or axis >= len(x.shape):
        raise ValueError(
            f"LayerNormalization axis {axis} is invalid for rank {len(x.shape)}"
        )

    # Normalization axes: from axis to end
    norm_axes = tuple(range(axis, len(x.shape)))

    mean = x.mean(axis=norm_axes, keepdims=True)
    # Use the two-pass formula for numerical stability
    x_centered = x - mean
    var = xp.mean(x_centered * x_centered, axis=norm_axes, keepdims=True)

    inv_std = 1.0 / xp.sqrt(var + epsilon)
    normalized = x_centered * inv_std

    # Apply scale and bias
    result = normalized * scale
    if bias is not None:
        result += bias

    result = result.astype(xp.float32, copy=False)
    num_outputs = int(attrs.get("_num_outputs", 1))
    if num_outputs < 1 or num_outputs > 3:
        raise ValueError("LayerNormalization supports one to three outputs")
    if num_outputs == 1:
        return result
    outputs = [result, mean.astype(xp.float32, copy=False)]
    if num_outputs == 3:
        outputs.append(inv_std.astype(xp.float32, copy=False))
    return outputs


def op_matmul(inputs: List[xp.ndarray], attrs: Dict[str, Any]) -> xp.ndarray:
    """MatMul: matrix multiplication.

    ONNX: standard matmul. For tensors with >2 dims, batch matmul is used.
    """
    a = xp.asarray(inputs[0], dtype=xp.float32)
    b = xp.asarray(inputs[1], dtype=xp.float32)
    result = xp.matmul(a, b)
    return result.astype(xp.float32, copy=False)


def op_constant(inputs: List[xp.ndarray], attrs: Dict[str, Any]) -> xp.ndarray:
    """Constant: outputs a constant value (already stored as weight).

    The value is extracted during model loading and stored in the weights dict.
    inputs list is empty for Constant nodes; the value is in the attributes.
    """
    # This is handled specially in the executor — the value is pre-loaded
    # as if it were a weight. If called directly, raise an error.
    raise RuntimeError("Constant nodes must be pre-loaded as weights")


def op_split(inputs: List[xp.ndarray], attrs: Dict[str, Any]) -> List[xp.ndarray]:
    """Split: split a tensor into a list of tensors along axis.

    ONNX:
        axis (default 0)
        split: list of sizes for each output (optional, defaults to equal split)
    """
    x = inputs[0]
    axis = int(attrs.get("axis", 0))
    if axis < 0:
        axis = len(x.shape) + axis
    if axis < 0 or axis >= len(x.shape):
        raise ValueError(f"Split axis {axis} is invalid for rank {len(x.shape)}")

    split_input = inputs[1] if len(inputs) > 1 else None
    if split_input is not None and int(split_input.size) > 0:
        split_sizes = [int(s) for s in to_host(split_input).reshape(-1).tolist()]
    elif attrs.get("split") is not None:
        # The attribute is retained for compatibility with pre-opset-13 graphs.
        split_attr = attrs["split"]
        split_sizes = [int(s) for s in split_attr]
    else:
        # Equal split
        num_outputs = int(attrs.get("_num_outputs", 2))
        if num_outputs <= 0 or x.shape[axis] % num_outputs != 0:
            raise ValueError("Equal Split requires a positive divisor of the axis size")
        size_per = x.shape[axis] // num_outputs
        split_sizes = [size_per] * num_outputs

    num_outputs = int(attrs.get("_num_outputs", len(split_sizes)))
    if len(split_sizes) != num_outputs:
        raise ValueError("Split size count must equal the number of outputs")
    if any(size < 0 for size in split_sizes) or sum(split_sizes) != x.shape[axis]:
        raise ValueError("Split sizes must be non-negative and sum to the axis size")

    # CuPy 14.1.1 requires ordinary Python integer boundaries here: passing a
    # CuPy ndarray makes its split routine treat the array as a scalar section
    # count.  Compute the tiny metadata list on the host without a GPU roundtrip.
    indices: List[int] = []
    boundary = 0
    for size in split_sizes[:-1]:
        boundary += int(size)
        indices.append(boundary)
    results = xp.split(x, indices, axis=axis)
    return list(results)


def op_reshape(inputs: List[xp.ndarray], attrs: Dict[str, Any]) -> xp.ndarray:
    """Reshape: reshape with special value handling (0 = copy from input, -1 = infer).

    ONNX opset 17:
        Inputs: data, shape (int64 tensor)
        allowzero: if 1, 0 means set dim to 0 (not copy)
    """
    x = inputs[0]
    target_shape = _resolve_reshape_shape(
        tuple(x.shape), inputs[1], int(attrs.get("allowzero", 0))
    )
    return x.reshape(target_shape)


def op_transpose(inputs: List[xp.ndarray], attrs: Dict[str, Any]) -> xp.ndarray:
    """Transpose: permute axes.

    ONNX: perm (list of ints) — defaults to reverse order.
    """
    x = inputs[0]
    perm = attrs.get("perm", None)
    if perm is None:
        perm = list(range(len(x.shape) - 1, -1, -1))
    else:
        perm = [int(p) for p in perm]
    if sorted(perm) != list(range(len(x.shape))):
        raise ValueError("Transpose perm must contain every axis exactly once")
    return xp.transpose(x, perm)


def op_softmax(inputs: List[xp.ndarray], attrs: Dict[str, Any]) -> xp.ndarray:
    """Softmax: numerically stable softmax along axis.

    ONNX opset 13+: axis (default -1)
    """
    x = xp.asarray(inputs[0], dtype=xp.float32)
    axis = int(attrs.get("axis", -1))
    if axis < 0:
        axis = len(x.shape) + axis
    if axis < 0 or axis >= len(x.shape):
        raise ValueError(f"Softmax axis {axis} is invalid for rank {len(x.shape)}")

    # Stable softmax: subtract max before exp
    max_val = x.max(axis=axis, keepdims=True)
    exp_x = xp.exp(x - max_val)
    result = exp_x / exp_x.sum(axis=axis, keepdims=True)
    return result.astype(xp.float32, copy=False)


def op_erf(inputs: List[xp.ndarray], attrs: Dict[str, Any]) -> xp.ndarray:
    """Erf: elementwise error function.

    Used for GELU activation: GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    """
    x = xp.asarray(inputs[0], dtype=xp.float32)
    return _erf_approx(x).astype(xp.float32, copy=False)


def op_mul(inputs: List[xp.ndarray], attrs: Dict[str, Any]) -> xp.ndarray:
    """Mul: elementwise multiplication with broadcasting."""
    a = xp.asarray(inputs[0], dtype=xp.float32)
    b = xp.asarray(inputs[1], dtype=xp.float32)
    return xp.multiply(a, b, dtype=xp.float32)


def op_div(inputs: List[xp.ndarray], attrs: Dict[str, Any]) -> xp.ndarray:
    """Div: elementwise division with broadcasting."""
    a = xp.asarray(inputs[0], dtype=xp.float32)
    b = xp.asarray(inputs[1], dtype=xp.float32)
    return xp.divide(a, b, dtype=xp.float32)


def op_sub(inputs: List[xp.ndarray], attrs: Dict[str, Any]) -> xp.ndarray:
    """Sub: elementwise subtraction with broadcasting."""
    a = xp.asarray(inputs[0], dtype=xp.float32)
    b = xp.asarray(inputs[1], dtype=xp.float32)
    return xp.subtract(a, b, dtype=xp.float32)


def op_exp(inputs: List[xp.ndarray], attrs: Dict[str, Any]) -> xp.ndarray:
    """Exp: elementwise exponential."""
    return xp.exp(xp.asarray(inputs[0], dtype=xp.float32))


def op_sqrt(inputs: List[xp.ndarray], attrs: Dict[str, Any]) -> xp.ndarray:
    """Sqrt: elementwise square root."""
    return xp.sqrt(xp.asarray(inputs[0], dtype=xp.float32))


# ── Operator dispatch table ─────────────────────────────────────────

OP_DISPATCH: Dict[str, Any] = {
    "Add": op_add,
    "Constant": op_constant,
    "Conv": op_conv,
    "Div": op_div,
    "Erf": op_erf,
    "Flatten": op_flatten,
    "Gather": op_gather,
    "Gemm": op_gemm,
    "GlobalAveragePool": op_global_average_pool,
    "LayerNormalization": op_layer_normalization,
    "MatMul": op_matmul,
    "Mul": op_mul,
    "Relu": op_relu,
    "Reshape": op_reshape,
    "Softmax": op_softmax,
    "Split": op_split,
    "Sqrt": op_sqrt,
    "Sub": op_sub,
    "Transpose": op_transpose,
    "Exp": op_exp,
}


# ── Fused operator implementations ─────────────────────────────────


def _execute_gemm_epilogue(
    inputs: List[xp.ndarray], attrs: Dict[str, Any], *, matmul_semantics: bool,
) -> xp.ndarray:
    """Execute FP32 GEMM, optional broadcast bias, and optional ReLU once."""
    global _GEMM_EPILOGUE_KERNEL
    if len(inputs) < 2:
        raise ValueError("Fused GEMM requires A and B inputs")
    a = xp.ascontiguousarray(xp.asarray(inputs[0], dtype=xp.float32))
    b = xp.ascontiguousarray(xp.asarray(inputs[1], dtype=xp.float32))
    if b.ndim != 2:
        raise ValueError("Fused GEMM requires a rank-2 B input")

    flatten_axis = attrs.get("_flatten_axis")
    if flatten_axis is not None:
        axis = int(flatten_axis)
        if axis < 0:
            axis += a.ndim
        if axis < 0 or axis > a.ndim:
            raise ValueError("Fused GEMM Flatten axis is out of range")
        a = a.reshape(
            math.prod(int(value) for value in a.shape[:axis]),
            math.prod(int(value) for value in a.shape[axis:]),
        )

    if matmul_semantics:
        if a.ndim < 2:
            raise ValueError("Fused MatMul+bias requires rank >= 2")
        leading_shape = tuple(int(value) for value in a.shape[:-1])
        a = a.reshape(-1, int(a.shape[-1]))
        trans_a = 0
        trans_b = 0
    else:
        if a.ndim != 2:
            raise ValueError("Fused Gemm requires rank-2 A after Flatten")
        leading_shape = ()
        trans_a = int(attrs.get("transA", 0))
        trans_b = int(attrs.get("transB", 0))
        if trans_a not in (0, 1) or trans_b not in (0, 1):
            raise ValueError("Gemm transA and transB must be 0 or 1")

    a_rows, a_cols = (int(a.shape[0]), int(a.shape[1]))
    b_rows, b_cols = (int(b.shape[0]), int(b.shape[1]))
    m = a_cols if trans_a else a_rows
    k = a_rows if trans_a else a_cols
    b_k = b_cols if trans_b else b_rows
    n = b_rows if trans_b else b_cols
    if k != b_k:
        raise ValueError(f"Fused GEMM inner dimensions differ: {k} and {b_k}")

    bias = inputs[2] if len(inputs) >= 3 else None
    has_bias = int(bias is not None)
    c_rows = c_cols = 1
    if bias is None:
        c = a
    else:
        c = xp.ascontiguousarray(xp.asarray(bias, dtype=xp.float32))
        c_shape = list(c.shape)
        while len(c_shape) > 2 and c_shape[0] == 1:
            c_shape.pop(0)
        if len(c_shape) == 0:
            c_rows = c_cols = 1
        elif len(c_shape) == 1:
            c_rows, c_cols = 1, int(c_shape[0])
        elif len(c_shape) == 2:
            c_rows, c_cols = int(c_shape[0]), int(c_shape[1])
        else:
            raise ValueError("Fused GEMM bias rank is not broadcastable")
        if c_rows not in (1, m) or c_cols not in (1, n):
            raise ValueError(
                f"Fused GEMM bias shape {tuple(c.shape)} cannot broadcast to {(m, n)}"
            )
        c = c.reshape(c_rows, c_cols)

    output_shape = leading_shape + (n,) if matmul_semantics else (m, n)
    out = _planned_output_for_shape(attrs, output_shape)
    if _GEMM_EPILOGUE_KERNEL is None:
        source = r'''
        extern "C" __global__ void c3_gemm_epilogue(
            const float* a, const float* b, const float* c, float* out,
            long long m, long long n, long long k,
            long long a_rows, long long a_cols,
            long long b_rows, long long b_cols,
            long long c_rows, long long c_cols,
            int trans_a, int trans_b, int has_bias, int relu,
            float alpha, float beta) {
          const long long col = (long long)blockIdx.x * blockDim.x + threadIdx.x;
          const long long row = (long long)blockIdx.y * blockDim.y + threadIdx.y;
          if (row >= m || col >= n) return;
          float sum = 0.0F;
          for (long long inner = 0; inner < k; ++inner) {
            const long long ai = trans_a
                ? inner * a_cols + row : row * a_cols + inner;
            const long long bi = trans_b
                ? col * b_cols + inner : inner * b_cols + col;
            sum = fmaf(a[ai], b[bi], sum);
          }
          float value = alpha * sum;
          if (has_bias) {
            const long long cr = c_rows == 1 ? 0 : row;
            const long long cc = c_cols == 1 ? 0 : col;
            value += beta * c[cr * c_cols + cc];
          }
          if (relu && value < 0.0F) value = 0.0F;
          out[row * n + col] = value;
        }
        '''
        _GEMM_EPILOGUE_KERNEL = cp.RawKernel(source, "c3_gemm_epilogue")
    block = (16, 16)
    grid = ((n + block[0] - 1) // block[0],
            (m + block[1] - 1) // block[1])
    _GEMM_EPILOGUE_KERNEL(
        grid, block,
        (
            a, b, c, out, cp.int64(m), cp.int64(n), cp.int64(k),
            cp.int64(a_rows), cp.int64(a_cols), cp.int64(b_rows),
            cp.int64(b_cols), cp.int64(c_rows), cp.int64(c_cols),
            cp.int32(trans_a), cp.int32(trans_b), cp.int32(has_bias),
            cp.int32(attrs.get("_activation") == "Relu"),
            cp.float32(attrs.get("alpha", 1.0)),
            cp.float32(attrs.get("beta", 1.0)),
        ),
    )
    return out


def op_fused_matmul_bias(inputs: List[xp.ndarray], attrs: Dict[str, Any]) -> xp.ndarray:
    """Single-kernel MatMul followed by a broadcast bias Add."""
    return _execute_gemm_epilogue(inputs, attrs, matmul_semantics=True)


def op_fused_gemm_epilogue(inputs: List[xp.ndarray],
                            attrs: Dict[str, Any]) -> xp.ndarray:
    """Single-kernel Gemm with optional absorbed Flatten and ReLU."""
    return _execute_gemm_epilogue(inputs, attrs, matmul_semantics=False)


def op_fused_conv_batchnorm(inputs: List[xp.ndarray], attrs: Dict[str, Any]) -> xp.ndarray:
    """Reference execution for fused Conv followed by inference BatchNorm."""
    if len(inputs) < 2:
        raise ValueError("FusedConv2dBatchNorm requires at least X, W inputs")
    if attrs.get("bn_folded"):
        return op_conv(inputs, attrs)
    offset = int(attrs.get("bn_parameter_offset", len(inputs)))
    conv_out = op_conv(inputs[:offset], attrs)
    if len(inputs) < offset + 4:
        raise ValueError("FusedConv2dBatchNorm is missing scale/bias/mean/variance")
    scale, bias, mean, var = (
        xp.asarray(x, dtype=xp.float32) for x in inputs[offset:offset + 4]
    )
    epsilon = float(attrs.get("bn_epsilon", 1e-5))
    channel_shape = [1, -1] + [1] * (conv_out.ndim - 2)
    scale = scale.reshape(channel_shape)
    bias = bias.reshape(channel_shape)
    mean = mean.reshape(channel_shape)
    var = var.reshape(channel_shape)
    return ((conv_out - mean) / xp.sqrt(var + epsilon) * scale + bias).astype(
        xp.float32, copy=False
    )


def _execute_fused_conv_epilogue(
    inputs: List[xp.ndarray], attrs: Dict[str, Any], *, has_residual: bool,
) -> xp.ndarray:
    """Execute Conv through the BLAS fallback plus one epilogue kernel.

    The former one-thread-per-output direct convolution was numerically correct
    but regressed released ResNet wall time by nearly an order of magnitude on
    the H200.  Keep the graph-level fusion while routing the expensive
    contraction through :func:`op_conv`, whose im2col contraction uses CuPy's
    BLAS-backed ``dot``.  Residual Add and ReLU remain one generated epilogue
    launch writing directly into the planned output.

    A future single-launch direct/implicit-GEMM kernel must not replace this
    path until cold H200 timing and the FP32 numerical gate both pass.
    """
    global _CONV_ACTIVATION_EPILOGUE
    global _CONV_RESIDUAL_ACTIVATION_EPILOGUE
    if len(inputs) < 2:
        raise ValueError("Fused Conv requires X and W inputs")
    residual_index = int(attrs.get("_residual_input_index", len(inputs)))
    conv_input_count = residual_index if has_residual else len(inputs)
    if has_residual and not 0 <= residual_index < len(inputs):
        raise ValueError("Fused Conv residual input index is invalid")
    if attrs.get("_activation") != "Relu":
        raise ValueError("Fused Conv epilogue currently requires Relu")

    conv = op_conv(list(inputs[:conv_input_count]), attrs)
    out = _planned_output_for_shape(attrs, tuple(int(v) for v in conv.shape))

    if has_residual:
        residual = xp.asarray(inputs[residual_index], dtype=xp.float32)
        if tuple(residual.shape) != tuple(conv.shape):
            raise ValueError(
                f"Fused Conv residual shape {residual.shape} != {conv.shape}"
            )
        if _CONV_RESIDUAL_ACTIVATION_EPILOGUE is None:
            _CONV_RESIDUAL_ACTIVATION_EPILOGUE = cp.ElementwiseKernel(
                "float32 conv, float32 residual",
                "float32 out",
                "out = fmaxf(conv + residual, 0.0f)",
                "c3_conv_residual_relu_epilogue",
            )
        _CONV_RESIDUAL_ACTIVATION_EPILOGUE(conv, residual, out)
    else:
        if _CONV_ACTIVATION_EPILOGUE is None:
            _CONV_ACTIVATION_EPILOGUE = cp.ElementwiseKernel(
                "float32 conv",
                "float32 out",
                "out = fmaxf(conv, 0.0f)",
                "c3_conv_relu_epilogue",
            )
        _CONV_ACTIVATION_EPILOGUE(conv, out)
    return out


def op_fused_conv_activation(inputs: List[xp.ndarray],
                             attrs: Dict[str, Any]) -> xp.ndarray:
    return _execute_fused_conv_epilogue(inputs, attrs, has_residual=False)


def op_fused_conv_residual_activation(
    inputs: List[xp.ndarray], attrs: Dict[str, Any]
) -> xp.ndarray:
    return _execute_fused_conv_epilogue(inputs, attrs, has_residual=True)


def op_fused_ew_chain(inputs: List[xp.ndarray], attrs: Dict[str, Any]) -> xp.ndarray:
    """FusedEWChain: chain of elementwise ops fused into one kernel.

    ``attrs['_ops']`` contains the ordered list of (op_type, op_attrs) pairs.
    The first op consumes the primary input; subsequent ops may reference
    additional inputs from the fused node's input list.
    """
    ops = attrs.get("_ops", [])
    if not ops:
        raise ValueError("FusedEWChain has no elementwise program")

    expression_for = {
        "Add": lambda a: f"({a[0]} + {a[1]})",
        "Sub": lambda a: f"({a[0]} - {a[1]})",
        "Mul": lambda a: f"({a[0]} * {a[1]})",
        "Div": lambda a: f"({a[0]} / {a[1]})",
        "Relu": lambda a: f"fmaxf({a[0]}, 0.0f)",
        "Erf": lambda a: f"erff({a[0]})",
        "Sqrt": lambda a: f"sqrtf({a[0]})",
        "Exp": lambda a: f"expf({a[0]})",
    }
    external = {index: f"in{index}" for index in range(len(inputs))}
    values: Dict[str, str] = {}
    statements: List[str] = []
    final_var = ""
    for index, entry in enumerate(ops):
        args: List[str] = []
        for ref in entry.get("inputs", []):
            if isinstance(ref, int) and ref in external:
                args.append(external[ref])
            elif isinstance(ref, str) and ref in values:
                args.append(values[ref])
            else:
                raise ValueError(f"FusedEWChain cannot resolve operand {ref!r}")
        op_type = entry.get("op", "")
        builder = expression_for.get(op_type)
        if builder is None:
            raise ValueError(f"Unknown op in FusedEWChain: {op_type}")
        arity = 1 if op_type in {"Relu", "Erf", "Sqrt", "Exp"} else 2
        if len(args) != arity:
            raise ValueError(f"FusedEWChain {op_type} expects {arity} operands")
        final_var = f"v{index}"
        statements.append(f"float {final_var} = {builder(args)};")
        output_name = entry.get("output", "")
        if output_name:
            values[output_name] = final_var
    statements.append(f"out = {final_var};")

    cache_key = json.dumps(ops, sort_keys=True, separators=(",", ":"))
    kernel = _ELEMENTWISE_FUSION_CACHE.get(cache_key)
    if kernel is None:
        digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:16]
        kernel = cp.ElementwiseKernel(
            ", ".join(f"float32 in{i}" for i in range(len(inputs))),
            "float32 out",
            " ".join(statements),
            f"c3_fused_ew_{digest}",
        )
        _ELEMENTWISE_FUSION_CACHE[cache_key] = kernel
    device_inputs = [xp.asarray(value, dtype=xp.float32) for value in inputs]
    output_shape = _broadcast_shape(
        *(tuple(value.shape) for value in device_inputs)
    )
    out = _planned_output_for_shape(attrs, output_shape)
    kernel(*device_inputs, out)
    return out


def op_fused_softmax_dropout(inputs: List[xp.ndarray], attrs: Dict[str, Any]) -> xp.ndarray:
    """FusedSoftmaxDropout: Softmax with inference-mode Dropout (identity).

    In inference mode, Dropout is a no-op, so this is just Softmax.
    """
    global _SOFTMAX_FUSION_KERNEL
    x = xp.ascontiguousarray(xp.asarray(inputs[0], dtype=xp.float32))
    axis = int(attrs.get("axis", -1))
    if axis < 0:
        axis += x.ndim
    if axis < 0 or axis >= x.ndim:
        raise ValueError(f"Softmax axis {axis} is out of range for rank {x.ndim}")
    if x.size == 0:
        return _planned_output(attrs, x)
    axis_size = int(x.shape[axis])
    inner = math.prod(int(dim) for dim in x.shape[axis + 1:])
    outer = math.prod(int(dim) for dim in x.shape[:axis])
    segments = outer * inner
    out = _planned_output(attrs, x)
    if _SOFTMAX_FUSION_KERNEL is None:
        source = r'''
        extern "C" __global__ void c3_fused_softmax(
            const float* x, float* out, long long axis_size, long long inner) {
          extern __shared__ float scratch[];
          const long long segment = (long long)blockIdx.x;
          const long long lane = (long long)threadIdx.x;
          const long long base = (segment / inner) * axis_size * inner
                               + (segment % inner);
          float local_max = -3.402823466e+38F;
          for (long long k = lane; k < axis_size; k += blockDim.x)
            local_max = fmaxf(local_max, x[base + k * inner]);
          scratch[lane] = local_max;
          __syncthreads();
          for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (lane < stride) scratch[lane] = fmaxf(scratch[lane], scratch[lane + stride]);
            __syncthreads();
          }
          const float maximum = scratch[0];
          float local_sum = 0.0F;
          for (long long k = lane; k < axis_size; k += blockDim.x)
            local_sum += expf(x[base + k * inner] - maximum);
          scratch[lane] = local_sum;
          __syncthreads();
          for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (lane < stride) scratch[lane] += scratch[lane + stride];
            __syncthreads();
          }
          const float denominator = scratch[0];
          for (long long k = lane; k < axis_size; k += blockDim.x)
            out[base + k * inner] = expf(x[base + k * inner] - maximum) / denominator;
        }
        '''
        _SOFTMAX_FUSION_KERNEL = cp.RawKernel(source, "c3_fused_softmax")
    block = 256
    _SOFTMAX_FUSION_KERNEL(
        (segments,), (block,),
        (x, out, cp.int64(axis_size), cp.int64(inner)),
        shared_mem=block * 4,
    )
    return out


def op_fused_residual_norm(inputs: List[xp.ndarray], attrs: Dict[str, Any]) -> xp.ndarray:
    """FusedResidualNorm: Add (residual) followed by LayerNorm.

    Lowered to: LayerNorm(Add(x, residual), weight, bias)
    """
    if len(inputs) < 2:
        raise ValueError("FusedResidualNorm requires at least x, residual inputs")
    if len(inputs) < 3 or inputs[2] is None:
        raise ValueError("FusedResidualNorm requires a LayerNorm scale")
    x = xp.ascontiguousarray(xp.asarray(inputs[0], dtype=xp.float32))
    residual = xp.ascontiguousarray(xp.asarray(inputs[1], dtype=xp.float32))
    if x.shape != residual.shape:
        raise ValueError("FusedResidualNorm requires equal residual input shapes")
    axis = int(attrs.get("axis", -1))
    if axis < 0:
        axis += x.ndim
    if axis < 0 or axis >= x.ndim:
        raise ValueError(f"LayerNorm axis {axis} is out of range for rank {x.ndim}")
    cols = math.prod(int(dim) for dim in x.shape[axis:])
    if cols == 0:
        return _planned_output(attrs, x)
    rows = int(x.size) // cols
    scale = xp.ascontiguousarray(
        xp.asarray(inputs[2], dtype=xp.float32)
    ).reshape(-1)
    if int(scale.size) != cols:
        raise ValueError("FusedResidualNorm scale size does not match normalized shape")
    has_bias = len(inputs) > 3 and inputs[3] is not None
    bias = None
    if has_bias:
        bias = xp.ascontiguousarray(
            xp.asarray(inputs[3], dtype=xp.float32)
        ).reshape(-1)
        if int(bias.size) != cols:
            raise ValueError("FusedResidualNorm bias size does not match normalized shape")
    out = _planned_output(attrs, x)
    if x.size == 0:
        return out
    kernel = _RESIDUAL_NORM_KERNELS.get(has_bias)
    if kernel is None:
        bias_parameter = ", const float* bias" if has_bias else ""
        bias_expression = " + bias[col]" if has_bias else ""
        source = f'''
        extern "C" __global__ void c3_fused_residual_norm(
            const float* x, const float* residual, const float* scale{bias_parameter},
            float* out, long long cols, float epsilon) {{
          extern __shared__ float scratch[];
          const long long row = (long long)blockIdx.x;
          const unsigned int lane = threadIdx.x;
          const long long base = row * cols;
          float local_sum = 0.0F;
          for (long long col = lane; col < cols; col += blockDim.x)
            local_sum += x[base + col] + residual[base + col];
          scratch[lane] = local_sum;
          __syncthreads();
          for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {{
            if (lane < stride) scratch[lane] += scratch[lane + stride];
            __syncthreads();
          }}
          const float mean = scratch[0] / (float)cols;
          float local_var = 0.0F;
          for (long long col = lane; col < cols; col += blockDim.x) {{
            const float centered = x[base + col] + residual[base + col] - mean;
            local_var += centered * centered;
          }}
          scratch[lane] = local_var;
          __syncthreads();
          for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {{
            if (lane < stride) scratch[lane] += scratch[lane + stride];
            __syncthreads();
          }}
          const float inv_std = rsqrtf(scratch[0] / (float)cols + epsilon);
          for (long long col = lane; col < cols; col += blockDim.x) {{
            const float normalized = (x[base + col] + residual[base + col] - mean) * inv_std;
            out[base + col] = normalized * scale[col]{bias_expression};
          }}
        }}
        '''
        kernel = cp.RawKernel(source, "c3_fused_residual_norm")
        _RESIDUAL_NORM_KERNELS[has_bias] = kernel
    epsilon = cp.float32(attrs.get("epsilon", 1e-5))
    if has_bias:
        args: Tuple[Any, ...] = (
            x, residual, scale, bias, out, cp.int64(cols), epsilon
        )
    else:
        args = (x, residual, scale, out, cp.int64(cols), epsilon)
    block = 256
    kernel((rows,), (block,), args, shared_mem=block * 4)
    return out


def op_fused_layer_normalization(
    inputs: List[xp.ndarray], attrs: Dict[str, Any]
) -> xp.ndarray:
    """Single-kernel ONNX LayerNormalization for one visible output."""
    global _LAYER_NORM_KERNEL
    if len(inputs) < 2 or inputs[1] is None:
        raise ValueError("Fused LayerNormalization requires X and Scale")
    x = xp.ascontiguousarray(xp.asarray(inputs[0], dtype=xp.float32))
    axis = int(attrs.get("axis", -1))
    if axis < 0:
        axis += x.ndim
    if axis < 0 or axis >= x.ndim:
        raise ValueError("Fused LayerNormalization axis is out of range")
    cols = math.prod(int(value) for value in x.shape[axis:])
    rows = int(x.size) // cols if cols else 0
    scale = xp.ascontiguousarray(
        xp.asarray(inputs[1], dtype=xp.float32)
    ).reshape(-1)
    if int(scale.size) != cols:
        raise ValueError("Fused LayerNormalization Scale size mismatch")
    has_bias = int(len(inputs) > 2 and inputs[2] is not None)
    bias = (
        xp.ascontiguousarray(
            xp.asarray(inputs[2], dtype=xp.float32)
        ).reshape(-1) if has_bias else scale
    )
    if has_bias and int(bias.size) != cols:
        raise ValueError("Fused LayerNormalization Bias size mismatch")
    out = _planned_output_for_shape(attrs, tuple(x.shape))
    if x.size == 0:
        return out
    if _LAYER_NORM_KERNEL is None:
        source = r'''
        extern "C" __global__ void c3_layer_norm(
            const float* x, const float* scale, const float* bias, float* out,
            long long cols, float epsilon, int has_bias) {
          extern __shared__ float scratch[];
          const long long row = (long long)blockIdx.x;
          const unsigned int lane = threadIdx.x;
          const long long base = row * cols;
          float local_sum = 0.0F;
          for (long long col = lane; col < cols; col += blockDim.x)
            local_sum += x[base + col];
          scratch[lane] = local_sum;
          __syncthreads();
          for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (lane < stride) scratch[lane] += scratch[lane + stride];
            __syncthreads();
          }
          const float mean = scratch[0] / (float)cols;
          float local_var = 0.0F;
          for (long long col = lane; col < cols; col += blockDim.x) {
            const float centered = x[base + col] - mean;
            local_var += centered * centered;
          }
          scratch[lane] = local_var;
          __syncthreads();
          for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (lane < stride) scratch[lane] += scratch[lane + stride];
            __syncthreads();
          }
          const float inv_std = rsqrtf(scratch[0] / (float)cols + epsilon);
          for (long long col = lane; col < cols; col += blockDim.x) {
            float value = (x[base + col] - mean) * inv_std * scale[col];
            if (has_bias) value += bias[col];
            out[base + col] = value;
          }
        }
        '''
        _LAYER_NORM_KERNEL = cp.RawKernel(source, "c3_layer_norm")
    block = 256
    _LAYER_NORM_KERNEL(
        (rows,), (block,),
        (x, scale, bias, out, cp.int64(cols),
         cp.float32(attrs.get("epsilon", 1e-5)), cp.int32(has_bias)),
        shared_mem=block * 4,
    )
    return out


def op_fused_attention_scores(
    inputs: List[xp.ndarray], attrs: Dict[str, Any]
) -> xp.ndarray:
    """Single-kernel rank-4 QK^T scaling, mask Add, and row Softmax."""
    global _ATTENTION_SCORES_KERNEL
    if len(inputs) != 4:
        raise ValueError("FusedAttentionScores requires Q, K, divisor, and mask")
    query = xp.ascontiguousarray(xp.asarray(inputs[0], dtype=xp.float32))
    key = xp.ascontiguousarray(xp.asarray(inputs[1], dtype=xp.float32))
    divisor = xp.ascontiguousarray(xp.asarray(inputs[2], dtype=xp.float32))
    mask = xp.ascontiguousarray(xp.asarray(inputs[3], dtype=xp.float32))
    if query.ndim != 4 or key.ndim != 4:
        raise ValueError("FusedAttentionScores requires rank-4 MatMul inputs")
    if int(divisor.size) != 1:
        raise ValueError("FusedAttentionScores divisor must be scalar")
    q0, q1, rows, inner = (int(value) for value in query.shape)
    k0, k1, key_inner, cols = (int(value) for value in key.shape)
    if inner != key_inner:
        raise ValueError("FusedAttentionScores MatMul inner dimensions differ")
    if q0 not in (1, k0) and k0 not in (1, q0):
        raise ValueError("FusedAttentionScores batch dimensions do not broadcast")
    if q1 not in (1, k1) and k1 not in (1, q1):
        raise ValueError("FusedAttentionScores head dimensions do not broadcast")
    out0, out1 = max(q0, k0), max(q1, k1)
    output_shape = (out0, out1, rows, cols)
    mask_shape = [1] * (4 - mask.ndim) + [int(value) for value in mask.shape]
    for mask_dim, output_dim in zip(mask_shape, output_shape):
        if mask_dim not in (1, output_dim):
            raise ValueError(
                f"Attention mask {tuple(mask.shape)} cannot broadcast to {output_shape}"
            )
    out = _planned_output_for_shape(attrs, output_shape)
    if _ATTENTION_SCORES_KERNEL is None:
        source = r'''
        extern "C" __global__ void c3_attention_scores(
            const float* query, const float* key, const float* divisor,
            const float* mask, float* out,
            int q0, int q1, int k0, int k1, int rows, int cols, int inner,
            int out_heads, int m0, int m1, int m2, int m3) {
          extern __shared__ float scratch[];
          const long long segment = (long long)blockIdx.x;
          const unsigned int lane = threadIdx.x;
          long long value = segment;
          const int row = value % rows; value /= rows;
          const int head = value % out_heads; value /= out_heads;
          const int batch = (int)value;
          const int qb = q0 == 1 ? 0 : batch;
          const int qh = q1 == 1 ? 0 : head;
          const int kb = k0 == 1 ? 0 : batch;
          const int kh = k1 == 1 ? 0 : head;
          float local_max = -3.402823466e+38F;
          for (int col = lane; col < cols; col += blockDim.x) {
            float score = 0.0F;
            for (int index = 0; index < inner; ++index) {
              const long long qi = (((long long)qb * q1 + qh) * rows + row) * inner + index;
              const long long ki = (((long long)kb * k1 + kh) * inner + index) * cols + col;
              score = fmaf(query[qi], key[ki], score);
            }
            const int mb = m0 == 1 ? 0 : batch;
            const int mh = m1 == 1 ? 0 : head;
            const int mr = m2 == 1 ? 0 : row;
            const int mc = m3 == 1 ? 0 : col;
            const long long mi = ((long long)(mb * m1 + mh) * m2 + mr) * m3 + mc;
            score = score / divisor[0] + mask[mi];
            local_max = fmaxf(local_max, score);
          }
          scratch[lane] = local_max;
          __syncthreads();
          for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (lane < stride) scratch[lane] = fmaxf(scratch[lane], scratch[lane + stride]);
            __syncthreads();
          }
          const float maximum = scratch[0];
          float local_sum = 0.0F;
          for (int col = lane; col < cols; col += blockDim.x) {
            float score = 0.0F;
            for (int index = 0; index < inner; ++index) {
              const long long qi = (((long long)qb * q1 + qh) * rows + row) * inner + index;
              const long long ki = (((long long)kb * k1 + kh) * inner + index) * cols + col;
              score = fmaf(query[qi], key[ki], score);
            }
            const int mb = m0 == 1 ? 0 : batch;
            const int mh = m1 == 1 ? 0 : head;
            const int mr = m2 == 1 ? 0 : row;
            const int mc = m3 == 1 ? 0 : col;
            const long long mi = ((long long)(mb * m1 + mh) * m2 + mr) * m3 + mc;
            score = score / divisor[0] + mask[mi];
            local_sum += expf(score - maximum);
          }
          scratch[lane] = local_sum;
          __syncthreads();
          for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (lane < stride) scratch[lane] += scratch[lane + stride];
            __syncthreads();
          }
          const float denominator = scratch[0];
          const long long output_base = segment * cols;
          for (int col = lane; col < cols; col += blockDim.x) {
            float score = 0.0F;
            for (int index = 0; index < inner; ++index) {
              const long long qi = (((long long)qb * q1 + qh) * rows + row) * inner + index;
              const long long ki = (((long long)kb * k1 + kh) * inner + index) * cols + col;
              score = fmaf(query[qi], key[ki], score);
            }
            const int mb = m0 == 1 ? 0 : batch;
            const int mh = m1 == 1 ? 0 : head;
            const int mr = m2 == 1 ? 0 : row;
            const int mc = m3 == 1 ? 0 : col;
            const long long mi = ((long long)(mb * m1 + mh) * m2 + mr) * m3 + mc;
            score = score / divisor[0] + mask[mi];
            out[output_base + col] = expf(score - maximum) / denominator;
          }
        }
        '''
        _ATTENTION_SCORES_KERNEL = cp.RawKernel(
            source, "c3_attention_scores"
        )
    block = 256
    segments = out0 * out1 * rows
    _ATTENTION_SCORES_KERNEL(
        (segments,), (block,),
        (
            query, key, divisor, mask, out,
            cp.int32(q0), cp.int32(q1), cp.int32(k0), cp.int32(k1),
            cp.int32(rows), cp.int32(cols), cp.int32(inner),
            cp.int32(out1),
            *(cp.int32(value) for value in mask_shape),
        ),
        shared_mem=block * 4,
    )
    return out


def op_fused_transpose_reshape(
    inputs: List[xp.ndarray], attrs: Dict[str, Any]
) -> xp.ndarray:
    """Write Transpose's contiguous order directly into Reshape's result."""
    global _TRANSPOSE_RESHAPE_KERNEL
    if len(inputs) < 2:
        raise ValueError("FusedTransposeReshape requires data and shape inputs")
    value = xp.ascontiguousarray(xp.asarray(inputs[0], dtype=xp.float32))
    rank = value.ndim
    if not 1 <= rank <= 4:
        raise ValueError("FusedTransposeReshape supports ranks one through four")
    perm = [int(item) for item in attrs.get("perm", range(rank - 1, -1, -1))]
    if sorted(perm) != list(range(rank)):
        raise ValueError("FusedTransposeReshape perm is invalid")
    transposed_shape = tuple(int(value.shape[index]) for index in perm)
    output_shape = _resolve_reshape_shape(
        transposed_shape, inputs[1], int(attrs.get("allowzero", 0))
    )
    out = _planned_output_for_shape(attrs, output_shape)
    if _TRANSPOSE_RESHAPE_KERNEL is None:
        source = r'''
        extern "C" __global__ void c3_transpose_reshape(
            const float* input, float* output, long long total, int rank,
            long long d0, long long d1, long long d2, long long d3,
            int p0, int p1, int p2, int p3) {
          const long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
          if (index >= total) return;
          const long long dims[4] = {d0, d1, d2, d3};
          const int perm[4] = {p0, p1, p2, p3};
          long long transposed_dims[4] = {1, 1, 1, 1};
          for (int axis = 0; axis < rank; ++axis)
            transposed_dims[axis] = dims[perm[axis]];
          long long remaining = index;
          long long output_coord[4] = {0, 0, 0, 0};
          for (int axis = rank - 1; axis >= 0; --axis) {
            output_coord[axis] = remaining % transposed_dims[axis];
            remaining /= transposed_dims[axis];
          }
          long long input_coord[4] = {0, 0, 0, 0};
          for (int axis = 0; axis < rank; ++axis)
            input_coord[perm[axis]] = output_coord[axis];
          long long input_index = 0;
          for (int axis = 0; axis < rank; ++axis)
            input_index = input_index * dims[axis] + input_coord[axis];
          output[index] = input[input_index];
        }
        '''
        _TRANSPOSE_RESHAPE_KERNEL = cp.RawKernel(
            source, "c3_transpose_reshape"
        )
    dimensions = list(map(int, value.shape)) + [1] * (4 - rank)
    permutation = perm + list(range(rank, 4))
    total = int(value.size)
    block = 256
    _TRANSPOSE_RESHAPE_KERNEL(
        ((total + block - 1) // block,), (block,),
        (
            value, out, cp.int64(total), cp.int32(rank),
            *(cp.int64(item) for item in dimensions),
            *(cp.int32(item) for item in permutation),
        ),
    )
    return out


def execute_op(op_type: str, inputs: List[xp.ndarray],
               attrs: Dict[str, Any]) -> Any:
    """Execute an ONNX operator.

    Args:
        op_type: The ONNX operator type (e.g., "Conv", "Gemm").
        inputs: List of input arrays from the configured backend.
        attrs: Node attributes dict.

    Returns:
        A backend array, or a list of backend arrays (for Split).
    """
    impl = OP_DISPATCH.get(op_type)
    if impl is None:
        raise ValueError(f"Unknown operator type: {op_type}")
    return impl(inputs, attrs)


# Register fused operators after all implementations are defined
_FUSED_DISPATCH = {
    "FusedMatMulBias": op_fused_matmul_bias,
    "FusedGemmEpilogue": op_fused_gemm_epilogue,
    "FusedConv2dBatchNorm": op_fused_conv_batchnorm,
    "FusedConvActivation": op_fused_conv_activation,
    "FusedConvResidualActivation": op_fused_conv_residual_activation,
    "FusedAttentionScores": op_fused_attention_scores,
    "FusedEWChain": op_fused_ew_chain,
    "FusedLayerNormalization": op_fused_layer_normalization,
    "FusedSoftmaxDropout": op_fused_softmax_dropout,
    "FusedResidualNorm": op_fused_residual_norm,
    "FusedTransposeReshape": op_fused_transpose_reshape,
}
OP_DISPATCH.update(_FUSED_DISPATCH)
