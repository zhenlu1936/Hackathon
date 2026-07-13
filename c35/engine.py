"""C3.5 array compute engine — ONNX operator implementations.

Implements all 17 required ONNX operators with opset-17 semantics.
The deployment CLI configures CuPy so numerical work stays on the CUDA device.
NumPy is retained only as an explicit development/reference backend for unit
tests and machines without CUDA.  Neither backend is the required AEC runtime.

Operators implemented:
    Flatten, Gemm, Relu, Conv, Add, GlobalAveragePool, Gather,
    LayerNormalization, MatMul, Constant, Split, Reshape, Transpose,
    Div, Softmax, Erf, Mul
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Unit tests import operator functions directly, so NumPy is the conservative
# module default.  ``c35.executor.load_and_infer`` must call
# ``configure_backend`` before constructing an executor; the deployment CLI
# defaults that call to CuPy and never falls back silently.
xp = np
_BACKEND_NAME = "numpy"


def configure_backend(name: str) -> Any:
    """Select the numerical array module and return it.

    CuPy import/device failures are intentionally propagated.  A server run
    requested with ``cupy`` must not quietly become a CPU run.
    """
    global xp, _BACKEND_NAME
    normalized = name.strip().lower()
    if normalized == "numpy":
        xp = np
    elif normalized == "cupy":
        try:
            import cupy as cp
        except ImportError as exc:
            raise RuntimeError(
                "CuPy backend requested but cupy is not installed; install the "
                "target environment dependency or use --backend numpy only for "
                "an explicitly disclosed development run"
            ) from exc
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("CuPy backend requested but no CUDA device is visible")
        xp = cp
    else:
        raise ValueError(f"Unsupported backend '{name}'; expected cupy or numpy")
    _BACKEND_NAME = normalized
    return xp


def backend_name() -> str:
    return _BACKEND_NAME


def to_device(value: Any) -> Any:
    return xp.asarray(value)


def to_host(value: Any) -> np.ndarray:
    if _BACKEND_NAME == "cupy":
        return xp.asnumpy(value)
    return np.asarray(value)


def array_module() -> Any:
    return xp


def ascontiguousarray(value: Any, dtype: Any = None) -> Any:
    return xp.ascontiguousarray(xp.asarray(value, dtype=dtype))


def synchronize() -> None:
    if _BACKEND_NAME == "cupy":
        xp.cuda.get_current_stream().synchronize()


def runtime_evidence() -> Dict[str, Any]:
    """Return structured evidence from the active numerical backend.

    CuPy's default pool retains allocations, so ``total_bytes`` is a stable
    process-local high-water proxy even when MIG hides the process from
    ``nvidia-smi --query-compute-apps``.
    """
    evidence: Dict[str, Any] = {"backend": _BACKEND_NAME}
    if _BACKEND_NAME != "cupy":
        return evidence
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
    return evidence

# ── Erf implementation ──────────────────────────────────────────────
# Rational Chebyshev approximation for the error function.
# Accurate to ~1e-7, no scipy dependency required.

_ERF_A = xp.float32(0.278393)
_ERF_B = xp.float32(0.230389)
_ERF_C = xp.float32(0.000972)
_ERF_D = xp.float32(0.078108)


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

    coeffs = xp.float32([0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429])
    poly = coeffs[0]*t + coeffs[1]*t2 + coeffs[2]*t3 + coeffs[3]*t4 + coeffs[4]*t5

    return sign * (1.0 - poly * xp.exp(-x * x))


# ── Operator implementations ────────────────────────────────────────


def op_flatten(inputs: List[xp.ndarray], attrs: Dict[str, Any]) -> xp.ndarray:
    """Flatten: flattens input into 2D, preserving axis 0 (batch).

    ONNX: axis (default 1) — flatten from axis to end.
    """
    x = inputs[0]
    axis = attrs.get("axis", 1)
    # Adjust for negative axis
    if axis < 0:
        axis = len(x.shape) + axis
    new_shape = x.shape[:axis] + (-1,)
    return x.reshape(new_shape).astype(xp.float32, copy=False)


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
        pads: [padH_begin, padW_begin, padH_end, padW_end] or [padH, padW]
        strides: [sH, sW] (default [1, 1])
        dilations: [dH, dW] (default [1, 1])
        group: int (default 1)

    Keeps computation in float32 for memory efficiency and speed.
    Uses xp.tensordot for BLAS-optimized contraction.
    """
    x = xp.asarray(inputs[0], dtype=xp.float32)
    w = xp.asarray(inputs[1], dtype=xp.float32)
    b = xp.asarray(inputs[2], dtype=xp.float32) if len(inputs) >= 3 and inputs[2] is not None else None

    kernel_shape = attrs.get("kernel_shape", [w.shape[-2], w.shape[-1]])
    kH = int(kernel_shape[0])
    kW = int(kernel_shape[-1]) if len(kernel_shape) > 1 else int(kernel_shape[0])

    pads = list(attrs.get("pads", [0, 0, 0, 0]))
    if len(pads) == 2:
        pads = [pads[0], pads[1], pads[0], pads[1]]
    pHT, pWL, pHB, pWR = [int(p) for p in pads]

    strides = list(attrs.get("strides", [1, 1]))
    sH = int(strides[0])
    sW = int(strides[-1]) if len(strides) > 1 else int(strides[0])

    dilations = list(attrs.get("dilations", [1, 1]))
    dH = int(dilations[0])
    dW = int(dilations[-1]) if len(dilations) > 1 else int(dilations[0])

    group = int(attrs.get("group", 1))

    N, C, H, W_in = x.shape
    O = w.shape[0]

    # Output spatial dims
    H_out = (H + pHT + pHB - dH * (kH - 1) - 1) // sH + 1
    W_out = (W_in + pWL + pWR - dW * (kW - 1) - 1) // sW + 1

    # Pad input
    x_padded = xp.pad(x, ((0, 0), (0, 0), (pHT, pHB), (pWL, pWR)),
                      mode='constant', constant_values=0)

    # Use sliding_window_view for clean im2col (numpy >= 1.20)
    # Extract spatial windows: (N, C, H_out, W_out, kH, kW)
    try:
        patches = xp.lib.stride_tricks.sliding_window_view(
            x_padded, (kH, kW), axis=(2, 3)
        )[:, :, ::sH, ::sW, :, :]

        # Apply dilation by striding within the window
        if dH > 1 or dW > 1:
            patches = patches[:, :, :, :, ::dH, ::dW]
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

    # Use xp.take which preserves indices shape in output
    result = xp.take(x, indices, axis=axis)
    return result.astype(xp.float32, copy=False)


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

    # Normalization axes: from axis to end
    norm_axes = tuple(range(axis, len(x.shape)))

    mean = x.mean(axis=norm_axes, keepdims=True)
    # Use the two-pass formula for numerical stability
    x_centered = x - mean
    var = xp.mean(x_centered * x_centered, axis=norm_axes, keepdims=True)

    normalized = x_centered / xp.sqrt(var + epsilon)

    # Apply scale and bias
    result = normalized * scale
    if bias is not None:
        result += bias

    return result.astype(xp.float32, copy=False)


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

    split_attr = attrs.get("split", None)
    if split_attr is not None:
        split_sizes = [int(s) for s in split_attr]
    else:
        # Equal split
        num_outputs = attrs.get("_num_outputs", 2)
        size_per = x.shape[axis] // num_outputs
        split_sizes = [size_per] * num_outputs

    # CuPy deliberately requires an ndarray here, while NumPy also accepts a
    # Python list.  Materialize backend-native split points for both paths.
    indices = xp.asarray(split_sizes, dtype=xp.int64).cumsum()[:-1]
    results = xp.split(x, indices, axis=axis)
    return [r.astype(xp.float32, copy=False) for r in results]


def op_reshape(inputs: List[xp.ndarray], attrs: Dict[str, Any]) -> xp.ndarray:
    """Reshape: reshape with special value handling (0 = copy from input, -1 = infer).

    ONNX opset 17:
        Inputs: data, shape (int64 tensor)
        allowzero: if 1, 0 means set dim to 0 (not copy)
    """
    x = inputs[0]
    # Shape metadata is tiny and drives Python control flow.  Copying it to the
    # host avoids scalar-by-scalar CUDA synchronization and gives ``reshape`` a
    # normal tuple of Python integers.
    target_shape_raw = to_host(inputs[1]).astype(np.int64, copy=False).tolist()

    allowzero = int(attrs.get("allowzero", 0))

    target_shape: List[int] = []
    for i, d in enumerate(target_shape_raw):
        if d == 0:
            if allowzero:
                target_shape.append(0)
            else:
                target_shape.append(x.shape[i] if i < len(x.shape) else 0)
        elif d == -1:
            target_shape.append(-1)
        else:
            target_shape.append(int(d))

    result = x.reshape(target_shape)
    return result.astype(xp.float32, copy=False)


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
    result = xp.transpose(x, perm)
    return result.astype(xp.float32, copy=False)


def op_softmax(inputs: List[xp.ndarray], attrs: Dict[str, Any]) -> xp.ndarray:
    """Softmax: numerically stable softmax along axis.

    ONNX opset 13+: axis (default -1)
    """
    x = xp.asarray(inputs[0], dtype=xp.float32)
    axis = int(attrs.get("axis", -1))
    if axis < 0:
        axis = len(x.shape) + axis

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
    "Transpose": op_transpose,
}


# ── Fused operator implementations ─────────────────────────────────


def op_fused_matmul_bias(inputs: List[xp.ndarray], attrs: Dict[str, Any]) -> xp.ndarray:
    """FusedMatMulBias: MatMul followed by bias Add.

    Lowered to: matmul(A, B) + bias
    """
    if len(inputs) < 2:
        raise ValueError("FusedMatMulBias requires at least A, B inputs")
    a, b = xp.asarray(inputs[0], dtype=xp.float32), xp.asarray(inputs[1], dtype=xp.float32)
    result = xp.dot(a, b)
    if len(inputs) >= 3 and inputs[2] is not None:
        result += xp.asarray(inputs[2], dtype=xp.float32)
    return result.astype(xp.float32, copy=False)


def op_fused_conv_batchnorm(inputs: List[xp.ndarray], attrs: Dict[str, Any]) -> xp.ndarray:
    """Reference execution for fused Conv followed by inference BatchNorm."""
    if len(inputs) < 2:
        raise ValueError("FusedConv2dBatchNorm requires at least X, W inputs")
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


def op_fused_ew_chain(inputs: List[xp.ndarray], attrs: Dict[str, Any]) -> xp.ndarray:
    """FusedEWChain: chain of elementwise ops fused into one kernel.

    ``attrs['_ops']`` contains the ordered list of (op_type, op_attrs) pairs.
    The first op consumes the primary input; subsequent ops may reference
    additional inputs from the fused node's input list.
    """
    ops = attrs.get("_ops", [])
    if not ops:
        # Fallback: just return the first input
        return xp.asarray(inputs[0], dtype=xp.float32)

    # Build a local value context from the fused node's inputs
    local_values = {f"_inp_{i}": xp.asarray(inp, dtype=xp.float32)
                    for i, inp in enumerate(inputs) if inp is not None}

    # The first op consumes _inp_0
    current = local_values.get("_inp_0")
    if current is None and inputs:
        current = xp.asarray(inputs[0], dtype=xp.float32)

    for op_entry in ops:
        op_type = op_entry["op"]
        op_attrs = op_entry.get("attrs", {})
        impl = _OP_DISPATCH.get(op_type)
        if impl is None:
            raise ValueError(f"Unknown op in FusedEWChain: {op_type}")
        # Resolve inputs from local context or global inputs list
        op_inputs = []
        for ref in op_entry.get("inputs", []):
            if isinstance(ref, int):
                op_inputs.append(xp.asarray(inputs[ref], dtype=xp.float32))
            elif isinstance(ref, str) and ref in local_values:
                op_inputs.append(local_values[ref])
            else:
                op_inputs.append(current)
        if not op_inputs:
            op_inputs = [current]
        current = impl(op_inputs, op_attrs)
        # Store result as intermediate
        out_name = op_entry.get("output", f"_out_{len(local_values)}")
        local_values[out_name] = current

    return xp.asarray(current, dtype=xp.float32)


def op_fused_softmax_dropout(inputs: List[xp.ndarray], attrs: Dict[str, Any]) -> xp.ndarray:
    """FusedSoftmaxDropout: Softmax with inference-mode Dropout (identity).

    In inference mode, Dropout is a no-op, so this is just Softmax.
    """
    return op_softmax(inputs, attrs)


def op_fused_residual_norm(inputs: List[xp.ndarray], attrs: Dict[str, Any]) -> xp.ndarray:
    """FusedResidualNorm: Add (residual) followed by LayerNorm.

    Lowered to: LayerNorm(Add(x, residual), weight, bias)
    """
    if len(inputs) < 2:
        raise ValueError("FusedResidualNorm requires at least x, residual inputs")
    x = xp.asarray(inputs[0], dtype=xp.float32)
    residual = xp.asarray(inputs[1], dtype=xp.float32)
    added = x + residual
    # Remaining inputs are weight and bias for LayerNorm
    ln_inputs = [added]
    if len(inputs) > 2:
        ln_inputs.append(xp.asarray(inputs[2], dtype=xp.float32))
    if len(inputs) > 3:
        ln_inputs.append(xp.asarray(inputs[3], dtype=xp.float32))
    return op_layer_normalization(ln_inputs, attrs)


# Copy of dispatch for FusedEWChain internal use (avoid circular import)
_OP_DISPATCH = {
    "Add": op_add,
    "Sub": op_sub,
    "Mul": op_mul,
    "Div": op_div,
    "Relu": op_relu,
    "Erf": op_erf,
    "Exp": op_exp,
    "Sqrt": op_sqrt,
}


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
    "FusedConv2dBatchNorm": op_fused_conv_batchnorm,
    "FusedEWChain": op_fused_ew_chain,
    "FusedSoftmaxDropout": op_fused_softmax_dropout,
    "FusedResidualNorm": op_fused_residual_norm,
}
OP_DISPATCH.update(_FUSED_DISPATCH)
