"""Operator decompositions — one function per ONNX op type.

Each function returns a non-empty list of KernelSpecRef.
"""

from __future__ import annotations

from typing import List

from c3common.ir.graph import Graph, Node
from c32.kernel_spec import KernelSpecRef, PrecisionProfile


def _inter_name(node_id: str, idx: int) -> str:
    return f"__c3_inter_{node_id}_{idx}__"


def _kernel_name(base: str, precision: PrecisionProfile) -> str:
    """Build a canonical kernel name from base + precision suffix.

    Maps ``fp32/fp16/bf16/fp8/fp4`` → ``f32/f16/bf16/f8/f4`` so that
    emitted names match what ``HardwareCapability`` advertises.
    """
    dt = precision.compute_dtype
    # Normalize precision suffix: fp32→f32, fp16→f16, fp8→f8, fp4→f4
    suffix = dt.replace("fp", "f") if dt.startswith("fp") else dt
    return f"{base}_{suffix}"


# ── Elementwise ops ────────────────────────────


def decompose_Add(node: Node, graph: Graph, precision: PrecisionProfile) -> List[KernelSpecRef]:
    return [KernelSpecRef(kernel_name=_kernel_name("add", precision),
                          inputs=list(node.inputs), outputs=list(node.outputs))]


def decompose_Mul(node: Node, graph: Graph, precision: PrecisionProfile) -> List[KernelSpecRef]:
    return [KernelSpecRef(kernel_name=_kernel_name("mul", precision),
                          inputs=list(node.inputs), outputs=list(node.outputs))]


def decompose_Div(node: Node, graph: Graph, precision: PrecisionProfile) -> List[KernelSpecRef]:
    return [KernelSpecRef(kernel_name=_kernel_name("div", precision),
                          inputs=list(node.inputs), outputs=list(node.outputs))]


def decompose_Relu(node: Node, graph: Graph, precision: PrecisionProfile) -> List[KernelSpecRef]:
    return [KernelSpecRef(kernel_name=_kernel_name("relu", precision),
                          inputs=list(node.inputs), outputs=list(node.outputs))]


def decompose_Erf(node: Node, graph: Graph, precision: PrecisionProfile) -> List[KernelSpecRef]:
    return [KernelSpecRef(kernel_name=_kernel_name("erf", precision),
                          inputs=list(node.inputs), outputs=list(node.outputs))]


# ── Layout / view / data-movement ops ──────────


def decompose_Flatten(node: Node, graph: Graph, precision: PrecisionProfile) -> List[KernelSpecRef]:
    axis = int(node.attributes.get("axis", 1))
    return [KernelSpecRef(kernel_name="flatten", inputs=list(node.inputs), outputs=list(node.outputs),
                          operator_params={"axis": axis})]


def decompose_Reshape(node: Node, graph: Graph, precision: PrecisionProfile) -> List[KernelSpecRef]:
    allowzero = int(node.attributes.get("allowzero", 0))
    return [KernelSpecRef(kernel_name="reshape", inputs=list(node.inputs), outputs=list(node.outputs),
                          operator_params={"allowzero": allowzero})]


def decompose_Transpose(node: Node, graph: Graph, precision: PrecisionProfile) -> List[KernelSpecRef]:
    # ``perm`` omitted means reverse the dimensions.  Preserve omission as
    # None instead of an empty permutation, which has different semantics.
    perm_attr = node.attributes.get("perm")
    perm = None if perm_attr is None else list(perm_attr)
    return [KernelSpecRef(kernel_name="transpose", inputs=list(node.inputs), outputs=list(node.outputs),
                          operator_params={"perm": perm})]


def decompose_Split(node: Node, graph: Graph, precision: PrecisionProfile) -> List[KernelSpecRef]:
    axis = int(node.attributes.get("axis", 0))
    split_sizes = list(node.attributes.get("split", [])) if "split" in node.attributes else []
    return [KernelSpecRef(kernel_name="split", inputs=list(node.inputs), outputs=list(node.outputs),
                          operator_params={"axis": axis, "split": split_sizes,
                                           "_num_outputs": len(node.outputs)})]


def decompose_Gather(node: Node, graph: Graph, precision: PrecisionProfile) -> List[KernelSpecRef]:
    axis = int(node.attributes.get("axis", 0))
    return [KernelSpecRef(kernel_name="gather", inputs=list(node.inputs), outputs=list(node.outputs),
                          operator_params={"axis": axis})]


def decompose_Constant(node: Node, graph: Graph, precision: PrecisionProfile) -> List[KernelSpecRef]:
    return [KernelSpecRef(kernel_name="constant", inputs=[], outputs=list(node.outputs),
                          operator_params=dict(node.attributes))]


# ── MatMul / Gemm ─────────────────────────────


def decompose_MatMul(node: Node, graph: Graph, precision: PrecisionProfile) -> List[KernelSpecRef]:
    return [KernelSpecRef(kernel_name=_kernel_name("matmul", precision),
                          inputs=list(node.inputs), outputs=list(node.outputs),
                          operator_params={"lowering_kind": "matmul"})]


def decompose_Linear(node: Node, graph: Graph, precision: PrecisionProfile) -> List[KernelSpecRef]:
    """Lower framework-style Linear using MatMul or Gemm semantics."""
    if len(node.inputs) >= 3 and node.inputs[2]:
        return decompose_Gemm(node, graph, precision)
    return decompose_MatMul(node, graph, precision)


def decompose_Gemm(node: Node, graph: Graph, precision: PrecisionProfile) -> List[KernelSpecRef]:
    """Gemm = alpha * A' * B' + beta * C.

    Decomposes into matmul followed by optional bias add.
    When bias is absent the matmul output IS the node output.
    When bias is present the matmul output is a named intermediate
    (``__c3_inter_*`` convention) consumed by the add_bias kernel.
    """
    nid = node.id
    has_bias = len(node.inputs) >= 3 and node.inputs[2]
    op_params = {
        "alpha": float(node.attributes.get("alpha", 1.0)),
        "beta": float(node.attributes.get("beta", 1.0)),
        "transA": int(node.attributes.get("transA", 0)),
        "transB": int(node.attributes.get("transB", 0)),
        "lowering_kind": "gemm",
    }

    if has_bias:
        matmul_out = _inter_name(nid, 0)
        return [
            KernelSpecRef(kernel_name=_kernel_name("matmul", precision),
                          inputs=list(node.inputs[:2]),
                          outputs=[matmul_out],
                          operator_params=dict(op_params)),
            KernelSpecRef(kernel_name=_kernel_name("add_bias", precision),
                          inputs=[matmul_out, node.inputs[2]],
                          outputs=list(node.outputs),
                          operator_params={"beta": op_params["beta"],
                                           "bias_axis": -1}),
        ]
    else:
        return [
            KernelSpecRef(kernel_name=_kernel_name("matmul", precision),
                          inputs=list(node.inputs[:2]),
                          outputs=list(node.outputs),
                          operator_params=dict(op_params)),
        ]


# ── Softmax ───────────────────────────────────


def decompose_Softmax(node: Node, graph: Graph, precision: PrecisionProfile) -> List[KernelSpecRef]:
    """Softmax(x) = exp(x - max) / sum(exp(x - max)).

    Preserves axis, stable-reduction semantics.
    All intermediate tensors use the ``__c3_inter_*`` convention.
    """
    nid, inp, out = node.id, node.inputs[0], node.outputs[0]
    axis = int(node.attributes.get("axis", -1))
    return [
        KernelSpecRef(kernel_name="reduce_max", inputs=[inp], outputs=[_inter_name(nid, 0)],
                      operator_params={"axis": axis, "keepdims": True}),
        KernelSpecRef(kernel_name="sub", inputs=[inp, _inter_name(nid, 0)], outputs=[_inter_name(nid, 1)]),
        KernelSpecRef(kernel_name="exp", inputs=[_inter_name(nid, 1)], outputs=[_inter_name(nid, 2)]),
        KernelSpecRef(kernel_name="reduce_sum", inputs=[_inter_name(nid, 2)], outputs=[_inter_name(nid, 3)],
                      operator_params={"axis": axis, "keepdims": True}),
        KernelSpecRef(kernel_name="div", inputs=[_inter_name(nid, 2), _inter_name(nid, 3)], outputs=[out]),
    ]


# ── LayerNormalization ────────────────────────


def decompose_LayerNormalization(node: Node, graph: Graph, precision: PrecisionProfile) -> List[KernelSpecRef]:
    """LayerNorm: (x - mean) / sqrt(var + eps) * weight + bias.

    All intermediate tensors use the ``__c3_inter_*`` convention.
    """
    nid, inp, out = node.id, node.inputs[0], node.outputs[0]
    weight = node.inputs[1] if len(node.inputs) > 1 else None
    bias = node.inputs[2] if len(node.inputs) > 2 else None
    axis = int(node.attributes.get("axis", -1))
    epsilon = float(node.attributes.get("epsilon", 1e-5))
    mean_out = node.outputs[1] if len(node.outputs) > 1 else _inter_name(nid, 0)
    centered = _inter_name(nid, 1)
    variance = _inter_name(nid, 3)
    denominator = _inter_name(nid, 5)
    normalized = _inter_name(nid, 6)

    refs = [
        KernelSpecRef(kernel_name="reduce_mean", inputs=[inp], outputs=[mean_out],
                      operator_params={"axis": axis, "axes_from": axis,
                                       "keepdims": True}),
        KernelSpecRef(kernel_name="sub", inputs=[inp, mean_out], outputs=[centered]),
        KernelSpecRef(kernel_name="mul", inputs=[centered, centered], outputs=[_inter_name(nid, 2)]),
        KernelSpecRef(kernel_name="reduce_mean", inputs=[_inter_name(nid, 2)], outputs=[variance],
                      operator_params={"axis": axis, "axes_from": axis,
                                       "keepdims": True}),
        KernelSpecRef(kernel_name="add", inputs=[variance], outputs=[_inter_name(nid, 4)],
                      operator_params={"epsilon": epsilon}),
        KernelSpecRef(kernel_name="sqrt", inputs=[_inter_name(nid, 4)], outputs=[denominator]),
    ]

    if len(node.outputs) > 2:
        inverse_std = node.outputs[2]
        refs.extend([
            KernelSpecRef(kernel_name="reciprocal", inputs=[denominator],
                          outputs=[inverse_std]),
            KernelSpecRef(kernel_name="mul", inputs=[centered, inverse_std],
                          outputs=[normalized]),
        ])
    else:
        refs.append(KernelSpecRef(kernel_name="div", inputs=[centered, denominator],
                                  outputs=[normalized]))

    if weight:
        refs.append(KernelSpecRef(kernel_name="mul",
                                  inputs=[normalized, weight],
                                  outputs=[_inter_name(nid, 7)] if bias else [out]))
    if bias:
        refs.append(KernelSpecRef(kernel_name="add",
                                  inputs=[_inter_name(nid, 7) if weight else normalized, bias],
                                  outputs=[out]))
    if not weight and not bias:
        refs[-1].outputs = [out]
    return refs


# ── Conv ──────────────────────────────────────


def _conv_kernel_shape(node: Node, graph: Graph) -> List[int]:
    ks = list(node.attributes.get("kernel_shape", []))
    if ks:
        return [int(value) for value in ks]
    if len(node.inputs) > 1:
        weight = graph.get_tensor(node.inputs[1])
        if weight is not None and len(weight.shape) >= 3:
            try:
                return [int(value) for value in weight.shape[2:]]
            except (TypeError, ValueError):
                pass
    return []


def _conv_has_3x3(node: Node, graph: Graph) -> bool:
    ks = _conv_kernel_shape(node, graph)
    return list(ks) == [3, 3] or list(ks) == [3]


def _conv_is_winograd_eligible(node: Node, graph: Graph) -> bool:
    strides = list(node.attributes.get("strides", [1, 1]))
    dilations = list(node.attributes.get("dilations", [1, 1]))
    group = int(node.attributes.get("group", 1))
    return (
        _conv_has_3x3(node, graph)
        and strides in ([1], [1, 1])
        and dilations in ([1], [1, 1])
        and group == 1
    )


def _conv_output_spatial(node: Node, graph: Graph, inp_tensor: Any) -> List[int]:
    """Compute output spatial dims [H_out, W_out] for a Conv node."""
    try:
        if inp_tensor is not None and inp_tensor.shape and len(inp_tensor.shape) == 4:
            H = int(inp_tensor.shape[-2])
            W = int(inp_tensor.shape[-1])
        else:
            return []
    except (ValueError, TypeError):
        return []

    kH, kW = _conv_kernel_shape(node, graph)[:2]
    kH, kW = int(kH), int(kW)

    strides = list(node.attributes.get("strides", [1, 1]))
    sH, sW = int(strides[0]), int(strides[-1]) if len(strides) > 1 else int(strides[0])

    dilations = list(node.attributes.get("dilations", [1, 1]))
    dH, dW = int(dilations[0]), int(dilations[-1]) if len(dilations) > 1 else int(dilations[0])

    pads = list(node.attributes.get("pads", [0, 0, 0, 0]))
    pHT, pWL, pHB, pWR = [int(p) for p in pads]

    H_out = (H + pHT + pHB - dH * (kH - 1) - 1) // sH + 1
    W_out = (W + pWL + pWR - dW * (kW - 1) - 1) // sW + 1

    return [H_out, W_out] if H_out > 0 and W_out > 0 else []


def decompose_Conv(node: Node, graph: Graph, precision: PrecisionProfile,
                   use_winograd: bool = True) -> List[KernelSpecRef]:
    nid = node.id
    inp, weight = node.inputs[0], node.inputs[1]
    has_bias = len(node.inputs) >= 3 and node.inputs[2]
    is_winograd_eligible = _conv_is_winograd_eligible(node, graph)
    op_params = {
        "kernel_shape": _conv_kernel_shape(node, graph),
        "auto_pad": node.attributes.get("auto_pad", "NOTSET"),
        "pads": list(node.attributes.get("pads", [0, 0, 0, 0])),
        "strides": list(node.attributes.get("strides", [1, 1])),
        "dilations": list(node.attributes.get("dilations", [1, 1])),
        "group": int(node.attributes.get("group", 1)),
    }

    if is_winograd_eligible and use_winograd:
        winograd_out = _inter_name(nid, 0) if has_bias else list(node.outputs)[0]
        refs = [KernelSpecRef(kernel_name=_kernel_name("winograd_forward", precision),
                              inputs=[inp, weight],
                              outputs=[winograd_out] if has_bias else list(node.outputs),
                              operator_params=dict(op_params))]
        if has_bias:
            refs.append(KernelSpecRef(kernel_name=_kernel_name("add_bias", precision),
                                      inputs=[winograd_out, node.inputs[2]],
                                      outputs=list(node.outputs),
                                      operator_params={"bias_axis": 1}))
        return refs
    else:
        im2col_out = _inter_name(nid, 0)
        # The contraction is always an internal NHWO matrix.  A distinct
        # reshape step is the sole producer of the public NCHW Conv output,
        # including bias-free Conv nodes.
        matmul_out = _inter_name(nid, 1)
        # Capture input spatial shape for downstream reshape.
        inp_tensor = graph.get_tensor(inp)
        input_spatial = []
        if inp_tensor is not None and inp_tensor.shape and len(inp_tensor.shape) == 4:
            input_spatial = list(inp_tensor.shape[-2:])
        # Compute output spatial dims for the reshape kernel.
        out_spatial = _conv_output_spatial(node, graph, inp_tensor)
        refs = [
            KernelSpecRef(kernel_name=_kernel_name("im2col", precision),
                          inputs=[inp], outputs=[im2col_out],
                          operator_params={**op_params,
                                           "lowering_kind": "conv_im2col"}),
            KernelSpecRef(kernel_name=_kernel_name("matmul", precision),
                          inputs=[im2col_out, weight],
                          outputs=[matmul_out],
                          operator_params={**op_params,
                                           "lowering_kind": "conv_contract",
                                           "_input_spatial": input_spatial}),
        ]
        if has_bias:
            reshape_out = _inter_name(nid, 2)
            refs.append(KernelSpecRef(
                kernel_name=_kernel_name("conv_reshape", precision),
                inputs=[matmul_out],
                outputs=[reshape_out],
                operator_params={**op_params,
                                 "_input_spatial": input_spatial,
                                 "_output_spatial": out_spatial},
            ))
            refs.append(KernelSpecRef(kernel_name=_kernel_name("add_bias", precision),
                                      inputs=[reshape_out, node.inputs[2]],
                                      outputs=list(node.outputs),
                                      operator_params={"bias_axis": 1}))
        else:
            refs.append(KernelSpecRef(
                kernel_name=_kernel_name("conv_reshape", precision),
                inputs=[matmul_out],
                outputs=list(node.outputs),
                operator_params={**op_params,
                                 "_input_spatial": input_spatial,
                                 "_output_spatial": out_spatial},
            ))
        return refs


# ── GlobalAveragePool ─────────────────────────


def decompose_GlobalAveragePool(node: Node, graph: Graph, precision: PrecisionProfile) -> List[KernelSpecRef]:
    """GlobalAveragePool: reduce HW to 1×1, preserving [N,C,H,W] layout.

    Does NOT introduce an unconditional squeeze — the output shape
    remains ``[N, C, 1, 1]`` in compliance with ONNX semantics.
    """
    return [
        KernelSpecRef(kernel_name=_kernel_name("reduce_mean", precision),
                      inputs=list(node.inputs), outputs=list(node.outputs),
                      operator_params={"axes": [-2, -1], "keepdims": True}),
    ]


# ── C3.3 executable fusion lowerings ──────────


def decompose_FusedMatMulBias(
    node: Node, graph: Graph, precision: PrecisionProfile,
) -> List[KernelSpecRef]:
    """One generated MatMul+bias epilogue kernel."""
    return [KernelSpecRef(
        kernel_name="fused_matmul_bias_f32",
        inputs=list(node.inputs),
        outputs=list(node.outputs),
        operator_params=dict(node.attributes),
    )]


def decompose_FusedConv2dBatchNorm(
    node: Node, graph: Graph, precision: PrecisionProfile,
) -> List[KernelSpecRef]:
    """Use one Conv lowering after runtime initializer folding.

    Before the executor materializes folded parameters, retain an explicit BN
    step so graph-only launch accounting never treats an unfurled Conv+BN as a
    single physical kernel.
    """
    if node.attributes.get("bn_folded"):
        return decompose_Conv(node, graph, precision, use_winograd=True)
    offset = int(node.attributes.get("bn_parameter_offset", len(node.inputs)))
    conv_out = _inter_name(node.id, 0)
    conv_node = Node(
        id=f"{node.id}_conv",
        name=node.name,
        op_type="Conv",
        inputs=list(node.inputs[:offset]),
        outputs=[conv_out],
        attributes=dict(node.attributes),
        domain=node.domain,
    )
    refs = decompose_Conv(conv_node, graph, precision, use_winograd=True)
    refs.append(KernelSpecRef(
        kernel_name="batchnorm_f32",
        inputs=[conv_out] + list(node.inputs[offset:offset + 4]),
        outputs=list(node.outputs),
        operator_params={"epsilon": node.attributes.get("bn_epsilon", 1e-5)},
    ))
    return refs


def decompose_FusedEWChain(
    node: Node, graph: Graph, precision: PrecisionProfile,
) -> List[KernelSpecRef]:
    return [KernelSpecRef(
        kernel_name="fused_ew_f32",
        inputs=list(node.inputs),
        outputs=list(node.outputs),
        operator_params=dict(node.attributes),
    )]


def decompose_FusedSoftmaxDropout(
    node: Node, graph: Graph, precision: PrecisionProfile,
) -> List[KernelSpecRef]:
    return [KernelSpecRef(
        kernel_name="fused_softmax_f32",
        inputs=list(node.inputs),
        outputs=list(node.outputs),
        operator_params=dict(node.attributes),
    )]


def decompose_FusedResidualNorm(
    node: Node, graph: Graph, precision: PrecisionProfile,
) -> List[KernelSpecRef]:
    return [KernelSpecRef(
        kernel_name="fused_residual_norm_f32",
        inputs=list(node.inputs),
        outputs=list(node.outputs),
        operator_params=dict(node.attributes),
    )]


def _single_fused_kernel(node: Node, kernel_name: str) -> List[KernelSpecRef]:
    return [KernelSpecRef(
        kernel_name=kernel_name,
        inputs=list(node.inputs),
        outputs=list(node.outputs),
        operator_params=dict(node.attributes),
    )]


def decompose_FusedGemmEpilogue(
    node: Node, graph: Graph, precision: PrecisionProfile,
) -> List[KernelSpecRef]:
    return _single_fused_kernel(node, "fused_gemm_epilogue_f32")


def decompose_FusedConvActivation(
    node: Node, graph: Graph, precision: PrecisionProfile,
) -> List[KernelSpecRef]:
    """Lower the BLAS Conv fallback followed by one ReLU epilogue.

    The deployment runtime intentionally does not use the former unqualified
    direct-convolution kernel.  Keep the metadata honest by exposing im2col,
    contraction, optional bias, and the generated epilogue as separate
    launches until a faster single-launch H200 kernel is qualified.
    """
    conv_out = _inter_name(node.id, 2)
    conv_node = Node(
        id=f"{node.id}_blas_conv",
        name=node.name,
        op_type="Conv",
        inputs=list(node.inputs),
        outputs=[conv_out],
        attributes=dict(node.attributes),
        domain=node.domain,
    )
    refs = decompose_Conv(conv_node, graph, precision, use_winograd=False)
    refs.append(KernelSpecRef(
        kernel_name="relu_f32",
        inputs=[conv_out],
        outputs=list(node.outputs),
        operator_params={"lowering_kind": "blas_conv_relu_epilogue"},
    ))
    return refs


def decompose_FusedConvResidualActivation(
    node: Node, graph: Graph, precision: PrecisionProfile,
) -> List[KernelSpecRef]:
    """Lower BLAS Conv plus one generated residual-Add/ReLU epilogue."""
    residual_index = int(node.attributes.get(
        "_residual_input_index", len(node.inputs)
    ))
    if not 0 <= residual_index < len(node.inputs):
        raise ValueError("Fused Conv residual input index is invalid")
    conv_out = _inter_name(node.id, 2)
    conv_node = Node(
        id=f"{node.id}_blas_conv",
        name=node.name,
        op_type="Conv",
        inputs=list(node.inputs[:residual_index]),
        outputs=[conv_out],
        attributes=dict(node.attributes),
        domain=node.domain,
    )
    refs = decompose_Conv(conv_node, graph, precision, use_winograd=False)
    refs.append(KernelSpecRef(
        kernel_name="fused_residual_relu_f32",
        inputs=[conv_out, node.inputs[residual_index]],
        outputs=list(node.outputs),
        operator_params={"lowering_kind": "blas_conv_residual_relu_epilogue"},
    ))
    return refs


def decompose_FusedAttentionScores(
    node: Node, graph: Graph, precision: PrecisionProfile,
) -> List[KernelSpecRef]:
    return _single_fused_kernel(node, "fused_attention_scores_f32")


def decompose_FusedLayerNormalization(
    node: Node, graph: Graph, precision: PrecisionProfile,
) -> List[KernelSpecRef]:
    return _single_fused_kernel(node, "fused_layer_norm_f32")


def decompose_FusedTransposeReshape(
    node: Node, graph: Graph, precision: PrecisionProfile,
) -> List[KernelSpecRef]:
    return _single_fused_kernel(node, "fused_transpose_reshape_f32")


# ── Dispatch table ────────────────────────────

DECOMPOSE_DISPATCH = {
    "Add": decompose_Add,
    "Constant": decompose_Constant,
    "Conv": decompose_Conv,
    "Conv2d": decompose_Conv,
    "Div": decompose_Div,
    "Erf": decompose_Erf,
    "Flatten": decompose_Flatten,
    "FusedConv2dBatchNorm": decompose_FusedConv2dBatchNorm,
    "FusedConvActivation": decompose_FusedConvActivation,
    "FusedConvResidualActivation": decompose_FusedConvResidualActivation,
    "FusedAttentionScores": decompose_FusedAttentionScores,
    "FusedEWChain": decompose_FusedEWChain,
    "FusedGemmEpilogue": decompose_FusedGemmEpilogue,
    "FusedLayerNormalization": decompose_FusedLayerNormalization,
    "FusedTransposeReshape": decompose_FusedTransposeReshape,
    "FusedMatMulBias": decompose_FusedMatMulBias,
    "FusedResidualNorm": decompose_FusedResidualNorm,
    "FusedSoftmaxDropout": decompose_FusedSoftmaxDropout,
    "Gather": decompose_Gather,
    "Gemm": decompose_Gemm,
    "GlobalAveragePool": decompose_GlobalAveragePool,
    "LayerNormalization": decompose_LayerNormalization,
    "LayerNorm": decompose_LayerNormalization,
    "Linear": decompose_Linear,
    "MatMul": decompose_MatMul,
    "Mul": decompose_Mul,
    "Relu": decompose_Relu,
    "Reshape": decompose_Reshape,
    "Softmax": decompose_Softmax,
    "Split": decompose_Split,
    "Transpose": decompose_Transpose,
}
