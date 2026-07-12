"""Operator decompositions — one function per ONNX op type.

Each function returns a non-empty list of KernelSpecRef.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from c3common.ir.graph import Graph, Node
from c32.kernel_spec import KernelSpecRef, KernelTuningParams, PrecisionProfile


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


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


def _default_tuning(
    problem_size: int,
    block_x: int = 256,
    smem_bytes: int = 0,
    max_threads: int = 1024,
    max_smem: int = 49152,
) -> KernelTuningParams:
    actual_block = min(block_x, max_threads)
    grid_x = max(1, _ceil_div(problem_size, actual_block))
    return KernelTuningParams(
        block_x=actual_block,
        grid_x=grid_x,
        smem_bytes=smem_bytes if smem_bytes <= max_smem else -1,
    )


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
    perm = list(node.attributes.get("perm", []))
    return [KernelSpecRef(kernel_name="transpose", inputs=list(node.inputs), outputs=list(node.outputs),
                          operator_params={"perm": perm})]


def decompose_Split(node: Node, graph: Graph, precision: PrecisionProfile) -> List[KernelSpecRef]:
    axis = int(node.attributes.get("axis", 0))
    split_sizes = list(node.attributes.get("split", [])) if "split" in node.attributes else []
    return [KernelSpecRef(kernel_name="split", inputs=list(node.inputs), outputs=list(node.outputs),
                          operator_params={"axis": axis, "split": split_sizes})]


def decompose_Gather(node: Node, graph: Graph, precision: PrecisionProfile) -> List[KernelSpecRef]:
    axis = int(node.attributes.get("axis", 0))
    return [KernelSpecRef(kernel_name="gather", inputs=list(node.inputs), outputs=list(node.outputs),
                          operator_params={"axis": axis})]


def decompose_Constant(node: Node, graph: Graph, precision: PrecisionProfile) -> List[KernelSpecRef]:
    return [KernelSpecRef(kernel_name="constant", inputs=[], outputs=list(node.outputs),
                          operator_params=dict(node.attributes))]


# ── MatMul / Gemm ─────────────────────────────


def _problem_size_matmul(node: Node, graph: Graph) -> Dict[str, int]:
    inp_tensors = [graph.get_tensor(n) for n in node.inputs if n]
    mat_shapes = []
    for t in inp_tensors:
        if t and len(t.shape) >= 2:
            dims = []
            for d in t.shape:
                try:
                    dims.append(int(d))
                except (ValueError, TypeError):
                    pass
            if len(dims) >= 2:
                mat_shapes.append(dims)
    if len(mat_shapes) >= 2:
        a_shape, b_shape = mat_shapes[0], mat_shapes[1]
        trans_b = node.attributes.get("transB", 0)
        m = a_shape[-2] if len(a_shape) >= 2 else 1
        k = a_shape[-1]
        n = b_shape[-2] if trans_b else b_shape[-1]
        return {"m": m, "n": n, "k": k}
    return {"m": 1, "n": 1, "k": 1}


def decompose_MatMul(node: Node, graph: Graph, precision: PrecisionProfile) -> List[KernelSpecRef]:
    return [KernelSpecRef(kernel_name=_kernel_name("matmul", precision),
                          inputs=list(node.inputs), outputs=list(node.outputs))]


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
                          operator_params={"beta": op_params["beta"]}),
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
    op_params = {"axis": axis}
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
    op_params = {"axis": axis, "epsilon": epsilon}

    refs = [
        KernelSpecRef(kernel_name="reduce_mean", inputs=[inp], outputs=[_inter_name(nid, 0)],
                      operator_params={"axis": axis, "keepdims": True}),
        KernelSpecRef(kernel_name="sub", inputs=[inp, _inter_name(nid, 0)], outputs=[_inter_name(nid, 1)]),
        KernelSpecRef(kernel_name="mul", inputs=[_inter_name(nid, 1), _inter_name(nid, 1)], outputs=[_inter_name(nid, 2)]),
        KernelSpecRef(kernel_name="reduce_mean", inputs=[_inter_name(nid, 2)], outputs=[_inter_name(nid, 3)],
                      operator_params={"axis": axis, "keepdims": True}),
        KernelSpecRef(kernel_name="add", inputs=[_inter_name(nid, 3)], outputs=[_inter_name(nid, 4)],
                      operator_params={"epsilon": epsilon}),
        KernelSpecRef(kernel_name="sqrt", inputs=[_inter_name(nid, 4)], outputs=[_inter_name(nid, 5)]),
        KernelSpecRef(kernel_name="div", inputs=[_inter_name(nid, 1), _inter_name(nid, 5)], outputs=[_inter_name(nid, 6)]),
    ]
    if weight:
        refs.append(KernelSpecRef(kernel_name="mul",
                                  inputs=[_inter_name(nid, 6), weight],
                                  outputs=[_inter_name(nid, 7)] if bias else [out]))
    if bias:
        refs.append(KernelSpecRef(kernel_name="add",
                                  inputs=[_inter_name(nid, 7) if weight else _inter_name(nid, 6), bias],
                                  outputs=[out]))
    if not weight and not bias:
        refs[-1].outputs = [out]
    return refs


# ── Conv ──────────────────────────────────────


def _conv_has_3x3(node: Node) -> bool:
    ks = node.attributes.get("kernel_shape", [])
    return list(ks) == [3, 3] or list(ks) == [3]


def _conv_stride(node: Node) -> int:
    s = node.attributes.get("strides", [1, 1])
    return int(s[0]) if isinstance(s, (list, tuple)) else int(s)


def decompose_Conv(node: Node, graph: Graph, precision: PrecisionProfile,
                   use_winograd: bool = True) -> List[KernelSpecRef]:
    nid = node.id
    inp, weight = node.inputs[0], node.inputs[1]
    has_bias = len(node.inputs) >= 3 and node.inputs[2]
    is_3x3_s1 = _conv_has_3x3(node) and _conv_stride(node) == 1
    op_params = {
        "kernel_shape": list(node.attributes.get("kernel_shape", [])),
        "pads": list(node.attributes.get("pads", [0, 0, 0, 0])),
        "strides": list(node.attributes.get("strides", [1, 1])),
        "dilations": list(node.attributes.get("dilations", [1, 1])),
        "group": int(node.attributes.get("group", 1)),
    }

    if is_3x3_s1 and use_winograd:
        winograd_out = _inter_name(nid, 0) if has_bias else list(node.outputs)[0]
        refs = [KernelSpecRef(kernel_name=_kernel_name("winograd_forward", precision),
                              inputs=[inp, weight],
                              outputs=[winograd_out] if has_bias else list(node.outputs),
                              operator_params=dict(op_params))]
        if has_bias:
            refs.append(KernelSpecRef(kernel_name=_kernel_name("add_bias", precision),
                                      inputs=[winograd_out, node.inputs[2]],
                                      outputs=list(node.outputs)))
        return refs
    else:
        im2col_out = _inter_name(nid, 0)
        matmul_out = _inter_name(nid, 1) if has_bias else list(node.outputs)[0]
        refs = [
            KernelSpecRef(kernel_name=_kernel_name("im2col", precision),
                          inputs=[inp], outputs=[im2col_out],
                          operator_params=dict(op_params)),
            KernelSpecRef(kernel_name=_kernel_name("matmul", precision),
                          inputs=[im2col_out, weight],
                          outputs=[matmul_out] if has_bias else list(node.outputs)),
        ]
        if has_bias:
            refs.append(KernelSpecRef(kernel_name=_kernel_name("add_bias", precision),
                                      inputs=[matmul_out, node.inputs[2]],
                                      outputs=list(node.outputs)))
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


# ── Dispatch table ────────────────────────────

DECOMPOSE_DISPATCH = {
    "Add": decompose_Add,
    "Constant": decompose_Constant,
    "Conv": decompose_Conv,
    "Div": decompose_Div,
    "Erf": decompose_Erf,
    "Flatten": decompose_Flatten,
    "Gather": decompose_Gather,
    "Gemm": decompose_Gemm,
    "GlobalAveragePool": decompose_GlobalAveragePool,
    "LayerNormalization": decompose_LayerNormalization,
    "MatMul": decompose_MatMul,
    "Mul": decompose_Mul,
    "Relu": decompose_Relu,
    "Reshape": decompose_Reshape,
    "Softmax": decompose_Softmax,
    "Split": decompose_Split,
    "Transpose": decompose_Transpose,
}
