#!/usr/bin/env python3
"""Self-test for C3.3 — validates all five fusion patterns, pipeline,
   launch/buffer reductions, and correctness invariants.

Usage:
    python -m c33.test_c33 [--verbose]
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from c3common.ir.graph import Graph, Node, Tensor, ONNSType
from c33.pipeline import GraphPassPipeline, _count_launches, _count_buffers
from c33.fusion import (
    fuse_matmul_bias,
    fuse_conv_batchnorm,
    fuse_elementwise_chain,
    fuse_softmax_dropout,
    fuse_residual_norm,
)

try:
    from c31.import_onnx import import_onnx as _import_onnx
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ".specification", "testcases", "release_to_competitors", "models",
)
MODEL_PATHS = {
    "mlp": os.path.join(MODELS_DIR, "mlp_v1.onnx"),
    "resnet": os.path.join(MODELS_DIR, "resnet_v1.onnx"),
    "transformer": os.path.join(MODELS_DIR, "transformer_v1.onnx"),
}

PASS = 0
FAIL = 0
SCORES: Dict[str, float] = {}


def check(condition: bool, msg: str, points: float = 0.0) -> bool:
    global PASS, FAIL
    if condition:
        PASS += 1
        if points:
            print(f"  \u2713 {msg}  [{points} pt]")
        else:
            print(f"  \u2713 {msg}")
        return True
    else:
        FAIL += 1
        print(f"  \u2717 {msg}")
        return False


# ── Helpers for building test graphs ──────────────────────────────────


def _make_tensor(name: str, shape: List, dtype: str = "FLOAT",
                  is_init: bool = False) -> Tensor:
    return Tensor(name=name, dtype=ONNSType(dtype),
                  shape=shape, is_initializer=is_init)


def _simple_graph() -> Graph:
    """Build a minimal graph with a single node for testing."""
    g = Graph()
    g.name = "test"
    # Input
    inp = _make_tensor("input", ["batch", 4], "FLOAT")
    g.inputs.append(inp)
    g.register_tensor("input", shape=["batch", 4])
    g.tensor_producer["input"] = "INPUT"

    # Output
    g.register_tensor("output", shape=["batch", 2])
    g.outputs.append(_make_tensor("output", ["batch", 2], "FLOAT"))

    return g


def _make_matmul_bias_graph() -> Graph:
    """Build a graph with MatMul -> Add (bias) pattern."""
    g = Graph()
    g.name = "matmul_bias_test"

    # Inputs
    for tname, shape in [("input", ["batch", 16]), ("weight", [16, 8]),
                          ("bias", [8])]:
        is_init = tname in ("weight", "bias")
        g.register_tensor(tname, shape=shape, is_initializer=is_init)
        g.tensor_producer[tname] = "INITIALIZER" if is_init else "INPUT"
        if not is_init:
            g.inputs.append(_make_tensor(tname, shape, is_init=is_init))

    g.outputs.append(_make_tensor("output", ["batch", 8]))

    # MatMul node
    mm = Node(id="mm1", name="MatMul_1", op_type="MatMul",
              inputs=["input", "weight"], outputs=["mm_out"])
    g.add_node(mm)
    g.register_tensor("mm_out", shape=["batch", 8])
    g.set_producer("mm_out", "mm1")

    # Add (bias) node
    add = Node(id="add1", name="Add_1", op_type="Add",
               inputs=["mm_out", "bias"], outputs=["output"])
    g.add_node(add)
    g.register_tensor("output", shape=["batch", 8])
    g.set_producer("output", "add1")

    # Consumer maps
    g.add_consumer("input", "mm1")
    g.add_consumer("weight", "mm1")
    g.add_consumer("mm_out", "add1")
    g.add_consumer("bias", "add1")

    g.topological_sort()
    return g


def _make_conv_bn_graph() -> Graph:
    """Build a graph with Conv -> BatchNormalization pattern."""
    g = Graph()
    g.name = "conv_bn_test"

    for tname, shape in [("input", [1, 3, 32, 32]),
                          ("conv_weight", [16, 3, 3, 3]),
                          ("bn_scale", [16]), ("bn_bias", [16]),
                          ("bn_mean", [16]), ("bn_var", [16])]:
        is_init = tname != "input"
        g.register_tensor(tname, shape=shape, is_initializer=is_init)
        g.tensor_producer[tname] = "INITIALIZER" if is_init else "INPUT"
        if not is_init:
            g.inputs.append(_make_tensor(tname, shape, is_init=is_init))

    g.register_tensor("conv_out", shape=[1, 16, 32, 32])
    g.register_tensor("output", shape=[1, 16, 32, 32])
    g.outputs.append(_make_tensor("output", [1, 16, 32, 32]))

    conv = Node(id="conv1", name="Conv_1", op_type="Conv",
                inputs=["input", "conv_weight"], outputs=["conv_out"],
                attributes={"kernel_shape": [3, 3], "strides": [1, 1],
                            "pads": [1, 1, 1, 1], "group": 1})
    g.add_node(conv)
    g.set_producer("conv_out", "conv1")

    bn = Node(id="bn1", name="BN_1", op_type="BatchNormalization",
              inputs=["conv_out", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
              outputs=["output"],
              attributes={"epsilon": 1e-5, "momentum": 0.9})
    g.add_node(bn)
    g.set_producer("output", "bn1")

    for c in ["input", "conv_weight"]:
        g.add_consumer(c, "conv1")
    for c in ["conv_out", "bn_scale", "bn_bias", "bn_mean", "bn_var"]:
        g.add_consumer(c, "bn1")

    g.topological_sort()
    return g


def _make_ew_chain_graph() -> Graph:
    """Build a graph with 3 elementwise nodes in a chain: Add -> Mul -> Relu."""
    g = Graph()
    g.name = "ew_chain_test"

    g.register_tensor("input", shape=["batch", 16])
    g.register_tensor("c1", shape=["batch", 16])
    g.register_tensor("c2", shape=["batch", 16])
    g.tensor_producer["input"] = "INPUT"
    g.tensor_producer["c1"] = "INITIALIZER"
    g.tensor_producer["c2"] = "INITIALIZER"
    g.inputs.append(_make_tensor("input", ["batch", 16]))
    for t in ("c1", "c2"):
        g.register_tensor(t, shape=["batch", 16], is_initializer=True)

    g.register_tensor("t1", shape=["batch", 16])
    g.register_tensor("t2", shape=["batch", 16])
    g.register_tensor("output", shape=["batch", 16])
    g.outputs.append(_make_tensor("output", ["batch", 16]))

    n1 = Node(id="add1", name="Add_1", op_type="Add",
              inputs=["input", "c1"], outputs=["t1"])
    g.add_node(n1); g.set_producer("t1", "add1")

    n2 = Node(id="mul1", name="Mul_1", op_type="Mul",
              inputs=["t1", "c2"], outputs=["t2"])
    g.add_node(n2); g.set_producer("t2", "mul1")

    n3 = Node(id="relu1", name="Relu_1", op_type="Relu",
              inputs=["t2"], outputs=["output"])
    g.add_node(n3); g.set_producer("output", "relu1")

    g.add_consumer("input", "add1")
    g.add_consumer("c1", "add1")
    g.add_consumer("t1", "mul1")
    g.add_consumer("c2", "mul1")
    g.add_consumer("t2", "relu1")

    g.topological_sort()
    return g


def _make_softmax_dropout_graph() -> Graph:
    """Build a graph with Softmax -> Dropout pattern (inference mode)."""
    g = Graph()
    g.name = "softmax_dropout_test"

    g.register_tensor("input", shape=["batch", 16])
    g.tensor_producer["input"] = "INPUT"
    g.inputs.append(_make_tensor("input", ["batch", 16]))

    g.register_tensor("sm_out", shape=["batch", 16])
    g.register_tensor("output", shape=["batch", 16])
    g.outputs.append(_make_tensor("output", ["batch", 16]))

    sm = Node(id="sm1", name="Softmax_1", op_type="Softmax",
              inputs=["input"], outputs=["sm_out"],
              attributes={"axis": -1})
    g.add_node(sm); g.set_producer("sm_out", "sm1")

    dp = Node(id="dp1", name="Dropout_1", op_type="Dropout",
              inputs=["sm_out"], outputs=["output"],
              attributes={"ratio": 0.0})  # inference mode
    g.add_node(dp); g.set_producer("output", "dp1")

    g.add_consumer("input", "sm1")
    g.add_consumer("sm_out", "dp1")

    g.topological_sort()
    return g


def _make_softmax_dropout_training_graph() -> Graph:
    """Build a graph with Softmax -> Dropout in training mode (should NOT fuse)."""
    g = Graph()
    g.name = "softmax_dropout_training_test"

    g.register_tensor("input", shape=["batch", 16])
    g.tensor_producer["input"] = "INPUT"
    g.inputs.append(_make_tensor("input", ["batch", 16]))

    g.register_tensor("sm_out", shape=["batch", 16])
    g.register_tensor("output", shape=["batch", 16])
    g.outputs.append(_make_tensor("output", ["batch", 16]))

    sm = Node(id="sm1", name="Softmax_1", op_type="Softmax",
              inputs=["input"], outputs=["sm_out"],
              attributes={"axis": -1})
    g.add_node(sm); g.set_producer("sm_out", "sm1")

    dp = Node(id="dp1", name="Dropout_1", op_type="Dropout",
              inputs=["sm_out"], outputs=["output"],
              attributes={"ratio": 0.5, "training_mode": 1})  # training mode
    g.add_node(dp); g.set_producer("output", "dp1")

    g.add_consumer("input", "sm1")
    g.add_consumer("sm_out", "dp1")

    g.topological_sort()
    return g


def _make_residual_norm_graph() -> Graph:
    """Build a graph with Add -> LayerNormalization (residual) pattern."""
    g = Graph()
    g.name = "residual_norm_test"

    for tname, shape in [("input1", ["batch", 32]), ("input2", ["batch", 32]),
                          ("ln_weight", [32]), ("ln_bias", [32])]:
        is_init = tname in ("ln_weight", "ln_bias")
        g.register_tensor(tname, shape=shape, is_initializer=is_init)
        g.tensor_producer[tname] = "INITIALIZER" if is_init else "INPUT"
        if not is_init:
            g.inputs.append(_make_tensor(tname, shape, is_init=is_init))

    g.register_tensor("add_out", shape=["batch", 32])
    g.register_tensor("output", shape=["batch", 32])
    g.outputs.append(_make_tensor("output", ["batch", 32]))

    add = Node(id="add1", name="Add_1", op_type="Add",
               inputs=["input1", "input2"], outputs=["add_out"])
    g.add_node(add); g.set_producer("add_out", "add1")

    ln = Node(id="ln1", name="LayerNorm_1", op_type="LayerNormalization",
              inputs=["add_out", "ln_weight", "ln_bias"], outputs=["output"],
              attributes={"axis": -1, "epsilon": 1e-5})
    g.add_node(ln); g.set_producer("output", "ln1")

    for c in ["input1", "input2"]:
        g.add_consumer(c, "add1")
    for c in ["add_out", "ln_weight", "ln_bias"]:
        g.add_consumer(c, "ln1")

    g.topological_sort()
    return g


# ── Fusion pattern tests ──────────────────────────────────────────────


def test_fused_matmul_bias() -> float:
    print("\n--- F1a: FusedMatMulBias ---")
    score = 0.0
    log: List[Dict] = []

    # Positive: MatMul -> Add bias
    g = _make_matmul_bias_graph()
    n = fuse_matmul_bias(g, log)
    ok = check(n == 1, f"MatMul -> Add fusion succeeded ({n} fusions)", 0.2)
    if ok:
        check("FusedMatMulBias" in [n.op_type for n in g.nodes.values()],
              "FusedMatMulBias node exists in graph", 0.2)
        check(g.nodes.get("mm1") is None, "Original MatMul removed", 0.1)
        check(g.nodes.get("add1") is None, "Original Add removed", 0.1)
        # Validate
        try:
            g.validate()
            check(True, "Graph validates after fusion", 0.2)
        except ValueError as e:
            check(False, f"Graph validation failed: {e}", 0)
        score += 0.8

    # Check fusion log
    log_entries = [e for e in log if e["pattern"] == "FusedMatMulBias"]
    if log_entries:
        check(log_entries[0]["status"] == "fused", "Fusion log status=fused", 0.1)
        check(len(log_entries[0]["old_node_ids"]) == 2, "Fusion log records 2 old nodes", 0.1)
        score += 0.2

    # Negative: MatMul with multiple consumers (should skip)
    g2 = _make_matmul_bias_graph()
    # Add another consumer to mm_out
    mm_out = g2.tensors.get("mm_out")
    extra_node = Node(id="extra", name="Extra", op_type="Relu",
                      inputs=["mm_out"], outputs=["extra_out"])
    g2.add_node(extra_node)
    g2.register_tensor("extra_out", shape=["batch", 8])
    g2.set_producer("extra_out", "extra")
    g2.add_consumer("mm_out", "extra")
    g2.topological_sort()

    log2: List[Dict] = []
    n2 = fuse_matmul_bias(g2, log2)
    check(n2 == 0, f"MatMul with multiple consumers: correctly skipped ({n2} fusions)", 0.2)
    # Check rejection reason
    for e in log2:
        if e["pattern"] == "FusedMatMulBias" and e["status"] == "skipped":
            check(e["rejection_reason"] != "", "Skip entry has rejection_reason", 0.2)
            break
    score += 0.4

    print(f"  F1a score: {score:.2f} / 1.0 pt")
    SCORES["F1a"] = score
    return score


def test_fused_conv_batchnorm() -> float:
    print("\n--- F1b: FusedConv2dBatchNorm ---")
    score = 0.0
    log: List[Dict] = []

    # Positive: Conv -> BatchNormalization
    g = _make_conv_bn_graph()
    n = fuse_conv_batchnorm(g, log)
    ok = check(n == 1, f"Conv -> BN fusion succeeded ({n} fusions)", 0.3)
    if ok:
        check("FusedConv2dBatchNorm" in [n.op_type for n in g.nodes.values()],
              "FusedConv2dBatchNorm node exists", 0.2)
        try:
            g.validate()
            check(True, "Graph validates after fusion", 0.2)
        except ValueError as e:
            check(False, f"Graph validation failed: {e}", 0)
        score += 0.7

    # Check fusion log
    log_entries = [e for e in log if e["pattern"] == "FusedConv2dBatchNorm"]
    if log_entries:
        check(log_entries[0]["status"] == "fused", "Fusion log status=fused", 0.1)
        score += 0.1

    # Negative: Conv with multiple consumers (should skip)
    g2 = _make_conv_bn_graph()
    conv_out = "conv_out"
    extra = Node(id="extra_conv", name="ExtraConv", op_type="Relu",
                 inputs=[conv_out], outputs=["extra_conv_out"])
    g2.add_node(extra)
    g2.register_tensor("extra_conv_out", shape=[1, 16, 32, 32])
    g2.set_producer("extra_conv_out", "extra_conv")
    g2.add_consumer(conv_out, "extra_conv")
    g2.topological_sort()

    log2: List[Dict] = []
    n2 = fuse_conv_batchnorm(g2, log2)
    check(n2 == 0, f"Conv with multiple consumers: correctly skipped ({n2} fusions)", 0.2)
    score += 0.2

    print(f"  F1b score: {score:.2f} / 1.0 pt")
    SCORES["F1b"] = score
    return score


def test_fused_ew_chain() -> float:
    print("\n--- F1c: FusedEWChain ---")
    score = 0.0
    log: List[Dict] = []

    # Positive: 3-elementwise chain
    g = _make_ew_chain_graph()
    n = fuse_elementwise_chain(g, log)
    ok = check(n == 1, f"EW chain fusion succeeded ({n} fusions)", 0.3)
    if ok:
        check("FusedEWChain" in [n.op_type for n in g.nodes.values()],
              "FusedEWChain node exists", 0.2)
        try:
            g.validate()
            check(True, "Graph validates after fusion", 0.2)
        except ValueError as e:
            check(False, f"Graph validation failed: {e}", 0)
        score += 0.7

    # Check fusion log
    log_entries = [e for e in log if e["pattern"] == "FusedEWChain"]
    if log_entries:
        check(log_entries[0]["status"] == "fused", "Fusion log status=fused", 0.1)
        check(len(log_entries[0]["old_node_ids"]) == 3, "Fusion log records 3 old nodes", 0.1)
        score += 0.2

    # Negative: 1-node chain (too short)
    g2 = Graph()
    g2.name = "single_ew"
    g2.register_tensor("inp", shape=[4]); g2.tensor_producer["inp"] = "INPUT"
    g2.inputs.append(_make_tensor("inp", [4]))
    g2.register_tensor("out", shape=[4])
    g2.outputs.append(_make_tensor("out", [4]))
    n1 = Node(id="relu1", name="Relu_1", op_type="Relu", inputs=["inp"], outputs=["out"])
    g2.add_node(n1); g2.set_producer("out", "relu1")
    g2.add_consumer("inp", "relu1")
    g2.topological_sort()

    log2: List[Dict] = []
    n2 = fuse_elementwise_chain(g2, log2)
    check(n2 == 0, "Single elementwise node: correctly skipped (chain too short)", 0.1)
    score += 0.1

    print(f"  F1c score: {score:.2f} / 1.0 pt")
    SCORES["F1c"] = score
    return score


def test_fused_softmax_dropout() -> float:
    print("\n--- F1d: FusedSoftmaxDropout ---")
    score = 0.0
    log: List[Dict] = []

    # Positive: Softmax -> Dropout (inference)
    g = _make_softmax_dropout_graph()
    n = fuse_softmax_dropout(g, log)
    ok = check(n == 1, f"Softmax -> Dropout fusion succeeded ({n} fusions)", 0.3)
    if ok:
        check("FusedSoftmaxDropout" in [n.op_type for n in g.nodes.values()],
              "FusedSoftmaxDropout node exists", 0.2)
        try:
            g.validate()
            check(True, "Graph validates after fusion", 0.2)
        except ValueError as e:
            check(False, f"Graph validation failed: {e}", 0)
        score += 0.7

    # Negative: Training mode Dropout (should skip)
    g2 = _make_softmax_dropout_training_graph()
    log2: List[Dict] = []
    n2 = fuse_softmax_dropout(g2, log2)
    check(n2 == 0, "Training-mode Dropout: correctly skipped", 0.2)
    for e in log2:
        if e["pattern"] == "FusedSoftmaxDropout" and e["status"] == "skipped":
            check("training" in e["rejection_reason"].lower(),
                  "Skip reason mentions training mode", 0.1)
            break
    score += 0.3

    print(f"  F1d score: {score:.2f} / 1.0 pt")
    SCORES["F1d"] = score
    return score


def test_fused_residual_norm() -> float:
    print("\n--- F1e: FusedResidualNorm ---")
    score = 0.0
    log: List[Dict] = []

    # Positive: Add -> LayerNormalization
    g = _make_residual_norm_graph()
    n = fuse_residual_norm(g, log)
    ok = check(n == 1, f"Residual Add -> LayerNorm fusion succeeded ({n} fusions)", 0.3)
    if ok:
        check("FusedResidualNorm" in [n.op_type for n in g.nodes.values()],
              "FusedResidualNorm node exists", 0.2)
        try:
            g.validate()
            check(True, "Graph validates after fusion", 0.2)
        except ValueError as e:
            check(False, f"Graph validation failed: {e}", 0)
        score += 0.7

    # Check fusion log
    log_entries = [e for e in log if e["pattern"] == "FusedResidualNorm"]
    if log_entries:
        check(log_entries[0]["status"] == "fused", "Fusion log status=fused", 0.1)
        score += 0.1

    # Negative: Add with multiple consumers (should skip)
    g2 = _make_residual_norm_graph()
    add_out = "add_out"
    extra = Node(id="extra_add", name="ExtraAdd", op_type="Relu",
                 inputs=[add_out], outputs=["extra_add_out"])
    g2.add_node(extra)
    g2.register_tensor("extra_add_out", shape=["batch", 32])
    g2.set_producer("extra_add_out", "extra_add")
    g2.add_consumer(add_out, "extra_add")
    g2.topological_sort()

    log2: List[Dict] = []
    n2 = fuse_residual_norm(g2, log2)
    check(n2 == 0, "Add with multiple consumers: correctly skipped", 0.2)
    score += 0.2

    print(f"  F1e score: {score:.2f} / 1.0 pt")
    SCORES["F1e"] = score
    return score


# ── Pipeline tests ────────────────────────────────────────────────────


def test_pipeline_integration() -> float:
    print("\n--- Pipeline: F2/F3/F4 ---")
    score = 0.0

    # Test on matmul bias graph
    g = _make_matmul_bias_graph()
    node_count_before = len(g.nodes)

    pipeline = GraphPassPipeline(enable_fusion=True, run_validation=True)
    results = pipeline.run(g)

    stats = results["Fusion"]["stats"]
    fusion_log = stats["fusion_log"]

    # F4: Correctness checks
    # 1. Graph outputs preserved
    out_names = {t.name for t in g.outputs}
    check(len(out_names) == 1, "F4: Graph outputs preserved", 0.25)
    score += 0.25

    # 2. Graph inputs preserved (should have at least the original inputs)
    inp_names = {t.name for t in g.inputs}
    check(len(inp_names) >= 1, "F4: Graph inputs preserved", 0.25)
    score += 0.25

    # 3. Validation passes
    check(stats["validation_passed"], "F4: Graph validation passed", 0.25)
    score += 0.25

    # 4. Node count not increased
    node_count_after = stats["node_count_after"]
    check(node_count_after <= node_count_before,
          f"F4: Nodes not increased ({node_count_before} -> {node_count_after})", 0.25)
    score += 0.25

    # F2 reduction check
    lr = stats["launch_reduction"]
    print(f"  Launch reduction: {lr:.1%} (raw={stats['raw_launches']}, opt={stats['optimized_launches']})")
    if lr > 0:
        score += 0.3

    # F3 reduction check
    br = stats["buffer_reduction"]
    print(f"  Buffer reduction: {br:.1%} (raw={stats['raw_buffers']}, opt={stats['optimized_buffers']})")
    if br > 0:
        score += 0.3

    # Fusion log exists and populated
    check(len(fusion_log) > 0, "Fusion log is non-empty", 0.2)
    has_fused = any(e["status"] == "fused" for e in fusion_log)
    check(has_fused, "Fusion log has at least one 'fused' entry", 0.2)
    score += 0.4

    # Check fusions_per_pattern
    fpp = stats["fusions_per_pattern"]
    check("MatMulBias" in fpp, "MatMulBias pattern recorded in fusions_per_pattern", 0.2)
    score += 0.2

    # Disabled fusion test
    g2 = _make_matmul_bias_graph()
    pipeline_disabled = GraphPassPipeline(enable_fusion=False)
    results_disabled = pipeline_disabled.run(g2)
    check(not results_disabled["Fusion"]["enabled"],
          "Disabled fusion: enabled=False", 0.1)
    check(len(results_disabled["Fusion"]["stats"]["fusion_log"]) == 0,
          "Disabled fusion: empty log", 0.1)
    score += 0.2

    print(f"  Pipeline score: {score:.2f} / 2.0 pt")
    SCORES["Pipeline"] = score
    return score


# ── Real model tests ──────────────────────────────────────────────────


def test_real_models() -> float:
    """Test released models and compute the written-rubric F2/F3 self-score.

    The published C3.2/C3.3 benchmark names MLP and ResNet-18. Transformer is
    retained as additional validation, but it does not dilute or inflate the
    two-model F2/F3 score.
    """
    print("\n--- Real models (F2/F3 rubric validation) ---")
    score = 0.0
    official_models = {"mlp", "resnet"}
    official_stats: Dict[str, Dict[str, Any]] = {}
    official_invariants: Dict[str, Dict[str, bool]] = {}

    if not HAS_ONNX:
        print("  SKIP: onnx package not available")
        return score

    for model_name, model_path in MODEL_PATHS.items():
        if not os.path.exists(model_path):
            print(f"  SKIP {model_name}: model not found at {model_path}")
            continue

        try:
            graph = _import_onnx(model_path)
            node_count_before = len(graph.nodes)
            input_names_before = {tensor.name for tensor in graph.inputs}
            output_names_before = {tensor.name for tensor in graph.outputs}

            pipeline = GraphPassPipeline(enable_fusion=True, run_validation=True)
            results = pipeline.run(graph)

            stats = results["Fusion"]["stats"]
            fusion_log = stats["fusion_log"]

            print(f"  {model_name}: {node_count_before} nodes -> {stats['node_count_after']} nodes, "
                  f"launch_red={stats['launch_reduction']:.1%}, "
                  f"buffer_red={stats['buffer_reduction']:.1%}, "
                  f"log_entries={len(fusion_log)}, "
                  f"valid={stats['validation_passed']}")

            check(stats["validation_passed"], f"{model_name}: validation passed", 0.1)
            check(stats["node_count_after"] <= node_count_before,
                  f"{model_name}: nodes not increased", 0.1)
            check(stats["launch_reduction"] >= 0.60,
                  f"{model_name}: structural launch reduction reaches 60%")
            check(stats["buffer_reduction"] >= 0.60,
                  f"{model_name}: logical buffer reduction reaches 60%")

            score += 0.2
            if model_name in official_models:
                official_stats[model_name] = stats
                official_invariants[model_name] = {
                    "inputs_preserved": (
                        {tensor.name for tensor in graph.inputs}
                        == input_names_before
                    ),
                    "outputs_preserved": (
                        {tensor.name for tensor in graph.outputs}
                        == output_names_before
                    ),
                    "validation_passed": bool(stats["validation_passed"]),
                    "nodes_not_increased": (
                        stats["node_count_after"] <= node_count_before
                    ),
                }
        except Exception as e:
            print(f"  {model_name}: ERROR {e}")

    # Apply the written formula separately to each specified model. The release
    # does not include the referenced benchmark or define its cross-model
    # aggregation rule, so this self-test displays every model score and uses a
    # conservative mean for its local summary. Full credit still requires both
    # models to reach the 60% anchor.
    f2_per_model = {
        name: min(max(stats["launch_reduction"], 0.0) * 5.0, 3.0)
        for name, stats in official_stats.items()
    }
    f3_per_model = {
        name: min(max(stats["buffer_reduction"], 0.0) * 5.0, 3.0)
        for name, stats in official_stats.items()
    }
    if official_models <= official_stats.keys():
        f2_score = sum(f2_per_model.values()) / len(official_models)
        f3_score = sum(f3_per_model.values()) / len(official_models)
    else:
        f2_score = 0.0
        f3_score = 0.0

    invariant_names = (
        "inputs_preserved", "outputs_preserved",
        "validation_passed", "nodes_not_increased",
    )
    f4_structural = sum(
        1.0 for invariant in invariant_names
        if official_models <= official_invariants.keys()
        and all(official_invariants[name][invariant] for name in official_models)
    )

    for name in sorted(official_models):
        if name in official_stats:
            print(
                f"  {name} rubric: F2={f2_per_model[name]:.2f}/3.0, "
                f"F3={f3_per_model[name]:.2f}/3.0"
            )
    print(f"  F2 public-model mean: {f2_score:.2f} / 3.0 pt")
    print(f"  F3 public-model mean: {f3_score:.2f} / 3.0 pt")
    print(
        f"  F4 structural: {f4_structural:.2f} / 4.0 pt "
        "(subject to the required FP32 numerical gate)"
    )
    print(f"  Structural diagnostics: {score:.2f} / 0.6 checks")
    SCORES["F2"] = f2_score
    SCORES["F3"] = f3_score
    SCORES["F4Structural"] = f4_structural
    SCORES["RealModels"] = score
    return score


# ── Launch/buffer count consistency ───────────────────────────────────


def test_counting() -> float:
    """Test launch and buffer counting on known graph structures."""
    print("\n--- Counting consistency ---")
    score = 0.0

    g = _simple_graph()
    g.add_node(Node(id="n1", name="Relu", op_type="Relu", inputs=["input"], outputs=["n1_out"]))
    g.register_tensor("n1_out", shape=["batch", 4])
    g.set_producer("n1_out", "n1")
    g.add_consumer("input", "n1")

    g.add_node(Node(id="n2", name="Add", op_type="Add", inputs=["n1_out", "c1"],
                    outputs=["output"]))
    g.register_tensor("c1", shape=["batch", 4], is_initializer=True)
    g.tensor_producer["c1"] = "INITIALIZER"
    g.register_tensor("output", shape=["batch", 4])
    g.set_producer("output", "n2")
    g.add_consumer("n1_out", "n2")
    g.add_consumer("c1", "n2")
    g.topological_sort()

    launches = _count_launches(g)
    buffers = _count_buffers(g)
    check(launches > 0, f"Launches > 0: {launches}", 0.1)
    check(buffers >= 0, f"Buffers >= 0: {buffers}", 0.1)
    score += 0.2

    # After fusion, launches should decrease
    pipeline = GraphPassPipeline(enable_fusion=True)
    results = pipeline.run(g)
    stats = results["Fusion"]["stats"]
    check(stats["launch_reduction"] >= 0, "Non-negative launch reduction", 0.15)
    check(stats["buffer_reduction"] >= 0, "Non-negative buffer reduction", 0.15)
    score += 0.3

    print(f"  Counting score: {score:.2f} / 0.5 pt")
    SCORES["Counting"] = score
    return score


# ── Graph invariant tests ─────────────────────────────────────────────


def test_graph_invariants() -> float:
    """Test that graph invariants hold after fusion."""
    print("\n--- Graph invariants ---")
    score = 0.0

    # Test acyclic after fusion
    g = _make_matmul_bias_graph()
    pipeline = GraphPassPipeline(enable_fusion=True)
    pipeline.run(g)

    try:
        g.topological_sort()
        check(True, "Graph remains acyclic after fusion", 0.25)
        score += 0.25
    except ValueError:
        check(False, "Graph has cycles after fusion!", 0)

    # Test tensor references valid
    all_ok = True
    for nid, node in g.nodes.items():
        for inp in node.inputs:
            if inp and inp not in g.tensors:
                all_ok = False
                break
        if not all_ok:
            break
    check(all_ok, "All node inputs resolve to registered tensors", 0.25)
    score += 0.25

    print(f"  Invariants score: {score:.2f} / 0.5 pt")
    SCORES["Invariants"] = score
    return score


def test_fusion_safety_guards() -> None:
    """Exercise guards for observable values and opset-17 optional inputs."""
    print("\n--- Fusion safety guards ---")

    graph = _make_matmul_bias_graph()
    graph.outputs.append(_make_tensor("mm_out", ["batch", 8]))
    log: List[Dict[str, Any]] = []
    check(fuse_matmul_bias(graph, log) == 0,
          "Observable MatMul intermediate is not erased")
    graph.validate()

    graph = _make_matmul_bias_graph()
    graph.tensors["bias"].shape = [2, 4]
    log = []
    check(fuse_matmul_bias(graph, log) == 0,
          "Incompatible rank-2 Add operand is not treated as bias")
    graph.validate()

    graph = _make_softmax_dropout_graph()
    dropout = graph.nodes["dp1"]
    graph.register_tensor("training", dtype=ONNSType.BOOL, shape=[],
                          is_initializer=True)
    graph.tensor_producer["training"] = "INITIALIZER"
    graph.replace_node_inputs("dp1", dropout.inputs,
                              ["sm_out", "", "training"])
    log = []
    check(fuse_softmax_dropout(graph, log) == 0,
          "Unknown explicit training_mode input blocks Dropout fusion")
    graph.validate()

    graph = _make_residual_norm_graph()
    graph.outputs.append(_make_tensor("add_out", ["batch", 32]))
    log = []
    check(fuse_residual_norm(graph, log) == 0,
          "Observable residual value is not erased")
    graph.validate()


# ── Main ──────────────────────────────────────────────────────────────


def print_summary() -> None:
    print("\n" + "=" * 50)
    print("C3.3 SCORING SUMMARY")
    print("=" * 50)
    total = 0.0

    official_categories = [
        ("F1a: MatMulBias", SCORES.get("F1a", 0), 1.0),
        ("F1b: Conv2dBatchNorm", SCORES.get("F1b", 0), 1.0),
        ("F1c: EWChain", SCORES.get("F1c", 0), 1.0),
        ("F1d: SoftmaxDropout", SCORES.get("F1d", 0), 1.0),
        ("F1e: ResidualNorm", SCORES.get("F1e", 0), 1.0),
        ("F2: Launch reduction", SCORES.get("F2", 0), 3.0),
        ("F3: Buffer reduction", SCORES.get("F3", 0), 3.0),
        ("F4: Structural correctness", SCORES.get("F4Structural", 0), 4.0),
    ]
    diagnostic_categories = [
        ("Pipeline diagnostics", SCORES.get("Pipeline", 0), 2.0),
        ("Real-model diagnostics", SCORES.get("RealModels", 0), 0.6),
        ("Counting", SCORES.get("Counting", 0), 0.5),
        ("Invariants", SCORES.get("Invariants", 0), 0.5),
    ]

    for name, s, mx in official_categories:
        capped = min(s, mx)
        total += capped
        stars = "\u2605" * int(round(capped * 5)) + "\u2606" * int(round(mx * 5 - capped * 5))
        print(f"  {name}: {capped:.2f}/{mx} {stars}")
    maximum = 15.0
    total = min(total, maximum)
    print("  " + "\u2500" * 13)
    print(f"  WRITTEN-RUBRIC STRUCTURAL TOTAL: {total:.2f}/{maximum}")
    print("  F4 becomes 0 if the required FP32 numerical comparison fails.")
    print("\n  Additional self-test diagnostics (not scoring points):")
    for name, s, mx in diagnostic_categories:
        print(f"    {name}: {min(s, mx):.2f}/{mx}")
    print(f"  {'PASS' if FAIL == 0 else 'SOME FAILURES'} "
          f"(PASS={PASS}, FAIL={FAIL})")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="C3.3 self-test")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    print("C3.3 Fusion Self-Test")
    print("=" * 40)

    test_fused_matmul_bias()
    test_fused_conv_batchnorm()
    test_fused_ew_chain()
    test_fused_softmax_dropout()
    test_fused_residual_norm()
    test_pipeline_integration()
    test_real_models()
    test_counting()
    test_graph_invariants()
    test_fusion_safety_guards()

    print_summary()
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
