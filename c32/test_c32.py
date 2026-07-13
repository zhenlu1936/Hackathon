#!/usr/bin/env python3
"""Self-test for C3.2 — validates all five scoring dimensions.

Usage:
    python -m c32.test_c32 [--verbose]
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from c31.import_onnx import import_onnx
from c3common.ir.graph import Graph, Node
from c32.api import strategy, hardware
from c32.hardware import HardwareCapability, set_hardware
from c32.kernel_spec import PrecisionProfile, KernelSpecRef, KernelTuningParams
from c32.strategy import ExecutionMode, SENSITIVE_OPS, TUMABLE_OPS

# CuPy is required on the AEC H200 for numerical verification.
# On macOS or other non-CUDA environments the numerical test is skipped.
try:
    import cupy as cp
    _CUPY_AVAILABLE = cp.cuda.runtime.getDeviceCount() >= 1
except Exception:
    cp = None  # type: ignore[assignment]
    _CUPY_AVAILABLE = False

MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
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
        return True
    else:
        FAIL += 1
        print(f"  \u2717 {msg}")
        return False


def load_all_graphs() -> Dict[str, Graph]:
    graphs = {}
    for name, path in MODEL_PATHS.items():
        if os.path.exists(path):
            graphs[name] = import_onnx(path)
            print(f"  Loaded {name}: {len(graphs[name].nodes)} nodes")
    return graphs


def test_full_fp32_numerical(graphs: Dict[str, Graph]) -> float:
    """FULL_FP32 numerical verification against golden outputs.

    Creates a FULL_FP32 strategy, decomposes every node, and executes
    through the connected C3.1→C3.5 CuPy pipeline on the AEC H200 device.
    Compares outputs against golden .npy tensors and checks top-1 agreement.

    Returns:
        Validation fraction in [0, 1]. This is a hard D1 gate, not an
        additional C3.2 scoring point.
    """
    print("\n=== FULL_FP32 Numerical Verification ===")
    score = 0.0

    if not _CUPY_AVAILABLE:
        print("  [SKIP] CuPy/CUDA device not available — numerical verification "
              "requires the AEC H200 GPU.")
        SCORES["FP32_NUM"] = 0.0
        return 0.0

    from pathlib import Path
    from c32.strategy import Strategy
    from c35.executor import CrossStageReferencePipeline

    ROOT = Path(__file__).resolve().parents[1]
    TESTDATA = (
        ROOT / ".specification" / "testcases" / "release_to_competitors"
        / "testdata" / "c35"
    )
    # ── Step 1: Verify FULL_FP32 decomposition ─────────────────
    fp32_strategy = Strategy(mode=ExecutionMode.FULL_FP32)
    all_fp32 = True
    all_decomposed = 0
    total_nodes = 0
    for gname, graph in graphs.items():
        for nid in graph.node_order:
            node = graph.nodes[nid]
            total_nodes += 1
            prec = fp32_strategy.select_precision(node, graph)
            if prec.compute_dtype != "fp32":
                all_fp32 = False
                print(f"  NON-FP32: {node.op_type}:{node.id} -> {prec.compute_dtype}")
            kernels = fp32_strategy.decompose(node, graph, prec)
            if kernels:
                all_decomposed += 1

    d0a_score = 0.4 if all_fp32 else 0.0
    check(all_fp32,
          f"All {total_nodes} nodes select fp32 in FULL_FP32 mode", d0a_score)
    score += d0a_score

    d0b_score = 0.1 if all_decomposed == total_nodes else 0.0
    check(all_decomposed == total_nodes,
          f"All {total_nodes} nodes decomposed ({all_decomposed}/{total_nodes})", d0b_score)
    score += d0b_score

    # ── Step 2: Execute and compare against golden ────────────
    models_ok = 0
    for gname in ("mlp", "resnet", "transformer"):
        model_path = ROOT / "models" / f"{gname}_v1.onnx"
        golden_path = TESTDATA / f"{gname}_v1" / "golden" / "logits.npy"

        if not model_path.is_file() or not golden_path.is_file():
            print(f"  [SKIP] {gname}: missing model or golden data")
            continue

        try:
            graph = import_onnx(str(model_path))
            pipeline = CrossStageReferencePipeline(
                graph, str(model_path), qualify_optimizations=True,
            )

            # Load a small feed batch (2 samples) for the qualification run
            input_dir = TESTDATA / f"{gname}_v1" / "input"
            import json
            manifest = json.loads((input_dir / "manifest.json").read_text())
            feed_dict = {
                entry["name"]: cp.load(input_dir / entry["file"])[:2]
                for entry in manifest["tensors"]
            }

            outputs = pipeline.run(feed_dict)

            # Compare against golden (first 2 samples)
            golden = cp.load(str(golden_path))[:2]
            logits = outputs["logits"]
            max_diff = float(cp.max(cp.abs(logits - golden)).item())

            allclose_ok = bool(cp.allclose(
                logits, golden, rtol=1e-3, atol=1e-3
            ))
            top1_match = float(cp.mean(
                cp.argmax(logits, axis=-1).reshape(-1)
                == cp.argmax(golden, axis=-1).reshape(-1)
            ).item())
            model_ok = allclose_ok and top1_match >= 0.99
            if model_ok:
                models_ok += 1

            print(
                f"  {gname}: max_abs_diff={max_diff:.6e}  "
                f"allclose={allclose_ok}  top1_match={top1_match:.4f}  "
                f"-> {'OK' if model_ok else 'FAIL'}"
            )
        except Exception as exc:
            print(f"  {gname}: EXCEPTION — {exc}")

    models_total = 3
    d0c_score = (models_ok / models_total) * 0.5
    check(models_ok == models_total,
          f"Golden comparison: {models_ok}/{models_total} models pass", d0c_score)
    score += d0c_score

    SCORES["FP32_NUM"] = score
    print(f"  FULL_FP32 numerical total: {score:.2f} / 1.0 pt")
    return score


def test_d1_precision(graphs: Dict[str, Graph]) -> float:
    print("\n=== D1: Precision routing (3 pt) ===")
    score = 0.0
    all_profiles: List[Tuple[str, Node, PrecisionProfile]] = []

    for gname, graph in graphs.items():
        for nid in graph.node_order:
            node = graph.nodes[nid]
            profile = strategy.select_precision(node, graph)
            all_profiles.append((gname, node, profile))

    sensitive_nodes = [(g, n, p) for g, n, p in all_profiles if n.op_type in SENSITIVE_OPS]
    fp32_count = sum(1 for _, _, p in sensitive_nodes if p.compute_dtype == "fp32")
    total_sens = len(sensitive_nodes)
    d1a_ratio = fp32_count / max(total_sens, 1)
    d1a_score = d1a_ratio * 1.5
    score += d1a_score
    check(d1a_ratio == 1.0, f"D1a: Sensitive ops FP32: {fp32_count}/{total_sens} = {d1a_ratio:.0%}", d1a_score)

    precisions_used = set(p.compute_dtype for _, _, p in all_profiles)
    diversity_targets = {"fp32", "fp16", "fp8", "fp4"}
    diversity_count = len(precisions_used & diversity_targets)
    d1b_score = diversity_count / 4.0
    score += d1b_score
    print(f"  D1b: Precisions used: {sorted(precisions_used)} (target coverage: {diversity_count}/4)  [{d1b_score:.2f} pt]")

    tunable_nodes = [(g, n, p) for g, n, p in all_profiles if n.op_type in TUMABLE_OPS]
    supported = hardware.supported_precisions()
    in_supported = sum(1 for _, _, p in tunable_nodes if p.compute_dtype in supported)
    total_tun = len(tunable_nodes)
    d1c_ratio = in_supported / max(total_tun, 1)
    d1c_score = d1c_ratio * 0.5
    score += d1c_score
    check(d1c_ratio == 1.0, f"D1c: Tunable ops in supported set: {in_supported}/{total_tun} = {d1c_ratio:.0%}", d1c_score)

    print(f"  D1 total: {score:.2f} / 3.0 pt")
    SCORES["D1"] = score
    return score


def test_d2_kernel_sequences(graphs: Dict[str, Graph]) -> float:
    print("\n=== D2: Kernel sequences (3 pt) ===")
    score = 0.0
    seq_cov, total_n = 0, 0
    key_found, key_total = 0, 0
    matmul_f, softmax_f, layernorm_f, conv_f = False, False, False, False

    for gname, graph in graphs.items():
        for nid in graph.node_order:
            node = graph.nodes[nid]
            total_n += 1
            precision = strategy.select_precision(node, graph)
            kernels = strategy.decompose(node, graph, precision)
            if len(kernels) > 0:
                seq_cov += 1
            knames = [k.kernel_name for k in kernels]
            if node.op_type in ("MatMul", "Gemm"):
                key_total += 1
                if any(k.startswith("matmul_") for k in knames):
                    matmul_f = True; key_found += 1
            elif node.op_type == "Softmax":
                key_total += 1
                if len(kernels) >= 4:
                    softmax_f = True; key_found += 1
            elif node.op_type == "LayerNormalization":
                key_total += 1
                expected = ["reduce_mean", "sub", "mul", "sqrt"]
                if all(any(k.startswith(e) or k == e for k in knames) for e in expected):
                    layernorm_f = True; key_found += 1
            elif node.op_type == "Conv":
                key_total += 1
                if any(k.startswith("winograd_forward_") or k.startswith("im2col_") for k in knames):
                    conv_f = True; key_found += 1

    cov = seq_cov / max(total_n, 1)
    score += cov * 1.0
    check(cov == 1.0, f"D2a: Kernel seq coverage: {seq_cov}/{total_n} = {cov:.0%}", cov * 1.0)

    key_r = key_found / max(key_total, 1)
    score += key_r * 2.0
    check(key_r == 1.0, f"D2b: Key sequences: {key_found}/{key_total} "
          f"(MatMul={matmul_f}, Softmax={softmax_f}, LayerNorm={layernorm_f}, Conv={conv_f})", key_r * 2.0)

    print(f"  D2 total: {score:.2f} / 3.0 pt")
    SCORES["D2"] = score
    return score


def test_d3_intermediates(graphs: Dict[str, Graph]) -> float:
    print("\n=== D3: Intermediate tensors (3 pt) ===")
    score = 0.0
    KEY_OPS = {"Softmax", "LayerNormalization", "Conv"}
    key_with, key_total = 0, 0
    all_with, all_total = 0, 0

    for gname, graph in graphs.items():
        for nid in graph.node_order:
            node = graph.nodes[nid]
            all_total += 1
            precision = strategy.select_precision(node, graph)
            kernels = strategy.decompose(node, graph, precision)
            node_outs = set(node.outputs)
            kern_outs = set()
            for k in kernels:
                kern_outs.update(k.outputs)
            inter = kern_outs - node_outs
            if inter:
                all_with += 1
                if node.op_type in KEY_OPS:
                    key_with += 1
            if node.op_type in KEY_OPS:
                key_total += 1

    kr = key_with / max(key_total, 1)
    tr = all_with / max(all_total, 1)
    score += kr * 2.0 + tr * 1.0
    check(kr >= 0.5, f"D3a: Key ops with intermediates: {key_with}/{key_total}", kr * 2.0)
    check(tr > 0, f"D3b: All nodes with intermediates: {all_with}/{all_total}", tr * 1.0)
    print(f"  D3 total: {score:.2f} / 3.0 pt")
    SCORES["D3"] = score
    return score


def test_d4_tuning(graphs: Dict[str, Graph]) -> float:
    print("\n=== D4: Tuning parameters (3 pt) ===")
    score = 0.0
    max_t = hardware.max_threads_per_block
    max_s = hardware.smem_bytes
    all_kernels: List[Tuple[str, str, KernelSpecRef]] = []

    for gname, graph in graphs.items():
        for nid in graph.node_order:
            node = graph.nodes[nid]
            prec = strategy.select_precision(node, graph)
            kernels = strategy.decompose(node, graph, prec)
            for k in kernels:
                if strategy._is_kernel_tunable(k.kernel_name):
                    k.tuning_params = strategy.tune_kernel(k, prec)
                    all_kernels.append((gname, nid, k))

    tunable = [k for _, _, k in all_kernels]
    with_p = sum(1 for k in tunable if k.tuning_params is not None)
    total = len(tunable) or 1
    tc = with_p / total
    score += min(tc / 0.9, 1.0) * 1.5
    check(tc >= 0.9, f"D4a: Tuning coverage: {with_p}/{total} = {tc:.0%}", min(tc / 0.9, 1.0) * 1.5)

    passed, total_chk = 0, 0
    for k in tunable:
        if k.tuning_params is None:
            total_chk += 3; continue
        tp = k.tuning_params
        total_chk += 3
        if 0 < tp.block_x <= max_t: passed += 1
        if tp.grid_x > 0: passed += 1
        if tp.smem_bytes <= max_s or tp.smem_bytes == -1: passed += 1
    vr = passed / max(total_chk, 1)
    score += vr * 1.5
    check(vr == 1.0, f"D4b: Tuning validity: {passed}/{total_chk} assertions passed", vr * 1.5)

    print(f"  D4 total: {score:.2f} / 3.0 pt")
    SCORES["D4"] = score
    return score


def test_d5_hardware() -> float:
    print("\n=== D5: Hardware coverage (3 pt) ===")
    score = 0.0
    supported = hardware.supported_precisions()
    pc = len(supported)
    d5a = min((pc - 1) / 2.0, 1.0) * 1.0 if pc >= 2 else 0.0
    score += d5a
    print(f"  D5a: {pc} precisions: {supported}  [{d5a:.2f} pt]")

    gk = hardware.gemm_kernels_available()
    d5b = 0.0
    if "matmul_f32" in gk and "matmul_f16" in gk:
        d5b = 0.5
    if any(k in gk for k in ("matmul_f8", "matmul_fp8")): d5b += 0.25
    if any(k in gk for k in ("matmul_f4", "matmul_fp4")): d5b += 0.25
    score += min(d5b, 1.0)
    print(f"  D5b: GEMM kernels: {sorted(gk)}  [{min(d5b,1.0):.2f} pt]")

    cs = hardware.conv_strategies_available()
    d5c = 1.0 if "im2col" in cs and "winograd" in cs else 0.5
    score += d5c
    print(f"  D5c: Conv strategies: {cs}  [{d5c:.2f} pt]")

    print(f"  D5 total: {score:.2f} / 3.0 pt")
    SCORES["D5"] = score
    return score


def print_summary() -> None:
    print("\n" + "=" * 50)
    print("C3.2 SCORING SUMMARY")
    print("=" * 50)
    total = 0.0
    for dim, mx in [("D1", 3.0), ("D2", 3.0), ("D3", 3.0), ("D4", 3.0), ("D5", 3.0)]:
        s = SCORES.get(dim, 0.0)
        total += s
        stars = "\u2605" * int(round(s)) + "\u2606" * int(round(mx - s))
        print(f"  {dim}: {s:.2f}/{mx} {stars}")
    print(f"  \u2500" * 13)
    print(f"  TOTAL: {total:.2f}/15.0")
    print(f"  {'PASS' if total >= 10 else 'NEEDS WORK'} ({total:.1f}/15)")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="C3.2 self-test")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    print(f"Hardware: {hardware.name}")
    print(f"  Supported precisions: {hardware.supported_precisions()}")
    print(f"  Max threads/block: {hardware.max_threads_per_block}")
    print(f"  Smem bytes: {hardware.smem_bytes}")
    print(f"  Conv strategies: {hardware.conv_strategies_available()}")

    print("\nLoading models...")
    graphs = load_all_graphs()
    if not graphs:
        print("ERROR: No models found!")
        return 1

    test_d1_precision(graphs)
    test_d2_kernel_sequences(graphs)
    test_d3_intermediates(graphs)
    test_d4_tuning(graphs)
    test_d5_hardware()
    fp32_numerical = test_full_fp32_numerical(graphs)
    if _CUPY_AVAILABLE and fp32_numerical < 1.0:
        # FULL_FP32 correctness is a hard D1 gate, not an extra scoring point.
        SCORES["D1"] = 0.0

    print_summary()
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
