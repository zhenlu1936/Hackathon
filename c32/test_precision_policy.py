"""Regression tests for deterministic engineering-based precision routing."""

from __future__ import annotations

import unittest

from c31.import_onnx import import_onnx
from c32.hardware import HardwareCapability
from c32.strategy import ExecutionMode, SENSITIVE_OPS, Strategy


class PrecisionPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.graphs = [
            import_onnx(f"models/{name}_v1.onnx")
            for name in ("mlp", "resnet", "transformer")
        ]

    def test_full_fp32_is_unchanged(self) -> None:
        strategy = Strategy(mode=ExecutionMode.FULL_FP32)
        for graph in self.graphs:
            for node in graph.nodes.values():
                self.assertEqual(strategy.select_precision(node, graph).compute_dtype, "fp32")

    def test_sensitive_operators_are_always_fp32(self) -> None:
        strategy = Strategy(mode=ExecutionMode.MIXED_PRECISION)
        checked = 0
        for graph in self.graphs:
            for node in graph.nodes.values():
                if node.op_type in SENSITIVE_OPS:
                    checked += 1
                    self.assertEqual(
                        strategy.select_precision(node, graph).compute_dtype,
                        "fp32",
                    )
        self.assertGreater(checked, 0)

    def test_public_union_uses_four_requested_precisions(self) -> None:
        strategy = Strategy(mode=ExecutionMode.MIXED_PRECISION)
        selected = {
            strategy.select_precision(node, graph).compute_dtype
            for graph in self.graphs
            for node in graph.nodes.values()
        }
        self.assertTrue({"fp32", "fp16", "fp8", "fp4"} <= selected)

    def test_selection_is_deterministic_supported_and_emitted(self) -> None:
        strategy = Strategy(mode=ExecutionMode.MIXED_PRECISION)
        supported = set(strategy.hardware.supported_precisions())
        for graph in self.graphs:
            for node in graph.nodes.values():
                first = strategy.select_precision(node, graph)
                second = strategy.select_precision(node, graph)
                self.assertEqual(first, second)
                self.assertIn(first.compute_dtype, supported)
                if node.op_type in {"MatMul", "Gemm", "Conv"}:
                    kernels = strategy.decompose(node, graph, first)
                    suffix = first.compute_dtype.replace("fp", "f")
                    self.assertTrue(
                        any(kernel.kernel_name.endswith(f"_{suffix}") for kernel in kernels),
                        (node.id, first, [kernel.kernel_name for kernel in kernels]),
                    )

    def test_unsupported_low_precision_falls_back_safely(self) -> None:
        hardware = HardwareCapability(
            supports_fp16=True,
            supports_bf16=False,
            supports_fp8=False,
            supports_fp4=False,
        )
        strategy = Strategy(hardware=hardware, mode=ExecutionMode.MIXED_PRECISION)
        selected = {
            strategy.select_precision(node, graph).compute_dtype
            for graph in self.graphs
            for node in graph.nodes.values()
        }
        self.assertTrue(selected <= {"fp32", "fp16"})
        self.assertIn("fp16", selected)

        # FP4 alone is insufficient because the policy is explicitly W4A16.
        fp4_without_fp16 = HardwareCapability(
            supports_fp16=False,
            supports_bf16=False,
            supports_fp8=False,
            supports_fp4=True,
        )
        strategy = Strategy(
            hardware=fp4_without_fp16,
            mode=ExecutionMode.MIXED_PRECISION,
        )
        selected = {
            strategy.select_precision(node, graph).compute_dtype
            for graph in self.graphs
            for node in graph.nodes.values()
        }
        self.assertEqual(selected, {"fp32"})


if __name__ == "__main__":
    unittest.main()
