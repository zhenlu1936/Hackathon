"""Independent regression checks for issues hidden by the self-score scripts."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest

import cupy as cp

import c32.api as c32_api
from c31.import_onnx import import_onnx
from c32.hardware import HardwareCapability, set_hardware
from c32.kernel_spec import ProblemSize
from c33.fusion import fuse_conv_batchnorm, fuse_elementwise_chain
from c33.test_c33 import _make_conv_bn_graph, _make_ew_chain_graph
from c34.scheduler import ExecutionScheduler
from c35.engine import execute_op, op_conv


MODELS = (
    Path(__file__).resolve().parents[1] / ".specification" / "testcases"
    / "release_to_competitors" / "models"
)


class ScoringRegressionTests(unittest.TestCase):
    def test_direct_hardware_switch_updates_public_references(self) -> None:
        old = replace(c32_api.hardware)
        try:
            tiny = HardwareCapability(
                name="tiny", max_threads_per_block=32, smem_bytes=1024,
                supports_fp16=False, supports_winograd=False,
            )
            set_hardware(tiny)
            self.assertEqual(c32_api.hardware.name, "tiny")
            self.assertEqual(c32_api.strategy.hardware.name, "tiny")

            graph = import_onnx(str(MODELS / "mlp_v1.onnx"))
            node = next(n for n in graph.nodes.values() if n.op_type == "Gemm")
            precision = c32_api.strategy.select_precision(node, graph)
            ref = c32_api.strategy.decompose(node, graph, precision)[0]
            tuning = c32_api.strategy.tune_kernel(
                ref, precision, ProblemSize(m=17, n=19, k=23)
            )
            self.assertTrue(tuning.is_valid(32, 1024))
        finally:
            set_hardware(old)

    def test_elementwise_fusion_preserves_external_operands(self) -> None:
        graph = _make_ew_chain_graph()
        self.assertEqual(fuse_elementwise_chain(graph, []), 1)
        fused = next(n for n in graph.nodes.values() if n.op_type == "FusedEWChain")
        self.assertEqual(fused.inputs, ["input", "c1", "c2"])
        for name in fused.inputs:
            self.assertIn(fused.id, graph.tensor_consumers[name])
        x = cp.array([[-2.0, 1.0]], dtype=cp.float32)
        c1 = cp.array([[3.0, 4.0]], dtype=cp.float32)
        c2 = cp.array([[10.0, 20.0]], dtype=cp.float32)
        actual = execute_op(fused.op_type, [x, c1, c2], fused.attributes)
        expected = cp.maximum((x + c1) * c2, 0)
        cp.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    def test_conv_bn_fusion_is_numerically_executable(self) -> None:
        graph = _make_conv_bn_graph()
        self.assertEqual(fuse_conv_batchnorm(graph, []), 1)
        fused = next(n for n in graph.nodes.values() if n.op_type == "FusedConv2dBatchNorm")
        self.assertEqual(fused.inputs[-4:], ["bn_scale", "bn_bias", "bn_mean", "bn_var"])

        # Deterministic ramps avoid depending on random.Generator methods that
        # are not uniformly available in the organizer's CuPy 14.1.1 build.
        x = cp.linspace(-1.0, 1.0, 75, dtype=cp.float32).reshape(1, 3, 5, 5)
        weight = cp.linspace(
            -0.5, 0.5, 16 * 3 * 3 * 3, dtype=cp.float32
        ).reshape(16, 3, 3, 3)
        scale = cp.linspace(0.5, 1.5, 16, dtype=cp.float32)
        bias = cp.linspace(-0.2, 0.2, 16, dtype=cp.float32)
        mean = cp.linspace(-0.1, 0.1, 16, dtype=cp.float32)
        var = cp.linspace(0.1, 1.6, 16, dtype=cp.float32)
        actual = execute_op(
            fused.op_type, [x, weight, scale, bias, mean, var], fused.attributes
        )
        conv = op_conv([x, weight], fused.attributes)
        expected = (
            (conv - mean.reshape(1, -1, 1, 1))
            / cp.sqrt(var.reshape(1, -1, 1, 1) + 1e-5)
            * scale.reshape(1, -1, 1, 1)
            + bias.reshape(1, -1, 1, 1)
        )
        cp.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    def test_all_public_execution_plans_have_complete_bindings(self) -> None:
        for model in ("mlp", "resnet", "transformer"):
            with self.subTest(model=model):
                model_path = MODELS / f"{model}_v1.onnx"
                plan = ExecutionScheduler(import_onnx(str(model_path))).build()
                self.assertEqual(plan.validate(), [])
                transfer_events = {
                    t.event_id for t in plan.transfers if t.event_id is not None
                }
                waited_events = {
                    event for step in plan.kernel_steps for event in step.depends_on
                    if event.startswith(("evt_wready_", "evt_input_ready_"))
                }
                self.assertTrue(waited_events <= transfer_events)


if __name__ == "__main__":
    unittest.main()
