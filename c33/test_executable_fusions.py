"""Dependency-light regressions for executable C3.3 released-model fusions."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import unittest

from c31.import_onnx import import_onnx
from c32.strategy import ExecutionMode, Strategy
from c33.pipeline import GraphPassPipeline, _count_launches
from c33.fusion import fuse_matmul_bias
from c33.test_c33 import _make_matmul_bias_graph
from c34.scheduler import ExecutionScheduler


MODELS = (
    Path(__file__).resolve().parents[1] / ".specification" / "testcases"
    / "release_to_competitors" / "models"
)


class ExecutableFusionStructureTests(unittest.TestCase):
    def _optimized(self, model: str):
        graph = import_onnx(str(MODELS / f"{model}_v1.onnx"))
        result = GraphPassPipeline().run(graph)
        return graph, result["Fusion"]["stats"]

    def test_released_model_reductions_report_qualified_lowerings(self) -> None:
        for model in ("mlp", "transformer"):
            with self.subTest(model=model):
                graph, stats = self._optimized(model)
                self.assertTrue(stats["validation_passed"])
                self.assertGreaterEqual(stats["launch_reduction"], 0.60)
                self.assertGreaterEqual(stats["buffer_reduction"], 0.60)
                graph.validate()

        # ResNet deliberately exposes the multi-stage BLAS fallback after the
        # unqualified direct kernel regressed H200 wall time.  Do not restore a
        # false one-launch claim merely to satisfy the structural anchor.
        graph, stats = self._optimized("resnet")
        self.assertTrue(stats["validation_passed"])
        self.assertLess(stats["launch_reduction"], 0.60)
        self.assertLess(stats["buffer_reduction"], 0.60)
        graph.validate()

    def test_qualified_single_launch_nodes_have_one_c32_kernel(self) -> None:
        strategy = Strategy(mode=ExecutionMode.FULL_FP32)
        expected = {
            "FusedAttentionScores", "FusedGemmEpilogue",
            "FusedLayerNormalization", "FusedTransposeReshape",
        }
        observed = set()
        for model in ("mlp", "resnet", "transformer"):
            graph, _ = self._optimized(model)
            for node in graph.nodes.values():
                if node.op_type not in expected:
                    continue
                observed.add(node.op_type)
                lowering = strategy.decompose(node, graph)
                self.assertEqual(len(lowering), 1)
                self.assertEqual(lowering[0].inputs, node.inputs)
                self.assertEqual(lowering[0].outputs, node.outputs)
        self.assertEqual(observed, expected)

    def test_conv_epilogues_advertise_blas_fallback_launches(self) -> None:
        graph, _ = self._optimized("resnet")
        strategy = Strategy(mode=ExecutionMode.FULL_FP32)
        expected_last = {
            "FusedConvActivation": "relu_f32",
            "FusedConvResidualActivation": "fused_residual_relu_f32",
        }
        observed = set()
        for node in graph.nodes.values():
            if node.op_type not in expected_last:
                continue
            observed.add(node.op_type)
            lowering = strategy.decompose(node, graph)
            names = [kernel.kernel_name for kernel in lowering]
            self.assertEqual(names[:-1], [
                "im2col_f32", "matmul_f32", "conv_reshape_f32",
                "add_bias_f32",
            ])
            self.assertEqual(names[-1], expected_last[node.op_type])
            self.assertNotIn("fused_conv_activation_f32", names)
            self.assertNotIn("fused_conv_residual_activation_f32", names)
        self.assertEqual(observed, set(expected_last))

    def test_optimized_graphs_build_complete_execution_plans(self) -> None:
        for model in ("mlp", "resnet", "transformer"):
            with self.subTest(model=model):
                graph, _ = self._optimized(model)
                plan = ExecutionScheduler(graph, batch_size=2).build()
                self.assertEqual(plan.validate(), [])
                self.assertEqual(
                    {step.node_id for step in plan.kernel_steps},
                    set(graph.nodes),
                )

    def test_mlp_absorbs_flatten_bias_and_relu_semantics(self) -> None:
        graph, _ = self._optimized("mlp")
        fused = [
            node for node in graph.nodes.values()
            if node.op_type == "FusedGemmEpilogue"
        ]
        self.assertEqual(len(fused), 3)
        first = next(node for node in fused if "_flatten_axis" in node.attributes)
        self.assertEqual(first.inputs[0], "input")
        self.assertEqual(first.attributes["_flatten_axis"], 1)
        self.assertEqual(first.attributes["_activation"], "Relu")
        self.assertEqual(_count_launches(graph), 3)

    def test_resnet_keeps_shortcut_convs_explicit(self) -> None:
        graph, _ = self._optimized("resnet")
        counts = Counter(node.op_type for node in graph.nodes.values())
        self.assertEqual(counts["FusedConvActivation"], 9)
        self.assertEqual(counts["FusedConvResidualActivation"], 8)
        self.assertEqual(counts["Conv"], 3)
        for node in graph.nodes.values():
            if node.op_type == "FusedConvResidualActivation":
                index = node.attributes["_residual_input_index"]
                self.assertLess(index, len(node.inputs))
                self.assertNotIn(node.inputs[index], node.inputs[:index])

    def test_nonlast_softmax_blocks_attention_score_fusion(self) -> None:
        graph = import_onnx(str(MODELS / "transformer_v1.onnx"))
        first_softmax = next(
            node for node in graph.nodes.values() if node.op_type == "Softmax"
        )
        first_softmax.attributes["axis"] = 2
        GraphPassPipeline().run(graph)
        count = sum(
            node.op_type == "FusedAttentionScores"
            for node in graph.nodes.values()
        )
        self.assertEqual(count, 3)
        graph.validate()

    def test_constant_metadata_is_not_counted_as_a_launch(self) -> None:
        graph = import_onnx(str(MODELS / "transformer_v1.onnx"))
        constants = sum(node.op_type == "Constant" for node in graph.nodes.values())
        strategy = Strategy(mode=ExecutionMode.FULL_FP32)
        all_refs = sum(
            len(strategy.decompose(node, graph))
            for node in graph.nodes.values()
        )
        self.assertEqual(constants, 36)
        self.assertEqual(all_refs - _count_launches(graph), constants)

    def test_batched_matmul_weight_is_not_mislabeled_single_launch(self) -> None:
        graph = _make_matmul_bias_graph()
        graph.tensors["weight"].shape = [2, 16, 8]
        log = []
        self.assertEqual(fuse_matmul_bias(graph, log), 0)
        self.assertIn("rank-2 B", log[0]["rejection_reason"])
        graph.validate()


if __name__ == "__main__":
    unittest.main()
