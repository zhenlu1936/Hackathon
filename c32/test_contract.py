"""Dependency-light C3.2 contract regressions.

These tests construct the shared IR directly, so decomposition, tuning, and
hardware-query logic remain testable on development hosts without ONNX/CUDA.
"""

from __future__ import annotations

import unittest

from c3common.ir.graph import Graph, Node, ONNSType
from c32.hardware import HardwareCapability
from c32.kernel_spec import KernelSpecRef, PrecisionProfile
from c32.strategy import ExecutionMode, Strategy


def _graph_with_tensors(**shapes: list[int]) -> Graph:
    graph = Graph()
    for name, shape in shapes.items():
        graph.register_tensor(name, dtype=ONNSType.FLOAT, shape=shape)
    return graph


class C32ContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.hardware = HardwareCapability(
            name="contract fixture",
            max_threads_per_block=128,
            max_block_dim=64,
            max_grid_dim=17,
            smem_bytes=8192,
            smem_bytes_total=16384,
            max_shared_memory_per_block=8192,
            supports_fp16=True,
            supports_bf16=True,
            supports_fp8=True,
            supports_fp4=True,
            verified=True,
            source="unit test",
        )
        self.strategy = Strategy(
            hardware=self.hardware, mode=ExecutionMode.FULL_FP32,
        )

    def test_hopper_properties_are_queried_conservatively(self) -> None:
        profile = HardwareCapability.from_cuda_properties({
            "name": b"fixture H200",
            "major": 9,
            "minor": 0,
            "maxThreadsPerBlock": 1024,
            "maxThreadsDim": (1024, 1024, 64),
            "maxGridSize": (2147483647, 65535, 65535),
            "sharedMemPerBlockOptin": 227328,
            "sharedMemPerMultiprocessor": 233472,
        })
        self.assertTrue(profile.verified)
        self.assertEqual(profile.compute_capability, (9, 0))
        self.assertTrue(profile.supports_fp8)
        self.assertFalse(profile.supports_fp4)
        self.assertEqual(profile.max_threads_per_block, 1024)

    def test_layernorm_three_outputs_form_a_connected_sequence(self) -> None:
        graph = _graph_with_tensors(
            x=[2, 4, 8], scale=[8], bias=[8], y=[2, 4, 8],
            mean=[2, 4, 1], inv_std=[2, 4, 1],
        )
        node = Node(
            id="norm", name="", op_type="LayerNorm",
            inputs=["x", "scale", "bias"],
            outputs=["y", "mean", "inv_std"],
            attributes={"axis": -1, "epsilon": 1e-6},
        )
        kernels = self.strategy.decompose(node, graph)
        names = [kernel.kernel_name for kernel in kernels]
        self.assertIn("reciprocal", names)
        self.assertTrue({"y", "mean", "inv_std"} <= {
            output for kernel in kernels for output in kernel.outputs
        })
        self.assertTrue(all(kernel.precision_profile is not None for kernel in kernels))

    def test_conv_strategy_uses_semantic_guards_and_infers_kernel_shape(self) -> None:
        graph = _graph_with_tensors(
            x=[1, 16, 8, 8], w=[32, 16, 3, 3], y=[1, 32, 8, 8],
        )
        eligible = Node(
            id="conv", name="", op_type="Conv2d",
            inputs=["x", "w"], outputs=["y"],
            attributes={"pads": [1, 1, 1, 1]},
        )
        kernels = self.strategy.decompose(eligible, graph)
        self.assertTrue(kernels[0].kernel_name.startswith("winograd_forward_"))

        grouped = Node(
            id="grouped", name="", op_type="Conv",
            inputs=["x", "w"], outputs=["y"],
            attributes={
                "kernel_shape": [3, 3], "pads": [1, 1, 1, 1],
                "group": 2,
            },
        )
        kernels = self.strategy.decompose(grouped, graph)
        self.assertTrue(kernels[0].kernel_name.startswith("im2col_"))
        self.assertEqual(kernels[1].operator_params["lowering_kind"], "conv_contract")
        self.assertNotIn("y", kernels[1].outputs)
        self.assertEqual(kernels[-1].outputs, ["y"])
        self.assertTrue(kernels[-1].kernel_name.startswith("conv_reshape_"))

    def test_linear_alias_emits_a_recognizable_matmul(self) -> None:
        graph = _graph_with_tensors(x=[4, 8], w=[8, 16], y=[4, 16])
        node = Node(
            id="linear", name="", op_type="Linear",
            inputs=["x", "w"], outputs=["y"],
        )
        kernels = self.strategy.decompose(node, graph)
        self.assertTrue(kernels[0].kernel_name.startswith("matmul_"))

    def test_tuning_enforces_all_declared_limits(self) -> None:
        profile = PrecisionProfile()
        refs = [
            KernelSpecRef("matmul_f32", inputs=["a", "b"], outputs=["c"]),
            KernelSpecRef("im2col_f32", inputs=["x"], outputs=["col"]),
            KernelSpecRef("reduce_sum", inputs=["x"], outputs=["y"]),
            KernelSpecRef("add", inputs=["x", "z"], outputs=["y"]),
        ]
        problem = {
            "m": 1_000_000, "n": 1_000_000, "k": 1_000_000,
            "elements": 1_000_000, "num_tiles": 1_000_000,
        }
        for ref in refs:
            params = self.strategy.tune_kernel(ref, profile, problem)
            self.assertTrue(params.is_valid(
                self.hardware.max_threads_per_block,
                self.hardware.smem_bytes,
                self.hardware.max_block_dim,
                self.hardware.max_grid_dim,
            ), (ref.kernel_name, params))

    def test_process_graph_rebuilds_missing_topological_order(self) -> None:
        graph = _graph_with_tensors(x=[2, 8], y=[2, 8])
        node = Node(
            id="relu", name="", op_type="Relu",
            inputs=["x"], outputs=["y"],
        )
        graph.add_node(node)
        graph.set_producer("x", "INPUT")
        graph.set_producer("y", node.id)
        self.assertEqual(graph.node_order, [])
        result = self.strategy.process_graph(graph)
        self.assertEqual(graph.node_order, [node.id])
        self.assertIn(node.id, result)
        self.assertIsNotNone(result[node.id]["kernels"][0].tuning_params)

    def test_sensitive_low_precision_and_disconnected_plans_are_rejected(self) -> None:
        graph = _graph_with_tensors(x=[2, 8], y=[2, 8])
        softmax = Node(
            id="softmax", name="", op_type="Softmax",
            inputs=["x"], outputs=["y"],
        )
        with self.assertRaisesRegex(ValueError, "must decompose in fp32"):
            self.strategy.decompose(
                softmax, graph, PrecisionProfile(
                    compute_dtype="fp16", accumulator_dtype="fp32",
                    input_dtype="fp16", output_dtype="fp32",
                ),
            )

        with self.assertRaisesRegex(ValueError, "unresolved tensors"):
            Strategy.validate_decomposition(softmax, [
                KernelSpecRef("bad", inputs=["missing"], outputs=["y"]),
            ])


if __name__ == "__main__":
    unittest.main()
