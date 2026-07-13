"""End-to-end checks for the connected C3.1 through C3.5 reference path."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import cupy as cp

from c31.import_onnx import import_onnx
from c35.executor import CrossStageReferencePipeline


ROOT = Path(__file__).resolve().parents[1]
MODELS = (
    ROOT / ".specification" / "testcases" / "release_to_competitors" / "models"
)
TESTDATA = (
    ROOT / ".specification" / "testcases" / "release_to_competitors"
    / "testdata" / "c35"
)


class CrossStagePipelineTests(unittest.TestCase):
    def _feed(self, model: str, count: int = 2) -> dict[str, cp.ndarray]:
        input_dir = TESTDATA / f"{model}_v1" / "input"
        manifest = json.loads((input_dir / "manifest.json").read_text())
        return {
            entry["name"]: cp.load(input_dir / entry["file"])[:count]
            for entry in manifest["tensors"]
        }

    def test_public_models_use_optimized_graph_and_valid_plan(self) -> None:
        expected_fusions = {"mlp": 0, "resnet": 1, "transformer": 1}
        for model, minimum_fusions in expected_fusions.items():
            with self.subTest(model=model):
                model_path = MODELS / f"{model}_v1.onnx"
                pipeline = CrossStageReferencePipeline(
                    import_onnx(str(model_path)), str(model_path),
                    qualify_optimizations=True,
                )
                outputs = pipeline.run(self._feed(model))
                self.assertIn("logits", outputs)
                self.assertEqual(outputs["logits"].shape[0], 2)
                self.assertLessEqual(pipeline.qualification_max_abs_diff, 1e-3)
                self.assertIsNotNone(pipeline.last_plan)
                self.assertEqual(pipeline.last_plan.validate(), [])
                stats = pipeline.fusion_result["Fusion"]["stats"]
                self.assertGreaterEqual(stats["total_fusions"], minimum_fusions)
                self.assertEqual(
                    {step.node_id for step in pipeline.last_plan.kernel_steps},
                    set(pipeline.graph.nodes),
                )
                trace = pipeline.optimized_executor.last_execution_trace
                self.assertEqual(len(trace), len(pipeline.last_plan.timeline))
                self.assertEqual(
                    [entry["kind"] for entry in trace],
                    [action.kind for action in pipeline.last_plan.timeline],
                )
                trace_kinds = {entry["kind"] for entry in trace}
                self.assertTrue(
                    {
                        "ALLOC", "H2D", "EVENT_WAIT", "KERNEL",
                        "EVENT_RECORD", "D2H",
                    } <= trace_kinds
                )

    def test_planned_executor_rejects_plan_graph_mismatch(self) -> None:
        model_path = MODELS / "mlp_v1.onnx"
        pipeline = CrossStageReferencePipeline(
            import_onnx(str(model_path)), str(model_path)
        )
        feed = self._feed("mlp", count=1)
        plan = pipeline._plan_for_batch(1)
        plan.kernel_steps[0].node_id = "__unexpected_plan_node__"
        with self.assertRaisesRegex(ValueError, "plan/optimized-graph mismatch"):
            pipeline.optimized_executor.run_planned(feed, plan)

    def test_model_tensors_are_uploaded_once_across_batch_plans(self) -> None:
        model_path = MODELS / "mlp_v1.onnx"
        pipeline = CrossStageReferencePipeline(
            import_onnx(str(model_path)), str(model_path)
        )
        feed = self._feed("mlp", count=1)
        first = pipeline.run(feed)
        first_logits = first["logits"].copy()
        cp.cuda.get_current_stream().synchronize()
        first_trace = list(pipeline.optimized_executor.last_execution_trace)
        first_resource_stats = (
            pipeline.optimized_executor.runtime_resource_stats()
        )
        second = pipeline.run(self._feed("mlp", count=2))
        second_trace = pipeline.optimized_executor.last_execution_trace
        second_resource_stats = (
            pipeline.optimized_executor.runtime_resource_stats()
        )

        model_tensors = set(pipeline.last_plan.weight_slots)
        first_model_h2d = [
            action for action in first_trace
            if action["kind"] == "H2D"
            and action["tensor_name"] in model_tensors
        ]
        second_model_h2d = [
            action for action in second_trace
            if action["kind"] == "H2D"
            and action["tensor_name"] in model_tensors
        ]
        self.assertTrue(first_model_h2d)
        self.assertTrue(all(a["status"] == "executed" for a in first_model_h2d))
        self.assertTrue(second_model_h2d)
        self.assertTrue(all(a["status"] == "resident" for a in second_model_h2d))
        self.assertTrue(cp.allclose(first_logits, second["logits"][:1]))
        self.assertEqual(
            second_resource_stats["stream_objects_created"],
            first_resource_stats["stream_objects_created"],
        )
        self.assertEqual(second_resource_stats["event_plan_count"], 2)

    def test_cuda_runtime_resources_are_reused_across_batches(self) -> None:
        model_path = MODELS / "mlp_v1.onnx"
        pipeline = CrossStageReferencePipeline(
            import_onnx(str(model_path)), str(model_path)
        )
        feed = self._feed("mlp", count=2)

        first = pipeline.run(feed)["logits"].copy()
        cp.cuda.get_current_stream().synchronize()
        first_stats = pipeline.optimized_executor.runtime_resource_stats()
        first_stream_objects = dict(pipeline.optimized_executor._streams)

        second = pipeline.run(feed)["logits"].copy()
        cp.cuda.get_current_stream().synchronize()
        second_stats = pipeline.optimized_executor.runtime_resource_stats()

        self.assertTrue(cp.allclose(first, second))
        self.assertGreater(first_stats["stream_objects_created"], 0)
        self.assertEqual(
            second_stats["stream_objects_created"],
            first_stats["stream_objects_created"],
        )
        self.assertEqual(
            second_stats["event_objects_created"],
            first_stats["event_objects_created"],
        )
        self.assertEqual(
            pipeline.optimized_executor._streams,
            first_stream_objects,
        )
        trace = pipeline.optimized_executor.memory_trace
        self.assertEqual(len(trace), 2)
        self.assertEqual(second_stats["memory_trace_dropped"], 0)
        self.assertEqual(trace[0]["batch_size"], 2)
        self.assertEqual(trace[1]["batch_size"], 2)
        self.assertEqual(
            trace[0]["logical_stream_count"],
            trace[1]["logical_stream_count"],
        )


if __name__ == "__main__":
    unittest.main()
