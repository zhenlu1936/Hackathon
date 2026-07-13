"""End-to-end checks for the connected C3.1 through C3.5 reference path."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import cupy as cp

from c31.import_onnx import import_onnx
from c35.executor import CrossStageReferencePipeline


ROOT = Path(__file__).resolve().parents[1]
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
                model_path = ROOT / "models" / f"{model}_v1.onnx"
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

    def test_planned_executor_rejects_plan_graph_mismatch(self) -> None:
        model_path = ROOT / "models" / "mlp_v1.onnx"
        pipeline = CrossStageReferencePipeline(
            import_onnx(str(model_path)), str(model_path)
        )
        feed = self._feed("mlp", count=1)
        plan = pipeline._plan_for_batch(1)
        removed_node = plan.kernel_steps[0].node_id
        plan.kernel_steps = [
            step for step in plan.kernel_steps if step.node_id != removed_node
        ]
        with self.assertRaisesRegex(ValueError, "plan/optimized-graph mismatch"):
            pipeline.optimized_executor.run_planned(feed, plan)


if __name__ == "__main__":
    unittest.main()
