"""Regression tests for the executable C3.4 timeline contract."""

from __future__ import annotations

import copy
from pathlib import Path
import unittest

from c31.import_onnx import import_onnx
from c34.scheduler import ExecutionScheduler


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / ".specification" / "testcases" / "release_to_competitors" / "models"


class ExecutablePlanTests(unittest.TestCase):
    def _plan(self, model: str):
        graph = import_onnx(str(MODELS / f"{model}_v1.onnx"))
        return ExecutionScheduler(graph, batch_size=1).build()

    def test_public_plans_have_complete_executable_timelines(self) -> None:
        for model in ("mlp", "resnet", "transformer"):
            with self.subTest(model=model):
                plan = self._plan(model)
                self.assertEqual(plan.validate(), [])
                transfer_refs = [
                    action.ref_index for action in plan.timeline
                    if action.kind in {"H2D", "D2H"}
                ]
                kernel_refs = [
                    action.ref_index for action in plan.timeline
                    if action.kind == "KERNEL"
                ]
                self.assertEqual(sorted(transfer_refs), list(range(len(plan.transfers))))
                self.assertEqual(sorted(kernel_refs), list(range(len(plan.kernel_steps))))

    def test_dynamic_int64_input_uses_dtype_aware_capacity(self) -> None:
        graph = import_onnx(str(MODELS / "transformer_v1.onnx"))
        plan = ExecutionScheduler(graph, batch_size=2).build()
        input_allocation = next(
            allocation for allocation in plan.allocations
            if allocation.tensor_name == "input_ids"
        )
        required_bytes = 2 * 18 * 8
        self.assertEqual(input_allocation.size_bytes, required_bytes)
        self.assertGreaterEqual(input_allocation.capacity_bytes, required_bytes)

    def test_cross_stream_waits_have_real_producer_records(self) -> None:
        plan = self._plan("transformer")
        signalled = {
            transfer.event_id for transfer in plan.transfers
            if transfer.event_id is not None
        } | {
            event_id for kernel in plan.kernel_steps for event_id in kernel.signals
        }
        waited = {
            event_id for kernel in plan.kernel_steps for event_id in kernel.depends_on
        } | {
            event_id for transfer in plan.transfers for event_id in transfer.depends_on
        }
        self.assertTrue(waited)
        self.assertLessEqual(waited, signalled)
        cross_events = {
            event.event_id for event in plan.events
            if event.event_id.startswith("evt_xs_")
        }
        kernel_signals = {
            event_id for kernel in plan.kernel_steps for event_id in kernel.signals
        }
        self.assertTrue(cross_events)
        self.assertLessEqual(cross_events, kernel_signals)

    def test_validator_rejects_wait_without_producer_signal(self) -> None:
        plan = copy.deepcopy(self._plan("mlp"))
        waited_event = next(
            event_id for kernel in plan.kernel_steps
            for event_id in kernel.depends_on
        )
        for transfer in plan.transfers:
            if transfer.event_id == waited_event:
                transfer.event_id = None
        for kernel in plan.kernel_steps:
            kernel.signals = [
                event_id for event_id in kernel.signals
                if event_id != waited_event
            ]
        issues = plan.validate()
        self.assertTrue(any("never signalled" in issue for issue in issues), issues)

    def test_weight_prefetch_is_interleaved_with_compute(self) -> None:
        plan = self._plan("transformer")
        first_kernel_action = next(
            action.step_index for action in plan.timeline if action.kind == "KERNEL"
        )
        staged_after_compute = [
            action for action in plan.timeline
            if action.kind == "H2D"
            and action.step_index > first_kernel_action
            and plan.transfers[action.ref_index].tensor_name in plan.weight_slots
        ]
        self.assertTrue(staged_after_compute)
        kernel_position = {
            action.ref_index: action.step_index for action in plan.timeline
            if action.kind == "KERNEL"
        }
        for action in staged_after_compute[:10]:
            transfer = plan.transfers[action.ref_index]
            first_use = plan.lifetimes[transfer.tensor_name].first_use
            self.assertLess(action.step_index, kernel_position[first_use])

    def test_cross_stream_arena_reuse_has_happens_before_event(self) -> None:
        plan = self._plan("transformer")
        reuse_events = {
            event.event_id for event in plan.events
            if event.event_id.startswith("evt_reuse_")
        }
        self.assertTrue(reuse_events)
        signals = {
            event_id for kernel in plan.kernel_steps for event_id in kernel.signals
        }
        waits = {
            event_id for kernel in plan.kernel_steps for event_id in kernel.depends_on
        }
        self.assertLessEqual(reuse_events, signals)
        self.assertLessEqual(reuse_events, waits)


if __name__ == "__main__":
    unittest.main()
