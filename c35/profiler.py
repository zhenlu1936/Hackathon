"""C3.5 deep profiler — instruments the full pipeline end-to-end.

Produces a profile report annotating every stage (parse, plan build, H2D,
kernel execution, D2H) with wall-clock timings, CuPy pool snapshots, and
per-operator breakdowns.

Usage — CLI (standalone):
    python3 -m c35.profiler \\
        --onnx models/resnet_v1.onnx \\
        --input .specification/testcases/release_to_competitors/testdata/c35/resnet_v1/input \\
        --output /tmp/profiled_resnet \\
        --batch-size 256 \\
        --report profile-resnet.json

Usage — programmatic:
    from c35.profiler import profiled_load_and_infer
    info, report = profiled_load_and_infer(model_path, input_dir, output_dir, batch_size=256)
    report.print_summary()
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import cupy as cp

from c35.instrument import (
    KernelTrace,
    ProfileReport,
    StageTimer,
    pool_snapshot,
)


def _describe_shapes(inputs: List[Any]) -> str:
    parts: List[str] = []
    for inp in inputs:
        if inp is None:
            parts.append("None")
        elif hasattr(inp, "shape"):
            parts.append(str(tuple(inp.shape)))
        else:
            parts.append(type(inp).__name__)
    return ", ".join(parts)


def _install_kernel_profiling(
    executor: Any, trace: Optional[KernelTrace]
) -> Optional[Any]:
    """Install a scoped planned-node timing hook and return the original."""
    if trace is None:
        return None
    original = executor._execute_planned_node

    def profiled(node, kernel_step, arena, allocations):
        with trace.record(
            str(node.id), str(node.op_type),
            shapes=_describe_shapes([
                executor.values.get(name) for name in node.inputs if name
            ]),
        ):
            return original(node, kernel_step, arena, allocations)

    executor._execute_planned_node = profiled
    return original


def _remove_kernel_profiling(executor: Any, original: Optional[Any]) -> None:
    """Restore the original executor method."""
    if original is not None:
        executor._execute_planned_node = original


# ── Main profiling entry point ─────────────────────────────────────

def profiled_load_and_infer(
    model_path: str,
    input_dir: str,
    output_dir: str,
    batch_size: Optional[int] = None,
    profile_kernels: bool = True,
) -> tuple:
    """Run the full C3.1→C3.5 pipeline with instrumentation.

    Mirrors the logic in ``c35.executor.load_and_infer`` but interleaves
    StageTimer, KernelTrace, and pool snapshots around every major step.

    Returns (info_dict, ProfileReport).
    """
    from c31.import_onnx import import_onnx
    from c33.pipeline import GraphPassPipeline
    from c34.scheduler import ExecutionScheduler
    from c35 import engine
    from c35.engine import require_device
    from c35.executor import PlannedGraphExecutor

    require_device()

    model_name = os.path.splitext(os.path.basename(model_path))[0]
    report = ProfileReport(model_name=model_name, batch_size=batch_size or 0)
    kernel_trace = KernelTrace() if profile_kernels else None

    overall_start = time.perf_counter()

    # ── Stage 1: Parse ONNX ─────────────────────────────────────────
    parse_timer = StageTimer("parse")
    with parse_timer:
        graph = import_onnx(model_path)
    report.add_stage(parse_timer)

    # ── Stage 2: Load & validate inputs ─────────────────────────────
    io_timer = StageTimer("io_load")
    with io_timer:
        input_manifest_path = os.path.join(input_dir, "manifest.json")
        with open(input_manifest_path, "r") as f:
            input_manifest = json.load(f)
        input_arrays: Dict[str, cp.ndarray] = {}
        for entry in input_manifest.get("tensors", []):
            name = entry["name"]
            file_path = os.path.join(input_dir, entry["file"])
            input_arrays[name] = cp.load(file_path, allow_pickle=False)

        total_samples = next(iter(input_arrays.values())).shape[0]
        actual_batch: int = (
            min(batch_size, total_samples)
            if (batch_size is not None and batch_size > 0)
            else total_samples
        )
    report.add_stage(io_timer)

    # ── Stage 3: Fusion (C3.3) ──────────────────────────────────────
    fusion_timer = StageTimer("fusion")
    with fusion_timer:
        pre_nodes = len(graph.nodes)
        fusion_result = GraphPassPipeline().run(graph)
        post_nodes = len(graph.nodes)
        fusion_stats = fusion_result["Fusion"]["stats"]
    report.extra["fusion"] = {
        "nodes_before": pre_nodes,
        "nodes_after": post_nodes,
        "launches_before": fusion_stats.get("raw_launches", 0),
        "launches_after": fusion_stats.get("optimized_launches", 0),
        "buffers_before": fusion_stats.get("raw_buffers", 0),
        "buffers_after": fusion_stats.get("optimized_buffers", 0),
        "fusions_per_pattern": fusion_stats.get("fusions_per_pattern", {}),
    }
    report.add_stage(fusion_timer)

    # ── Stage 4: Build execution plan (C3.4) ────────────────────────
    plan_timer = StageTimer("plan_build")
    with plan_timer:
        scheduler = ExecutionScheduler(graph, batch_size=actual_batch)
        plan = scheduler.build()

        # Count kernels by op type
        kernel_by_op: Dict[str, int] = {}
        for ks in plan.kernel_steps:
            node = graph.nodes.get(ks.node_id)
            if node is not None:
                kernel_by_op[node.op_type] = kernel_by_op.get(node.op_type, 0) + 1

        report.extra["plan"] = {
            "batch_size": actual_batch,
            "num_kernels": len(plan.kernel_steps),
            "num_allocations": len(plan.allocations),
            "num_transfers": len(plan.transfers),
            "num_events": len(plan.events),
            "num_timeline_steps": len(plan.timeline),
            "peak_reserved_bytes": plan.pool_stats.peak_reserved_bytes,
            "requested_bytes": plan.pool_stats.requested_bytes,
            "internal_fragmentation": plan.pool_stats.internal_fragmentation,
            "reuse_hits": plan.pool_stats.reuse_hits,
            "free_list_blocks": plan.pool_stats.free_list_blocks,
            "num_compute_streams": plan.num_compute_streams,
            "kernels_by_op": kernel_by_op,
        }
    report.add_stage(plan_timer)

    # ── Stage 5: Create executor ────────────────────────────────────
    exec_init_timer = StageTimer("executor_init")
    with exec_init_timer:
        executor = PlannedGraphExecutor(graph, model_path)
        original_execute = _install_kernel_profiling(executor, kernel_trace)
    report.add_stage(exec_init_timer)

    # ── Stage 6: Inference (batched) ─────────────────────────────────
    infer_timer = StageTimer("inference")
    try:
        with infer_timer:
            pool_before = pool_snapshot()
            report.add_pool_sample("before_inference",
                                   pool_before.get("used_bytes", 0),
                                   pool_before.get("total_bytes", 0))

            t_infer = time.perf_counter()
            output_name = graph.outputs[0].name if graph.outputs else "logits"
            final_device_output: Optional[cp.ndarray] = None
            num_batches = (total_samples + actual_batch - 1) // actual_batch
            # Cache plans per batch size — the last partial batch may differ.
            plan_cache: Dict[int, Any] = {actual_batch: plan}

            for start in range(0, total_samples, actual_batch):
                end = min(start + actual_batch, total_samples)
                this_batch_size = end - start
                batch_feed: Dict[str, cp.ndarray] = {}
                for name, arr in input_arrays.items():
                    batch_feed[name] = engine.to_device(arr[start:end])

                if this_batch_size not in plan_cache:
                    plan_cache[this_batch_size] = (
                        ExecutionScheduler(graph, batch_size=this_batch_size).build()
                    )
                batch_plan = plan_cache[this_batch_size]

                batch_outputs = executor.run_planned(batch_feed, batch_plan)
                batch_output = batch_outputs[output_name]

                if final_device_output is None:
                    final_shape = (total_samples,) + tuple(batch_output.shape[1:])
                    final_device_output = cp.empty(final_shape, dtype=cp.float32)
                cp.copyto(final_device_output[start:end], batch_output)

            engine.synchronize()
            infer_elapsed = time.perf_counter() - t_infer

            pool_after = pool_snapshot()
            report.add_pool_sample("after_inference",
                                   pool_after.get("used_bytes", 0),
                                   pool_after.get("total_bytes", 0))

        report.extra["inference_wall_s"] = infer_elapsed
        report.extra["num_batches"] = num_batches
        report.extra["pool_delta_bytes"] = (
            pool_after.get("total_bytes", 0) - pool_before.get("total_bytes", 0)
        )
    finally:
        _remove_kernel_profiling(executor, original_execute)

    report.add_stage(infer_timer)

    # ── Stage 7: Write outputs ──────────────────────────────────────
    write_timer = StageTimer("write_outputs")
    with write_timer:
        os.makedirs(output_dir, exist_ok=True)
        output_file = "logits.npy"
        output_path = os.path.join(output_dir, output_file)
        if final_device_output is not None:
            cp.save(output_path, final_device_output)
        output_manifest = {
            "tensors": [{
                "name": output_name,
                "file": output_file,
                "dtype": "float32",
                "shape": list(final_device_output.shape) if final_device_output is not None else [],
            }]
        }
        with open(os.path.join(output_dir, "manifest.json"), "w") as f:
            json.dump(output_manifest, f, indent=2)
    report.add_stage(write_timer)

    # ── Finalise ────────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - overall_start
    report.extra["total_wall_s"] = total_elapsed

    if kernel_trace is not None:
        report.add_kernel_trace(kernel_trace)

    # Per-batch kernel stats (first batch vs average of rest)
    if kernel_trace is not None and kernel_trace.records:
        all_records = kernel_trace.records
        kernels_per_batch = (
            report.extra["plan"]["num_kernels"]
            if "plan" in report.extra else len(all_records) // max(1, num_batches)
        )
        if kernels_per_batch > 0 and len(all_records) >= kernels_per_batch:
            first_batch = all_records[:kernels_per_batch]
            first_total = sum(r["wall_ns"] for r in first_batch)
            rest_total = sum(r["wall_ns"] for r in all_records[kernels_per_batch:])
            rest_batches = max(1, num_batches - 1)
            report.extra["kernel_batch_stats"] = {
                "kernels_per_batch": kernels_per_batch,
                "first_batch_wall_s": first_total / 1e9,
                "avg_rest_batch_wall_s": (rest_total / rest_batches) / 1e9,
                "first_batch_overhead_ratio": (
                    (first_total / 1e9) / max(1e-9, (rest_total / rest_batches) / 1e9)
                ),
            }

    info: Dict[str, Any] = {
        "parse_time_s": parse_timer.elapsed_s,
        "compile_time_s": (fusion_timer.elapsed_s + plan_timer.elapsed_s
                           + exec_init_timer.elapsed_s),
        "infer_time_s": infer_elapsed,
        "total_time_s": total_elapsed,
        "samples": total_samples,
        "output_shape": (list(final_device_output.shape)
                         if final_device_output is not None else []),
        "backend": "connected C3 cupy on AEC H200",
    }

    return info, report


# ── Standalone CLI ──────────────────────────────────────────────────

def _validate_batch_size(value: str) -> int:
    try:
        bs = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"batch-size must be an integer, got '{value}'"
        )
    if bs <= 0:
        raise argparse.ArgumentTypeError(f"batch-size must be positive, got {bs}")
    return bs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="C3.5 Deep Profiler — instrument the full inference pipeline"
    )
    parser.add_argument("--onnx", required=True, help="Path to ONNX model")
    parser.add_argument("--input", required=True,
                        help="Directory with manifest.json + .npy inputs")
    parser.add_argument("--output", required=True, help="Directory to write outputs")
    parser.add_argument("--batch-size", type=_validate_batch_size, default=None,
                        help="Batch size (default: all samples)")
    parser.add_argument("--report", default=None,
                        help="Path to write JSON profile report (default: <output>/profile.json)")
    parser.add_argument("--no-kernel-profile", action="store_true",
                        help="Disable per-kernel timing (faster)")
    args = parser.parse_args()

    report_path = args.report or os.path.join(args.output, "profile.json")

    print("C3.5 Deep Profiler", file=sys.stderr)
    print(f"  Model:       {args.onnx}", file=sys.stderr)
    print(f"  Input:       {args.input}", file=sys.stderr)
    print(f"  Output:      {args.output}", file=sys.stderr)
    print(f"  Batch size:  {args.batch_size or 'all'}", file=sys.stderr)
    print(f"  Report:      {report_path}", file=sys.stderr)
    print(file=sys.stderr)

    info, report = profiled_load_and_infer(
        args.onnx, args.input, args.output,
        batch_size=args.batch_size,
        profile_kernels=not args.no_kernel_profile,
    )

    # Print summary
    print(file=sys.stderr)
    print("Inference complete:", file=sys.stderr)
    print(f"  Samples:     {info['samples']}", file=sys.stderr)
    print(f"  Output shape:{info['output_shape']}", file=sys.stderr)
    print(f"  Parse time:  {info['parse_time_s']:.3f}s", file=sys.stderr)
    print(f"  Infer time:  {info['infer_time_s']:.3f}s", file=sys.stderr)
    print(f"  Total time:  {info['total_time_s']:.3f}s", file=sys.stderr)
    print(f"  Backend:     {info['backend']}", file=sys.stderr)

    fusion_info = report.extra.get("fusion", {})
    print(f"  Fusion:      {fusion_info.get('nodes_before','?')} -> "
          f"{fusion_info.get('nodes_after','?')} nodes", file=sys.stderr)

    plan_info = report.extra.get("plan", {})
    print(f"  C3.4 plan:   {plan_info.get('num_kernels','?')} kernels, "
          f"{plan_info.get('num_allocations','?')} allocations", file=sys.stderr)

    # Print human-readable profile summary
    report.print_summary()

    # Write JSON report
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report.json())
    print(f"Profile report written to {report_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
