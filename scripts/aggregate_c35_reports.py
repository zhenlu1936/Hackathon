#!/usr/bin/env python3
"""Aggregate repeated C3.5 standard reports using arithmetic means."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any


TIMING_PATTERNS = {
    "samples": r"^\s*Samples:\s+(\d+)",
    "parse_time_s": r"^\s*Parse time:\s+([0-9.]+)s",
    "infer_time_s": r"^\s*Infer time:\s+([0-9.]+)s",
    "total_time_s": r"^\s*Total time:\s+([0-9.]+)s",
    "fusion_nodes_before": r"^\s*Fusion:\s+(\d+)\s+->",
    "fusion_nodes_after": r"^\s*Fusion:\s+\d+\s+->\s+(\d+)",
    "plan_kernels": r"^\s*C3\.4 plan:\s+(\d+)\s+kernels",
    "plan_allocations": r"^\s*C3\.4 plan:\s+\d+\s+kernels,\s+(\d+)\s+allocations",
}

TOP_LEVEL_METRICS = (
    "wall_time_s",
    "peak_gpu_memory_bytes",
    "accuracy",
    "max_abs_diff",
)

BACKEND_METRICS = (
    "batch_count",
    "max_plan_arena_bytes",
    "pool_reserved_bytes",
    "pool_reserved_bytes_after_last_batch",
    "pool_reserved_bytes_before_first_batch",
    "pool_used_bytes",
    "runtime_event_objects",
    "runtime_stream_objects",
)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _extract_timings(stderr_tail: str) -> dict[str, float]:
    extracted: dict[str, float] = {}
    for name, pattern in TIMING_PATTERNS.items():
        match = re.search(pattern, stderr_tail, re.MULTILINE)
        if match:
            extracted[name] = float(match.group(1))
    return extracted


def _collect(result: dict[str, Any]) -> dict[str, float | None]:
    metrics = {name: _number(result.get(name)) for name in TOP_LEVEL_METRICS}
    backend = result.get("backend_evidence") or {}
    metrics.update(
        {name: _number(backend.get(name)) for name in BACKEND_METRICS}
    )
    metrics.update(_extract_timings(result.get("stderr_tail", "")))
    return metrics


def _mean(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    if len(present) != len(values):
        raise ValueError("a numeric metric is missing from only some input reports")
    return statistics.fmean(present)


def aggregate(paths: list[Path]) -> dict[str, Any]:
    if len(paths) < 2:
        raise ValueError("at least two C3.5 reports are required")

    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    batch_sizes = {report.get("batch_size") for report in reports}
    if len(batch_sizes) != 1:
        raise ValueError(f"batch sizes differ: {sorted(batch_sizes)}")

    by_run = []
    expected_models: list[str] | None = None
    for report in reports:
        results = {result["model"]: result for result in report["results"]}
        models = sorted(results)
        if expected_models is None:
            expected_models = models
        elif models != expected_models:
            raise ValueError("input reports contain different model sets")
        by_run.append(results)

    results = []
    assert expected_models is not None
    for model in expected_models:
        model_runs = [run[model] for run in by_run]
        samples = [_collect(result) for result in model_runs]
        metric_names = sorted({name for sample in samples for name in sample})
        means = {name: _mean([sample.get(name) for sample in samples]) for name in metric_names}
        results.append(
            {
                "model": model,
                "run_count": len(model_runs),
                "runs_passed": sum(bool(result.get("passed")) for result in model_runs),
                "all_passed": all(bool(result.get("passed")) for result in model_runs),
                "mean": means,
                "source_values": {
                    name: [sample.get(name) for sample in samples]
                    for name in metric_names
                },
            }
        )

    preflight = reports[0].get("cupy_preflight", {})
    device_names = {
        report.get("cupy_preflight", {}).get("device_name") for report in reports
    }
    if len(device_names) != 1:
        raise ValueError("input reports were produced on different devices")

    return {
        "format_version": "c35-mean-1.0",
        "aggregation": {
            "method": "arithmetic mean",
            "run_count": len(reports),
            "source_reports": [path.name for path in paths],
        },
        "batch_size": reports[0].get("batch_size"),
        "device": {
            "name": preflight.get("device_name"),
            "cupy_version": preflight.get("cupy_version"),
            "cuda_runtime_version": preflight.get("cuda_runtime_version"),
            "device_id": preflight.get("device_id"),
        },
        "all_passed": all(result["all_passed"] for result in results),
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    report = aggregate(args.reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.output} from {len(args.reports)} reports")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
