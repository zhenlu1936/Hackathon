"""C3.3 — Operator fusion and graph optimization.

Public API:

    from c33.pipeline import GraphPassPipeline
    pipeline = GraphPassPipeline(enable_fusion=True)
    results = pipeline.run(graph)
    # results['Fusion']['stats'] contains fusion_log, launch/buffer reductions
"""

from __future__ import annotations

from c33.pipeline import GraphPassPipeline

__all__ = ["GraphPassPipeline"]

