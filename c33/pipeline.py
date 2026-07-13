"""GraphPassPipeline — orchestrates graph optimization passes for C3.3.

Usage:
    pipeline = GraphPassPipeline(enable_fusion=True)
    results = pipeline.run(graph)

    # Access fusion log
    log = results['Fusion']['stats']['fusion_log']

    # Access before/after counts
    stats = results['Fusion']['stats']
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List

from c3common.ir.graph import Graph

from c33.fusion import (
    fuse_matmul_bias,
    fuse_conv_batchnorm,
    fuse_elementwise_chain,
    fuse_softmax_dropout,
    fuse_residual_norm,
    fuse_gemm_epilogues,
    fuse_conv_epilogues,
    fuse_attention_scores,
    fuse_transpose_reshape,
    fuse_layer_normalization_kernels,
    cleanup_dead_tensors,
)


# ── Launch/buffer counting helpers ──────────────────────────────────


def _count_launches(graph: Graph) -> int:
    """Count the kernels emitted by the actual C3.2 lowering."""
    from c32.strategy import ExecutionMode, Strategy
    strategy = Strategy(mode=ExecutionMode.FULL_FP32)
    launches = 0
    for nid in graph.node_order:
        node = graph.nodes.get(nid)
        if node is None:
            continue
        for kernel in strategy.decompose(node, graph):
            # Constants are uploaded before graph execution.  The C3.5
            # executors return immediately for Constant nodes, so their
            # metadata KernelSpecRef is not a physical device launch.
            if kernel.kernel_name == "constant":
                continue
            launches += 1

    return max(launches, 1)


def _count_buffers(graph: Graph) -> int:
    """Count graph and C3.2-lowering intermediate buffers.

    Graph edges and named outputs internal to multi-kernel decompositions both
    require logical storage in the executable lowering.  The old graph-only
    count omitted Gemm/Conv/Softmax/LayerNorm workspaces and therefore did not
    use the same lowering model as launch counting.
    """
    from c32.strategy import ExecutionMode, Strategy
    input_names = {t.name for t in graph.inputs}
    output_names = {t.name for t in graph.outputs}

    buffers = 0
    for tname, tensor in graph.tensors.items():
        if tname in input_names:
            continue
        if tensor.is_initializer or tensor.is_constant:
            continue
        # Count as buffer if it's produced by a node and consumed by another node
        producer = graph.tensor_producer.get(tname)
        if producer is None:
            continue
        # Skip if only an output (final output, not intermediate)
        consumers = graph.tensor_consumers.get(tname, [])
        active_consumers = [c for c in consumers if c in graph.nodes]
        if not active_consumers and tname in output_names:
            continue
        buffers += 1

    strategy = Strategy(mode=ExecutionMode.FULL_FP32)
    for node_id in graph.node_order:
        node = graph.nodes.get(node_id)
        if node is None:
            continue
        node_outputs = {name for name in node.outputs if name}
        lowering_intermediates = {
            output
            for kernel in strategy.decompose(node, graph)
            for output in kernel.outputs
            if output and output not in node_outputs
        }
        buffers += len(lowering_intermediates)

    return max(buffers, 0)


# ── Pipeline passes ─────────────────────────────────────────────────


PASS_ORDER = [
    ("Conv2dBatchNorm", fuse_conv_batchnorm),
    ("MatMulBias", fuse_matmul_bias),
    ("ResidualNorm", fuse_residual_norm),
    ("SoftmaxDropout", fuse_softmax_dropout),
    ("AttentionScores", fuse_attention_scores),
    ("TransposeReshape", fuse_transpose_reshape),
    ("GemmEpilogue", fuse_gemm_epilogues),
    ("ConvEpilogue", fuse_conv_epilogues),
    ("LayerNormalizationKernel", fuse_layer_normalization_kernels),
    ("EWChain", fuse_elementwise_chain),
]


class GraphPassPipeline:
    """Orchestrate graph optimization passes.

    Args:
        enable_fusion: Enable fusion passes (default True).
        run_validation: Run graph validation after each pass (default True).
        enable_dead_cleanup: Remove dead tensors after passes (default True).
    """

    def __init__(
        self,
        enable_fusion: bool = True,
        run_validation: bool = True,
        enable_dead_cleanup: bool = True,
    ) -> None:
        self.enable_fusion = enable_fusion
        self.run_validation = run_validation
        self.enable_dead_cleanup = enable_dead_cleanup

    def run(self, graph: Graph) -> Dict[str, Any]:
        """Run the full optimization pipeline on a graph.

        Args:
            graph: The input graph to optimize (modified in-place).

        Returns:
            A dict with pass results and statistics:

            {
                'Fusion': {
                    'enabled': True/False,
                    'stats': {
                        'fusion_log': [...],
                        'raw_launches': int,
                        'optimized_launches': int,
                        'launch_reduction': float,
                        'raw_buffers': int,
                        'optimized_buffers': int,
                        'buffer_reduction': float,
                        'fusions_per_pattern': {...},
                        'node_count_before': int,
                        'node_count_after': int,
                        'validation_passed': bool,
                    }
                }
            }
        """
        if not self.enable_fusion:
            return self._make_disabled_result(graph)

        # Ensure topological order
        if not graph.node_order:
            graph.topological_sort()

        fusion_log: List[Dict[str, Any]] = []

        # Count before
        raw_launches = _count_launches(graph)
        raw_buffers = _count_buffers(graph)
        node_count_before = len(graph.nodes)

        fusions_per_pattern: Dict[str, int] = {}
        total_fusions = 0

        # Run each pass
        for pass_name, pass_fn in PASS_ORDER:
            snapshot = copy.deepcopy(graph)
            log_len = len(fusion_log)
            try:
                n = pass_fn(graph, fusion_log)
                if self.run_validation:
                    graph.validate()
                if n > 0:
                    fusions_per_pattern[pass_name] = n
                    total_fusions += n
            except Exception as exc:
                graph.__dict__.clear()
                graph.__dict__.update(snapshot.__dict__)
                del fusion_log[log_len:]
                # Log error but continue pipeline
                fusion_log.append({
                    "pattern": pass_name,
                    "status": "error",
                    "old_node_ids": [],
                    "new_node_id": "",
                    "removed_tensors": [],
                    "rejection_reason": f"Pass raised exception: {exc}",
                })

        # Cleanup dead tensors
        if self.enable_dead_cleanup:
            cleanup_dead_tensors(graph)

        # Re-sort and validate
        try:
            graph.topological_sort()
            graph.validate()
            validation_passed = True
        except ValueError:
            validation_passed = False

        # Count after
        optimized_launches = _count_launches(graph)
        optimized_buffers = _count_buffers(graph)
        node_count_after = len(graph.nodes)

        # Compute reductions
        launch_reduction = (
            (raw_launches - optimized_launches) / max(raw_launches, 1)
        ) if raw_launches > 0 else 0.0

        buffer_reduction = (
            (raw_buffers - optimized_buffers) / max(raw_buffers, 1)
        ) if raw_buffers > 0 else 0.0

        stats = {
            "fusion_log": fusion_log,
            "raw_launches": raw_launches,
            "optimized_launches": optimized_launches,
            "launch_reduction": launch_reduction,
            "raw_buffers": raw_buffers,
            "optimized_buffers": optimized_buffers,
            "buffer_reduction": buffer_reduction,
            "fusions_per_pattern": fusions_per_pattern,
            "total_fusions": total_fusions,
            "node_count_before": node_count_before,
            "node_count_after": node_count_after,
            "validation_passed": validation_passed,
        }

        return {
            "Fusion": {
                "enabled": True,
                "stats": stats,
            }
        }

    @staticmethod
    def _make_disabled_result(graph: Graph) -> Dict[str, Any]:
        """Return result when fusion is disabled."""
        if not graph.node_order:
            graph.topological_sort()

        launches = _count_launches(graph)
        buffers = _count_buffers(graph)

        return {
            "Fusion": {
                "enabled": False,
                "stats": {
                    "fusion_log": [],
                    "raw_launches": launches,
                    "optimized_launches": launches,
                    "launch_reduction": 0.0,
                    "raw_buffers": buffers,
                    "optimized_buffers": buffers,
                    "buffer_reduction": 0.0,
                    "fusions_per_pattern": {},
                    "total_fusions": 0,
                    "node_count_before": len(graph.nodes),
                    "node_count_after": len(graph.nodes),
                    "validation_passed": True,
                },
            }
        }
