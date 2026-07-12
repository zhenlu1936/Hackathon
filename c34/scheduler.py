"""Execution scheduler — builds the complete C3.4 execution plan.

Integrates all five features A–E:

A: Device pool allocations + weight preload (H2D before first kernel use).
B: Intermediate tensor lifetime analysis with slot reuse.
C: Free-list pool with best-fit reuse policy.
D: Weight prefetch: async copy stream overlaps H2D with compute.
E: Stream-level parallelism: dependency-aware multi-stream scheduling
   with events for cross-stream producer/consumer edges.

Usage:
    from c34.scheduler import ExecutionScheduler
    scheduler = ExecutionScheduler(graph, batch_size=1)
    plan = scheduler.build()
    print(plan.summary())
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from c3common.ir.graph import Graph
from c32.kernel_spec import KernelSpecRef
from c32.strategy import Strategy, ExecutionMode
from c34.execution_plan import (
    Allocation, Transfer, KernelStep, EventDep,
    LifetimeInterval, ExecutionPlan, PoolStats,
)
from c34.lifetime import compute_lifetimes, find_overlap_groups
from c34.memory_pool import DeviceMemoryPool, FitPolicy


# ── Constants ──────────────────────────────────────────────────────

COPY_STREAM = 0         # dedicated copy stream ID
FIRST_COMPUTE_STREAM = 1  # first compute stream ID


class ExecutionScheduler:
    """Build the C3.4 execution plan from a Graph.

    Integrates AEC device allocation, lifetime reuse, pool fragmentation,
    weight prefetch, and multi-stream scheduling into one plan.

    Args:
        graph: The computation graph IR (after decomposition / fusion).
        batch_size: Concretized batch dimension for size calculations.
        enable_prefetch: Enable async weight prefetch (Feature D).
        enable_multi_stream: Enable stream-level parallelism (Feature E).
        pool_policy: Free-list selection policy ("best_fit", "first_fit", "size_class").
    """

    def __init__(
        self,
        graph: Graph,
        batch_size: int = 1,
        enable_prefetch: bool = True,
        enable_multi_stream: bool = True,
        pool_policy: str = FitPolicy.BEST_FIT,
    ) -> None:
        self.graph = graph
        self.batch_size = batch_size
        self.enable_prefetch = enable_prefetch
        self.enable_multi_stream = enable_multi_stream
        self.pool_policy = pool_policy

        # Internal state
        self._strategy = Strategy(mode=ExecutionMode.FULL_FP32)
        self._pool = DeviceMemoryPool(policy=pool_policy)

        # Accumulators
        self._allocations: List[Allocation] = []
        self._transfers: List[Transfer] = []
        self._kernel_steps: List[KernelStep] = []
        self._events: List[EventDep] = []
        self._weight_slots: Dict[str, str] = {}  # tensor_name -> alloc_id
        self._alloc_map: Dict[str, str] = {}      # tensor_name -> alloc_id
        self._event_counter: int = 0
        self._step_counter: int = 0

        # Kernel schedule (built during build())
        self._kernel_schedule: List[KernelSpecRef] = []
        # Map kernel index -> originating node_id
        self._kernel_node_map: Dict[int, str] = {}

    # ── Main entry ─────────────────────────────────────────────────

    def build(self) -> ExecutionPlan:
        """Build the complete execution plan.

        Returns:
            ExecutionPlan with all allocations, transfers, kernels,
            events, lifetimes, and pool stats populated.
        """
        # Step 1: Build flat kernel schedule from decomposed nodes
        self._build_kernel_schedule()

        # Step 2: Compute tensor lifetimes
        lifetimes = compute_lifetimes(
            self.graph, self._kernel_schedule, self.batch_size,
        )

        # Step 3: Pre-allocate and upload weights (Feature A)
        self._allocate_weights(lifetimes)

        # Step 4: Allocate intermediates with lifetime reuse (Feature B + C)
        # NOTE: _alloc_map is *not* purged after this step so that
        # _schedule_kernels can look up the bindings for every kernel
        # input and output.
        self._allocate_intermediates(lifetimes)

        # Step 5: Schedule kernel steps with stream assignments (Feature D + E)
        self._schedule_kernels(lifetimes)

        # Step 6: Schedule D2H transfers for graph outputs
        self._schedule_output_readback(lifetimes)

        # Step 7: Build the final plan
        plan = ExecutionPlan(
            model_name=self.graph.name,
            batch_size=self.batch_size,
            allocations=self._allocations,
            transfers=self._transfers,
            kernel_steps=self._kernel_steps,
            events=self._events,
            lifetimes=lifetimes,
            pool_stats=self._pool.stats(),
            weight_slots=self._weight_slots,
            num_compute_streams=self._compute_num_streams(),
            copy_stream_id=COPY_STREAM,
        )

        return plan

    # ── Step 1: Kernel schedule ────────────────────────────────────

    def _build_kernel_schedule(self) -> None:
        """Decompose every node and build a flat kernel execution schedule.

        Also registers intermediate tensors in the graph's tensor table
        with inferred shapes/sizes so that lifetime analysis can compute
        correct byte sizes.
        """
        if not self.graph.node_order:
            self.graph.topological_sort()

        for nid in self.graph.node_order:
            node = self.graph.nodes.get(nid)
            if node is None:
                continue

            precision = self._strategy.select_precision(node, self.graph)
            kernels = self._strategy.decompose(node, self.graph, precision)

            for krn in kernels:
                idx = len(self._kernel_schedule)
                self._kernel_schedule.append(krn)
                self._kernel_node_map[idx] = nid

                # Register intermediate tensor shapes for lifetime sizing
                self._register_intermediate_tensors(node, krn)

    # ── Intermediate tensor registration ───────────────────────────

    def _register_intermediate_tensors(
        self, node: Any, krn: KernelSpecRef
    ) -> None:
        """Register intermediate tensors from kernel outputs in the graph.

        Derives shapes from the parent node's input/output shapes so that
        lifetime analysis produces correct byte sizes.
        """
        from c3common.ir.graph import ONNSType

        # Determine output dtype from the parent node
        out_dtype = ONNSType.FLOAT
        for out_name in node.outputs:
            out_t = self.graph.tensors.get(out_name)
            if out_t is not None and out_t.dtype != ONNSType.UNDEFINED:
                out_dtype = out_t.dtype
                break

        # Estimate output shape
        out_shape = self._infer_output_shape(node, krn)

        for out_name in krn.outputs:
            if not out_name:
                continue
            if out_name in node.outputs:
                continue  # already a registered node output
            if out_name in self.graph.tensors:
                continue  # already registered

            # Register intermediate tensor with inferred shape
            self.graph.register_tensor(
                name=out_name,
                dtype=out_dtype,
                shape=out_shape,
            )

    def _infer_output_shape(
        self, node: Any, krn: KernelSpecRef
    ) -> List[int]:
        """Infer output shape of a kernel step from its parent node's shapes.

        Resolves symbolic batch dims using self.batch_size.
        """
        def _resolve(shape_list) -> List[int]:
            result = []
            for d in shape_list:
                if d is None:
                    return []
                try:
                    result.append(int(d))
                except (ValueError, TypeError):
                    if isinstance(d, str) and d.lower() in ("n", "batch", "b"):
                        result.append(self.batch_size)
                    else:
                        return []
            return result

        # Try to get shape from parent node outputs first
        for out_name in node.outputs:
            out_t = self.graph.tensors.get(out_name)
            if out_t is not None and out_t.shape:
                resolved = _resolve(out_t.shape)
                if resolved:
                    return resolved

        # For matmul intermediates (e.g. fc1_matmul_out), infer from inputs
        if krn.kernel_name.startswith("matmul_"):
            shapes = []
            for inp_name in krn.inputs:
                inp_t = self.graph.tensors.get(inp_name)
                if inp_t is not None and inp_t.shape:
                    resolved = _resolve(inp_t.shape)
                    if resolved:
                        shapes.append(resolved)
            if len(shapes) >= 2:
                a, b = shapes[0], shapes[1]
                trans_b = node.attributes.get("transB", 0)
                m = a[-2] if len(a) >= 2 else 1
                n = b[-2] if trans_b else b[-1]
                if len(a) == 3:
                    return [a[0], m, n]
                elif len(a) >= 4:
                    return a[:-2] + [m, n]
                return [m, n]

        # For elementwise ops, output shape = first input shape
        if krn.kernel_name.startswith(("add_", "mul_", "div_", "sub_", "relu_",
                                         "erf_", "sqrt", "exp", "reduce_")):
            for inp_name in krn.inputs:
                inp_t = self.graph.tensors.get(inp_name)
                if inp_t is not None and inp_t.shape:
                    resolved = _resolve(inp_t.shape)
                    if resolved:
                        return resolved

        # For flatten: output = [batch, product_of_remaining_dims]
        if krn.kernel_name == "flatten":
            for inp_name in krn.inputs:
                inp_t = self.graph.tensors.get(inp_name)
                if inp_t is not None and inp_t.shape:
                    resolved = _resolve(inp_t.shape)
                    if len(resolved) >= 2:
                        prod = 1
                        for d in resolved[1:]:
                            prod *= d
                        return [resolved[0], prod]

        # Default: try parent node output shape
        for out_name in node.outputs:
            out_t = self.graph.tensors.get(out_name)
            if out_t is not None and out_t.shape:
                resolved = _resolve(out_t.shape)
                if resolved:
                    return resolved

        return []

    # ── Step 3: Weight allocation (Feature A) ──────────────────────

    def _allocate_weights(self, lifetimes: Dict[str, LifetimeInterval]) -> None:
        """Allocate device slots for all weights/initializers and create H2D transfers.

        Weights are model-lifetime resident: allocated once, never freed.
        Each H2D transfer is assigned a ``weight_ready`` event so that
        kernels know when the data is on-device.
        """
        for tname, interval in lifetimes.items():
            if not interval.is_weight:
                continue
            if interval.size_bytes <= 0:
                continue

            # Allocate from pool
            slot_id = self._pool.alloc(interval.size_bytes)
            alloc_id = f"alloc_w_{tname}"

            alloc = Allocation(
                alloc_id=alloc_id,
                tensor_name=tname,
                slot_id=slot_id,
                size_bytes=interval.size_bytes,
                is_weight=True,
            )
            self._allocations.append(alloc)
            self._weight_slots[tname] = alloc_id
            self._alloc_map[tname] = alloc_id

            # Create weight-ready event for async prefetch
            event_id = f"evt_weight_ready_{tname}"
            self._event_counter += 1

            # Create H2D transfer that signals the event when done
            transfer = Transfer(
                kind="H2D",
                tensor_name=tname,
                alloc_id=alloc_id,
                size_bytes=interval.size_bytes,
                stream_id=COPY_STREAM,
                event_id=event_id,
            )
            self._transfers.append(transfer)

            # Register the event so kernels can depend on it
            evt = EventDep(
                event_id=event_id,
                src_stream=COPY_STREAM,
                dst_stream=FIRST_COMPUTE_STREAM,  # will be refined per-kernel
                description=f"Weight ready: {tname}",
            )
            self._events.append(evt)

    # ── Step 4: Intermediate allocation (Feature B + C) ────────────

    def _allocate_intermediates(self, lifetimes: Dict[str, LifetimeInterval]) -> None:
        """Allocate intermediate tensors with lifetime-based slot reuse.

        Uses an interval-based allocation strategy:
        1. Sort tensors by first_use.
        2. Before each kernel step, allocate tensors whose first_use == step.
        3. After each kernel step, free tensors whose last_use == step.
        4. Pool free list provides reuse (Feature C).

        IMPORTANT: ``_alloc_map`` is preserved with every binding so that
        ``_schedule_kernels`` can look up input/output alloc_ids.
        Pool-level free() is called for lifetime management, but the
        mapping stays intact.
        """
        # Collect non-weight, non-input intermediates
        intermediates = [
            li for li in lifetimes.values()
            if not li.is_weight and not li.is_input and li.size_bytes > 0
        ]
        # Sort by first_use
        intermediates.sort(key=lambda li: (li.first_use, li.last_use))

        # Track which tensors to free after each step (pool-level only)
        free_after: Dict[int, List[str]] = {}
        for li in intermediates:
            free_after.setdefault(li.last_use, []).append(li.tensor_name)

        # Track which tensors to alloc before each step
        alloc_before: Dict[int, List[LifetimeInterval]] = {}
        for li in intermediates:
            alloc_before.setdefault(li.first_use, []).append(li)

        num_steps = len(self._kernel_schedule)

        for step_idx in range(num_steps + 1):  # +1 for post-last-step cleanup
            # Before step: allocate tensors first used at this step
            if step_idx < num_steps:
                for li in alloc_before.get(step_idx, []):
                    if li.tensor_name in self._alloc_map:
                        continue  # already allocated

                    slot_id = self._pool.alloc(li.size_bytes)
                    alloc_id = f"alloc_{li.tensor_name}"
                    if len(alloc_id) > 64:
                        alloc_id = f"alloc_inter_{slot_id}_{len(self._allocations)}"

                    alloc = Allocation(
                        alloc_id=alloc_id,
                        tensor_name=li.tensor_name,
                        slot_id=slot_id,
                        size_bytes=li.size_bytes,
                    )
                    self._allocations.append(alloc)
                    # Keep in _alloc_map for kernel binding lookups
                    self._alloc_map[li.tensor_name] = alloc_id

            # After step: free pool slots (but keep _alloc_map bindings intact)
            for tname in free_after.get(step_idx, []):
                if tname in self._alloc_map and tname not in self._weight_slots:
                    alloc_id = self._alloc_map[tname]
                    # Find the slot_id for this alloc_id and free from pool
                    for a in self._allocations:
                        if a.alloc_id == alloc_id and not a.is_weight:
                            self._pool.free(a.slot_id)
                            break
                    # NOTE: we do NOT pop from _alloc_map — kernel steps
                    # created later need the binding for output lookup.

    # ── Step 5: Kernel scheduling (Feature D + E) ──────────────────

    def _schedule_kernels(self, lifetimes: Dict[str, LifetimeInterval]) -> None:
        """Schedule kernel steps with stream assignments and events.

        Feature D: Weight prefetch — H2D transfers for weights are placed on
        the copy stream. Kernels that consume a weight must wait for its
        weight-ready event. Weights for layer k+1 can be uploaded while
        layer k computes.

        Feature E: Multi-stream parallelism — independent kernel groups
        (no data dependencies) are assigned to different compute streams.
        Cross-stream producer/consumer edges have event dependencies.
        """
        # ── Stream assignment (Feature E) ──
        stream_assignments = self._assign_streams() if self.enable_multi_stream else {}

        # ── Weight-ready events (Feature D) ──
        weight_events: Dict[str, str] = {}  # weight_tensor_name -> event_id
        if self.enable_prefetch:
            weight_events = self._create_weight_events()

        # ── Walk kernel schedule and create KernelSteps ──
        for step_idx, krn in enumerate(self._kernel_schedule):
            node_id = self._kernel_node_map.get(step_idx, "unknown")
            stream_id = stream_assignments.get(step_idx, FIRST_COMPUTE_STREAM)

            # Map inputs/outputs to alloc_ids
            input_bindings: Dict[str, str] = {}
            for inp in krn.inputs:
                if inp and inp in self._alloc_map:
                    input_bindings[inp] = self._alloc_map[inp]

            output_bindings: Dict[str, str] = {}
            for out in krn.outputs:
                if out and out in self._alloc_map:
                    output_bindings[out] = self._alloc_map[out]

            # Dependencies: weight events + cross-stream events
            depends_on: List[str] = []

            # Weight dependencies (Feature D)
            if self.enable_prefetch:
                for inp in krn.inputs:
                    if inp in weight_events:
                        depends_on.append(weight_events[inp])

            # Cross-stream event dependencies (Feature E)
            if self.enable_multi_stream:
                cross_events = self._cross_stream_deps(step_idx, stream_assignments)
                depends_on.extend(cross_events)

            # Signals: this kernel signals weight-ready events for downstream uses
            signals: List[str] = []
            if self.enable_prefetch:
                for out in krn.outputs:
                    if out in weight_events:
                        signals.append(weight_events[out])

            # Tuning params
            tuning = None
            if krn.tuning_params is not None:
                tp = krn.tuning_params
                tuning = {
                    "block_x": tp.block_x,
                    "grid_x": tp.grid_x,
                    "smem_bytes": tp.smem_bytes,
                }

            ks = KernelStep(
                step_index=self._step_counter,
                kernel_name=krn.kernel_name,
                node_id=node_id,
                inputs=input_bindings,
                outputs=output_bindings,
                stream_id=stream_id,
                depends_on=depends_on,
                signals=signals,
                tuning_params=tuning,
            )
            self._kernel_steps.append(ks)
            self._step_counter += 1

    # ── Output readback (D2H) ──────────────────────────────────────

    def _schedule_output_readback(self, lifetimes: Dict[str, LifetimeInterval]) -> None:
        """Create D2H transfers for graph outputs."""
        output_names = {t.name for t in self.graph.outputs}
        for tname in output_names:
            if tname not in self._alloc_map:
                # Allocate if not already done
                li = lifetimes.get(tname)
                size = li.size_bytes if li else 0
                if size <= 0:
                    continue
                slot_id = self._pool.alloc(size)
                alloc_id = f"alloc_out_{tname}"
                alloc = Allocation(
                    alloc_id=alloc_id,
                    tensor_name=tname,
                    slot_id=slot_id,
                    size_bytes=size,
                    is_output=True,
                )
                self._allocations.append(alloc)
                self._alloc_map[tname] = alloc_id

            alloc_id = self._alloc_map[tname]
            li = lifetimes.get(tname)
            size = li.size_bytes if li else 0

            transfer = Transfer(
                kind="D2H",
                tensor_name=tname,
                alloc_id=alloc_id,
                size_bytes=size,
                stream_id=COPY_STREAM,
            )
            self._transfers.append(transfer)

    # ── Stream assignment (Feature E) ──────────────────────────────

    def _assign_streams(self) -> Dict[int, int]:
        """Assign compute streams to kernel steps based on DAG dependencies.

        Independent subgraphs (no data dependencies) get different streams.
        Returns: {kernel_step_index -> stream_id}

        Algorithm:
        1. Build a DAG of kernels based on original graph edges.
        2. Compute a "depth" for each kernel (longest path from inputs).
        3. Group kernels by depth — kernels at the same depth that are
           independent get different streams (up to a max).
        """
        num_steps = len(self._kernel_schedule)
        if num_steps <= 1:
            return {0: FIRST_COMPUTE_STREAM}

        # Build kernel-level dependency graph
        # A kernel j depends on kernel i if j reads a tensor that i produces.
        kernel_deps: Dict[int, Set[int]] = {}  # step -> {dependent steps}
        kernel_rdeps: Dict[int, Set[int]] = {}  # step -> {steps it depends on}

        for i in range(num_steps):
            kernel_deps.setdefault(i, set())
            kernel_rdeps.setdefault(i, set())

        for i in range(num_steps):
            krn_i = self._kernel_schedule[i]
            for out in krn_i.outputs:
                if not out:
                    continue
                # Find all kernels that consume this output
                for j in range(i + 1, num_steps):
                    krn_j = self._kernel_schedule[j]
                    if out in krn_j.inputs:
                        kernel_deps[i].add(j)
                        kernel_rdeps[j].add(i)

        # Compute depth (longest path from source kernels)
        depth: Dict[int, int] = {}
        max_streams = 4

        # Topological order with depths
        visited: Set[int] = set()

        def _compute_depth(step: int) -> int:
            if step in depth:
                return depth[step]
            if step in visited:
                return 0  # cycle (shouldn't happen)
            visited.add(step)

            max_pred_depth = -1
            for pred in kernel_rdeps.get(step, set()):
                max_pred_depth = max(max_pred_depth, _compute_depth(pred))

            depth[step] = max_pred_depth + 1
            return depth[step]

        for i in range(num_steps):
            _compute_depth(i)

        # Assign streams: round-robin within each depth level
        stream_assignments: Dict[int, int] = {}
        depth_counters: Dict[int, int] = {}

        for i in range(num_steps):
            d = depth.get(i, 0)
            cnt = depth_counters.get(d, 0)
            stream_id = FIRST_COMPUTE_STREAM + (cnt % max_streams)
            depth_counters[d] = cnt + 1
            stream_assignments[i] = stream_id

        return stream_assignments

    # ── Weight prefetch events (Feature D) ─────────────────────────

    def _create_weight_events(self) -> Dict[str, str]:
        """Create weight-ready events for async weight prefetch.

        Strategy: For layer-by-layer weight consumption, create an event
        per weight tensor. The H2D transfer signals the event when complete.
        Kernels that consume the weight wait on the event.

        To simulate "prefetch overlap": weight H2D for layer k+1 can be
        issued on the copy stream while compute for layer k runs on the
        compute stream. The event ensures layer k+1 compute waits for
        its weight.

        Returns:
            Dict mapping weight_tensor_name -> event_id.
        """
        weight_events: Dict[str, str] = {}

        # Group weight tensors by their consumers (layer grouping)
        for tname, alloc_id in self._weight_slots.items():
            evt_id = f"evt_wready_{tname}"
            weight_events[tname] = evt_id

            # Create event dependency: H2D -> signals event -> kernels wait
            evt = EventDep(
                event_id=evt_id,
                src_stream=COPY_STREAM,       # H2D on copy stream
                dst_stream=FIRST_COMPUTE_STREAM,  # compute waits
                description=f"Weight ready for {tname}",
            )
            self._events.append(evt)
            self._event_counter += 1

        return weight_events

    # ── Cross-stream dependencies (Feature E) ──────────────────────

    def _cross_stream_deps(
        self,
        step_idx: int,
        stream_assignments: Dict[int, int],
    ) -> List[str]:
        """Find cross-stream event dependencies for a kernel step.

        If a kernel reads a tensor produced by a kernel on a different stream,
        insert an event dependency to ensure correct ordering.

        Returns:
            List of event IDs this kernel must wait on.
        """
        deps: List[str] = []
        my_stream = stream_assignments.get(step_idx, FIRST_COMPUTE_STREAM)
        krn = self._kernel_schedule[step_idx]

        for inp in krn.inputs:
            if not inp:
                continue
            # Find the producer step
            for prev_step in range(step_idx):
                prev_krn = self._kernel_schedule[prev_step]
                if inp in prev_krn.outputs:
                    prev_stream = stream_assignments.get(prev_step, FIRST_COMPUTE_STREAM)
                    if prev_stream != my_stream:
                        # Cross-stream dependency needed
                        evt_id = f"evt_xs_{prev_step}_{step_idx}_{inp}"
                        # Check if event already exists
                        existing = {e.event_id for e in self._events}
                        if evt_id not in existing:
                            evt = EventDep(
                                event_id=evt_id,
                                src_stream=prev_stream,
                                dst_stream=my_stream,
                                description=(
                                    f"Cross-stream: kernel {prev_step} "
                                    f"(stream {prev_stream}) -> kernel {step_idx} "
                                    f"(stream {my_stream}) via {inp}"
                                ),
                            )
                            self._events.append(evt)
                            self._event_counter += 1
                        deps.append(evt_id)
                    break

        return deps

    # ── Helpers ────────────────────────────────────────────────────

    def _compute_num_streams(self) -> int:
        """Count number of distinct compute streams used."""
        streams = {FIRST_COMPUTE_STREAM}
        for ks in self._kernel_steps:
            streams.add(ks.stream_id)
        return len(streams)
