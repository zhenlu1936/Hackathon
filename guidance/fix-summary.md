# Remaining Problems — Fix Summary

Generated: 2026-07-12

This file documents every problem from `remaining-problems.md` that was addressed, what changed, and which files were modified.

---

## Fix 1: P0 — Deterministic Precision Selection

**Problem:** `c32/strategy.py` used `coverage_mode=True` and a round-robin `_precision_counter`, so identical calls could return different precisions depending on evaluator call order. This resembled score targeting.

**Change:** Replaced the stateful selection with a deterministic `ExecutionMode` enum (`FULL_FP32` / `MIXED_PRECISION`) and a priority-ordered precision policy. In `FULL_FP32` mode every operator is fp32. In `MIXED_PRECISION` mode, sensitive ops stay fp32 while tunable ops (`MatMul`, `Gemm`, `Conv`) pick the highest-priority precision the hardware supports (fp32 > fp16 > bf16 > fp8 > fp4). The result is purely a function of the node, graph, mode, and hardware — never call order.

**Files changed:**
- `c32/strategy.py` — added `ExecutionMode` enum, `PRECISION_PRIORITY` list, rewrote `select_precision` and `_select_tunable_precision`, added `refresh_hardware()`.
- `c32/api.py` — updated module-level `Strategy` to use `ExecutionMode.FULL_FP32`, added `_refresh_globals()` to keep `strategy` and `hardware` in sync after `set_hardware()`.
- `c34/scheduler.py` — changed `Strategy(coverage_mode=False)` → `Strategy(mode=ExecutionMode.FULL_FP32)`.

**Verification:**
```python
s = Strategy(mode=ExecutionMode.FULL_FP32)
p1 = s.select_precision(node, graph)
p2 = s.select_precision(node, graph)
assert p1 == p2  # deterministic
```

---

## Fix 2: P1 — Kernel Name Normalization

**Problem:** Hardware API advertises `matmul_f32`, `matmul_f16`, etc. but decompositions emitted `matmul_fp32`, `matmul_fp16`. This mismatch meant claimed kernel diversity was false.

**Change:** Added `_kernel_name(base, precision)` helper in `c32/decompositions.py` that canonicalizes `fp32→f32`, `fp16→f16`, `fp8→f8`, `fp4→f4`. All decomposition functions now use this helper. Also fixed the `tune_kernel` name prefixes in `strategy.py` to match the canonical names.

**Files changed:**
- `c32/decompositions.py` — added `_kernel_name()`, updated all decomposition functions.

**Verification:**
- C3.2 self-test still reports `matmul_f32`, `matmul_f16`, etc. under D5b.
- All 17 operator decompositions emit canonical names.

---

## Fix 3: P1 — Gemm Without Bias Dataflow Bug

**Problem:** `decompose_Gemm` always emitted a `matmul` kernel with output `{nid}_matmul_out`, then conditionally added `add_bias`. When bias was absent, the matmul output was never connected to the node output — leaving a broken plan.

**Change:** When bias is absent, the `matmul` kernel now writes directly to the node's declared output. When bias is present, intermediate tensors use the documented `__c3_inter_*` convention.

**Files changed:**
- `c32/decompositions.py` — rewrote `decompose_Gemm`.

**Verification:**
```python
kernels = strategy.decompose(gemm_node, graph)
if no_bias:
    assert kernels[-1].outputs == list(node.outputs)  # connected
```

---

## Fix 4: P1 — Operator Parameters Retention

**Problem:** `KernelSpecRef` carried only kernel names and I/O tensor names — no Conv pads/strides, no Gemm alpha/beta/trans, no Softmax axis, no LayerNorm epsilon. Many decompositions could not preserve ONNX semantics.

**Change:** Added `operator_params: Dict[str, Any]` to `KernelSpecRef`. Every decomposition function now populates it with the node's ONNX attributes and derived parameters:
- **Conv:** `kernel_shape`, `pads`, `strides`, `dilations`, `group`
- **Gemm:** `alpha`, `beta`, `transA`, `transB`
- **Softmax:** `axis`, `keepdims`
- **LayerNormalization:** `axis`, `epsilon`
- **Transpose:** `perm`
- **Gather:** `axis`
- **Split:** `axis`, `split`
- **Reshape:** `allowzero`
- **Flatten:** `axis`
- **GlobalAveragePool:** no squeeze — preserves `[N,C,1,1]`

**Files changed:**
- `c32/kernel_spec.py` — added `operator_params` field.
- `c32/decompositions.py` — populated `operator_params` in all 17 decomposers.
- `c32/strategy.py` — default decomposer passes `node.attributes`.

---

## Fix 5: P1 — Tuning Hardware Limit Clamping

**Problem:** `tune_kernel` generated launch parameters without validating against actual hardware thread/smem limits. The `smem_bytes` calculation for matmul didn't clamp properly (`smem_bytes=min(k*4+1024, max_smem)` could still exceed limits after other tuning).

**Change:** Added `_clamp_smem()` helper that enforces the valid range: `-1` (dynamic) is passed through, `0 <= smem_bytes <= max_smem` is valid, anything exceeding the limit returns `-1` (infeasible). Applied clamping to all tuning paths.

**Files changed:**
- `c32/strategy.py` — added `_clamp_smem()`, updated all tuning branches.

---

## Fix 6: P0/P1 — C3.4 Scheduler: Output Bindings & Weight Events

### 6a: Preserved allocation bindings for kernel steps

**Problem:** `_allocate_intermediates` popped entries from `_alloc_map` as it freed pool slots, so when `_schedule_kernels` later looked up input/output bindings, the map was empty. Observed empty output bindings were 9/9 (MLP), 76/76 (ResNet), 253/253 (Transformer).

**Change:** `_allocate_intermediates` now frees pool slots but keeps `_alloc_map` entries intact. `_schedule_kernels` can now resolve every logical tensor to its physical `alloc_id`.

### 6b: Weight transfers now signal events

**Problem:** H2D weight transfers had no `event_id`. Kernels that consume a weight couldn't wait for it. Observed transfers with `event_id` were 0/7 (MLP), 0/43 (ResNet), 0/56 (Transformer).

**Change:** Each weight H2D transfer now creates a `evt_weight_ready_{tname}` event, registered in `self._events`, and the transfer carries it. Kernels that consume a weight can depend on this event.

### 6c: ExecutionPlan.validate requires bindings

**Problem:** `validate()` only checked bindings that happened to exist — an entirely missing output map passed validation.

**Change:** Added checks for empty `inputs` / `outputs` dicts. Every kernel step must have at least one input binding and at least one output binding.

**Files changed:**
- `c34/scheduler.py` — `_allocate_intermediates`, `_allocate_weights` (weight events).
- `c34/execution_plan.py` — `validate()` empty-check added.

---

## Fix 7: P1 — C3.5 Fused Operator Dispatch

**Problem:** C3.3 creates fused nodes (`FusedMatMulBias`, `FusedConv2dBatchNorm`, `FusedEWChain`, `FusedSoftmaxDropout`, `FusedResidualNorm`) but C3.5's engine had no dispatch for them — every fused node was rejected as unknown.

**Change:** Added implementations for all five fused operator types:
- **FusedMatMulBias:** lowered to matmul + bias add.
- **FusedConv2dBatchNorm:** lowered to Conv (weights folded at fusion time).
- **FusedEWChain:** walks the chain of elementwise ops, resolves inputs from the fused node's input list.
- **FusedSoftmaxDropout:** inference-mode Dropout is identity → just Softmax.
- **FusedResidualNorm:** Add(residual) + LayerNorm.

Also added missing primitive ops (`op_sub`, `op_exp`, `op_sqrt`) needed by the fused implementations.

**Files changed:**
- `c35/engine.py` — added 5 fused op implementations, 3 missing primitives, updated dispatch.

---

## Fix 8: P1 — Submission Cleanliness

**Problem:** Repository contained `__pycache__/`, `.pyc`, and `.DS_Store` files. No policy existed for excluding generated artifacts from the submission archive.

**Change:** Extended `.gitignore` to cover OS artifacts, Python caches, generated DAGs/outputs/caches, IDE config, and development binaries. Deleted all existing bytecode and `.DS_Store` files from the tree.

**Files changed:**
- `.gitignore` — extended ignore patterns.

**Verification:**
```bash
find . -name '__pycache__' -o -name '*.pyc' -o -name '.DS_Store'
# produces no results
```

---

## Fix 9: P0 — Hardware API Staleness

**Problem:** `set_hardware()` updated the global `_default_hardware` but `c32.api.strategy` and `c32.api.hardware` module-level singletons were not refreshed.

**Change:** `c32/api.py` now includes `_refresh_globals()` called automatically after `set_hardware()`. The module-level `strategy` and `hardware` always reflect the active target.

**Files changed:**
- `c32/api.py` — `_refresh_globals()`, patched `set_hardware`.

---

## Items NOT addressed (require organizer input or AEC hardware)

These problems from `remaining-problems.md` remain open because they depend on the organizer-provided AEC toolchain or evaluator benchmark:

| Problem | Reason |
|---|---|
| No executable AEC compiler/runtime path | Requires the AEC GPGPU stack (compiler, runtime, device) |
| No FULL_FP32 numerical validation on AEC | Requires AEC hardware |
| C3.3 launch/buffer counting from actual decompositions | Currently fixed estimates; requires full AEC pipeline |
| C3.5 NumPy CPU path → AEC GPU path | Requires AEC runtime replacement |
| Offline wheels / evaluator dependency declaration | Requires organizer-provided dependencies list |
| CuPy module decision (`cupy-cuda12x` vs organizer image) | Requires organizer clarification |
| C3.2/C3.3 benchmark API (`bench_c32_c33.py`) | Not released to competitors |
| C3.5 memory points conflict (10 vs 15) | Requires organizer clarification |
| Conv+BN evaluation without original BN parameters | Requires organizer clarification |
| Build/JIT time in timing | Requires organizer clarification |

---

## Modified files (complete list)

```
.gitignore
c32/api.py
c32/decompositions.py
c32/kernel_spec.py
c32/strategy.py
c34/execution_plan.py
c34/scheduler.py
c35/engine.py
```

## Verified passing after fixes

- `c32/test_c32.py` — 13.42/15 (all structural checks pass; self-score is acknowledged as inflated in `remaining-problems.md`)
- `c35.test_c35.C35SpecificationTests` — 14/14 tests pass (all 17 operators, all 3 public models, all batch sizes)
