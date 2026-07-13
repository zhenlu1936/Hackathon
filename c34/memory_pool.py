"""Device memory pool with free-list reuse and fragmentation management.

Implements C3.4 features A and C:
- A: Device allocation/free abstractions with weight upload path.
- C: Free-list reuse with best-fit policy, coalescing, and statistics tracking.

The pool manages physical device "slots" identified by integer IDs.
Each slot has a size and alignment. Freed slots enter a free list
and can be reused by subsequent allocations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from c34.execution_plan import PoolStats


# ── Memory block ────────────────────────────────────────────────────

@dataclass
class _MemoryBlock:
    """A contiguous block of device memory (allocated or free)."""

    slot_id: int
    offset: int = 0
    size_bytes: int = 0
    is_free: bool = True

    @property
    def end(self) -> int:
        return self.offset + self.size_bytes


# ── Free-list policy enum ──────────────────────────────────────────

class FitPolicy:
    """Enumeration of free-block selection policies."""
    BEST_FIT = "best_fit"
    FIRST_FIT = "first_fit"
    SIZE_CLASS = "size_class"


# ── Device memory pool ─────────────────────────────────────────────

class DeviceMemoryPool:
    """Manages device memory allocations with free-list reuse.

    Features A + C:
    - alloc(size, alignment) → slot_id (with reuse from free list)
    - free(slot_id) → returns block to free list
    - Coalescing of adjacent free blocks
    - Best-fit policy for block selection
    - Statistics tracking for code review and C3.5 tuning

    Usage in the scheduler pipeline:

        pool = DeviceMemoryPool(policy="best_fit")
        slot = pool.alloc(size)          # allocate or reuse
        # ... kernels use the slot ...
        pool.free(slot)                  # returns to free list
        stats = pool.stats()             # PoolStats for ExecutionPlan
    """

    def __init__(
        self,
        policy: str = FitPolicy.BEST_FIT,
        alignment: int = 64,
        total_device_bytes: int = 8 * 1024 * 1024 * 1024,  # 8 GiB default
    ) -> None:
        self._policy = policy
        self._alignment = alignment
        self._total_bytes = total_device_bytes

        # Internal state
        self._next_slot_id: int = 0
        self._free_list: List[_MemoryBlock] = []  # sorted by size for best-fit
        self._active: Dict[int, _MemoryBlock] = {}  # slot_id -> allocated block
        self._coalesce_counter: int = 0
        self._reuse_hits: int = 0
        self._total_allocs: int = 0
        self._total_frees: int = 0
        self._peak_active_bytes: int = 0
        self._peak_reserved_bytes: int = 0
        self._requested_bytes: int = 0
        self._reserved_bytes: int = 0

    # ── Public API ─────────────────────────────────────────────────

    def alloc(self, size_bytes: int, alignment: int = 64) -> int:
        """Allocate a device memory slot.

        Tries the free list first; falls back to fresh allocation.
        Uses best-fit policy when multiple free blocks qualify.

        Returns: slot_id (int)
        """
        if size_bytes <= 0:
            raise ValueError(f"Invalid allocation size: {size_bytes}")

        size_bytes = self._align_up(size_bytes, alignment)
        self._requested_bytes += size_bytes
        self._total_allocs += 1

        # Try free list first
        slot_id = self._try_reuse(size_bytes, alignment)
        if slot_id >= 0:
            self._reuse_hits += 1
            return slot_id

        # Fresh allocation
        slot_id = self._next_slot_id
        self._next_slot_id += 1

        block = _MemoryBlock(
            slot_id=slot_id,
            offset=self._reserved_bytes,
            size_bytes=size_bytes,
            is_free=False,
        )
        self._active[slot_id] = block
        self._reserved_bytes += size_bytes
        self._peak_reserved_bytes = max(self._peak_reserved_bytes, self._reserved_bytes)

        return slot_id

    def free(self, slot_id: int) -> None:
        """Free a device memory slot, returning it to the free list."""
        if slot_id not in self._active:
            raise ValueError(f"Slot {slot_id} is not allocated")

        block = self._active.pop(slot_id)
        block.is_free = True
        self._free_list.append(block)
        self._total_frees += 1

        # Coalesce adjacent free blocks
        self._coalesce()

    def stats(self) -> PoolStats:
        """Return current pool statistics."""
        active_bytes = sum(b.size_bytes for b in self._active.values())
        free_bytes = sum(b.size_bytes for b in self._free_list)

        # Internal fragmentation: wasted bytes inside allocated blocks
        # (reserved but not actively used — allocated but idle free blocks)
        internal_frag = sum(b.size_bytes for b in self._free_list)

        return PoolStats(
            requested_bytes=self._requested_bytes,
            reserved_bytes=self._reserved_bytes,
            active_bytes=active_bytes,
            peak_reserved_bytes=self._peak_reserved_bytes,
            internal_fragmentation=internal_frag,
            reuse_hits=self._reuse_hits,
            total_allocs=self._total_allocs,
            total_frees=self._total_frees,
            free_list_blocks=len(self._free_list),
        )

    def describe(self, slot_id: int) -> Tuple[int, int]:
        """Return ``(offset_bytes, capacity_bytes)`` for an active slot.

        The scheduler stores this physical range in every Allocation so the
        CuPy runtime can construct tensor views into one device arena instead
        of allocating an unrelated array per logical tensor.
        """
        block = self._active.get(slot_id)
        if block is None:
            raise ValueError(f"Slot {slot_id} is not allocated")
        return block.offset, block.size_bytes

    # ── Reuse logic ────────────────────────────────────────────────

    def _try_reuse(self, size_bytes: int, alignment: int) -> int:
        """Try to find a free block that fits. Returns slot_id or -1."""
        # Filter candidate blocks
        candidates = [b for b in self._free_list if b.size_bytes >= size_bytes]
        if not candidates:
            return -1

        if self._policy == FitPolicy.BEST_FIT:
            # Pick the smallest block that fits
            candidates.sort(key=lambda b: b.size_bytes)
        elif self._policy == FitPolicy.FIRST_FIT:
            pass  # use in-order (already in free_list order)
        elif self._policy == FitPolicy.SIZE_CLASS:
            # Round up to power-of-2 size class, find exact or next size class
            size_class = self._size_class(size_bytes)
            exact = [b for b in candidates if self._size_class(b.size_bytes) == size_class]
            if exact:
                candidates = exact
            else:
                candidates.sort(key=lambda b: self._size_class(b.size_bytes))

        chosen = candidates[0]
        self._free_list.remove(chosen)

        # If the block is larger than needed, split it
        if chosen.size_bytes > size_bytes:
            remainder = _MemoryBlock(
                slot_id=self._next_slot_id,
                offset=chosen.offset + size_bytes,
                size_bytes=chosen.size_bytes - size_bytes,
                is_free=True,
            )
            self._next_slot_id += 1
            self._free_list.append(remainder)
            chosen.size_bytes = size_bytes

        chosen.is_free = False
        self._active[chosen.slot_id] = chosen
        return chosen.slot_id

    def _coalesce(self) -> None:
        """Merge adjacent free blocks to reduce fragmentation."""
        if len(self._free_list) < 2:
            return

        # Sort by offset
        self._free_list.sort(key=lambda b: b.offset)

        merged: List[_MemoryBlock] = []
        current = self._free_list[0]

        for block in self._free_list[1:]:
            if current.end == block.offset:
                # Adjacent blocks — merge
                current.size_bytes += block.size_bytes
                self._coalesce_counter += 1
            else:
                merged.append(current)
                current = block

        merged.append(current)
        self._free_list = merged

    # ── Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _align_up(size: int, alignment: int) -> int:
        return ((size + alignment - 1) // alignment) * alignment

    @staticmethod
    def _size_class(size_bytes: int) -> int:
        """Map a size to a power-of-2 size class."""
        if size_bytes <= 0:
            return 0
        # Find next power of 2
        p = 1
        while p < size_bytes:
            p <<= 1
        return p
