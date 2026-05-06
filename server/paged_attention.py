"""PagedAttention-style KV memory: block pool + per-sequence block tables.

This is a **vLLM-shaped mimic** for scheduling and observability: a global
pool of **physical** block IDs, and per-request **block tables** that map
**logical** block indices to physical blocks—same story as vLLM’s PagedAttention
memory manager, without GPU kernels or real KV tensors.

What is faithful to the *idea*:
  - fixed ``block_size`` tokens per physical block;
  - append-only growth of a sequence’s table as logical length increases;
  - ``resolve(logical_token) -> (physical_block_id, slot)`` like a page walk;
  - internal fragmentation from rounding demand up to whole blocks.

What is **not** here: fused attention over block tables, prefix-block sharing,
or any coupling to ``mlx_lm`` caches (those stay dense inside MLX).
"""

from __future__ import annotations

import threading
from typing import Dict, List, Tuple


class BlockPoolExhausted(Exception):
    """Raised when allocate() needs more physical blocks than the pool has free."""


class PagedKVCacheManager:
    """Global physical block pool + per-sequence logical→physical block tables."""

    def __init__(self, block_size: int, num_blocks: int) -> None:
        if block_size < 1 or num_blocks < 1:
            raise ValueError("block_size and num_blocks must be positive")
        self.block_size = block_size
        self.num_blocks = num_blocks
        self._lock = threading.Lock()
        self._free: set[int] = set(range(num_blocks))
        self._tables: Dict[str, List[int]] = {}
        self._token_counts: Dict[str, int] = {}
        self._peak_used_physical_blocks: int = 0

    def allocate(self, request_id: str, num_tokens: int) -> None:
        """Grow the sequence’s block table until it can hold ``num_tokens`` KV slots."""
        if num_tokens < 0:
            raise ValueError("num_tokens must be non-negative")
        with self._lock:
            current = self._token_counts.get(request_id, 0)
            if num_tokens <= current:
                return
            table = self._tables.setdefault(request_id, [])
            capacity = len(table) * self.block_size
            table_len_before = len(table)
            try:
                while capacity < num_tokens:
                    if not self._free:
                        raise BlockPoolExhausted(
                            f"need blocks for {num_tokens} tokens; "
                            f"only {capacity} slots available; pool exhausted"
                        )
                    block_id = self._free.pop()
                    table.append(block_id)
                    capacity += self.block_size
                self._token_counts[request_id] = num_tokens
                used_phys = self.num_blocks - len(self._free)
                self._peak_used_physical_blocks = max(
                    self._peak_used_physical_blocks, used_phys
                )
            except BlockPoolExhausted:
                while len(table) > table_len_before:
                    self._free.add(table.pop())
                if not table:
                    self._tables.pop(request_id, None)
                    self._token_counts.pop(request_id, None)
                raise

    def free(self, request_id: str) -> None:
        """Return all physical blocks for this sequence to the pool."""
        with self._lock:
            table = self._tables.pop(request_id, None)
            self._token_counts.pop(request_id, None)
            if not table:
                return
            for block_id in table:
                self._free.add(block_id)

    def block_table(self, request_id: str) -> List[int]:
        """Copy of the sequence’s logical block order → physical block IDs."""
        with self._lock:
            table = self._tables.get(request_id)
            if table is None:
                raise KeyError(f"unknown sequence {request_id}")
            return list(table)

    def sequence_footprint(self, request_id: str) -> dict[str, int]:
        """Per-sequence stats: vLLM-style “how many slots did we reserve vs use?”."""
        with self._lock:
            table = self._tables.get(request_id)
            if table is None:
                raise KeyError(f"unknown sequence {request_id}")
            logical_blocks = len(table)
            reserved_slots = logical_blocks * self.block_size
            committed = self._token_counts.get(request_id, 0)
            return {
                "logical_blocks": logical_blocks,
                "reserved_token_slots": reserved_slots,
                "committed_kv_tokens": committed,
                "unused_reserved_slots": max(0, reserved_slots - committed),
            }

    def resolve(
        self, request_id: str, logical_token_index: int
    ) -> Tuple[int, int]:
        """Map logical token index to (physical_block_id, slot_offset) — a “page walk”."""
        if logical_token_index < 0:
            raise ValueError("logical_token_index must be non-negative")
        with self._lock:
            table = self._tables.get(request_id)
            if not table:
                raise KeyError(f"unknown sequence {request_id}")
            block_idx = logical_token_index // self.block_size
            if block_idx >= len(table):
                raise IndexError("logical_token_index out of allocated range")
            slot_offset = logical_token_index % self.block_size
            physical_block_id = table[block_idx]
            return (physical_block_id, slot_offset)

    def available_blocks(self) -> int:
        """Number of free physical blocks in the pool."""
        with self._lock:
            return len(self._free)

    def metrics(self) -> dict[str, float | int]:
        """Pool + fragmentation stats useful when comparing to vLLM-style charts."""
        with self._lock:
            used_physical_blocks = self.num_blocks - len(self._free)
            total_slots = self.num_blocks * self.block_size
            committed = sum(self._token_counts.get(rid, 0) for rid in self._tables)
            reserved = used_physical_blocks * self.block_size
            unused_reserved = max(0, reserved - committed)
            slot_util = (committed / total_slots) if total_slots else 0.0
            return {
                "total_blocks": self.num_blocks,
                "free_blocks": len(self._free),
                "used_physical_blocks": used_physical_blocks,
                "active_sequences": len(self._tables),
                "committed_kv_tokens": committed,
                "reserved_token_slots": reserved,
                "unused_reserved_slots": unused_reserved,
                "pool_slot_utilization": round(slot_util, 6),
                "peak_used_physical_blocks": self._peak_used_physical_blocks,
            }
