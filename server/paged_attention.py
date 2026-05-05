"""PagedAttention-style KV memory: block pool + per-sequence block tables.

Pure Python simulation: no attention math, GPU ops, or model inference.
"""

from __future__ import annotations

import threading
from typing import Dict, List, Tuple


class BlockPoolExhausted(Exception):
    """Raised when allocate() needs more physical blocks than the pool has free."""


class PagedKVCacheManager:
    """Fixed-size block pool with one block table per sequence (no prefix sharing)."""

    def __init__(self, block_size: int, num_blocks: int) -> None:
        if block_size < 1 or num_blocks < 1:
            raise ValueError("block_size and num_blocks must be positive")
        self.block_size = block_size
        self.num_blocks = num_blocks
        self._lock = threading.Lock()
        self._free: set[int] = set(range(num_blocks))
        self._tables: Dict[str, List[int]] = {}
        self._token_counts: Dict[str, int] = {}

    def allocate(self, request_id: str, num_tokens: int) -> None:
        """Reserve enough blocks so this sequence can store at least num_tokens tokens."""
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

    def resolve(
        self, request_id: str, logical_token_index: int
    ) -> Tuple[int, int]:
        """Map logical token index to (physical_block_id, slot_offset)."""
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

    def metrics(self) -> dict[str, int]:
        with self._lock:
            return {
                "total_blocks": self.num_blocks,
                "free_blocks": len(self._free),
                "active_sequences": len(self._tables),
            }
