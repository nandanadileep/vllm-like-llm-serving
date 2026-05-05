"""PagedAttention-style KV memory: block pool + per-sequence block tables.

vLLM pairs this non-contiguous KV layout with a fused CUDA kernel that gathers
K/V using a block table. This module implements the allocator and logical→physical
mapping in pure Python; it does not ship a GPU attention implementation.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class LogicalMapping:
    """Maps a logical token position to a physical block id and intra-block slot."""

    physical_block: int
    slot: int


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

    def ensure_tokens(self, seq_id: str, total_tokens: int) -> None:
        """Grow KV storage so the sequence can hold at least total_tokens tokens."""
        if total_tokens < 0:
            raise ValueError("total_tokens must be non-negative")
        with self._lock:
            current = self._token_counts.get(seq_id, 0)
            if total_tokens <= current:
                return
            table = self._tables.setdefault(seq_id, [])
            capacity = len(table) * self.block_size
            while capacity < total_tokens:
                if not self._free:
                    raise RuntimeError("PagedKVCache OOM: no free blocks")
                block_id = self._free.pop()
                table.append(block_id)
                capacity += self.block_size
            self._token_counts[seq_id] = total_tokens

    def release_sequence(self, seq_id: str) -> None:
        with self._lock:
            table = self._tables.pop(seq_id, None)
            self._token_counts.pop(seq_id, None)
            if not table:
                return
            for block_id in table:
                self._free.add(block_id)

    def resolve(self, seq_id: str, logical_token_index: int) -> LogicalMapping:
        if logical_token_index < 0:
            raise ValueError("logical_token_index must be non-negative")
        with self._lock:
            table = self._tables.get(seq_id)
            if not table:
                raise KeyError(f"unknown sequence {seq_id}")
            block_idx = logical_token_index // self.block_size
            if block_idx >= len(table):
                raise IndexError("logical_token_index out of allocated range")
            slot = logical_token_index % self.block_size
            return LogicalMapping(physical_block=table[block_idx], slot=slot)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "kv_total_blocks": self.num_blocks,
                "kv_free_blocks": len(self._free),
                "kv_active_sequences": len(self._tables),
            }
