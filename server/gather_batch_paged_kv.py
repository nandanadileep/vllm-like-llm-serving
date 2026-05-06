"""Gather-based paged KV storage for ``mlx_lm`` batched generation.

``mlx_lm`` attention still calls ``mx.fast.scaled_dot_product_attention`` on
**dense** ``(B, H, T, D)`` keys/values. This module keeps a **parallel** layout
where KV lives in a global physical tensor ``(num_blocks, block_size, H, D)``
plus per-row block tables, then **materializes** the same dense view via
``mx.take`` each step — paged *allocation* + gather, dense SDPA.

This is intentionally conservative: after every ``BatchKVCache`` mutating
operation we **re-scatter** the active prefix from the dense buffers into the
physical pool, then **gather** it back out. That is slower than incremental
bookkeeping but hard to desync from upstream ``mlx_lm`` behavior.

Enable with ``MLX_GATHER_PAGED_KV=1`` (see ``server.scheduler``).
"""

from __future__ import annotations

import os
from typing import List, Sequence

import mlx.core as mx

from mlx_lm.models.cache import BatchKVCache as _OriginalBatchKVCache


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


class GatherBatchPagedKVCache(_OriginalBatchKVCache):
    """Drop-in ``BatchKVCache`` with physical block tensors + gather return path."""

    def __init__(self, left_padding: Sequence[int]) -> None:
        super().__init__(list(left_padding))
        self._block_size = max(1, _env_int("PAGED_BLOCK_SIZE", 16))
        self._num_phys_blocks = max(1, _env_int("GATHER_PAGED_NUM_BLOCKS", 512))
        self._phys_k: mx.array | None = None
        self._phys_v: mx.array | None = None
        self._free: set[int] = set()
        self._tables: List[List[int]] = [[] for _ in range(len(left_padding))]

    def _lazy_init_phys(self, n_kv_heads: int, k_dim: int, v_dim: int, dtype) -> None:
        if self._phys_k is not None:
            return
        nb, bs = self._num_phys_blocks, self._block_size
        self._phys_k = mx.zeros((nb, bs, n_kv_heads, k_dim), dtype=dtype)
        self._phys_v = mx.zeros((nb, bs, n_kv_heads, v_dim), dtype=dtype)
        self._reset_free()

    def _reset_free(self) -> None:
        self._free = set(range(self._num_phys_blocks))

    def _alloc_block(self) -> int:
        if not self._free:
            raise RuntimeError(
                "GatherBatchPagedKVCache: physical block pool exhausted "
                f"(num_blocks={self._num_phys_blocks})."
            )
        return self._free.pop()

    def _rebuild_tables_from_dense(self) -> None:
        """Scatter active dense prefix into physical blocks (per batch row)."""
        if self.keys is None:
            self._reset_free()
            self._tables = [[] for _ in range(len(self.left_padding))]
            return

        self._reset_free()
        for row in self._tables:
            row.clear()

        B = int(self.keys.shape[0])
        H = int(self.keys.shape[1])
        T = int(self._idx)
        Dk = int(self.keys.shape[3])
        Dv = int(self.values.shape[3])

        self._lazy_init_phys(H, Dk, Dv, self.keys.dtype)

        assert self._phys_k is not None and self._phys_v is not None

        for b in range(B):
            for t in range(T):
                li = t // self._block_size
                if li == len(self._tables[b]):
                    self._tables[b].append(self._alloc_block())
                block_id = self._tables[b][li]
                slot = t % self._block_size
                self._phys_k[block_id, slot, :, :] = self.keys[b, :, t, :]
                self._phys_v[block_id, slot, :, :] = self.values[b, :, t, :]

    def _gather_prefix(self) -> tuple[mx.array, mx.array]:
        """Rebuild dense (B,H,T,*) views from the physical pool."""
        if self.keys is None:
            raise RuntimeError("GatherBatchPagedKVCache: gather with empty cache")

        B = int(self.keys.shape[0])
        H = int(self.keys.shape[1])
        T = int(self._idx)
        Dk = int(self.keys.shape[3])
        Dv = int(self.values.shape[3])

        assert self._phys_k is not None and self._phys_v is not None

        flat_k = self._phys_k.reshape(-1, H, Dk)
        flat_v = self._phys_v.reshape(-1, H, Dv)

        idx_list: list[int] = []
        for b in range(B):
            for t in range(T):
                li = t // self._block_size
                block_id = self._tables[b][li]
                slot = t % self._block_size
                idx_list.append(block_id * self._block_size + slot)

        idx = mx.array(idx_list, dtype=mx.int32).reshape(B, T)
        idx_flat = idx.reshape(-1)

        gk = mx.take(flat_k, idx_flat, axis=0).reshape(B, T, H, Dk).transpose(0, 2, 1, 3)
        gv = mx.take(flat_v, idx_flat, axis=0).reshape(B, T, H, Dv).transpose(0, 2, 1, 3)
        return gk, gv

    def update_and_fetch(self, keys, values):
        super().update_and_fetch(keys, values)
        self._rebuild_tables_from_dense()
        return self._gather_prefix()

    def prepare(self, *, left_padding=None, lengths=None, right_padding=None):
        super().prepare(
            left_padding=left_padding, lengths=lengths, right_padding=right_padding
        )
        if self.keys is not None:
            self._rebuild_tables_from_dense()

    def finalize(self):
        super().finalize()
        if self.keys is not None:
            self._rebuild_tables_from_dense()

    def filter(self, batch_indices):
        super().filter(batch_indices)
        self._tables = [self._tables[int(i)] for i in batch_indices]
        if self.keys is not None:
            self._rebuild_tables_from_dense()

    def extend(self, other: _OriginalBatchKVCache):
        n_other = int(other.offset.shape[0])
        if hasattr(other, "_tables"):
            tables_other = list(other._tables)  # type: ignore[attr-defined]
        else:
            tables_other = [[] for _ in range(n_other)]
        super().extend(other)
        self._tables.extend(tables_other)
        if self.keys is not None:
            self._rebuild_tables_from_dense()

    def trim(self, n):
        out = super().trim(n)
        if self.keys is not None:
            self._rebuild_tables_from_dense()
        return out

    @classmethod
    def merge(cls, caches):
        merged = _OriginalBatchKVCache.merge(caches)
        pad = merged.left_padding.tolist()
        out = cls(pad)
        out.keys = merged.keys
        out.values = merged.values
        out.offset = merged.offset
        out._idx = merged._idx
        out._right_padding = merged._right_padding
        out._tables = [[] for _ in range(int(merged.offset.shape[0]))]
        if out.keys is not None:
            out._lazy_init_phys(
                int(out.keys.shape[1]),
                int(out.keys.shape[3]),
                int(out.values.shape[3]),
                out.keys.dtype,
            )
            out._rebuild_tables_from_dense()
        return out

    @property
    def state(self):
        return super().state

    @state.setter
    def state(self, v):
        self.keys, self.values, self.offset, self.left_padding = v
        self._idx = self.keys.shape[2]
        if self.keys is not None:
            self._lazy_init_phys(
                int(self.keys.shape[1]),
                int(self.keys.shape[3]),
                int(self.values.shape[3]),
                self.keys.dtype,
            )
            self._rebuild_tables_from_dense()


def install_gather_batch_paged_kv() -> None:
    """Monkey-patch ``mlx_lm`` to use :class:`GatherBatchPagedKVCache` globally.

    ``mlx_lm.generate`` binds ``BatchKVCache`` at import time, so both the
    ``models.cache`` module and the ``generate`` module must be patched.
    """
    import mlx_lm.generate as mlx_gen
    import mlx_lm.models.cache as mlx_cache

    mlx_cache.BatchKVCache = GatherBatchPagedKVCache
    mlx_gen.BatchKVCache = GatherBatchPagedKVCache


def uninstall_gather_batch_paged_kv() -> None:
    """Restore the original ``BatchKVCache`` class (mainly for tests)."""
    import mlx_lm.generate as mlx_gen
    import mlx_lm.models.cache as mlx_cache

    mlx_cache.BatchKVCache = _OriginalBatchKVCache
    mlx_gen.BatchKVCache = _OriginalBatchKVCache
