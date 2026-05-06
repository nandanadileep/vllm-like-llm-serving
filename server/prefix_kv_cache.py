"""Exact-match MLX prompt KV cache for ``mlx_lm.batch_generate``.

Stores ``prompt_caches`` entries keyed by the full tokenized prompt (SHA-256
over token ids). This is intentionally **narrow**: identical prompts reuse KV;
prefix-of-prompt sharing without a warm pass is not implemented here.

Thread-safe for concurrent readers; the scheduler worker performs writes.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from typing import Any, List, Optional


class PrefixKVCache:
    def __init__(self, enabled: bool, max_entries: int, ttl_sec: float) -> None:
        self.enabled = enabled
        self.max_entries = max(1, max_entries)
        self.ttl_sec = max(0.0, ttl_sec)
        self._lock = threading.Lock()
        self._entries: "OrderedDict[str, tuple[Any, float]]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _hash_token_ids(token_ids: List[int]) -> str:
        h = hashlib.sha256()
        for t in token_ids:
            h.update(int(t).to_bytes(4, byteorder="little", signed=False))
        return h.hexdigest()

    def lookup(self, token_ids: List[int]) -> Optional[Any]:
        if not self.enabled:
            return None
        key = self._hash_token_ids(token_ids)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.misses += 1
                return None
            caches, ts = entry
            if self.ttl_sec > 0 and (time.monotonic() - ts) > self.ttl_sec:
                del self._entries[key]
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            self.hits += 1
            return caches

    def store(self, token_ids: List[int], caches: Any) -> None:
        if not self.enabled or caches is None:
            return
        key = self._hash_token_ids(token_ids)
        with self._lock:
            self._entries[key] = (caches, time.monotonic())
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def metrics(self) -> dict[str, int]:
        with self._lock:
            return {
                "prefix_cache_hits": self.hits,
                "prefix_cache_misses": self.misses,
                "prefix_cache_entries": len(self._entries),
            }
