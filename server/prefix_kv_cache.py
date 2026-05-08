"""MLX prompt KV cache for ``mlx_lm.batch_generate``.

- **Exact match**: full prompt token ids as key.
- **Shared prefix**: optional static prefix (see ``PREFIX_CACHE_SHARED_TEXT`` in
  the scheduler) warmed once; any prompt whose tokens start with that prefix
  can reuse the warmed ``prompt_caches`` for the prefix span.

Thread-safe; the scheduler worker performs writes.
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
        self.shared_prefix_hits = 0
        self.misses = 0

    @staticmethod
    def _hash_token_ids(token_ids: List[int]) -> str:
        h = hashlib.sha256()
        for t in token_ids:
            h.update(int(t).to_bytes(4, byteorder="little", signed=False))
        return h.hexdigest()

    def _take(self, key: str) -> Optional[Any]:
        entry = self._entries.get(key)
        if entry is None:
            return None
        caches, ts = entry
        if self.ttl_sec > 0 and (time.monotonic() - ts) > self.ttl_sec:
            del self._entries[key]
            return None
        self._entries.move_to_end(key)
        return caches

    def lookup(self, token_ids: List[int]) -> Optional[Any]:
        if not self.enabled:
            return None
        key = self._hash_token_ids(token_ids)
        with self._lock:
            caches = self._take(key)
            if caches is None:
                self.misses += 1
                return None
            self.hits += 1
            return caches

    def lookup_for_prompt(
        self,
        token_ids: List[int],
        shared_prefix: Optional[List[int]],
    ) -> Optional[Any]:
        """Prefer exact entry, then a warmed shared-prefix entry if applicable."""
        if not self.enabled:
            return None
        with self._lock:
            exact = self._take(self._hash_token_ids(token_ids))
            if exact is not None:
                self.hits += 1
                return exact
            if (
                shared_prefix
                and len(token_ids) >= len(shared_prefix)
                and token_ids[: len(shared_prefix)] == shared_prefix
            ):
                shared = self._take(self._hash_token_ids(shared_prefix))
                if shared is not None:
                    self.shared_prefix_hits += 1
                    return shared
            self.misses += 1
            return None

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
                "prefix_cache_shared_hits": self.shared_prefix_hits,
                "prefix_cache_misses": self.misses,
                "prefix_cache_entries": len(self._entries),
            }
