"""MLX prompt KV cache for ``mlx_lm.batch_generate``.

Stores prompt caches in a token trie so requests can reuse the longest cached
prefix, not just exact full-prompt matches.

Thread-safe; the scheduler worker performs writes.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class _TrieNode:
    children: Dict[int, "_TrieNode"] = field(default_factory=dict)
    cache: Optional[Any] = None
    last_used: float = 0.0
    token_count: int = 0
    parent: Optional["_TrieNode"] = None
    token: Optional[int] = None


class PrefixKVCache:
    def __init__(self, enabled: bool, max_entries: int, ttl_sec: float) -> None:
        self.enabled = enabled
        self.max_entries = max(1, max_entries)
        self.ttl_sec = max(0.0, ttl_sec)
        self._block_size = 16
        self._lock = threading.Lock()
        self._root = _TrieNode()
        self._leaves: List[_TrieNode] = []
        self._entry_count = 0
        self.hits = 0
        self.shared_prefix_hits = 0
        self.misses = 0
        self.matched_prefix_tokens = 0

    def _cache_expired(self, node: _TrieNode, now: float) -> bool:
        return (
            node.cache is not None
            and self.ttl_sec > 0
            and (now - node.last_used) > self.ttl_sec
        )

    def _prune_empty_ancestors(self, node: _TrieNode) -> None:
        while node.parent is not None and node.cache is None and not node.children:
            parent = node.parent
            if node.token is not None and parent.children.get(node.token) is node:
                del parent.children[node.token]
            node = parent

    def _clear_cache(self, node: _TrieNode) -> None:
        if node.cache is not None:
            node.cache = None
            node.last_used = 0.0
            self._entry_count -= 1
            if node in self._leaves:
                self._leaves.remove(node)
            self._prune_empty_ancestors(node)

    def _evict_if_needed(self) -> None:
        while self._entry_count > self.max_entries:
            if not self._leaves:
                self._entry_count = 0
                return
            self._clear_cache(min(self._leaves, key=lambda node: node.last_used))

    def lookup_prefix(self, token_ids: List[int]) -> Tuple[Optional[Any], int]:
        """Return the longest cached prefix and the number of matched tokens."""
        if not self.enabled:
            return None, 0
        now = time.monotonic()
        with self._lock:
            node = self._root
            best_cache: Optional[Any] = None
            best_len = 0
            for idx, token_id in enumerate(token_ids, start=1):
                child = node.children.get(int(token_id))
                if child is None:
                    break
                node = child
                if self._cache_expired(node, now):
                    self._clear_cache(node)
                if node.cache is not None:
                    best_cache = node.cache
                    best_len = idx
                    node.last_used = now
            if best_cache is None:
                self.misses += 1
                return None, 0
            self.hits += 1
            self.matched_prefix_tokens += best_len
            if best_len != len(token_ids):
                self.shared_prefix_hits += 1
            return best_cache, best_len

    def lookup(self, token_ids: List[int]) -> Optional[Any]:
        cache, _ = self.lookup_prefix(token_ids)
        return cache

    def lookup_for_prompt(
        self,
        token_ids: List[int],
        shared_prefix: Optional[List[int]],
    ) -> Optional[Any]:
        """Compatibility wrapper; the trie already finds the best prefix."""
        return self.lookup(token_ids)

    def store(self, token_ids: List[int], caches: Any) -> None:
        if not self.enabled or caches is None:
            return
        aligned_len = (len(token_ids) // self._block_size) * self._block_size
        if aligned_len == 0:
            return
        now = time.monotonic()
        with self._lock:
            node = self._root
            for idx, token_id in enumerate(token_ids[:aligned_len], start=1):
                token = int(token_id)
                child = node.children.get(token)
                if child is None:
                    child = _TrieNode(token_count=idx, parent=node, token=token)
                    node.children[token] = child
                node = child
            if node.cache is None:
                self._entry_count += 1
                self._leaves.append(node)
            node.cache = caches
            node.last_used = now
            self._evict_if_needed()

    def metrics(self) -> dict[str, int]:
        with self._lock:
            return {
                "prefix_cache_hits": self.hits,
                "prefix_cache_shared_hits": self.shared_prefix_hits,
                "prefix_cache_misses": self.misses,
                "prefix_cache_entries": self._entry_count,
                "matched_prefix_tokens": self.matched_prefix_tokens,
                "prefix_cache_matched_prefix_tokens": self.matched_prefix_tokens,
            }
