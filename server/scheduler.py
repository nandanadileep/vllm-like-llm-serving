"""Request scheduling and batched inference.

Parity notes for writing about this stack versus vLLM:

- **Continuous batching (decode)**: `mlx_lm.batch_generate` drives MLX’s
  `BatchGenerator`, which interleaves chunked prefill and batched decode steps
  for all sequences inserted in that wave (see upstream ``mlx_lm.generate``).
- **This file’s PagedKVCacheManager**: a Python block-pool *mimic* (global
  physical blocks, per-sequence block tables, ``resolve()`` page walks,
  fragmentation-style metrics). It is not wired into MLX attention; vLLM’s
  PagedAttention is GPU-side block tables inside the model.
- **Optional ``MLX_GATHER_PAGED_KV=1``**: monkey-patches ``mlx_lm``’s
  ``BatchKVCache`` with :mod:`server.gather_batch_paged_kv` so KV is also stored
  in a physical block tensor and **re-materialized via ``mx.take`` each step**
  before ``scaled_dot_product_attention`` (dense SDPA, paged backing store).
- **Optional features (env-gated)**:
  - ``PREFIX_CACHE_ENABLED``: ``mlx_lm`` prompt cache reuse (exact + optional
    shared prefix via ``PREFIX_CACHE_SHARED_TEXT`` warm).
  - ``STREAM_USE_MLX``: SSE yields real ``stream_generate`` segments (MLX lock
    interleaves with the batch worker one step at a time).
  - ``MLX_GLOBAL_BATCH_GENERATOR``: one long-lived ``BatchGenerator`` loop
    (continuous admission) instead of wave-only ``batch_generate``.
  - ``SPECULATIVE_DECODE`` + ``DRAFT_MODEL_PATH``: speculative decode via
    ``mlx_lm.generate`` (single) and, when supported by installed ``mlx_lm``,
    ``batch_generate`` (multi-request waves).
- **Not attempted here**: CUDA kernels, tensor/pipeline parallel, or vLLM-style
  fused paged attention kernels.
"""

import os
import queue as _queue_module
import threading
import time
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Dict, Iterator, List, Optional

import mlx_lm
import psutil
from mlx_lm import stream_generate as mlx_stream_generate

from server.paged_attention import BlockPoolExhausted, PagedKVCacheManager
from server.prefix_kv_cache import PrefixKVCache


@dataclass
class RequestItem:
    request_id: str
    prompt: str
    user_id: str
    """Tokenizer output for ``prompt``; used for token-accurate prefill/KV."""
    prompt_token_ids: List[int] = field(default_factory=list)
    created_at: float = field(default_factory=time.monotonic)
    done_event: threading.Event = field(default_factory=threading.Event)
    result: Optional[str] = None
    prefill_cursor: int = 0
    is_prefilling: bool = True
    max_tokens: int = 200
    ttft: Optional[float] = None
    prompt_cache: Optional[Any] = None
    partial_cache: Optional[Any] = None
    prefill_chunk_start: int = 0  # token index where the current chunk begins


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_truthy(name: str) -> bool:
    value = os.getenv(name)
    if value is None:
        return False
    return value.strip().lower() in ("1", "true", "yes", "on")


class Scheduler:
    def __init__(self, batch_size: int = 4, batch_timeout: float = 0.05) -> None:
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.prefill_chunk_size = _env_int("PREFILL_CHUNK_SIZE", 128)
        self.use_global_batch = _env_truthy("MLX_GLOBAL_BATCH_GENERATOR")
        self.queue: List[RequestItem] = []
        self.lock = threading.Lock()
        self.kv = PagedKVCacheManager(
            block_size=_env_int("PAGED_BLOCK_SIZE", 16),
            num_blocks=_env_int("PAGED_NUM_BLOCKS", 256),
        )
        self._prefix_cache = PrefixKVCache(
            enabled=_env_truthy("PREFIX_CACHE_ENABLED"),
            max_entries=_env_int("PREFIX_CACHE_MAX_ENTRIES", 256),
            ttl_sec=float(os.getenv("PREFIX_CACHE_TTL_SEC") or 0.0),
        )
        self._model_path = os.getenv("MODEL_PATH", "mlx-community/Llama-3.2-1B-Instruct-4bit")
        self.model = None
        self.tokenizer = None
        self._draft_model = None
        self._draft_model_path: Optional[str] = None
        self._batch_spec_decode_unsupported = False
        self._model_ready = threading.Event()
        self._stream_queues: Dict[str, "_queue_module.Queue[Optional[str]]"] = {}
        self.total_batches = 0
        self.total_wait_time = 0.0
        self.total_processed = 0
        self.total_preemptions = 0
        self.total_prefill_chunks = 0
        self.total_tokens_generated = 0
        self.total_ttft = 0.0
        self.max_queue_length = 0
        self.global_batch_steps = 0
        self.speculative_generations = 0
        self._process = psutil.Process(os.getpid())
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._batch_loop, daemon=True)
        self._worker.start()

    def _kv_target_after_prefill_chunk(self, item: RequestItem) -> int:
        """Cumulative prompt tokens the simulated KV must cover after this chunk."""
        nt = len(item.prompt_token_ids)
        if nt == 0:
            return max(1, item.prefill_cursor + 1)
        chunk = min(self.prefill_chunk_size, max(0, nt - item.prefill_cursor))
        return max(1, item.prefill_cursor + chunk)

    def _kv_target_decode(self, item: RequestItem) -> int:
        """Upper-bound KV footprint: full prompt plus max new tokens."""
        nt = len(item.prompt_token_ids)
        return max(1, nt + item.max_tokens)

    def _encode_prompt_tokens(self, prompt: str) -> List[int]:
        """Match mlx_lm string encoding so batch and single-path prompts align."""
        tok = self.tokenizer
        bos = getattr(tok, "bos_token", None)
        add_special_tokens = bos is None or not prompt.startswith(bos)
        return list(tok.encode(prompt, add_special_tokens=add_special_tokens))

    def format_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Render OpenAI-style chat messages through the loaded model template."""
        self._model_ready.wait()
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def count_completion_tokens(self, text: str) -> int:
        self._model_ready.wait()
        return len(self.tokenizer.encode(text))

    def _speculative_enabled(self) -> bool:
        return _env_truthy("SPECULATIVE_DECODE") and bool(os.getenv("DRAFT_MODEL_PATH"))

    def _ensure_draft_model(self) -> None:
        path = os.getenv("DRAFT_MODEL_PATH")
        if not path:
            return
        if self._draft_model is not None and path == self._draft_model_path:
            return
        self._draft_model, _ = mlx_lm.load(path)
        self._draft_model_path = path

    def _speculative_generate_single(self, item: RequestItem) -> Optional[str]:
        """Single-stream speculative decode via ``mlx_lm.generate``."""
        if not self._speculative_enabled():
            return None
        self._ensure_draft_model()
        if self._draft_model is None:
            return None
        text = mlx_lm.generate(
            self.model,
            self.tokenizer,
            prompt=item.prompt,
            max_tokens=item.max_tokens,
            verbose=False,
            draft_model=self._draft_model,
            num_draft_tokens=_env_int("NUM_DRAFT_TOKENS", 4),
        )
        with self.lock:
            self.speculative_generations += 1
        return text

    def _batch_speculative_kwargs(self) -> Dict[str, Any]:
        """Best-effort speculative kwargs for ``batch_generate``.

        Some ``mlx_lm`` versions may not support batched speculative arguments.
        We guard usage and permanently disable retries on first incompatibility.
        """
        if not self._speculative_enabled() or self._batch_spec_decode_unsupported:
            return {}
        self._ensure_draft_model()
        if self._draft_model is None:
            return {}
        return {
            "draft_model": self._draft_model,
            "num_draft_tokens": _env_int("NUM_DRAFT_TOKENS", 4),
        }

    def _run_inference_batch(self, items: List[RequestItem]) -> None:
        """Batched ``batch_generate``, with optional prefix caches and speculative fast path."""
        if not items:
            return
        if (
            len(items) == 1
            and self._speculative_enabled()
            and items[0].prompt_cache is None
            and items[0].partial_cache is None
        ):
            now = time.monotonic()
            items[0].ttft = now - items[0].created_at
            text = self._speculative_generate_single(items[0])
            if text is not None:
                items[0].result = text
                tokens_out = len(self.tokenizer.encode(text))
                with self.lock:
                    self.total_tokens_generated += tokens_out
                    self.total_ttft += items[0].ttft
                return
        now = time.monotonic()
        for item in items:
            item.ttft = now - item.created_at

        prompts: List[List[int]] = []
        prompt_caches: Optional[List[Any]] = None
        per_item_caches: List[Optional[Any]] = []
        for item in items:
            if item.prompt_cache is not None:
                prompts.append([])
                per_item_caches.append(item.prompt_cache)
            else:
                prompts.append(list(item.prompt_token_ids[item.prefill_chunk_start :]))
                per_item_caches.append(item.partial_cache)
        max_tokens = [item.max_tokens for item in items]
        if self._prefix_cache.enabled:
            for idx, it in enumerate(items):
                if per_item_caches[idx] is None:
                    cached, matched_tokens = self._prefix_cache.lookup_prefix(
                        list(it.prompt_token_ids)
                    )
                    if cached is not None:
                        per_item_caches[idx] = cached
                        prompts[idx] = list(it.prompt_token_ids[matched_tokens:])
                        if matched_tokens == len(it.prompt_token_ids):
                            it.prompt_cache = cached
        if any(c is not None for c in per_item_caches):
            prompt_caches = per_item_caches
        return_caches = self._prefix_cache.enabled
        batch_kwargs: Dict[str, Any] = {
            "prompt_caches": prompt_caches,
            "max_tokens": max_tokens,
            "verbose": False,
            "return_prompt_caches": return_caches,
        }
        batch_kwargs.update(self._batch_speculative_kwargs())
        try:
            batch_response = mlx_lm.batch_generate(
                self.model,
                self.tokenizer,
                prompts,
                **batch_kwargs,
            )
            if "draft_model" in batch_kwargs:
                with self.lock:
                    self.speculative_generations += len(items)
        except TypeError:
            if "draft_model" not in batch_kwargs:
                raise
            # Installed mlx_lm does not accept batched speculative args.
            self._batch_spec_decode_unsupported = True
            batch_response = mlx_lm.batch_generate(
                self.model,
                self.tokenizer,
                prompts,
                prompt_caches=prompt_caches,
                max_tokens=max_tokens,
                verbose=False,
                return_prompt_caches=return_caches,
            )
        for item, text in zip(items, batch_response.texts):
            item.result = text
            tokens_out = len(self.tokenizer.encode(text))
            with self.lock:
                self.total_tokens_generated += tokens_out
                self.total_ttft += item.ttft
        if return_caches and batch_response.caches is not None:
            for item, caches in zip(items, batch_response.caches):
                if caches is not None:
                    item.prompt_cache = caches
                    self._prefix_cache.store(list(item.prompt_token_ids), caches)

    def _run_prefill_chunks(self, items: List[RequestItem]) -> None:
        if not items:
            return
        prompts = [
            list(item.prompt_token_ids[item.prefill_chunk_start : item.prefill_cursor])
            for item in items
        ]
        caches = [item.partial_cache for item in items]
        has_any_cache = any(c is not None for c in caches)
        try:
            response = mlx_lm.batch_generate(
                self.model,
                self.tokenizer,
                prompts,
                prompt_caches=caches if has_any_cache else None,
                max_tokens=[0] * len(items),
                verbose=False,
                return_prompt_caches=True,
            )
        except (AssertionError, ValueError):
            response = mlx_lm.batch_generate(
                self.model,
                self.tokenizer,
                prompts,
                prompt_caches=caches if has_any_cache else None,
                max_tokens=[1] * len(items),
                verbose=False,
                return_prompt_caches=True,
            )
        if response.caches is None:
            # Avoid reprocessing the same chunk if this mlx_lm build declines
            # to return prompt caches for prefill-only calls.
            for item in items:
                item.prefill_chunk_start = item.prefill_cursor
            return
        for item, cache in zip(items, response.caches):
            item.partial_cache = cache
            item.prefill_chunk_start = item.prefill_cursor

    def submit_request(self, prompt: str, user_id: str, request_id: str, max_tokens: int = 200) -> str:
        self._model_ready.wait()
        prompt_token_ids = self._encode_prompt_tokens(prompt)
        item = RequestItem(
            request_id=request_id,
            prompt=prompt,
            user_id=user_id,
            prompt_token_ids=prompt_token_ids,
            max_tokens=max_tokens,
            is_prefilling=not self.use_global_batch,
            prefill_cursor=len(prompt_token_ids) if self.use_global_batch else 0,
        )
        with self.lock:
            self.queue.append(item)
            if len(self.queue) > self.max_queue_length:
                self.max_queue_length = len(self.queue)
        item.done_event.wait()
        if item.result is None:
            raise RuntimeError("Scheduler completed request without a result")
        return item.result

    def submit_request_stream_tokens(
        self, prompt: str, user_id: str, request_id: str, max_tokens: int = 200
    ) -> Iterator[str]:
        self._model_ready.wait()
        if self.use_global_batch:
            # Real token-by-token streaming from the global batch loop.
            q: "_queue_module.Queue[Optional[str]]" = _queue_module.Queue()
            with self.lock:
                self._stream_queues[request_id] = q

            prompt_token_ids = self._encode_prompt_tokens(prompt)
            item = RequestItem(
                request_id=request_id,
                prompt=prompt,
                user_id=user_id,
                prompt_token_ids=prompt_token_ids,
                max_tokens=max_tokens,
                is_prefilling=False,
                prefill_cursor=len(prompt_token_ids),
            )
            with self.lock:
                self.queue.append(item)
                if len(self.queue) > self.max_queue_length:
                    self.max_queue_length = len(self.queue)

            try:
                while True:
                    token = q.get()
                    if token is None:
                        break
                    yield token
            finally:
                with self.lock:
                    self._stream_queues.pop(request_id, None)
            return

        # Fallback: full generation, word-split (non-global-batch mode).
        result = self.submit_request(prompt, user_id, request_id, max_tokens=max_tokens)
        for word in result.split():
            yield word

    def submit_request_stream(
        self, prompt: str, user_id: str, request_id: str, max_tokens: int = 200
    ) -> Iterator[str]:
        yield from self.submit_request_stream_tokens(
            prompt=prompt,
            user_id=user_id,
            request_id=request_id,
            max_tokens=max_tokens,
        )

    def get_metrics(self) -> dict[str, float | int]:
        with self.lock:
            avg_wait_time = (
                self.total_wait_time / self.total_processed
                if self.total_processed > 0
                else 0.0
            )
            avg_ttft = (
                self.total_ttft / self.total_processed
                if self.total_processed > 0
                else 0.0
            )
            metrics = {
                "total_batches": self.total_batches,
                "total_processed": self.total_processed,
                "avg_wait_time": avg_wait_time,
                "avg_ttft": avg_ttft,
                "max_queue_length": self.max_queue_length,
                "total_preemptions": self.total_preemptions,
                "total_prefill_chunks": self.total_prefill_chunks,
                "total_tokens_generated": self.total_tokens_generated,
                "memory_mb": self._process.memory_info().rss / 1024 / 1024,
                "global_batch_steps": self.global_batch_steps,
                "speculative_generations": self.speculative_generations,
            }
        metrics.update(self.kv.metrics())
        metrics.update(self._prefix_cache.metrics())
        return metrics

    def _batch_loop(self) -> None:
        if _env_truthy("MLX_GATHER_PAGED_KV"):
            from server.gather_batch_paged_kv import install_gather_batch_paged_kv

            install_gather_batch_paged_kv()
        self.model, self.tokenizer = mlx_lm.load(self._model_path)
        self._model_ready.set()
        if self.use_global_batch:
            self._global_batch_loop()
            return
        while not self._stop_event.is_set():
            batch = self._get_next_batch()
            if not batch:
                time.sleep(0.001)
                continue
            self._process_batch(batch)

    def _global_batch_loop(self) -> None:
        """Continuous ``BatchGenerator`` loop: admit from the queue and step decode."""
        gen_mod = import_module("mlx_lm.generate")
        bg = gen_mod.BatchGenerator(
            self.model,
            stop_tokens=[[t] for t in self.tokenizer.eos_token_ids],
            prefill_batch_size=self.batch_size,
            completion_batch_size=self.batch_size,
        )
        uid_to_item: Dict[int, RequestItem] = {}
        gen_tokens: Dict[int, List[int]] = {}
        ttft_marked: set[int] = set()

        while not self._stop_event.is_set():
            self._global_try_admit(bg, uid_to_item, gen_tokens, ttft_marked)
            responses = bg.next_generated()
            if not responses:
                if not uid_to_item:
                    time.sleep(0.001)
                continue
            with self.lock:
                self.global_batch_steps += 1
            for r in responses:
                uid = int(r.uid)
                item = uid_to_item.get(uid)
                if item is None:
                    continue
                if uid not in ttft_marked:
                    item.ttft = time.monotonic() - item.created_at
                    ttft_marked.add(uid)
                if r.finish_reason is None:
                    token = int(r.token)
                    gen_tokens.setdefault(uid, []).append(token)
                    token_text = self.tokenizer.decode([token])
                    q = self._stream_queues.get(item.request_id)
                    if q is not None:
                        q.put(token_text)
                    continue
                if r.finish_reason != "stop":
                    gen_tokens.setdefault(uid, []).append(int(r.token))
                tokens = gen_tokens.pop(uid, [])
                text = self.tokenizer.decode(tokens) if tokens else ""
                item.result = text
                tokens_out = len(self.tokenizer.encode(text))
                now = time.monotonic()
                wait = now - item.created_at
                with self.lock:
                    self.total_processed += 1
                    self.total_wait_time += wait
                    self.total_tokens_generated += tokens_out
                    self.total_ttft += item.ttft
                if self._prefix_cache.enabled and r.prompt_cache is not None:
                    self._prefix_cache.store(list(item.prompt_token_ids), r.prompt_cache)
                q = self._stream_queues.pop(item.request_id, None)
                if q is not None:
                    q.put(None)
                self.kv.free(item.request_id)
                item.done_event.set()
                uid_to_item.pop(uid, None)
                ttft_marked.discard(uid)

    def _global_try_admit(
        self,
        bg: Any,
        uid_to_item: Dict[int, RequestItem],
        gen_tokens: Dict[int, List[int]],
        ttft_marked: set[int],
    ) -> None:
        """Pull up to ``batch_size`` waiting requests and ``insert`` them."""
        with self.lock:
            if not self.queue:
                return
            admit: List[RequestItem] = []
            while self.queue and len(admit) < self.batch_size:
                admit.append(self.queue.pop(0))
        if not admit:
            return
        kept: List[RequestItem] = []
        for item in admit:
            try:
                self.kv.allocate(
                    item.request_id,
                    len(item.prompt_token_ids) + item.max_tokens,
                )
            except BlockPoolExhausted:
                for it in kept:
                    self.kv.free(it.request_id)
                with self.lock:
                    self.queue.insert(0, item)
                    for it in reversed(kept):
                        self.queue.insert(0, it)
                with self.lock:
                    self.total_preemptions += 1
                return
            kept.append(item)
        prompts = [list(it.prompt_token_ids) for it in kept]
        max_tokens = [it.max_tokens for it in kept]
        caches_arg: Optional[List[Any]]
        if self._prefix_cache.enabled:
            caches_arg = []
            for idx, it in enumerate(kept):
                cache, matched_tokens = self._prefix_cache.lookup_prefix(
                    list(it.prompt_token_ids)
                )
                # BatchGenerator cannot admit an empty prompt segment. If a
                # cache covers the whole prompt, fall back to the full prompt.
                if cache is not None and matched_tokens < len(it.prompt_token_ids):
                    caches_arg.append(cache)
                    prompts[idx] = list(it.prompt_token_ids[matched_tokens:])
                else:
                    caches_arg.append(None)
            if not any(c is not None for c in caches_arg):
                caches_arg = None
        else:
            caches_arg = None
        uids = bg.insert(prompts, max_tokens, caches=caches_arg)
        with self.lock:
            self.total_batches += 1
        for uid, it in zip(uids, kept):
            uid_to_item[int(uid)] = it
            gen_tokens[int(uid)] = []
            ttft_marked.discard(int(uid))

    def _get_next_batch(self) -> List[RequestItem]:
        """Pop up to batch_size requests; chunking is applied in _process_batch."""
        with self.lock:
            if not self.queue:
                return []

            oldest_wait = time.monotonic() - self.queue[0].created_at
            should_dispatch = (
                len(self.queue) >= self.batch_size
                or oldest_wait >= self.batch_timeout
            )
            if not should_dispatch:
                return []

            batch = self.queue[: self.batch_size]
            del self.queue[: len(batch)]
            return batch

    def _reset_prefill_state(self, item: RequestItem) -> None:
        """KV was freed; restart prefill bookkeeping (no swap-to-CPU)."""
        item.prefill_cursor = 0
        item.is_prefilling = True
        item.prompt_cache = None
        item.partial_cache = None
        item.prefill_chunk_start = 0

    def _allocate_batch_with_preemption(self, batch: List[RequestItem]) -> List[RequestItem]:
        """Allocate KV for all items in batch, preempting latest-arrived on exhaustion."""
        working = list(batch)
        while True:
            try:
                for item in working:
                    if item.is_prefilling:
                        n = self._kv_target_after_prefill_chunk(item)
                    else:
                        n = self._kv_target_decode(item)
                    self.kv.allocate(item.request_id, num_tokens=n)
                return working
            except BlockPoolExhausted:
                if not working:
                    raise RuntimeError(
                        "BlockPoolExhausted with empty working batch"
                    ) from None
                victim = max(working, key=lambda x: x.created_at)
                self.kv.free(victim.request_id)
                self._reset_prefill_state(victim)
                working.remove(victim)
                with self.lock:
                    self.queue.insert(0, victim)
                    self.total_preemptions += 1
                if not working:
                    return []

    def _process_batch(self, batch: List[RequestItem]) -> None:
        working = self._allocate_batch_with_preemption(batch)
        if not working:
            return
        now = time.monotonic()
        total_batch_wait = sum(now - item.created_at for item in working)
        decode_finished: List[RequestItem] = []
        try:
            still_prefilling: List[RequestItem] = []
            to_decode: List[RequestItem] = []

            for item in working:
                if item.is_prefilling:
                    nt = len(item.prompt_token_ids)
                    chunk = min(
                        self.prefill_chunk_size,
                        max(0, nt - item.prefill_cursor),
                    )
                    item.prefill_chunk_start = item.prefill_cursor
                    item.prefill_cursor += chunk
                    with self.lock:
                        self.total_prefill_chunks += 1
                    if item.prefill_cursor >= nt:
                        item.is_prefilling = False
                        try:
                            self.kv.allocate(item.request_id, num_tokens=self._kv_target_decode(item))
                            to_decode.append(item)
                        except BlockPoolExhausted:
                            self._reset_prefill_state(item)
                            with self.lock:
                                self.queue.insert(0, item)
                                self.total_preemptions += 1
                    else:
                        still_prefilling.append(item)
                else:
                    to_decode.append(item)

            self._run_prefill_chunks(still_prefilling)
            for item in still_prefilling:
                with self.lock:
                    self.queue.append(item)

            completed_decodes = len(to_decode)
            if to_decode:
                self._run_inference_batch(to_decode)
                for item in to_decode:
                    item.done_event.set()
                    decode_finished.append(item)

            with self.lock:
                self.total_batches += 1
                self.total_processed += completed_decodes
                self.total_wait_time += total_batch_wait
        finally:
            for item in decode_finished:
                self.kv.free(item.request_id)
