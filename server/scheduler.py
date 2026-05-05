import os
import threading
import time
from dataclasses import dataclass, field
from typing import Iterator, List, Optional

from server.paged_attention import BlockPoolExhausted, PagedKVCacheManager


@dataclass
class RequestItem:
    request_id: str
    prompt: str
    user_id: str
    created_at: float = field(default_factory=time.monotonic)
    done_event: threading.Event = field(default_factory=threading.Event)
    result: Optional[str] = None
    prefill_cursor: int = 0
    is_prefilling: bool = True


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


class Scheduler:
    def __init__(self, batch_size: int = 4, batch_timeout: float = 0.05) -> None:
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.prefill_chunk_size = _env_int("PREFILL_CHUNK_SIZE", 128)
        self.queue: List[RequestItem] = []
        self.lock = threading.Lock()
        self.kv = PagedKVCacheManager(
            block_size=_env_int("PAGED_BLOCK_SIZE", 16),
            num_blocks=_env_int("PAGED_NUM_BLOCKS", 256),
        )
        self.total_batches = 0
        self.total_wait_time = 0.0
        self.total_processed = 0
        self.total_preemptions = 0
        self.total_prefill_chunks = 0
        self.max_queue_length = 0
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._batch_loop, daemon=True)
        self._worker.start()

    @staticmethod
    def _prompt_words(prompt: str) -> List[str]:
        return prompt.split()

    def _prompt_word_count(self, prompt: str) -> int:
        return max(1, len(self._prompt_words(prompt)))

    def _kv_target_after_prefill_chunk(self, item: RequestItem) -> int:
        """Cumulative word-token count KV must cover after this prefill chunk."""
        words = self._prompt_words(item.prompt)
        nw = len(words)
        if nw == 0:
            return max(1, item.prefill_cursor + 1)
        chunk = min(self.prefill_chunk_size, max(0, nw - item.prefill_cursor))
        return max(1, item.prefill_cursor + chunk)

    def _kv_target_decode(self, item: RequestItem) -> int:
        return self._prompt_word_count(item.prompt)

    def submit_request(self, prompt: str, user_id: str, request_id: str) -> str:
        item = RequestItem(request_id=request_id, prompt=prompt, user_id=user_id)
        with self.lock:
            self.queue.append(item)
            if len(self.queue) > self.max_queue_length:
                self.max_queue_length = len(self.queue)
        item.done_event.wait()
        if item.result is None:
            raise RuntimeError("Scheduler completed request without a result")
        return item.result

    def submit_request_stream(
        self, prompt: str, user_id: str, request_id: str
    ) -> Iterator[str]:
        """Run full batch inference, then yield stub result as word tokens (not incremental decode)."""
        result = self.submit_request(prompt, user_id, request_id)
        for word in result.split():
            yield word

    def get_metrics(self) -> dict[str, float | int]:
        with self.lock:
            avg_wait_time = (
                self.total_wait_time / self.total_processed
                if self.total_processed > 0
                else 0.0
            )
            metrics = {
                "total_batches": self.total_batches,
                "avg_wait_time": avg_wait_time,
                "max_queue_length": self.max_queue_length,
                "total_preemptions": self.total_preemptions,
                "total_prefill_chunks": self.total_prefill_chunks,
            }
        metrics.update(self.kv.metrics())
        return metrics

    def _batch_loop(self) -> None:
        while not self._stop_event.is_set():
            batch = self._get_next_batch()
            if not batch:
                time.sleep(0.001)
                continue
            self._process_batch(batch)

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
            time.sleep(0.02 * len(working))
            completed_decodes = 0
            for item in working:
                if item.is_prefilling:
                    words = self._prompt_words(item.prompt)
                    nw = len(words)
                    chunk = min(
                        self.prefill_chunk_size,
                        max(0, nw - item.prefill_cursor),
                    )
                    item.prefill_cursor += chunk
                    with self.lock:
                        self.total_prefill_chunks += 1
                    if item.prefill_cursor >= nw:
                        item.is_prefilling = False
                    if item.is_prefilling:
                        with self.lock:
                            self.queue.append(item)
                    else:
                        item.result = f"processed: {item.prompt}"
                        item.done_event.set()
                        decode_finished.append(item)
                        completed_decodes += 1
                else:
                    item.result = f"processed: {item.prompt}"
                    item.done_event.set()
                    decode_finished.append(item)
                    completed_decodes += 1
            with self.lock:
                self.total_batches += 1
                self.total_processed += completed_decodes
                self.total_wait_time += total_batch_wait
        finally:
            for item in decode_finished:
                self.kv.free(item.request_id)
