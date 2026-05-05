import os
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

from server.paged_attention import PagedKVCacheManager


@dataclass
class RequestItem:
    request_id: str
    prompt: str
    user_id: str
    created_at: float = field(default_factory=time.monotonic)
    done_event: threading.Event = field(default_factory=threading.Event)
    result: Optional[str] = None


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
        self.queue: List[RequestItem] = []
        self.lock = threading.Lock()
        self.kv = PagedKVCacheManager(
            block_size=_env_int("PAGED_BLOCK_SIZE", 16),
            num_blocks=_env_int("PAGED_NUM_BLOCKS", 256),
        )
        self.total_batches = 0
        self.total_wait_time = 0.0
        self.total_processed = 0
        self.max_queue_length = 0
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._batch_loop, daemon=True)
        self._worker.start()

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
            }
        metrics.update(self.kv.stats())
        return metrics

    def _batch_loop(self) -> None:
        while not self._stop_event.is_set():
            batch = self._get_next_batch()
            if not batch:
                time.sleep(0.001)
                continue
            self._process_batch(batch)

    def _get_next_batch(self) -> List[RequestItem]:
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

    def _process_batch(self, batch: List[RequestItem]) -> None:
        for item in batch:
            # Toy prefill length: one logical token slot per character (deterministic).
            self.kv.ensure_tokens(item.request_id, total_tokens=len(item.prompt))
        now = time.monotonic()
        total_batch_wait = sum(now - item.created_at for item in batch)
        try:
            # Simulate inference cost scaling with batch size.
            time.sleep(0.02 * len(batch))
            with self.lock:
                self.total_batches += 1
                self.total_processed += len(batch)
                self.total_wait_time += total_batch_wait
            for item in batch:
                item.result = f"processed: {item.prompt}"
                item.done_event.set()
        finally:
            for item in batch:
                self.kv.release_sequence(item.request_id)

