import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional
from uuid import uuid4


@dataclass
class RequestItem:
    request_id: str
    prompt: str
    user_id: str
    created_at: float = field(default_factory=time.monotonic)
    done_event: threading.Event = field(default_factory=threading.Event)
    result: Optional[str] = None


class Scheduler:
    def __init__(self, batch_size: int = 4, batch_timeout: float = 0.05) -> None:
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.queue: List[RequestItem] = []
        self.lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._batch_loop, daemon=True)
        self._worker.start()

    def submit_request(self, prompt: str, user_id: str) -> str:
        item = RequestItem(request_id=str(uuid4()), prompt=prompt, user_id=user_id)
        with self.lock:
            self.queue.append(item)
        item.done_event.wait()
        if item.result is None:
            raise RuntimeError("Scheduler completed request without a result")
        return item.result

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
            item.result = f"processed: {item.prompt}"
            item.done_event.set()

