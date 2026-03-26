"""Dynamic batch scheduler for serving."""
import time


class DynamicBatchScheduler:
    """Collects requests and processes them in batches."""
    def __init__(self, max_batch_size=8, max_wait_ms=50):
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.pending = []
        self.pending_since = []

    def add_request(self, request):
        self.pending.append(request)
        self.pending_since.append(time.monotonic())

    def should_process(self):
        if not self.pending:
            return False
        if len(self.pending) >= self.max_batch_size:
            return True
        oldest_age_ms = (time.monotonic() - self.pending_since[0]) * 1000.0
        return oldest_age_ms >= self.max_wait_ms

    def get_batch(self):
        batch = self.pending[:self.max_batch_size]
        batch_count = len(batch)
        self.pending = self.pending[self.max_batch_size:]
        self.pending_since = self.pending_since[batch_count:]
        return batch

    def clear(self):
        self.pending.clear()
        self.pending_since.clear()
