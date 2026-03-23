"""Dynamic batch scheduler for serving."""


class DynamicBatchScheduler:
    """Collects requests and processes them in batches."""
    def __init__(self, max_batch_size=8, max_wait_ms=50):
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.pending = []

    def add_request(self, request):
        self.pending.append(request)

    def should_process(self):
        return len(self.pending) >= self.max_batch_size

    def get_batch(self):
        batch = self.pending[:self.max_batch_size]
        self.pending = self.pending[self.max_batch_size:]
        return batch

    def clear(self):
        self.pending.clear()
