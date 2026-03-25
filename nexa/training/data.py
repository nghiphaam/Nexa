"""Data loading utilities."""
import os
import numpy as np
import torch
from nexa.utils.device import is_cuda_device, is_xla_device


class ChunkDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, block_size):
        self.data_path = data_path
        self.block_size = block_size
        bytes_size = os.path.getsize(data_path)
        num_tokens = bytes_size // 2
        if num_tokens < block_size + 1:
            self.length = 0
        else:
            self.length = (num_tokens - block_size - 1) // block_size
            if self.length == 0 and num_tokens >= block_size + 1:
                self.length = 1
        self.data = None

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.data is None:
            self.data = np.memmap(self.data_path, dtype=np.uint16, mode="r")
        start = idx * self.block_size
        chunk = self.data[start: start + self.block_size + 1]
        return torch.from_numpy(chunk.astype(np.int32))


class DataLoaderLite:
    def __init__(self, data_path, batch_size, block_size, device, eos_id=None):
        self.device = device
        self.eos_id = eos_id
        self.batch_size = batch_size
        self.block_size = block_size
        self.data_path = data_path

        if is_xla_device(device):
            self.data = np.memmap(data_path, dtype=np.uint16, mode="r")
            num_tokens = len(self.data)
            if num_tokens < block_size + 1:
                self.length = 0
            else:
                self.length = (num_tokens - block_size - 1) // block_size
                if self.length == 0 and num_tokens >= block_size + 1:
                    self.length = 1
            if self.length == 0:
                raise ValueError(
                    f"Dataset {data_path} is too small for block_size={block_size}: got {num_tokens} tokens"
                )
            self.loader = None
            self.iter = None
        else:
            self.dataset = ChunkDataset(data_path, block_size)
            n_w = min(4, os.cpu_count() or 1) if is_cuda_device(device) else 0
            self.loader = torch.utils.data.DataLoader(
                self.dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                num_workers=n_w, pin_memory=is_cuda_device(device),
                persistent_workers=(n_w > 0 and is_cuda_device(device)),
                prefetch_factor=2 if n_w > 0 else None,
            )
            self.iter = iter(self.loader)
            self.data = None


class CUDAPrefetcher:
    def __init__(self, dataloader_lite):
        self.base_loader = dataloader_lite
        self.device = dataloader_lite.device
        self.eos_id = dataloader_lite.eos_id
        self.batch_size = dataloader_lite.batch_size
        self.block_size = dataloader_lite.block_size

        if is_xla_device(self.device):
            self.data = dataloader_lite.data
            self.length = dataloader_lite.length
            self.loader = None
            self.iter = None
            self.stream = None
        else:
            self.loader = dataloader_lite.loader
            self.iter = iter(self.loader)
            self.stream = torch.cuda.Stream() if is_cuda_device(self.device) else None
            self.data = None

        self.next_x = None
        self.next_y = None
        self.preload()

    def preload(self):
        if is_xla_device(self.device):
            indices = np.random.randint(0, self.length, size=self.batch_size)
            chunks = []
            for idx in indices:
                start = idx * self.block_size
                chunk = self.data[start: start + self.block_size + 1]
                chunks.append(torch.from_numpy(chunk.astype(np.int32)))
            chunk = torch.stack(chunks)
            x = chunk[:, :-1].contiguous()
            y = chunk[:, 1:].contiguous().clone()
            if self.eos_id is not None:
                y[x == self.eos_id] = -100
            self.next_x = x.to(self.device)
            self.next_y = y.to(self.device)
            return

        try:
            chunk = next(self.iter)
        except StopIteration:
            self.iter = iter(self.loader)
            chunk = next(self.iter)

        x = chunk[:, :-1].contiguous()
        y = chunk[:, 1:].contiguous().clone()
        if self.eos_id is not None:
            y[x == self.eos_id] = -100

        if is_cuda_device(self.device):
            with torch.cuda.stream(self.stream):
                self.next_x = x.to(self.device, non_blocking=True)
                self.next_y = y.to(self.device, non_blocking=True)
        else:
            self.next_x = x.to(self.device)
            self.next_y = y.to(self.device)

    def next_batch(self):
        if is_cuda_device(self.device):
            torch.cuda.current_stream().wait_stream(self.stream)
        x, y = self.next_x, self.next_y
        self.preload()
        return x, y
