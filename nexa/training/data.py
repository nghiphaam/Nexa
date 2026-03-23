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
        self.length = ((bytes_size // 2) - 1) // block_size
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
        self.dataset = ChunkDataset(data_path, block_size)

        n_w = min(4, os.cpu_count() or 1) if (is_cuda_device(device) or is_xla_device(device)) else 0
        self.loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, shuffle=True, drop_last=True,
            num_workers=n_w, pin_memory=is_cuda_device(device),
            persistent_workers=(n_w > 0 and (is_cuda_device(device) or is_xla_device(device))),
            prefetch_factor=2 if n_w > 0 else None,
        )
        self.iter = iter(self.loader)


class CUDAPrefetcher:
    def __init__(self, dataloader_lite):
        self.base_loader = dataloader_lite
        self.loader = dataloader_lite.loader
        self.iter = iter(self.loader)
        self.stream = torch.cuda.Stream() if is_cuda_device(dataloader_lite.device) else None
        self.device = dataloader_lite.device
        self.eos_id = dataloader_lite.eos_id
        self.next_x = None
        self.next_y = None
        self.preload()

    def preload(self):
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
