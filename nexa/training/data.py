"""Data loading utilities."""
import os

import numpy as np
import torch

from nexa.utils.device import get_rank, get_world_size, is_cuda_device, is_distributed, is_xla_device


def resolve_token_dtype(vocab_size=None, token_dtype=None):
    if token_dtype is not None:
        if token_dtype in (np.uint16, 'uint16', 'np.uint16'):
            return np.uint16
        if token_dtype in (np.uint32, 'uint32', 'np.uint32'):
            return np.uint32
        raise ValueError(f'Unsupported token dtype: {token_dtype}')
    return np.uint32 if vocab_size is not None and int(vocab_size) >= 65536 else np.uint16


def _num_chunks(num_tokens, block_size):
    if num_tokens < block_size + 1:
        return 0
    return ((num_tokens - block_size - 1) // block_size) + 1


class ChunkDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, block_size, token_dtype=None, vocab_size=None):
        self.data_path = data_path
        self.block_size = block_size
        self.token_dtype = resolve_token_dtype(vocab_size=vocab_size, token_dtype=token_dtype)
        bytes_size = os.path.getsize(data_path)
        num_tokens = bytes_size // np.dtype(self.token_dtype).itemsize
        self.length = _num_chunks(num_tokens, block_size)
        self.data = None

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.data is None:
            self.data = np.memmap(self.data_path, dtype=self.token_dtype, mode='r')
        start = idx * self.block_size
        chunk = self.data[start: start + self.block_size + 1]
        return torch.from_numpy(chunk.astype(np.int64))


class DataLoaderLite:
    def __init__(self, data_path, batch_size, block_size, device, eos_id=None, token_dtype=None, vocab_size=None):
        self.device = device
        self.eos_id = eos_id
        self.batch_size = batch_size
        self.block_size = block_size
        self.data_path = data_path
        self.token_dtype = resolve_token_dtype(vocab_size=vocab_size, token_dtype=token_dtype)
        self.sampler = None
        self._epoch = 0

        if is_xla_device(device):
            self.data = np.memmap(data_path, dtype=self.token_dtype, mode='r')
            num_tokens = len(self.data)
            self.length = _num_chunks(num_tokens, block_size)
            if self.length == 0:
                raise ValueError(
                    f"Dataset {data_path} is too small for block_size={block_size}: got {num_tokens} tokens"
                )
            self.loader = None
            self.iter = None
        else:
            self.dataset = ChunkDataset(data_path, block_size, token_dtype=self.token_dtype)
            if len(self.dataset) == 0:
                bytes_size = os.path.getsize(data_path)
                num_tokens = bytes_size // np.dtype(self.token_dtype).itemsize
                raise ValueError(
                    f"Dataset {data_path} is too small for block_size={block_size}: got {num_tokens} tokens"
                )

            n_w = min(4, os.cpu_count() or 1) if is_cuda_device(device) else 0
            if is_cuda_device(device) and is_distributed():
                self.sampler = torch.utils.data.distributed.DistributedSampler(
                    self.dataset,
                    num_replicas=get_world_size(),
                    rank=get_rank(),
                    shuffle=True,
                    drop_last=True,
                )
            self.loader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=batch_size,
                shuffle=self.sampler is None,
                sampler=self.sampler,
                drop_last=True,
                num_workers=n_w,
                pin_memory=is_cuda_device(device),
                persistent_workers=(n_w > 0 and is_cuda_device(device)),
                prefetch_factor=2 if n_w > 0 else None,
            )
            self.iter = self.new_iter()
            self.data = None

    def new_iter(self):
        if self.sampler is not None:
            self.sampler.set_epoch(self._epoch)
            self._epoch += 1
        return iter(self.loader)


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
                chunks.append(torch.from_numpy(chunk.astype(np.int64)))
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
            self.iter = self.base_loader.new_iter()
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
