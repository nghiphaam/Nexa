"""Model components: RMSNorm, RoPE, KVCache, FeedForward, TransformerBlock."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * norm).type_as(x) * self.weight


def precompute_rope_freqs(dim, max_seq_len, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(xq, xk, freqs_cos, freqs_sin):
    # FIX #7: Handle shape mismatch gracefully
    seq_len = xq.shape[-2]
    freq_len = freqs_cos.shape[0]
    if seq_len != freq_len:
        if seq_len <= freq_len:
            freqs_cos = freqs_cos[:seq_len]
            freqs_sin = freqs_sin[:seq_len]
        else:
            raise RuntimeError(
                f"RoPE overflow: seq_len={seq_len} > freq_len={freq_len}. "
                f"Increase block_size in config."
            )
    xq_r = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_r = xk.float().reshape(*xk.shape[:-1], -1, 2)

    fc = freqs_cos.unsqueeze(0).unsqueeze(0)
    fs = freqs_sin.unsqueeze(0).unsqueeze(0)

    xq_out = torch.stack(
        [xq_r[..., 0] * fc - xq_r[..., 1] * fs, xq_r[..., 0] * fs + xq_r[..., 1] * fc],
        dim=-1,
    ).flatten(-2)

    xk_out = torch.stack(
        [xk_r[..., 0] * fc - xk_r[..., 1] * fs, xk_r[..., 0] * fs + xk_r[..., 1] * fc],
        dim=-1,
    ).flatten(-2)

    return xq_out.type_as(xq), xk_out.type_as(xk)


class KVCache:
    def __init__(self, max_batch_size, max_seq_len, n_kv_head, head_dim, device, dtype, n_global_tokens=0):
        self.k_cache = torch.zeros((max_batch_size, n_kv_head, max_seq_len, head_dim), device=device, dtype=dtype)
        self.v_cache = torch.zeros((max_batch_size, n_kv_head, max_seq_len, head_dim), device=device, dtype=dtype)
        self.pos = 0
        self.max_len = max_seq_len
        self.n_global_tokens = max(0, min(n_global_tokens, max_seq_len))
        self.slot_seq = torch.full((max_batch_size, max_seq_len), -1, device=device, dtype=torch.long)
        self.write_seq = 0
        self.committed_seq = -1
        self.filled = 0
        self._tail_start = self.n_global_tokens

    def reset(self):
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.slot_seq.fill_(-1)
        self.pos = 0
        self.write_seq = 0
        self.committed_seq = -1
        self.filled = 0
        self._tail_start = self.n_global_tokens

    def update(self, k_val, v_val, w=None):
        bsz, n_h, seq_len, hd = k_val.shape
        w = min(w or self.max_len, self.max_len)

        if self.write_seq > 1_000_000:
            self.reset()
        if self.filled > self.max_len or self.write_seq < 0 or self.committed_seq > self.write_seq - 1:
            self.reset()

        orig_seq_len = seq_len
        seq_ids = torch.arange(self.write_seq, self.write_seq + orig_seq_len, device=k_val.device, dtype=torch.long)
        global_keep = min(self.n_global_tokens, w)

        if seq_len > w:
            if global_keep > 0 and w > global_keep:
                keep_prefix = min(global_keep, orig_seq_len)
                keep_tail = max(0, w - keep_prefix)
                keep_idx = torch.cat([
                    torch.arange(keep_prefix, device=k_val.device),
                    torch.arange(orig_seq_len - keep_tail, orig_seq_len, device=k_val.device),
                ])
            else:
                keep_idx = torch.arange(orig_seq_len - w, orig_seq_len, device=k_val.device)
            k_val = k_val.index_select(2, keep_idx)
            v_val = v_val.index_select(2, keep_idx)
            seq_ids = seq_ids.index_select(0, keep_idx)
            seq_len = k_val.size(2)

        offset = 0
        if global_keep > 0 and self.write_seq < global_keep:
            n_prefix = min(seq_len, global_keep - self.write_seq)
            prefix_slots = torch.arange(self.write_seq, self.write_seq + n_prefix, device=k_val.device)
            current_seq = self.slot_seq[0, prefix_slots]
            new_seq = seq_ids[:n_prefix]
            update_mask = (current_seq == -1) | (new_seq >= current_seq)
            if update_mask.any():
                update_slots = prefix_slots[update_mask]
                update_idx = torch.nonzero(update_mask).squeeze(-1)
                self.k_cache[:, :, update_slots, :] = k_val[:, :, update_idx, :]
                self.v_cache[:, :, update_slots, :] = v_val[:, :, update_idx, :]
                self.slot_seq[:, update_slots] = seq_ids[update_idx].unsqueeze(0)
            offset = n_prefix

        if offset < seq_len:
            tail_len = seq_len - offset
            tail_capacity = max(1, w - global_keep)
            if self.pos == 0 and tail_len <= tail_capacity:
                tail_slots = torch.arange(self._tail_start, self._tail_start + tail_len, device=k_val.device)
                self._tail_start = global_keep + ((self._tail_start - global_keep + tail_len) % tail_capacity)
                self.pos = (self.pos + tail_len) % tail_capacity
            else:
                tail_slots = global_keep + ((self.pos + torch.arange(tail_len, device=k_val.device)) % tail_capacity)
                self.pos = int((self.pos + tail_len) % tail_capacity)
            self.k_cache[:, :, tail_slots, :] = k_val[:, :, offset:, :]
            self.v_cache[:, :, tail_slots, :] = v_val[:, :, offset:, :]
            self.slot_seq[:, tail_slots] = seq_ids[offset:].unsqueeze(0)

        self.write_seq += orig_seq_len
        self.committed_seq = self.write_seq - 1
        valid = self.slot_seq >= 0
        self.filled = int(valid[0].sum().item())
        return self.get_kv_ordered()

    def rollback(self, n_tokens):
        if n_tokens <= 0:
            return
        old_write_seq = self.write_seq
        n_tail_back = max(0, old_write_seq - max(self.n_global_tokens, old_write_seq - n_tokens))
        self.committed_seq = max(-1, self.committed_seq - n_tokens)
        self.write_seq = self.committed_seq + 1
        dirty_mask = self.slot_seq > self.committed_seq
        dirty_idx = torch.nonzero(dirty_mask[0], as_tuple=False).squeeze(-1)
        if dirty_idx.numel() > 0:
            self.k_cache.index_fill_(2, dirty_idx, 0)
            self.v_cache.index_fill_(2, dirty_idx, 0)
        self.slot_seq[dirty_mask] = -1
        if n_tail_back > 0:
            tail_capacity = max(1, self.max_len - self.n_global_tokens)
            self.pos = (self.pos - n_tail_back) % tail_capacity
            self._tail_start = self.n_global_tokens + ((self._tail_start - self.n_global_tokens - n_tail_back) % tail_capacity)
        valid = self.slot_seq >= 0
        self.filled = int(valid[0].sum().item())

    def get_kv_ordered(self):
        seq = self.slot_seq[0]
        valid_mask = (seq >= 0) & (seq <= self.committed_seq)
        valid_idx = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0:
            self.filled = 0
            return self.k_cache[:, :, :0, :], self.v_cache[:, :, :0, :]
        valid_seq = seq[valid_idx]
        order = torch.argsort(valid_seq)
        idx = valid_idx[order]
        self.filled = idx.numel()
        k_out = self.k_cache.index_select(2, idx)
        v_out = self.v_cache.index_select(2, idx)
        return k_out, v_out


def repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    B, n_kv, T, hd = x.shape
    return x[:, :, None, :, :].expand(B, n_kv, n_rep, T, hd).reshape(B, n_kv * n_rep, T, hd)


class FeedForward(nn.Module):
    """SwiGLU FFN."""
    def __init__(self, config):
        super().__init__()
        hidden = int(4 * config.n_embd * 2 / 3)
        hidden = 64 * ((hidden + 63) // 64)
        self.w1 = nn.Linear(config.n_embd, hidden, bias=False)
        self.w2 = nn.Linear(hidden, config.n_embd, bias=False)
        self.w3 = nn.Linear(config.n_embd, hidden, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
