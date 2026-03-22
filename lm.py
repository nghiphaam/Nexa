import os
import time
import math
import random
import argparse
import contextlib
from dataclasses import dataclass

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch_xla

    _HAS_XLA = True
except Exception:
    torch_xla = None
    _HAS_XLA = False


def safe_cuda_alloc():
    try:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            torch.zeros((1,), device="cuda")
            return True
        return False
    except Exception:
        return False


def safe_xla_alloc():
    if not _HAS_XLA:
        return False
    try:
        device = torch_xla.device()
        torch.zeros((1,), device=device)
        return True
    except Exception:
        return False


def get_xla_device():
    if not _HAS_XLA:
        raise RuntimeError("torch_xla is not available")
    return torch_xla.device()


def is_cuda_device(device):
    return str(device).startswith("cuda")


def is_xla_device(device):
    return str(device).startswith("xla")


def configure_tf32_runtime():
    if not torch.cuda.is_available():
        return
    if getattr(torch.backends.cuda, "_tf32_configured", False):
        return
    try:
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        torch.backends.cudnn.conv.fp32_precision = "tf32"
    except Exception:
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
    torch.backends.cuda._tf32_configured = True


# ---------------------------------------------------------------------------
# TOKENIZER
# ---------------------------------------------------------------------------

DEFAULT_VOCAB_SIZE = 50261

EOS_TOKEN = "<|endoftext|>"
PAD_TOKEN = "<|pad|>"
SYS_TOKEN = "<|system|>"
USR_TOKEN = "<|user|>"
AST_TOKEN = "<|assistant|>"

# ---------------------------------------------------------------------------
# NEXA TOKENIZER — same interface as HuggingFace tokenizers
# ---------------------------------------------------------------------------


class NexaTokenizer:
    """Wraps tiktoken BPE to match HuggingFace tokenizers interface.
    Expanded vocab to 50,261 to avoid collisions with GPT-2 merges.
    """

    def __init__(self):
        import tiktoken
        import re

        self.enc = tiktoken.get_encoding("gpt2")
        self._base_vocab_size = self.enc.n_vocab  # 50257
        self._vocab_size = 50261
        self._eos_id = self.enc.eot_token  # 50256
        self.supports_special_role_tokens = True

        self.special_map = {
            EOS_TOKEN: 50256,
            PAD_TOKEN: 50257,
            SYS_TOKEN: 50258,
            USR_TOKEN: 50259,
            AST_TOKEN: 50260,
        }
        self.id_to_special = {v: k for k, v in self.special_map.items()}

        # Regex for splitting by special tokens to handle them correctly during encoding
        pattern = "|".join(re.escape(k) for k in self.special_map.keys())
        self.special_pattern = re.compile(f"({pattern})")

    def get_vocab_size(self):
        return self._vocab_size

    def token_to_id(self, token):
        if token in self.special_map:
            return self.special_map[token]
        raise ValueError(f"Unknown token: {token}")

    def encode(self, text):
        parts = self.special_pattern.split(text)
        ids = []
        for p in parts:
            if p in self.special_map:
                ids.append(self.special_map[p])
            elif p:
                ids.extend(self.enc.encode_ordinary(p))
        return type("E", (), {"ids": ids})()

    def encode_batch(self, texts):
        return [self.encode(t) for t in texts if t and t.strip()]

    def decode(self, ids, skip_special_tokens=True):
        result = []
        buf = []
        for i in ids:
            if i in self.id_to_special:
                if buf:
                    result.append(self.enc.decode(buf))
                    buf = []
                if not skip_special_tokens:
                    result.append(self.id_to_special[i])
            elif i < self._base_vocab_size:
                buf.append(i)
        if buf:
            result.append(self.enc.decode(buf))
        return "".join(result)


_cached_tokenizer = {}


def load_tokenizer():
    """Load tokenizer — Nexa default."""
    global _cached_tokenizer
    if "nexa" not in _cached_tokenizer:
        print("Tokenizer: Nexa BPE (vocab=50,261)")
        _cached_tokenizer["nexa"] = NexaTokenizer()
    return _cached_tokenizer["nexa"]


def get_vocab_size():
    return 50261


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------


@dataclass
class Config:
    # Data
    data_dir: str = "data"

    # Model
    vocab_size: int = DEFAULT_VOCAB_SIZE
    eos_id: int = None
    block_size: int = 384
    sliding_window: int = 384  # Match block_size to prevent drift
    n_embd: int = 1792
    n_head: int = 16  # 1792 / 16 = 112 head_dim
    n_kv_head: int = 4  # GQA ratio = 4:1
    n_layer: int = 28
    dropout: float = 0.1

    # Training
    batch_size: int = 2
    grad_accum_steps: int = 16
    lr: float = 2e-4
    min_lr: float = 2e-5
    weight_decay: float = 0.1
    max_iters: int = 10000
    warmup_iters: int = 500
    grad_clip: float = 1.0

    # Eval / Log
    eval_interval: int = 500
    eval_iters: int = 100
    log_interval: int = 50

    # Generation
    gen_len: int = 200
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    min_p: float = 0.05
    repetition_penalty: float = 1.1
    num_samples: int = 3
    reasoning_loss_weight: float = 0.0
    critic_score_loss_weight: float = 0.0
    critic_rank_loss_weight: float = 1.0
    n_global_tokens: int = 16
    memory_state_scale: float = 0.0  # Overkill for scratch pre-train
    memory_train_dropout: float = 0.3
    speculative_disable_steps: int = 16

    # System
    device: str = "cuda" if safe_cuda_alloc() else "cpu"
    dtype: str = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    use_grad_ckpt: bool = False
    compile_model: bool = True
    compile_mode: str = "reduce-overhead"  # max-autotune chỉ tốt trên Ampere+
    seed: int = 1337
    checkpoint_dir: str = "checkpoints"
    preset: str = "auto"


# ---------------------------------------------------------------------------
# MODEL COMPONENTS
# ---------------------------------------------------------------------------


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
    def __init__(
        self,
        max_batch_size,
        max_seq_len,
        n_kv_head,
        head_dim,
        device,
        dtype,
        n_global_tokens=0,
    ):
        self.k_cache = torch.zeros(
            (max_batch_size, n_kv_head, max_seq_len, head_dim),
            device=device,
            dtype=dtype,
        )
        self.v_cache = torch.zeros(
            (max_batch_size, n_kv_head, max_seq_len, head_dim),
            device=device,
            dtype=dtype,
        )
        self.pos = 0
        self.max_len = max_seq_len
        self.n_global_tokens = max(0, min(n_global_tokens, max_seq_len))
        # NEW: logical sequence tracking for exact temporal order
        self.slot_seq = torch.full(
            (max_batch_size, max_seq_len), -1, device=device, dtype=torch.long
        )
        self.write_seq = 0  # next absolute seq id
        self.committed_seq = -1  # max valid seq id (after rollback)
        self.filled = 0

    def reset(self):
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.slot_seq.fill_(-1)
        self.pos = 0
        self.write_seq = 0
        self.committed_seq = -1
        self.filled = 0

    def update(self, k_val, v_val, w=None):
        bsz, n_h, seq_len, hd = k_val.shape
        w = min(w or self.max_len, self.max_len)
        if (
            self.filled > self.max_len
            or self.write_seq < 0
            or self.committed_seq >= self.write_seq
        ):
            self.reset()
        assert (self.slot_seq <= self.committed_seq).all(), (
            "KVCache slot_seq exceeds committed_seq before update"
        )

        orig_seq_len = seq_len
        seq_ids = torch.arange(
            self.write_seq,
            self.write_seq + orig_seq_len,
            device=k_val.device,
            dtype=torch.long,
        )
        global_keep = min(self.n_global_tokens, w)
        if seq_len > w:
            if global_keep > 0 and w > global_keep:
                keep_prefix = min(global_keep, orig_seq_len)
                keep_tail = max(0, w - keep_prefix)
                keep_idx = torch.cat(
                    [
                        torch.arange(keep_prefix, device=k_val.device),
                        torch.arange(
                            orig_seq_len - keep_tail, orig_seq_len, device=k_val.device
                        ),
                    ]
                )
            else:
                keep_idx = torch.arange(
                    orig_seq_len - w, orig_seq_len, device=k_val.device
                )
            k_val = k_val.index_select(2, keep_idx)
            v_val = v_val.index_select(2, keep_idx)
            seq_ids = seq_ids.index_select(0, keep_idx)
            seq_len = k_val.size(2)

        offset = 0
        if global_keep > 0 and self.write_seq < global_keep:
            n_prefix = min(seq_len, global_keep - self.write_seq)
            prefix_slots = torch.arange(
                self.write_seq, self.write_seq + n_prefix, device=k_val.device
            )
            self.k_cache[:, :, prefix_slots, :] = k_val[:, :, :n_prefix, :]
            self.v_cache[:, :, prefix_slots, :] = v_val[:, :, :n_prefix, :]
            self.slot_seq[:, prefix_slots] = seq_ids[:n_prefix].unsqueeze(0)
            offset = n_prefix

        if offset < seq_len:
            tail_len = seq_len - offset
            tail_capacity = max(1, w - global_keep)
            tail_slots = global_keep + (
                (self.pos + torch.arange(tail_len, device=k_val.device)) % tail_capacity
            )
            self.k_cache[:, :, tail_slots, :] = k_val[:, :, offset:, :]
            self.v_cache[:, :, tail_slots, :] = v_val[:, :, offset:, :]
            self.slot_seq[:, tail_slots] = seq_ids[offset:].unsqueeze(0)
            self.pos = int((self.pos + tail_len) % tail_capacity)

        self.write_seq += orig_seq_len

        self.committed_seq = self.write_seq - 1
        return self.get_kv_ordered()

    def rollback(self, n_tokens):
        if n_tokens <= 0:
            return
        old_write_seq = self.write_seq
        # Only rollback pos for tokens that were in the tail (past n_global_tokens)
        n_tail_back = max(
            0, old_write_seq - max(self.n_global_tokens, old_write_seq - n_tokens)
        )

        self.committed_seq = max(-1, self.committed_seq - n_tokens)
        self.write_seq = self.committed_seq + 1
        # CLEANUP: invalidate dirty slots > committed_seq
        dirty_mask = self.slot_seq > self.committed_seq
        dirty_idx = torch.nonzero(dirty_mask[0], as_tuple=False).squeeze(-1)
        if dirty_idx.numel() > 0:
            self.k_cache.index_fill_(2, dirty_idx, 0)
            self.v_cache.index_fill_(2, dirty_idx, 0)
        self.slot_seq[dirty_mask] = -1

        # Rollback pos correctly for circular buffer
        if n_tail_back > 0:
            tail_capacity = max(1, self.max_len - self.n_global_tokens)
            self.pos = (self.pos - n_tail_back) % tail_capacity

        # recompute filled from cleaned slot_seq
        valid = self.slot_seq >= 0
        self.filled = int(valid[0].sum().item())
        assert (self.slot_seq <= self.committed_seq).all(), (
            "KVCache slot_seq exceeds committed_seq after rollback"
        )

    def get_kv_ordered(self):
        # bsz=1 assumption for chat
        seq = self.slot_seq[0]
        valid_mask = (seq >= 0) & (seq <= self.committed_seq)
        valid_idx = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0:
            self.filled = 0
            return self.k_cache[:, :, :0, :], self.v_cache[:, :, :0, :]

        # Sort by logical sequence to fix temporal order perfectly
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
    return (
        x[:, :, None, :, :]
        .expand(B, n_kv, n_rep, T, hd)
        .reshape(B, n_kv * n_rep, T, hd)
    )


# ---------------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------------


class CausalSelfAttention(nn.Module):
    """GQA (Grouped Query Attention)."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.n_rep = self.n_head // self.n_kv_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        kv_dim = self.n_kv_head * self.head_dim
        self.wq = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.wk = nn.Linear(config.n_embd, kv_dim, bias=False)
        self.wv = nn.Linear(config.n_embd, kv_dim, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x, freqs_cos, freqs_sin, kv_cache=None):
        B, T, C = x.size()
        q = self.wq(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        q, k = apply_rope(q, k, freqs_cos, freqs_sin)

        new_cache = None
        if kv_cache is not None:
            w = (
                getattr(self.config, "sliding_window", None)
                if hasattr(self.config, "sliding_window")
                else None
            )
            k, v = kv_cache.update(k, v, w)
            new_cache = kv_cache

        k_exp = repeat_kv(k, self.n_rep)
        v_exp = repeat_kv(v, self.n_rep)

        attn_mask = None
        use_causal = T > 1
        if kv_cache is None:
            local_w = getattr(self.config, "sliding_window", None)
            n_global = min(getattr(self.config, "n_global_tokens", 0), T)
            if local_w is not None and 0 < local_w < T:
                q_pos = torch.arange(T, device=x.device).view(T, 1)
                k_pos = torch.arange(T, device=x.device).view(1, T)
                attn_mask = (k_pos <= q_pos) & (k_pos >= (q_pos - (local_w - 1)))
                if n_global > 0:
                    attn_mask |= k_pos < n_global
                attn_mask = attn_mask.view(1, 1, T, T)
                use_causal = False

        y = F.scaled_dot_product_attention(
            q,
            k_exp,
            v_exp,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=use_causal,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y, new_cache


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


class TransformerBlock(nn.Module):
    """Parallel Attention + FFN (PaLM-style)."""

    def __init__(self, config):
        super().__init__()
        self.ln = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = FeedForward(config)

    def forward(self, x, freqs_cos, freqs_sin, kv_cache=None):
        h = self.ln(x)
        attn_out, new_cache = self.attn(h, freqs_cos, freqs_sin, kv_cache)
        ffn_out = self.mlp(h)
        x = x + attn_out + ffn_out
        return x, new_cache

    def forward_ckpt(self, x, freqs_cos, freqs_sin):
        from torch.utils.checkpoint import checkpoint

        h = self.ln(x)

        def attn_fn(h_in, fc, fs):
            out, _ = self.attn(h_in, fc, fs, None)
            return out

        attn_out = checkpoint(attn_fn, h, freqs_cos, freqs_sin, use_reentrant=False)
        ffn_out = checkpoint(self.mlp, h, use_reentrant=False)
        x = x + attn_out + ffn_out
        return x


class NexaModel(nn.Module):
    def __init__(self, config, use_grad_ckpt=False):
        super().__init__()
        self.config = config
        self.use_grad_ckpt = use_grad_ckpt
        head_dim = config.n_embd // config.n_head

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList(
                    [TransformerBlock(config) for _ in range(config.n_layer)]
                ),
                ln_f=RMSNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.critic_adapter = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd),
            nn.SiLU(),
            nn.Linear(config.n_embd, config.n_embd),
        )
        self.critic_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.critic_score_head = nn.Linear(config.n_embd, 1)
        self.memory_gate = nn.Linear(config.n_embd, config.n_embd)
        self.memory_query_gate = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.memory_value_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.memory_scale_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

        freqs_cos, freqs_sin = precompute_rope_freqs(head_dim, config.block_size * 2)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight") or pn.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        n_params = sum(p.numel() for p in self.parameters())
        n_embd_params = config.vocab_size * config.n_embd
        n_non_embd = n_params - n_embd_params
        print(f"Model params : {n_params:,} ({n_params / 1e6:.1f}M)")
        print(
            f"  Embedding  : {n_embd_params:,} ({n_embd_params / 1e6:.1f}M) [vocab={config.vocab_size}]"
        )
        print(f"  Transformer: {n_non_embd:,} ({n_non_embd / 1e6:.1f}M)")

        # Special tokens (like SYS, USR, AST) require this explicitly so they can be trained in SFT/Pretrain
        self.transformer.wte.weight.requires_grad_(True)
        self._current_entropy = None
        self._current_entropy_norm = None
        self._entropy_ema = None
        self._entropy_var_ema = None
        self._reflect_cooldown = 0
        self._spec_accept_ema = None
        self._adaptive_gamma = None
        self._spec_disable_steps = 0

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _apply_memory_state(self, x, memory_state=None, memory_query_state=None):
        if memory_state is None:
            return x
        if memory_state.dim() == 1:
            memory_state = memory_state.unsqueeze(0)
        memory_state = memory_state.clamp(-3.0, 3.0).to(device=x.device, dtype=x.dtype)
        scale = getattr(self.config, "memory_state_scale", 0.08)
        memory_value = self.memory_value_proj(memory_state).unsqueeze(1) * scale
        memory_scale = (
            torch.tanh(self.memory_scale_proj(memory_state)).unsqueeze(1) * scale
        )
        gate_logits = self.memory_gate(x)
        if memory_query_state is not None:
            if memory_query_state.dim() == 1:
                memory_query_state = memory_query_state.unsqueeze(0)
            memory_query_state = memory_query_state.to(device=x.device, dtype=x.dtype)
            if torch.norm(memory_query_state.float(), dim=-1).mean().item() >= 0.25:
                gate_logits = gate_logits + self.memory_query_gate(
                    memory_query_state
                ).unsqueeze(1)
        gate = torch.sigmoid(gate_logits)
        return x * (1.0 + gate * memory_scale) + gate * memory_value

    def _project_logits(self, x, head="main"):
        if head == "critic":
            return self.critic_head(x.detach())
        return self.lm_head(x)

    def forward(
        self,
        idx,
        targets=None,
        reasoning_targets=None,
        critic_labels=None,
        memory_state=None,
        memory_query_state=None,
        return_aux=False,
    ):
        B, T = idx.size()
        assert T <= self.freqs_cos.size(0), (
            f"Sequence length {T} > max block_size {self.freqs_cos.size(0)}"
        )

        x = self.transformer.drop(self.transformer.wte(idx))
        x = self._apply_memory_state(x, memory_state, memory_query_state)
        freqs_cos = self.freqs_cos[:T]
        freqs_sin = self.freqs_sin[:T]
        for block in self.transformer.h:
            if self.use_grad_ckpt and self.training:
                x = block.forward_ckpt(x, freqs_cos, freqs_sin)
            else:
                x, _ = block(x, freqs_cos, freqs_sin, None)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # Optimize: skip critic heads entirely if not needed
        need_critic = reasoning_targets is not None or critic_labels is not None or (return_aux and reasoning_targets is not None)
        critic_logits = None
        critic_score = None
        aux = {}

        if need_critic:
            critic_x = self.critic_adapter(x.detach())
            critic_logits = self.critic_head(critic_x)
            critic_score = torch.sigmoid(self.critic_score_head(critic_x[:, -1, :]))
            aux["critic_score"] = critic_score.detach()

        loss = None
        if targets is not None:
            lm_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100
            )
            loss = lm_loss
            aux["lm_loss"] = lm_loss.detach()
            if reasoning_targets is not None:
                reasoning_loss = F.cross_entropy(
                    critic_logits.view(-1, critic_logits.size(-1)),
                    reasoning_targets.view(-1),
                    ignore_index=-100,
                )
                loss = (
                    loss
                    + getattr(self.config, "reasoning_loss_weight", 0.1)
                    * reasoning_loss
                )
                aux["reasoning_loss"] = reasoning_loss.detach()
        if critic_labels is not None:
            critic_score_loss = F.binary_cross_entropy(
                critic_score,
                critic_labels.to(device=critic_score.device, dtype=critic_score.dtype),
            )
            loss = (
                critic_score_loss
                * getattr(self.config, "critic_score_loss_weight", 0.03)
                if loss is None
                else loss
                + getattr(self.config, "critic_score_loss_weight", 0.03)
                * critic_score_loss
            )
            aux["critic_score_loss"] = critic_score_loss.detach()
        if return_aux:
            return logits, loss, aux
        return logits, loss

    def _update_entropy_stats(self, logits):
        with torch.no_grad():
            _lp = F.log_softmax(logits.float(), dim=-1)
            ent = -(_lp.exp() * _lp).sum(dim=-1).mean()
        e = float(ent.item())
        e_norm = e / max(math.log(logits.size(-1)), 1e-8)
        self._current_entropy = e
        self._current_entropy_norm = e_norm
        if self._entropy_ema is None:
            self._entropy_ema = e_norm
            self._entropy_var_ema = 0.0
        else:
            beta = 0.95
            delta = e_norm - self._entropy_ema
            self._entropy_ema = beta * self._entropy_ema + (1 - beta) * e_norm
            self._entropy_var_ema = beta * self._entropy_var_ema + (1 - beta) * (
                delta * delta
            )
        return e

    def _sample_token(
        self,
        logits,
        generated,
        temperature,
        top_k,
        top_p,
        min_p,
        repetition_penalty,
        seen_mask=None,
        track_entropy=True,
    ):
        """Sample with top-k, top-p, min-p, and repetition penalty.
        Pass `seen_mask` (bool tensor [vocab_size]) for O(1) penalty; otherwise falls back to O(n) scan.
        """
        if repetition_penalty != 1.0:
            vocab_size = logits.size(-1)
            for b in range(logits.size(0)):
                if seen_mask is not None:
                    # O(1) path: pre-built boolean mask
                    score = logits[b].clone()
                    logits[b] = torch.where(
                        seen_mask,
                        torch.where(
                            score > 0,
                            score / repetition_penalty,
                            score * repetition_penalty,
                        ),
                        score,
                    )
                else:
                    # Fallback O(n) scan for standalone generate()
                    seen = generated[b]
                    seen = seen[seen > 0]
                    if seen.numel() > 0:
                        seen = seen.clamp(0, vocab_size - 1)
                        score = logits[b].clone()
                        logits[b, seen] = torch.where(
                            score[seen] > 0,
                            score[seen] / repetition_penalty,
                            score[seen] * repetition_penalty,
                        )

        temperature = max(float(temperature), 1e-5)
        logits = logits / temperature
        raw_logits = logits.clone()

        # NaN/Inf guard — preserve distribution shape
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)

        # Min-P filtering
        if min_p > 0.0:
            probs = F.softmax(logits, dim=-1)
            max_probs = probs.max(dim=-1, keepdim=True).values
            mask = probs < (min_p * max_probs)
            logits[mask] = float("-inf")

        # Top-K filtering
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        # Top-P (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[sorted_mask] = float("-inf")
            logits = (
                torch.empty_like(logits)
                .fill_(float("-inf"))
                .scatter_(1, sorted_indices, sorted_logits)
            )

        if track_entropy:
            self._update_entropy_stats(raw_logits)
        probs = F.softmax(logits, dim=-1)
        # Renormalize to ensure sum=1 after filtering
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return torch.multinomial(probs, num_samples=1)

    def _should_reflect(self, temperature: float) -> bool:
        if self._current_entropy is None:
            return False
        norm = self._current_entropy_norm or 0.0
        std = math.sqrt(max(self._entropy_var_ema or 0.0, 1e-8))
        z = (norm - (self._entropy_ema or norm)) / std
        z_th = 1.0 + 0.3 * max(0.0, temperature - 0.7)
        norm_th = 0.25 + 0.1 * max(0.0, temperature)
        return (
            norm > norm_th and z > z_th and (getattr(self, "_reflect_cooldown", 0) <= 0)
        )

    @torch.no_grad()
    def generate_stream(
        self,
        idx,
        max_new_tokens,
        temperature=1.0,
        top_k=0,
        top_p=0.9,
        min_p=0.05,
        repetition_penalty=1.1,
        head="main",
        memory_state=None,
        memory_query_state=None,
    ):
        try:
            self.eval()
            B, T = idx.size()
            if B != 1:
                out = self.generate(
                    idx,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    min_p=min_p,
                    repetition_penalty=repetition_penalty,
                    head=head,
                    memory_state=memory_state,
                    memory_query_state=memory_query_state,
                )
                for tok in out[0, T:].tolist():
                    yield tok
                return

            n_layers = len(self.transformer.h)
            if T + max_new_tokens > self.config.block_size:
                idx = idx[:, -(self.config.block_size - max_new_tokens) :]
                T = idx.size(1)

            max_cache_len = min(
                T + max_new_tokens,
                getattr(self.config, "sliding_window", None) or self.config.block_size,
            )
            head_dim = self.config.n_embd // self.config.n_head
            caches = [
                KVCache(
                    B,
                    max_cache_len,
                    self.config.n_kv_head,
                    head_dim,
                    idx.device,
                    self.transformer.wte.weight.dtype,
                    n_global_tokens=getattr(self.config, "n_global_tokens", 0),
                )
                for _ in range(n_layers)
            ]

            x = self.transformer.drop(self.transformer.wte(idx))
            x = self._apply_memory_state(x, memory_state, memory_query_state)
            freqs_cos = self.freqs_cos[:T]
            freqs_sin = self.freqs_sin[:T]
            for i, block in enumerate(self.transformer.h):
                x, caches[i] = block(x, freqs_cos, freqs_sin, kv_cache=caches[i])
            x = self.transformer.ln_f(x)
            logits = self._project_logits(x[:, -1, :], head=head)

            all_so_far = idx.clone()
            seen_mask = torch.zeros(
                self.config.vocab_size, dtype=torch.bool, device=idx.device
            )
            init_ids = idx[0]
            init_ids = init_ids[(init_ids > 0) & (init_ids < self.config.vocab_size)]
            if init_ids.numel() > 0:
                seen_mask[init_ids.unique()] = True

            eos_id = getattr(self.config, "eos_id", -1)
            current_len = T
            for _ in range(max_new_tokens):
                idx_next = self._sample_token(
                    logits,
                    all_so_far,
                    temperature,
                    top_k,
                    top_p,
                    min_p,
                    repetition_penalty,
                    seen_mask=seen_mask,
                )
                token_id = int(idx_next.item())
                yield token_id
                if 0 <= token_id < self.config.vocab_size:
                    seen_mask[token_id] = True
                if eos_id >= 0 and token_id == eos_id:
                    break
                all_so_far = torch.cat([all_so_far, idx_next], dim=1)
                x = self.transformer.wte(idx_next)
                x = self._apply_memory_state(x, memory_state, memory_query_state)
                pos = current_len % self.freqs_cos.size(0)
                fc = self.freqs_cos[pos : pos + 1]
                fs = self.freqs_sin[pos : pos + 1]
                for i, block in enumerate(self.transformer.h):
                    x, caches[i] = block(x, fc, fs, kv_cache=caches[i])
                x = self.transformer.ln_f(x)
                logits = self._project_logits(x[:, -1, :], head=head)
                current_len += 1
        except Exception:
            return

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens,
        temperature=1.0,
        top_k=0,
        top_p=0.9,
        min_p=0.05,
        repetition_penalty=1.1,
        caches=None,
        draft_model=None,
        gamma=4,
        head="main",
        memory_state=None,
        memory_query_state=None,
    ):
        self.eval()
        if getattr(self, "_spec_disable_steps", 0) > 0:
            draft_model = None
            self._spec_disable_steps = max(0, self._spec_disable_steps - max_new_tokens)
        if draft_model:
            draft_model.eval()
        if hasattr(self, "_reflect_cooldown"):
            self._reflect_cooldown = max(0, self._reflect_cooldown - 1)
        if draft_model and getattr(self, "_adaptive_gamma", None) is None:
            self._adaptive_gamma = max(2, int(gamma))
        n_layers = len(self.transformer.h)
        idx.size(1)

        B, T = idx.size()

        if T + max_new_tokens > self.config.block_size:
            idx = idx[:, -(self.config.block_size - max_new_tokens) :]
            T = idx.size(1)

        if caches is None:
            max_cache_len = min(
                T + max_new_tokens,
                getattr(self.config, "sliding_window", None) or self.config.block_size,
            )
            head_dim = self.config.n_embd // self.config.n_head
            caches = [
                KVCache(
                    B,
                    max_cache_len,
                    self.config.n_kv_head,
                    head_dim,
                    idx.device,
                    self.transformer.wte.weight.dtype,
                    n_global_tokens=getattr(self.config, "n_global_tokens", 0),
                )
                for _ in range(n_layers)
            ]
        else:
            max_cache_len = caches[0].max_len

        if T > max_cache_len:
            idx = idx[:, -max_cache_len:]
            T = idx.size(1)

        draft_caches = None
        if draft_model:
            n_layers_d = len(draft_model.transformer.h)
            head_dim_d = draft_model.config.n_embd // draft_model.config.n_head
            draft_dtype = next(draft_model.parameters()).dtype
            draft_caches = [
                KVCache(
                    B,
                    max_cache_len,
                    draft_model.config.n_kv_head,
                    head_dim_d,
                    idx.device,
                    draft_dtype,
                    n_global_tokens=getattr(draft_model.config, "n_global_tokens", 0),
                )
                for _ in range(n_layers_d)
            ]

        # --- PREFILL ---
        x = self.transformer.drop(self.transformer.wte(idx))
        x = self._apply_memory_state(x, memory_state, memory_query_state)
        freqs_cos = self.freqs_cos[:T]
        freqs_sin = self.freqs_sin[:T]
        for i, block in enumerate(self.transformer.h):
            x, caches[i] = block(x, freqs_cos, freqs_sin, kv_cache=caches[i])
        x = self.transformer.ln_f(x)
        logits = self._project_logits(x[:, -1, :], head=head)

        if draft_model:
            xd = draft_model.transformer.drop(draft_model.transformer.wte(idx))
            fcd = draft_model.freqs_cos[:T]
            fsd = draft_model.freqs_sin[:T]
            for i, block in enumerate(draft_model.transformer.h):
                xd, draft_caches[i] = block(xd, fcd, fsd, kv_cache=draft_caches[i])

        # Allocate persistent buffer for sampling tensor
        max_total_len = T + max_new_tokens
        all_so_far = torch.zeros(
            (B, max_total_len), dtype=torch.long, device=idx.device
        )
        all_so_far[:, :T] = idx
        current_len = T
        seen_mask = None
        if B == 1:
            seen_mask = torch.zeros(
                self.config.vocab_size, dtype=torch.bool, device=idx.device
            )
            init_ids = idx[0]
            init_ids = init_ids[(init_ids > 0) & (init_ids < self.config.vocab_size)]
            if init_ids.numel() > 0:
                seen_mask[init_ids.unique()] = True

        idx_next = self._sample_token(
            logits,
            all_so_far[:, :current_len],
            temperature,
            top_k,
            top_p,
            min_p,
            repetition_penalty,
            seen_mask=seen_mask,
        )
        all_so_far[:, current_len] = idx_next.squeeze()
        if seen_mask is not None:
            seen_mask[idx_next.squeeze().clamp(0, self.config.vocab_size - 1)] = True
        current_len += 1

        eos_id = getattr(self.config, "eos_id", -1)

        while current_len < T + max_new_tokens:
            if eos_id >= 0 and (idx_next.squeeze(-1) == eos_id).all():
                break

            if draft_model is None:
                # Normal Auto-regressive
                x = self.transformer.wte(idx_next)
                x = self._apply_memory_state(x, memory_state, memory_query_state)
                pos = (current_len - 1) % self.freqs_cos.size(0)
                fc = self.freqs_cos[pos : pos + 1]
                fs = self.freqs_sin[pos : pos + 1]
                for i, block in enumerate(self.transformer.h):
                    x, caches[i] = block(x, fc, fs, kv_cache=caches[i])
                x = self.transformer.ln_f(x)
                logits = self._project_logits(x[:, -1, :], head=head)

                idx_next = self._sample_token(
                    logits,
                    all_so_far[:, :current_len],
                    temperature,
                    top_k,
                    top_p,
                    min_p,
                    repetition_penalty,
                    seen_mask=seen_mask,
                )
                all_so_far[:, current_len] = idx_next.squeeze()
                if seen_mask is not None:
                    seen_mask[
                        idx_next.squeeze().clamp(0, self.config.vocab_size - 1)
                    ] = True
                current_len += 1
            else:
                # Speculative Decoding — Entropy-Adaptive Gamma
                drafted_ids = []
                idx_d = idx_next
                gamma_max = max(2, int(getattr(self, "_adaptive_gamma", gamma)))

                # 1. Draft Phase (run first step to get real-context logits for entropy)
                old_len = current_len
                # Run one step to get actual context-aware logits
                xd0 = draft_model.transformer.wte(idx_d)
                pos_d0 = (old_len - 1) % draft_model.freqs_cos.size(0)
                fcd0 = draft_model.freqs_cos[pos_d0 : pos_d0 + 1]
                fsd0 = draft_model.freqs_sin[pos_d0 : pos_d0 + 1]
                for i, block in enumerate(draft_model.transformer.h):
                    xd0, draft_caches[i] = block(
                        xd0, fcd0, fsd0, kv_cache=draft_caches[i]
                    )
                xd0 = draft_model.transformer.ln_f(xd0)
                logits_d0 = draft_model.lm_head(xd0[:, -1, :])

                # Entropy from draft model logits (evaluates drafting confidence)
                _prob0 = torch.softmax(logits_d0.float(), dim=-1)
                _entropy = -(_prob0 * (_prob0 + 1e-9).log()).sum().item()
                _max_entropy = math.log(draft_model.config.vocab_size)
                _conf = 1.0 - (_entropy / _max_entropy)
                entropy_norm = _entropy / max(_max_entropy, 1e-8)
                if entropy_norm > 0.75:
                    gamma_max = max(2, gamma_max - 1)
                elif entropy_norm < 0.35 and _conf > 0.65:
                    gamma_max = min(max(2, gamma * 2), gamma_max + 1)
                effective_gamma = max(2, int(round((0.5 + 0.5 * _conf) * gamma_max)))

                # First drafted token from above run
                idx_d = torch.argmax(logits_d0, dim=-1, keepdim=True)
                drafted_ids.append(idx_d.item())

                for step in range(1, effective_gamma):
                    if old_len + step >= max_total_len:
                        break
                    xd = draft_model.transformer.wte(idx_d)
                    pos_d = (old_len - 1 + step) % draft_model.freqs_cos.size(0)
                    fcd = draft_model.freqs_cos[pos_d : pos_d + 1]
                    fsd = draft_model.freqs_sin[pos_d : pos_d + 1]
                    for i, block in enumerate(draft_model.transformer.h):
                        xd, draft_caches[i] = block(
                            xd, fcd, fsd, kv_cache=draft_caches[i]
                        )
                    xd = draft_model.transformer.ln_f(xd)
                    logits_d = draft_model.lm_head(xd[:, -1, :])
                    # Greedy sampling for draft model
                    idx_d = torch.argmax(logits_d, dim=-1, keepdim=True)
                    drafted_ids.append(idx_d.item())

                if not drafted_ids:
                    break

                # 2. Verification Phase
                batch_input = [idx_next.item()] + drafted_ids
                new_t = torch.tensor([batch_input], device=idx.device)
                x = self.transformer.wte(new_t)
                x = self._apply_memory_state(x, memory_state, memory_query_state)
                max_len = self.freqs_cos.size(0)
                pos_range = (
                    torch.arange(
                        old_len - 1, old_len - 1 + len(batch_input), device=x.device
                    )
                    % max_len
                )
                fc = self.freqs_cos[pos_range]
                fs = self.freqs_sin[pos_range]

                for i, block in enumerate(self.transformer.h):
                    x, caches[i] = block(x, fc, fs, kv_cache=caches[i])
                x = self.transformer.ln_f(x)
                logits_v = self._project_logits(
                    x, head=head
                )  # [B, len(batch_input), V]

                # 3. Acceptance Phase
                accepted_drafts = 0
                rejected = False
                branch_seen_mask = seen_mask.clone() if seen_mask is not None else None
                for i in range(len(drafted_ids)):
                    l_i = logits_v[:, i, :]
                    idx_v = self._sample_token(
                        l_i,
                        all_so_far[:, : old_len + i],
                        temperature,
                        top_k,
                        top_p,
                        min_p,
                        repetition_penalty,
                        seen_mask=branch_seen_mask,
                        track_entropy=False,
                    )
                    all_so_far[:, old_len + i] = idx_v.squeeze()
                    if branch_seen_mask is not None:
                        branch_seen_mask[
                            idx_v.squeeze().clamp(0, self.config.vocab_size - 1)
                        ] = True
                    if idx_v.item() == drafted_ids[i]:
                        accepted_drafts += 1
                        continue
                    idx_next = idx_v
                    rejected = True
                    break

                if rejected:
                    current_len = old_len + accepted_drafts + 1
                else:
                    # All drafts matched; sample one fresh continuation from the extra verifier logit.
                    l_next = logits_v[:, len(drafted_ids), :]
                    idx_next = self._sample_token(
                        l_next,
                        all_so_far[:, : old_len + len(drafted_ids)],
                        temperature,
                        top_k,
                        top_p,
                        min_p,
                        repetition_penalty,
                        seen_mask=branch_seen_mask,
                    )
                    all_so_far[:, old_len + len(drafted_ids)] = idx_next.squeeze()
                    if branch_seen_mask is not None:
                        branch_seen_mask[
                            idx_next.squeeze().clamp(0, self.config.vocab_size - 1)
                        ] = True
                    current_len = old_len + len(drafted_ids) + 1

                if branch_seen_mask is not None:
                    seen_mask = branch_seen_mask

                accept_rate = accepted_drafts / max(1, len(drafted_ids))
                prev_accept_ema = (
                    self._spec_accept_ema
                    if self._spec_accept_ema is not None
                    else accept_rate
                )
                self._spec_accept_ema = 0.8 * prev_accept_ema + 0.2 * accept_rate
                if self._spec_accept_ema < 0.2:
                    self._spec_disable_steps = max(
                        getattr(self.config, "speculative_disable_steps", 16),
                        len(drafted_ids) * 2,
                    )
                    self._adaptive_gamma = 2
                elif self._spec_accept_ema < 0.3 or _conf < 0.35:
                    self._adaptive_gamma = max(2, gamma_max - 1)
                elif self._spec_accept_ema > 0.7 and entropy_norm < 0.45:
                    self._adaptive_gamma = min(max(2, gamma * 2), gamma_max + 1)
                else:
                    self._adaptive_gamma = max(2, gamma_max)

                # 4. Rollback KV Caches if rejected
                if accepted_drafts < len(drafted_ids):
                    n_reject = len(drafted_ids) - accepted_drafts
                    for c in draft_caches:
                        c.rollback(n_reject)
                    for c in caches:
                        c.rollback(n_reject)
                else:
                    # Draft cache is still one token behind after the final greedy draft; advance once to sync.
                    x_sync = draft_model.transformer.wte(
                        torch.tensor([[drafted_ids[-1]]], device=idx.device)
                    )
                    pos_sync = (
                        old_len + len(drafted_ids) - 1
                    ) % draft_model.freqs_cos.size(0)
                    fcd_sync = draft_model.freqs_cos[pos_sync : pos_sync + 1]
                    fsd_sync = draft_model.freqs_sin[pos_sync : pos_sync + 1]
                    for i, block in enumerate(draft_model.transformer.h):
                        x_sync, draft_caches[i] = block(
                            x_sync, fcd_sync, fsd_sync, kv_cache=draft_caches[i]
                        )

        self.train()
        return all_so_far[:, :current_len]


# ---------------------------------------------------------------------------
# TRAINING UTILITIES
# ---------------------------------------------------------------------------


def get_random_batch(data, block_size, batch_size, device):
    """Random batch for eval."""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64))
            for i in ix
        ]
    )
    if is_cuda_device(device):
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


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
        chunk = self.data[start : start + self.block_size + 1]
        return torch.from_numpy(chunk.astype(np.int32))


class DataLoaderLite:
    """Robust chunked dataloader using PyTorch native DataLoader with pin_memory and multiproc."""

    def __init__(self, data_path, batch_size, block_size, device, eos_id=None):
        self.device = device
        self.eos_id = eos_id
        self.batch_size = batch_size
        self.block_size = block_size
        self.dataset = ChunkDataset(data_path, block_size)

        n_w = (
            min(4, os.cpu_count() or 1)
            if (is_cuda_device(device) or is_xla_device(device))
            else 0
        )
        self.loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=n_w,
            pin_memory=is_cuda_device(device),
            persistent_workers=(
                n_w > 0 and (is_cuda_device(device) or is_xla_device(device))
            ),
            prefetch_factor=2 if n_w > 0 else None,
        )
        self.iter = iter(self.loader)


class CUDAPrefetcher:
    """Async prefetcher for pure GPU utilization."""

    def __init__(self, dataloader_lite):
        self.base_loader = dataloader_lite
        self.loader = dataloader_lite.loader
        self.iter = iter(self.loader)
        self.stream = (
            torch.cuda.Stream() if is_cuda_device(dataloader_lite.device) else None
        )
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


def get_lr(it, config):
    """Linear warmup -> cosine decay -> floor."""
    if it < config.warmup_iters:
        return config.lr * (it + 1) / config.warmup_iters
    if it >= config.max_iters:
        return config.min_lr
    progress = (it - config.warmup_iters) / (config.max_iters - config.warmup_iters)
    return config.min_lr + 0.5 * (config.lr - config.min_lr) * (
        1.0 + math.cos(math.pi * progress)
    )


def make_amp_context(device, dtype):
    if not is_cuda_device(device):
        return contextlib.nullcontext()
    if dtype not in (torch.float16, torch.bfloat16):
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    try:
        return torch.amp.autocast(device_type=device, dtype=dtype)
    except Exception:
        return contextlib.nullcontext()


def safe_load_model_state(model, state_dict, label="checkpoint"):
    try:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(
                f"[warn] {label} loaded with missing={len(missing)} unexpected={len(unexpected)}"
            )
        return True
    except Exception as e:
        print(f"[warn] {label} incompatible -> partial load skipped: {e}")
        return False


def apply_preset_args(args):
    if args.preset == "low":
        args.block_size = 128
        args.batch_size = 8
        args.n_layer = 4
        args.n_embd = 256
        args.n_head = 4
        args.n_kv_head = 1
        args.use_grad_ckpt = False
        args.compile = False
    elif args.preset == "mid":
        args.block_size = 384
        args.batch_size = 32
        args.n_layer = 12
        args.n_embd = 768
        args.n_head = 12
        args.n_kv_head = 4
    elif args.preset == "high":
        args.block_size = 512
        args.batch_size = 128
        args.n_layer = 16
        args.n_embd = 1024
        args.n_head = 16
        args.n_kv_head = 4
        args.use_grad_ckpt = True
    return args


@torch.no_grad()
def estimate_loss(model, train_data, val_data, config, amp_ctx):
    model.eval()
    out = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            x, y = get_random_batch(
                data, config.block_size, config.batch_size, config.device
            )
            with amp_ctx:
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def generate_sample(model, tokenizer, prompt, max_tokens, config, amp_ctx):
    idx = torch.tensor(
        [tokenizer.encode(prompt).ids], dtype=torch.long, device=config.device
    )
    model.eval()
    with amp_ctx:
        out = model.generate(
            idx,
            max_new_tokens=max_tokens,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            min_p=config.min_p,
            repetition_penalty=config.repetition_penalty,
        )
    model.train()
    return tokenizer.decode(out[0].tolist())


def configure_optimizer(model, config, device):
    """Weight decay per param group: only decay 2D weights (matmul weights).

    Do NOT decay:
    - 1D params: RMSNorm weights, biases
    - Embeddings are tied to lm_head so they get decay through that
    """
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    n_decay = sum(p.numel() for p in decay_params)
    n_nodecay = sum(p.numel() for p in no_decay_params)
    print(f"Weight decay : {n_decay:,} params WITH decay, {n_nodecay:,} params WITHOUT")

    param_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    use_fused = is_cuda_device(device)
    try:
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=config.lr,
            betas=(0.9, 0.95),
            fused=use_fused,
        )
        fused_str = "Fused" if use_fused else "Standard"
    except TypeError:
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=config.lr,
            betas=(0.9, 0.95),
        )
        fused_str = "Standard"

    return optimizer, fused_str


# ---------------------------------------------------------------------------
# AUTO CONFIG
# ---------------------------------------------------------------------------


def auto_config(config: Config) -> Config:
    """Auto-tune config based on detected GPU capabilities."""
    if is_xla_device(config.device):
        config.dtype = "bfloat16"
        config.compile_model = False
        target_tokens = (
            50000
            if config.preset == "auto"
            else {"low": 16000, "mid": 50000, "high": 120000}.get(config.preset, 50000)
        )
        config.grad_accum_steps = max(
            1, target_tokens // max(1, config.batch_size * config.block_size)
        )
        if config.sliding_window is None:
            config.sliding_window = max(config.block_size, 512)
        print(f"[auto_config] TPU/XLA device={config.device}")
        print(
            f"[auto_config] dtype={config.dtype}, compile=disabled, "
            f"batch={config.batch_size}, accum={config.grad_accum_steps}, "
            f"sliding_window={config.sliding_window}, grad_ckpt={config.use_grad_ckpt}"
        )
        return config

    if not torch.cuda.is_available():
        config.dtype = "float32"
        config.compile_model = False
        config.batch_size = min(config.batch_size, 1)
        target_tokens = 50000
        config.grad_accum_steps = max(
            1, target_tokens // max(1, config.batch_size * config.block_size)
        )
        return config

    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024**3)
    sm = props.major

    # dtype
    config.dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"

    # compile mode
    config.compile_mode = "max-autotune" if sm >= 8 else "reduce-overhead"

    # batch scaling
    if config.preset == "auto":
        if vram_gb < 16:
            config.batch_size = 1
            config.block_size = min(config.block_size, 256)
            config.use_grad_ckpt = True
        elif vram_gb < 40:
            config.batch_size = 2
        else:
            config.batch_size = 4

    target_tokens = (
        20000
        if config.preset == "auto"
        else {"low": 8000, "mid": 20000, "high": 50000}.get(config.preset, 20000)
    )
    config.grad_accum_steps = max(
        1, target_tokens // max(1, config.batch_size * config.block_size)
    )

    # grad checkpoint
    if config.block_size * config.n_layer > 4000:
        config.use_grad_ckpt = True

    # sliding window theo KV budget (30% VRAM) — chỉ set nếu user chưa chỉ định
    if config.sliding_window is None:
        head_dim = config.n_embd // config.n_head
        bytes_per_elem = 2 if config.dtype in ("float16", "bfloat16") else 4
        bytes_per_token = config.n_kv_head * head_dim * 2 * bytes_per_elem  # k+v
        kv_budget = vram_gb * (1024**3) * 0.3
        max_tokens = kv_budget // (bytes_per_token * config.n_layer * config.batch_size)
        config.sliding_window = int(min(max_tokens, config.block_size * 2))

    print(
        f"[auto_config] GPU={props.name} ({vram_gb:.1f}GB, sm_{props.major}{props.minor})"
    )
    print(
        f"[auto_config] dtype={config.dtype}, compile={config.compile_mode}, "
        f"batch={config.batch_size}, accum={config.grad_accum_steps}, "
        f"sliding_window={config.sliding_window}, grad_ckpt={config.use_grad_ckpt}"
    )

    return config


# ---------------------------------------------------------------------------
# MAIN TRAINING
# ---------------------------------------------------------------------------


def train(config: Config):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        configure_tf32_runtime()
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    config = auto_config(config)

    device = config.device
    tokenizer = load_tokenizer()
    config.vocab_size = tokenizer.get_vocab_size()
    assert config.vocab_size == tokenizer.get_vocab_size(), (
        "Tokenizer vocab mismatch between config and runtime tokenizer"
    )
    config.eos_id = tokenizer.token_to_id(EOS_TOKEN)

    print("=" * 65)
    print("  Nexa 1  (~1B, Reasoning + GQA)")
    print("=" * 65)
    print(f"\nDevice       : {device}")
    if is_cuda_device(device):
        props = torch.cuda.get_device_properties(0)
        print(f"GPU          : {props.name} (sm_{props.major}{props.minor})")
        print(f"VRAM         : {props.total_memory / (1024**3):.1f} GB")
        if props.major < 8:
            print("SDPA backend : memory-efficient (no flash_attn)")
        else:
            print("SDPA backend : flash_attention_2")
    elif is_xla_device(device):
        print("Accelerator  : TPU/XLA")

    # Data
    train_path = os.path.join(config.data_dir, "train.bin")
    val_path = os.path.join(config.data_dir, "val.bin")

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise RuntimeError(
            f"Missing {train_path} or {val_path}. Run pre_train.py first, then train with --data-dir."
        )

    train_data = np.memmap(train_path, dtype=np.uint16, mode="r")
    val_data = np.memmap(val_path, dtype=np.uint16, mode="r")

    print(f"\nTrain tokens : {len(train_data):,}")
    print(f"Val tokens   : {len(val_data):,}")
    print(f"Tokenizer    : Nexa BPE (vocab={config.vocab_size})")

    print(
        f"\nArchitecture : block={config.block_size}, embd={config.n_embd}, "
        f"q_head={config.n_head}, kv_head={config.n_kv_head}, layer={config.n_layer}"
    )
    print(
        f"GQA ratio    : {config.n_head}:{config.n_kv_head} "
        f"({config.n_head // config.n_kv_head} Q per KV group)"
    )
    print("Block style  : Parallel Attn+FFN (PaLM)")

    model = NexaModel(config).to(device)

    # Resume from checkpoint if exists
    resume_iter = 0
    ckpt_path = os.path.join(config.checkpoint_dir, "best.pt")
    if os.path.exists(ckpt_path):
        print(f"\nResuming from {ckpt_path}...")
        map_location = "cpu" if is_xla_device(device) else device
        ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
        safe_load_model_state(model, ckpt["model"], label="resume checkpoint")
        resume_iter = ckpt.get("iter", 0)
        print(f"  Resumed at iter {resume_iter}, val_loss={ckpt.get('val_loss', '?')}")

    if config.compile_model and hasattr(torch, "compile"):
        print(f"torch.compile enabled (mode={config.compile_mode})")
        try:
            model = torch.compile(model, mode=config.compile_mode)
        except Exception as e:
            print(f"[warn] torch.compile failed: {e}. Falling back to eager mode.")

    # Optimizer with weight decay groups
    optimizer, fused_str = configure_optimizer(model, config, device)

    # Restore optimizer + scaler state if resuming
    if resume_iter > 0:
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            print("  Restored optimizer state")

    micro_bs = config.batch_size
    tok_per_step = micro_bs * config.grad_accum_steps * config.block_size
    print(f"Optimizer    : AdamW ({fused_str}, lr={config.lr}, min_lr={config.min_lr})")
    print(
        f"Batch        : {micro_bs} x {config.grad_accum_steps} accum x {config.block_size} = {tok_per_step:,} tok/step"
    )
    print(f"Max iters    : {config.max_iters:,}")
    print(
        f"LR schedule  : warmup({config.warmup_iters}) -> cosine -> floor({config.min_lr})"
    )

    # AMP
    use_amp = is_cuda_device(device)
    dtype_obj = getattr(torch, config.dtype, torch.float32)
    amp_ctx = (
        make_amp_context(device, dtype_obj) if use_amp else contextlib.nullcontext()
    )
    scaler = (
        torch.cuda.amp.GradScaler(enabled=(config.dtype == "float16"))
        if use_amp
        else None
    )

    # Restore scaler state if resuming
    if resume_iter > 0 and scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
        print("  Restored scaler state")

    print(f"\n{'\u2500' * 65}")
    print("  TRAINING")
    print(f"{'\u2500' * 65}")

    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    base_loader = DataLoaderLite(
        train_path, config.batch_size, config.block_size, device, eos_id=eos_id
    )
    train_loader = CUDAPrefetcher(base_loader)

    if resume_iter > 0:
        print(
            "  Note: Resuming with ChunkDataset skips pointer restore due to randomized sampling."
        )

    t0 = time.time()
    best_val = ckpt.get("val_loss", float("inf")) if resume_iter > 0 else float("inf")

    for it in range(resume_iter, config.max_iters + 1):
        lr = get_lr(it, config)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Eval
        if it % config.eval_interval == 0 or it == config.max_iters:
            losses = estimate_loss(model, train_data, val_data, config, amp_ctx)
            elapsed = time.time() - t0
            ppl = math.exp(min(losses["val"], 20))
            print(
                f"\n  step {it:>5d} | "
                f"train {losses['train']:.4f} | "
                f"val {losses['val']:.4f} | "
                f"ppl {ppl:.2f} | "
                f"lr {lr:.2e} | "
                f"{elapsed:.0f}s"
            )

            if losses["val"] < best_val:
                best_val = losses["val"]
                os.makedirs(config.checkpoint_dir, exist_ok=True)
                ckpt_data = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "iter": it,
                    "val_loss": best_val,
                    "config": config,
                    "meta": {
                        "tokenizer": "nexa",
                        "vocab_size": config.vocab_size,
                        "dtype": config.dtype,
                        "sliding_window": config.sliding_window,
                        "block_size": config.block_size,
                    },
                }
                if scaler is not None:
                    ckpt_data["scaler"] = scaler.state_dict()
                torch.save(ckpt_data, os.path.join(config.checkpoint_dir, "best.pt"))
                print(f"  ** best model saved (val={best_val:.4f})")

            sample = generate_sample(
                model, tokenizer, "The meaning of life is", 100, config, amp_ctx
            )
            print(f"  >> {sample[:200]}...\n")

        if it == config.max_iters:
            break

        # Train step
        accum_loss = 0.0
        micro_step = 0
        skip_step = False
        optimizer.zero_grad(set_to_none=True)  # zero ONCE before all micro steps
        while micro_step < config.grad_accum_steps:
            x, y = train_loader.next_batch()

            with amp_ctx:
                _, loss = model(x, y)
                loss = loss / config.grad_accum_steps

            if not torch.isfinite(loss):
                print(f"\n[warn] NaN loss detected! step={it}, lr={lr}. Skipping step.")
                optimizer.zero_grad(set_to_none=True)
                accum_loss = 0.0
                skip_step = True
                break

            try:
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and config.batch_size > 1:
                    print(
                        f"\n[warn] CUDA OOM! Auto-downscaling batch_size {config.batch_size} -> {config.batch_size // 2}"
                    )
                    if hasattr(torch.cuda, "empty_cache"):
                        torch.cuda.empty_cache()
                    config.batch_size //= 2
                    base_loader = DataLoaderLite(
                        train_path,
                        config.batch_size,
                        config.block_size,
                        device,
                        eos_id=eos_id,
                    )
                    train_loader = CUDAPrefetcher(base_loader)
                    optimizer.zero_grad(set_to_none=True)
                    micro_step = 0
                    accum_loss = 0.0
                    continue
                else:
                    raise e

            accum_loss += loss.item()
            micro_step += 1

        if skip_step:
            continue

        norm = -1.0
        if scaler is not None:
            if config.grad_clip > 0:
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.grad_clip
                ).item()
            prev_scale = scaler.get_scale()
            if is_xla_device(device):
                optimizer.step()
            else:
                scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < prev_scale:
                print(
                    f"\n⚠️  AMP overflow step={it} scale {prev_scale:.0f}→{scaler.get_scale():.0f} (loss={accum_loss:.4f}, norm={norm:.2f})"
                )
        else:
            if config.grad_clip > 0:
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.grad_clip
                ).item()
            if is_xla_device(device):
                import torch_xla.core.xla_model as xm

                xm.optimizer_step(optimizer, barrier=False)
                xm.mark_step()
            else:
                optimizer.step()
        loss = accum_loss

        # Stability checks
        if not math.isfinite(loss):
            print(f"\n⚠️  NaN/Inf loss at step {it} — skipping")
            optimizer.zero_grad(set_to_none=True)
            continue

        if norm > 100:
            print(f"\n⚠️  Gradient explode at step {it}: norm={norm:.1f}")

        bad_grad = any(
            p.grad is not None and not torch.isfinite(p.grad).all()
            for p in model.parameters()
        )
        if bad_grad:
            print(f"\n⚠️  Bad gradient at step {it} — skipping optimizer step")
            optimizer.zero_grad(set_to_none=True)
            continue

        # Periodic VRAM defrag
        if it % 200 == 0 and is_cuda_device(device):
            torch.cuda.empty_cache()

        if it > 0 and it % config.log_interval == 0:
            elapsed = time.time() - t0
            tok_per_step = (
                config.batch_size * config.grad_accum_steps * config.block_size
            )
            tps = (it * tok_per_step) / elapsed
            rem_iters = config.max_iters - it
            eta_s = rem_iters * (elapsed / it)
            eta_str = f"{eta_s / 60:.1f}m" if eta_s < 3600 else f"{eta_s / 3600:.1f}h"

            gpu_str = ""
            if is_cuda_device(device):
                mem_alloc = torch.cuda.memory_allocated() / 1e9
                mem_res = torch.cuda.memory_reserved() / 1e9
                gpu_str = f" | VRAM: {mem_alloc:.1f}/{mem_res:.1f}G"

            norm_str = f" | norm {norm:.2f}" if norm >= 0 else ""

            print(
                f"  step {it:>5d}/{config.max_iters} | loss {loss:.4f} | lr {lr:.2e}{norm_str} | {tps:,.0f} tok/s | ETA: {eta_str}{gpu_str}"
            )

    total = time.time() - t0
    print(f"\n{'\u2500' * 65}")
    print(f"  DONE in {total:.0f}s ({total / 60:.1f}min)")
    print(f"  Best val loss: {best_val:.4f} | PPL: {math.exp(min(best_val, 20)):.2f}")
    print(f"{'\u2500' * 65}")

    prompts = [
        "The meaning of life is",
        "In a world where",
        "Scientists have discovered",
    ]
    for i, prompt in enumerate(prompts):
        text = generate_sample(
            model, tokenizer, prompt, config.gen_len, config, amp_ctx
        )
        print(f"\n--- Sample {i + 1} (prompt: '{prompt}') ---")
        print(text)

    print(f"\n{'=' * 65}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--block-size", type=int, default=384)
    p.add_argument("--n-embd", type=int, default=1792)
    p.add_argument("--n-head", type=int, default=16)
    p.add_argument("--n-kv-head", type=int, default=4)
    p.add_argument(
        "--use-grad-ckpt", action=argparse.BooleanOptionalAction, default=False
    )
    p.add_argument(
        "--sliding-window",
        type=int,
        default=None,
        help="Sliding window size (default: use config value 512)",
    )
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--n-layer", type=int, default=28)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--min-lr", type=float, default=2e-5)
    p.add_argument("--max-iters", type=int, default=10000)
    p.add_argument("--warmup-iters", type=int, default=500)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--gen-len", type=int, default=200)
    p.add_argument("--num-samples", type=int, default=3)
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--compile-mode",
        type=str,
        default="max-autotune",
        choices=["default", "reduce-overhead", "max-autotune"],
    )
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--generate", action="store_true", help="Generate only")
    p.add_argument("--prompt", type=str, default="The meaning of life is")
    p.add_argument("--demo", action="store_true", help="Run tiny demo configuration")
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu", "xla", "tpu"],
    )
    p.add_argument(
        "--preset", type=str, default="auto", choices=["auto", "low", "mid", "high"]
    )
    p.add_argument("--low-vram", action="store_true", help="Force low VRAM profile")
    p.add_argument("--cpu-only", action="store_true", help="Force CPU mode")
    p.add_argument(
        "--dtype", type=str, default="auto", choices=["auto", "fp16", "bf16", "fp32"]
    )

    args = p.parse_args()

    import random

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        configure_tf32_runtime()
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    args = apply_preset_args(args)

    # Device config
    cuda_works = safe_cuda_alloc()
    xla_works = safe_xla_alloc()
    wants_xla = args.device in ("xla", "tpu")
    if wants_xla and not xla_works:
        raise RuntimeError("XLA/TPU requested but torch_xla device is not available")

    if args.cpu_only or args.device == "cpu":
        device_str = "cpu"
        dtype_str = "float32"
        vram_gb = 0
        args.compile = False
        args.batch_size = min(args.batch_size, 8)
    elif wants_xla or (args.device == "auto" and not cuda_works and xla_works):
        device_str = str(get_xla_device())
        dtype_str = "bfloat16"
        vram_gb = 0
        args.compile = False
    elif cuda_works:
        device_str = "cuda"
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        dtype_str = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    else:
        device_str = "cpu"
        dtype_str = "float32"
        vram_gb = 0
        args.compile = False
        args.batch_size = min(args.batch_size, 8)

    if args.dtype != "auto":
        dtype_map = {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}
        dtype_str = dtype_map[args.dtype]

    if (
        args.compile_mode == "max-autotune"
        and is_cuda_device(device_str)
        and torch.cuda.get_device_properties(0).major < 8
    ):
        args.compile_mode = "reduce-overhead"

    use_grad_ckpt = args.use_grad_ckpt
    if args.demo:
        args.block_size = min(args.block_size, 128)
        args.n_embd = min(args.n_embd, 128)
        args.n_head = min(args.n_head, 4)
        args.n_kv_head = min(args.n_kv_head, 1)
        args.n_layer = min(args.n_layer, 4)
        args.batch_size = min(args.batch_size, 4)
        args.max_iters = min(args.max_iters, 20)
        args.eval_interval = 10
        args.log_interval = 5
    elif args.device == "auto" or is_cuda_device(device_str):
        if vram_gb < 8 or args.low_vram:
            args.batch_size = min(args.batch_size, 8)
            args.block_size = min(args.block_size, 128)
            use_grad_ckpt = True
        elif vram_gb < 16:
            args.batch_size = min(args.batch_size, 16)
            args.block_size = min(args.block_size, 256)
        else:
            args.batch_size = min(args.batch_size, 32)
            args.block_size = min(args.block_size, 384)

        if args.block_size * args.n_layer > 4000:
            use_grad_ckpt = True

    if not args.compile or device_str == "cpu" or is_xla_device(device_str):
        args.compile = False

    vocab_size = get_vocab_size()
    config = Config(
        data_dir=args.data_dir,
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_kv_head=args.n_kv_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        min_lr=args.min_lr,
        max_iters=args.max_iters,
        warmup_iters=args.warmup_iters,
        temperature=args.temperature,
        top_k=args.top_k,
        gen_len=args.gen_len,
        num_samples=args.num_samples,
        compile_model=args.compile,
        compile_mode=args.compile_mode,
        eval_interval=args.eval_interval,
        log_interval=args.log_interval,
        checkpoint_dir=args.checkpoint_dir,
        device=device_str,
        dtype=dtype_str,
        use_grad_ckpt=use_grad_ckpt,
        sliding_window=args.sliding_window if args.sliding_window is not None else None,
        preset=args.preset,
    )

    if args.generate:
        tokenizer = load_tokenizer()
        assert config.vocab_size == tokenizer.get_vocab_size(), (
            "Tokenizer vocab mismatch between config and runtime tokenizer"
        )
        ckpt_path = os.path.join(config.checkpoint_dir, "best.pt")
        assert os.path.exists(ckpt_path), f"No checkpoint at {ckpt_path}"
        map_location = "cpu" if is_xla_device(config.device) else config.device
        ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
        model = NexaModel(config).to(config.device)
        safe_load_model_state(model, ckpt["model"], label="generate checkpoint")
        dtype = next(model.parameters()).dtype
        amp_ctx = make_amp_context(config.device, dtype)
        print(f"\nLoaded checkpoint (val_loss={ckpt['val_loss']:.4f})")
        text = generate_sample(
            model, tokenizer, args.prompt, config.gen_len, config, amp_ctx
        )
        print(f"\nPrompt: {args.prompt}")
        print(f"Generated:\n{text}")
        return

    train(config)


if __name__ == "__main__":
    main()
