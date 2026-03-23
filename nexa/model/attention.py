"""Causal Self-Attention with GQA and TransformerBlock."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from nexa.model.components import RMSNorm, FeedForward, apply_rope, repeat_kv


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
            w = getattr(self.config, "sliding_window", None) if hasattr(self.config, "sliding_window") else None
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
            q, k_exp, v_exp,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=use_causal,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y, new_cache


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
