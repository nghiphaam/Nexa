"""Cross-modal attention fusion for Nexa 1.6 multimodal inputs."""
from __future__ import annotations

import torch.nn as nn


class _FusionBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int, hidden_size: int, dropout: float):
        super().__init__()
        self.text_to_image = nn.MultiheadAttention(
            embed_dim=n_embd,
            num_heads=n_head,
            dropout=dropout,
            batch_first=True,
        )
        self.image_to_text = nn.MultiheadAttention(
            embed_dim=n_embd,
            num_heads=n_head,
            dropout=dropout,
            batch_first=True,
        )
        self.text_norm1 = nn.LayerNorm(n_embd)
        self.image_norm1 = nn.LayerNorm(n_embd)
        self.text_norm2 = nn.LayerNorm(n_embd)
        self.image_norm2 = nn.LayerNorm(n_embd)
        self.text_ff = nn.Sequential(
            nn.Linear(n_embd, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_embd),
        )
        self.image_ff = nn.Sequential(
            nn.Linear(n_embd, hidden_size),
            nn.GELU(),
                        nn.Dropout(dropout),
            nn.Linear(hidden_size, n_embd),
        )

    def forward(self, text_tokens, image_tokens):
        if image_tokens is None or image_tokens.size(1) == 0:
            return text_tokens, image_tokens
        text_ctx, _ = self.text_to_image(text_tokens, image_tokens, image_tokens, need_weights=False)
        image_ctx, _ = self.image_to_text(image_tokens, text_tokens, text_tokens, need_weights=False)
        text_tokens = self.text_norm1(text_tokens + text_ctx)
        image_tokens = self.image_norm1(image_tokens + image_ctx)
        text_tokens = self.text_norm2(text_tokens + self.text_ff(text_tokens))
        image_tokens = self.image_norm2(image_tokens + self.image_ff(image_tokens))
        return text_tokens, image_tokens


class CrossModalAttentionFusion(nn.Module):
    """Bidirectional cross-modal fusion between text and image token streams."""

    def __init__(self, n_embd: int, n_head: int = 8, hidden_size: int = 2304, n_layer: int = 2, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            _FusionBlock(n_embd=n_embd, n_head=n_head, hidden_size=hidden_size, dropout=dropout)
            for _ in range(max(1, n_layer))
        ])

    def forward(self, text_tokens, image_tokens):
        for layer in self.layers:
            text_tokens, image_tokens = layer(text_tokens, image_tokens)
        return text_tokens, image_tokens
