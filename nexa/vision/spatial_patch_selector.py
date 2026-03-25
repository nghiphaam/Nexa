"""Spatial patch selector for reducing image tokens."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialPatchSelector(nn.Module):
    def __init__(self, num_tokens=64):
        super().__init__()
        self.num_tokens = num_tokens
        self.attention = None  # optional top-k upgrade later

    def forward(self, features, use_attention=False):
        # features: [B, N, D]
        B, N, D = features.shape

        if use_attention and self.attention is not None:
            # Top-k by attention scores
            scores = self.attention(features).squeeze(-1)
            _, indices = torch.topk(scores, self.num_tokens, dim=1)
            indices = indices.unsqueeze(-1).expand(-1, -1, D)
            selected = torch.gather(features, 1, indices)
        else:
            # Simple adaptive pooling for v1.6
            selected = F.adaptive_avg_pool1d(
                features.transpose(1, 2),
                self.num_tokens
            ).transpose(1, 2)

        return selected
