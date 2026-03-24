"""Vision projector: maps vision features to text embedding space."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionProjector(nn.Module):
    def __init__(self, vision_dim=768, n_embd=2048, num_tokens=64):
        super().__init__()
        self.n_embd = n_embd
        self.proj = nn.Sequential(
            nn.Linear(vision_dim, n_embd),
            nn.GELU(),
            nn.Linear(n_embd, n_embd),
            nn.LayerNorm(n_embd)
        )
        self.image_pos_emb = nn.Parameter(torch.randn(num_tokens, n_embd) * 0.02)

    def forward(self, vision_features):
        # Normalize while preserving relative magnitude
        norm = vision_features.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        vision_features = vision_features / norm

        # Project
        projected = self.proj(vision_features)

        # Scale to match text embedding distribution
        projected = projected * (self.n_embd ** -0.5)

        # Add image-specific positional embedding
        projected = projected + self.image_pos_emb.unsqueeze(0)

        return projected
