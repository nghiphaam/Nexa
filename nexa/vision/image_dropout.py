"""Image dropout with learnable null token."""
import random
import torch
import torch.nn as nn


class ImageDropout(nn.Module):
    def __init__(self, n_embd, p=0.1):
        super().__init__()
        self.p = p
        self.null_token = nn.Parameter(torch.randn(1, 1, n_embd) * 0.02)

    def forward(self, image_tokens, training=True):
        if training and random.random() < self.p:
            B, N, D = image_tokens.shape
            return self.null_token.expand(B, N, D)
        return image_tokens
