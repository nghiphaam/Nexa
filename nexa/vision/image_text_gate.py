"""Image-text gate to balance image and text contributions."""
import torch
import torch.nn as nn


class ImageTextGate(nn.Module):
    def __init__(self, n_embd, temperature=1.0, freeze_steps=1000):
        super().__init__()
        self.gate = nn.Linear(n_embd, 1)
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.freeze_steps = freeze_steps
        self.current_step = 0

    def forward(self, image_proj):
        # Clamp temperature to prevent divergence
        temp = self.temperature.clamp(0.1, 10.0)

        # Freeze gate for first N steps
        if self.current_step < self.freeze_steps:
            alpha = torch.full(
                (image_proj.size(0), 1),
                0.75,
                device=image_proj.device,
                dtype=image_proj.dtype
            )
        else:
            logit = self.gate(image_proj.mean(dim=1)) / temp
            # Ensure minimum 50% signal
            alpha = 0.5 + 0.5 * torch.sigmoid(logit)

        image_proj = alpha.unsqueeze(1) * image_proj
        return image_proj, alpha

    def step(self):
        self.current_step += 1
