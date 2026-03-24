"""Vision encoder using pretrained SigLIP (frozen)."""
import torch
import torch.nn as nn


class VisionEncoder(nn.Module):
    def __init__(self, model_name: str = "google/siglip-base-patch16-224"):
        super().__init__()
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(model_name)
        self.model.requires_grad_(False)  # FROZEN

        # Get hidden size
        if hasattr(self.model.config, "hidden_size"):
            self.hidden_size = self.model.config.hidden_size
        elif hasattr(self.model.config, "vision_config"):
            self.hidden_size = self.model.config.vision_config.hidden_size
        else:
            self.hidden_size = 768  # fallback

    @torch.no_grad()
    def forward(self, images):
        out = self.model(images)

        # Safe access to features
        if hasattr(out, "last_hidden_state"):
            features = out.last_hidden_state
        elif hasattr(out, "vision_model"):
            features = out.vision_model.last_hidden_state
        else:
            features = out[0]

        # Drop CLS token if present
        if features.size(1) > 1:
            features = features[:, 1:, :]

        return features
