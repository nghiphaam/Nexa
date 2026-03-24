"""Multimodal model: NexaModel + vision support."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from nexa.model.nexa_model import NexaModel
from nexa.vision import (
    VisionEncoder,
    VisionProjector,
    ImageTextGate,
    ImageDropout,
    SpatialPatchSelector,
    ImageProcessor,
)


class MultimodalModel(nn.Module):
    def __init__(self, config, vision_model_name="google/siglip-base-patch16-224", num_image_tokens=64):
        super().__init__()
        self.config = config
        self.num_image_tokens = num_image_tokens

        # Base text model
        self.text_model = NexaModel(config)

        # Vision modules
        self.vision_encoder = VisionEncoder(vision_model_name)
        self.patch_selector = SpatialPatchSelector(num_image_tokens)
        self.projector = VisionProjector(
            vision_dim=self.vision_encoder.hidden_size,
            n_embd=config.n_embd,
            num_tokens=num_image_tokens,
        )
        self.gate = ImageTextGate(config.n_embd)
        self.image_dropout = ImageDropout(config.n_embd)
        self.image_processor = ImageProcessor(vision_model_name)

        # Special token ids (to be set externally after tokenizer resize)
        self.image_start_id = None
        self.image_end_id = None

    def set_image_token_ids(self, image_start_id: int, image_end_id: int):
        self.image_start_id = image_start_id
        self.image_end_id = image_end_id

    def encode_image(self, images, training=False):
        """Encode images into pseudo text tokens."""
        if isinstance(images, str) or not isinstance(images, torch.Tensor):
            images = self.image_processor(images)

        if images.dim() == 3:
            images = images.unsqueeze(0)

        # Vision encoding
        features = self.vision_encoder(images)

        # Select patches
        features = self.patch_selector(features)

        # Project to text space
        image_tokens = self.projector(features)

        # Gate image signal
        image_tokens, gate_alpha = self.gate(image_tokens)

        # Image dropout during training
        image_tokens = self.image_dropout(image_tokens, training=training)

        return image_tokens, gate_alpha

    def forward(self, input_ids=None, labels=None, images=None, **kwargs):
        """
        Forward pass.

        If images provided, prepend image pseudo tokens to text embeddings.
        """
        if images is None:
            # Text-only path
            logits, loss = self.text_model(input_ids, labels)
            return {
                "logits": logits,
                "loss": loss,
                "image_features": None,
                "text_features": None,
                "gate_alpha": None,
            }

        # Encode image
        image_tokens, gate_alpha = self.encode_image(images, training=self.training)

        # Get text embeddings
        text_emb = self.text_model.transformer.wte(input_ids)

        # Concatenate image tokens + text embeddings
        x = torch.cat([image_tokens, text_emb], dim=1)

        # Handle labels: pad image token positions with ignore_index
        if labels is not None:
            B = labels.size(0)
            ignore_labels = torch.full(
                (B, self.num_image_tokens),
                -100,
                device=labels.device,
                dtype=labels.dtype,
            )
            full_labels = torch.cat([ignore_labels, labels], dim=1)
        else:
            full_labels = None

        # Forward through text model manually with embeddings
        logits, loss = self._forward_with_embeddings(x, full_labels)

        return {
            "logits": logits,
            "loss": loss,
            "image_features": image_tokens,
            "text_features": text_emb,
            "gate_alpha": gate_alpha,
        }

    def _forward_with_embeddings(self, x, labels=None):
        """Forward pass using pre-computed embeddings."""
        B, T, C = x.size()

        # Check block size
        if T > self.config.block_size:
            x = x[:, :self.config.block_size, :]
            T = self.config.block_size
            if labels is not None:
                labels = labels[:, :self.config.block_size]

        x = self.text_model.transformer.drop(x)

        # Position embeddings via RoPE
        pos = torch.arange(0, T, device=x.device)
        freqs_cos = self.text_model.freqs_cos[pos]
        freqs_sin = self.text_model.freqs_sin[pos]

        # Transformer blocks
        for block in self.text_model.transformer.h:
            x, _ = block(x, freqs_cos, freqs_sin, kv_cache=None)

        x = self.text_model.transformer.ln_f(x)
        logits = self.text_model.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids, images=None, max_new_tokens=100, **kwargs):
        """Generate text with optional image."""
        if images is None:
            return self.text_model.generate(input_ids, max_new_tokens, **kwargs)

        raise NotImplementedError(
            "Multimodal generation is not implemented yet. "
            "Use forward(..., images=...) for training/evaluation only."
        )
