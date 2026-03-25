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

        self.text_model = NexaModel(config)

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

        features = self.vision_encoder(images)

        features = self.patch_selector(features)

        image_tokens = self.projector(features)

        image_tokens, gate_alpha = self.gate(image_tokens)

        image_tokens = self.image_dropout(image_tokens, training=training)

        return image_tokens, gate_alpha

    def forward(self, input_ids=None, labels=None, images=None, **kwargs):
        """
        Forward pass.

        If images provided, prepend image pseudo tokens to text embeddings.
        """
        if images is None:
            logits, loss = self.text_model(input_ids, labels)
            return {
                "logits": logits,
                "loss": loss,
                "image_features": None,
                "text_features": None,
                "gate_alpha": None,
            }

        image_tokens, gate_alpha = self.encode_image(images, training=self.training)

        text_emb = self.text_model.transformer.wte(input_ids)

        x = torch.cat([image_tokens, text_emb], dim=1)

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

        if T > self.config.block_size:
            x = x[:, :self.config.block_size, :]
            T = self.config.block_size
            if labels is not None:
                labels = labels[:, :self.config.block_size]

        x = self.text_model.transformer.drop(x)

        pos = torch.arange(0, T, device=x.device)
        freqs_cos = self.text_model.freqs_cos[pos]
        freqs_sin = self.text_model.freqs_sin[pos]

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
        """Generate text with optional image.

        Args:
            input_ids: (B, T) token ids, may contain <image_start> and <image_end> markers
            images: Optional images (B, C, H, W) or list of PIL images
            max_new_tokens: Number of tokens to generate
            **kwargs: Sampling parameters (temperature, top_k, top_p, etc.)

        Returns:
            Generated token ids (B, T + max_new_tokens)
        """
        if images is None:
            return self.text_model.generate(input_ids, max_new_tokens, **kwargs)

        image_tokens, _ = self.encode_image(images, training=False)
        B = image_tokens.size(0)

        text_emb = self.text_model.transformer.wte(input_ids)  # (B, T, n_embd)

        if self.image_start_id is None or self.image_end_id is None:
            raise ValueError("Image token IDs not set. Call set_image_token_ids() first.")

        full_emb = []
        for b in range(B):
            seq = input_ids[b]
            emb = text_emb[b]

            start_pos = (seq == self.image_start_id).nonzero(as_tuple=True)[0]
            end_pos = (seq == self.image_end_id).nonzero(as_tuple=True)[0]

            if len(start_pos) > 0 and len(end_pos) > 0:
                start_idx = start_pos[0].item()
                end_idx = end_pos[0].item()

                before = emb[:start_idx]
                after = emb[end_idx + 1:]
                img_emb = image_tokens[b]

                img_emb, _ = self.gate(img_emb)

                seq_emb = torch.cat([before, img_emb, after], dim=0)
            else:
                seq_emb = emb

            full_emb.append(seq_emb)

        max_len = max(e.size(0) for e in full_emb)
        padded_emb = []
        for emb in full_emb:
            if emb.size(0) < max_len:
                pad = torch.zeros(max_len - emb.size(0), emb.size(1), device=emb.device, dtype=emb.dtype)
                emb = torch.cat([emb, pad], dim=0)
            padded_emb.append(emb)

        x = torch.stack(padded_emb, dim=0)  # (B, T', n_embd)

        generated = []
        for _ in range(max_new_tokens):
            T = x.size(1)
            if T > self.config.block_size:
                x_input = x[:, -self.config.block_size:]
                T = self.config.block_size
            else:
                x_input = x

            pos = torch.arange(0, T, device=x.device)
            freqs_cos = self.text_model.freqs_cos[pos]
            freqs_sin = self.text_model.freqs_sin[pos]

            h = self.text_model.transformer.drop(x_input)
            for block in self.text_model.transformer.h:
                h, _ = block(h, freqs_cos, freqs_sin, kv_cache=None)
            h = self.text_model.transformer.ln_f(h)
            logits = self.text_model.lm_head(h)  # (B, T, vocab_size)

            next_token_logits = logits[:, -1, :]  # (B, vocab_size)

            temperature = kwargs.get('temperature', 0.8)
            top_k = kwargs.get('top_k', 50)
            top_p = kwargs.get('top_p', 0.9)
            min_p = kwargs.get('min_p', 0.05)

            next_token_logits = next_token_logits / temperature

            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')

            if min_p > 0.0:
                probs = F.softmax(next_token_logits, dim=-1)
                max_prob = probs.max(dim=-1, keepdim=True)[0]
                indices_to_remove = probs < (min_p * max_prob)
                next_token_logits[indices_to_remove] = float('-inf')

            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            generated.append(next_token)

            next_emb = self.text_model.transformer.wte(next_token)  # (B, 1, n_embd)
            x = torch.cat([x, next_emb], dim=1)

            if self.config.eos_id is not None and (next_token == self.config.eos_id).all():
                break

        if generated:
            generated_ids = torch.cat(generated, dim=1)  # (B, max_new_tokens)
            return generated_ids
        else:
            return torch.empty(B, 0, dtype=torch.long, device=input_ids.device)
