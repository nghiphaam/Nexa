"""Multimodal model for Nexa 1.6: language backbone + vision fusion."""
from __future__ import annotations

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
    CrossModalAttentionFusion,
)


class MultimodalModel(nn.Module):
    """Language model with image encoding and cross-modal attention fusion."""

    def __init__(self, config, vision_model_name=None, num_image_tokens=None):
        super().__init__()
        self.config = config
        self.num_image_tokens = num_image_tokens or getattr(config, "num_image_tokens", 64)
        vision_model_name = vision_model_name or getattr(config, "vision_model_name", "google/siglip-base-patch16-224")

        self.text_model = NexaModel(config, use_grad_ckpt=getattr(config, "use_grad_ckpt", False))
        self.vision_encoder = VisionEncoder(vision_model_name)
        self.patch_selector = SpatialPatchSelector(self.num_image_tokens)
        self.projector = VisionProjector(
            vision_dim=self.vision_encoder.hidden_size,
            n_embd=config.n_embd,
            num_tokens=self.num_image_tokens,
        )
        self.cross_modal_fusion = CrossModalAttentionFusion(
            n_embd=config.n_embd,
            n_head=getattr(config, "cross_modal_heads", 8),
            hidden_size=getattr(config, "multimodal_hidden_size", config.n_embd),
            n_layer=getattr(config, "cross_modal_layers", 2),
            dropout=getattr(config, "multimodal_dropout", config.dropout),
        )
        self.gate = ImageTextGate(config.n_embd)
        self.image_dropout = ImageDropout(config.n_embd)
        self.image_processor = ImageProcessor(vision_model_name)
        self.image_start_id = None
        self.image_end_id = None

    def set_image_token_ids(self, image_start_id: int, image_end_id: int):
        self.image_start_id = image_start_id
        self.image_end_id = image_end_id

    def _match_image_batch(self, image_tokens, batch_size: int):
        if image_tokens.size(0) == batch_size:
            return image_tokens
        if image_tokens.size(0) == 1 and batch_size > 1:
            return image_tokens.repeat(batch_size, 1, 1)
        raise ValueError(
            f"Image batch size ({image_tokens.size(0)}) must match text batch size ({batch_size}) or be 1"
        )

    def encode_image(self, images, training: bool = False):
        """Encode images into fused pseudo-token representations."""
        if images is None:
            raise ValueError("images must not be None when calling encode_image()")

        if isinstance(images, (list, tuple)):
            processed = self.image_processor.batch_process(images)
        elif isinstance(images, str) or not isinstance(images, torch.Tensor):
            processed = self.image_processor(images)
        else:
            processed = images

        if processed.dim() == 3:
            processed = processed.unsqueeze(0)

        device = next(self.parameters()).device
        processed = processed.to(device=device)
        features = self.vision_encoder(processed)
        features = self.patch_selector(features)
        image_tokens = self.projector(features)
        image_tokens, gate_alpha = self.gate(image_tokens)
        image_tokens = self.image_dropout(image_tokens, training=training)
        return image_tokens, gate_alpha

    def forward(self, input_ids=None, labels=None, images=None, **kwargs):
        """Run a multimodal forward pass.

        Args:
            input_ids: token ids with shape (B, T)
            labels: optional language modeling labels
            images: optional image batch or image paths

        Returns:
            dict with logits, loss, and intermediate multimodal features.
        """
        if input_ids is None:
            raise ValueError("input_ids must be provided")

        if images is None:
            logits, loss = self.text_model(input_ids, labels)
            return {
                "logits": logits,
                "loss": loss,
                "image_features": None,
                "text_features": self.text_model.transformer.wte(input_ids),
                "gate_alpha": None,
            }

        image_tokens, gate_alpha = self.encode_image(images, training=self.training)
        image_tokens = self._match_image_batch(image_tokens, input_ids.size(0))
        text_emb = self.text_model.transformer.wte(input_ids)
        text_emb, image_tokens = self.cross_modal_fusion(text_emb, image_tokens)
        x = torch.cat([image_tokens, text_emb], dim=1)

        if labels is not None:
            batch_size = labels.size(0)
            ignore_labels = torch.full(
                (batch_size, image_tokens.size(1)),
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
        batch_size, seq_len, _ = x.size()
        if seq_len > self.config.block_size:
            x = x[:, : self.config.block_size, :]
            seq_len = self.config.block_size
            if labels is not None:
                labels = labels[:, : self.config.block_size]

        x = self.text_model.transformer.drop(x)
        pos = torch.arange(0, seq_len, device=x.device)
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

    def _build_generation_embeddings(self, input_ids: torch.Tensor, images):
        image_tokens, gate_alpha = self.encode_image(images, training=False)
        image_tokens = self._match_image_batch(image_tokens, input_ids.size(0))
        text_emb = self.text_model.transformer.wte(input_ids)
        fused_text, fused_image = self.cross_modal_fusion(text_emb, image_tokens)

        full_embeddings = []
        for batch_idx in range(input_ids.size(0)):
            seq = input_ids[batch_idx]
            text_row = fused_text[batch_idx]
            image_row = fused_image[batch_idx]

            if self.image_start_id is not None and self.image_end_id is not None:
                start_pos = (seq == self.image_start_id).nonzero(as_tuple=True)[0]
                end_pos = (seq == self.image_end_id).nonzero(as_tuple=True)[0]
                if len(start_pos) > 0 and len(end_pos) > 0 and start_pos[0].item() < end_pos[0].item():
                    start_idx = start_pos[0].item()
                    end_idx = end_pos[0].item()
                    seq_emb = torch.cat([text_row[:start_idx], image_row, text_row[end_idx + 1 :]], dim=0)
                else:
                    seq_emb = torch.cat([image_row, text_row], dim=0)
            else:
                seq_emb = torch.cat([image_row, text_row], dim=0)
            full_embeddings.append(seq_emb)

        max_len = max(emb.size(0) for emb in full_embeddings)
        padded = []
        for emb in full_embeddings:
            if emb.size(0) < max_len:
                pad = torch.zeros(max_len - emb.size(0), emb.size(1), device=emb.device, dtype=emb.dtype)
                emb = torch.cat([emb, pad], dim=0)
            padded.append(emb)
        return torch.stack(padded, dim=0), gate_alpha

    def _slice_images(self, images, index, batch_size):
        if images is None:
            return None
        if torch.is_tensor(images):
            if images.dim() > 0 and images.size(0) == batch_size:
                return images[index : index + 1]
            return images
        if isinstance(images, (list, tuple)) and len(images) == batch_size:
            return images[index]
        return images

    @torch.no_grad()
    def generate(self, input_ids, images=None, max_new_tokens=None, **kwargs):
        """Generate tokens or text with optional images.

        Set return_dict=True and pass tokenizer=... to receive both token ids and decoded text.
        """
        if images is None:
            return self.text_model.generate(input_ids, max_new_tokens=max_new_tokens, **kwargs)

        tokenizer = kwargs.get("tokenizer")
        return_dict = bool(kwargs.get("return_dict", False))
        include_prompt = bool(kwargs.get("include_prompt", True))
        temperature = kwargs.get("temperature", getattr(self.config, "temperature", 0.8))
        top_k = kwargs.get("top_k", getattr(self.config, "top_k", 50))
        top_p = kwargs.get("top_p", getattr(self.config, "top_p", 0.9))
        min_p = kwargs.get("min_p", getattr(self.config, "min_p", 0.05))
        repetition_penalty = kwargs.get("repetition_penalty", getattr(self.config, "repetition_penalty", 1.1))
        eos_id = kwargs.get("eos_id", getattr(self.config, "eos_id", None))
        max_new_tokens = int(max_new_tokens or getattr(self.config, "gen_len", 200))

        token_history, prompt_lengths = self.text_model._prepare_generation_inputs(input_ids, tokenizer=tokenizer)
        if token_history.size(0) > 1:
            sample_outputs = []
            for batch_idx, prompt_length in enumerate(prompt_lengths):
                sample_outputs.append(
                    self.generate(
                        token_history[batch_idx, :prompt_length].unsqueeze(0),
                        images=self._slice_images(images, batch_idx, token_history.size(0)),
                        max_new_tokens=max_new_tokens,
                        tokenizer=tokenizer,
                        return_dict=True,
                        include_prompt=include_prompt,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        min_p=min_p,
                        repetition_penalty=repetition_penalty,
                        eos_id=eos_id,
                    )
                )
            return self.text_model._merge_generation_outputs(
                sample_outputs,
                tokenizer=tokenizer,
                return_dict=return_dict,
                include_prompt=include_prompt,
            )

        if prompt_lengths[0] != token_history.size(1):
            token_history = token_history[:, :prompt_lengths[0]]

        prompt_embeddings, _ = self._build_generation_embeddings(token_history, images)
        batch_size = token_history.size(0)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=token_history.device)
        eos_enabled = eos_id is not None

        for _ in range(max_new_tokens):
            x_input = prompt_embeddings[:, -self.config.block_size :]
            seq_len = x_input.size(1)
            pos = torch.arange(0, seq_len, device=x_input.device)
            freqs_cos = self.text_model.freqs_cos[pos]
            freqs_sin = self.text_model.freqs_sin[pos]

            hidden = self.text_model.transformer.drop(x_input)
            for block in self.text_model.transformer.h:
                hidden, _ = block(hidden, freqs_cos, freqs_sin, kv_cache=None)
            hidden = self.text_model.transformer.ln_f(hidden)
            logits = self.text_model.lm_head(hidden)[:, -1, :]
            next_token = self.text_model._sample_token(
                logits,
                token_history,
                temperature,
                top_k,
                top_p,
                min_p,
                repetition_penalty,
            )

            if eos_enabled and finished.any():
                next_token = next_token.clone()
                next_token[finished] = int(eos_id)

            token_history = torch.cat([token_history, next_token], dim=1)
            next_emb = self.text_model.transformer.wte(next_token)
            prompt_embeddings = torch.cat([prompt_embeddings, next_emb], dim=1)

            if eos_enabled:
                finished = finished | (next_token.squeeze(-1) == int(eos_id))
                if finished.all():
                    break

        return self.text_model._build_generation_output(
            token_history,
            prompt_lengths,
            tokenizer=tokenizer,
            return_dict=return_dict,
            include_prompt=include_prompt,
        )

    @torch.no_grad()
    def generate_stream(self, input_ids, images=None, max_new_tokens=None, **kwargs):
        """Stream multimodal generation.

        For batched multimodal inputs, falls back to batched generation and emits step-wise events.
        """
        if images is None:
            yield from self.text_model.generate_stream(input_ids, max_new_tokens=max_new_tokens, **kwargs)
            return

        tokenizer = kwargs.get("tokenizer")
        return_dict = bool(kwargs.get("return_dict", False))
        temperature = kwargs.get("temperature", getattr(self.config, "temperature", 0.8))
        top_k = kwargs.get("top_k", getattr(self.config, "top_k", 50))
        top_p = kwargs.get("top_p", getattr(self.config, "top_p", 0.9))
        min_p = kwargs.get("min_p", getattr(self.config, "min_p", 0.05))
        repetition_penalty = kwargs.get("repetition_penalty", getattr(self.config, "repetition_penalty", 1.1))
        eos_id = kwargs.get("eos_id", getattr(self.config, "eos_id", None))
        max_new_tokens = int(max_new_tokens or getattr(self.config, "gen_len", 200))

        token_history, prompt_lengths = self.text_model._prepare_generation_inputs(input_ids, tokenizer=tokenizer)
        if token_history.size(0) > 1:
            out = self.generate(
                token_history,
                images=images,
                max_new_tokens=max_new_tokens,
                tokenizer=tokenizer,
                return_dict=True,
                include_prompt=True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                eos_id=eos_id,
            )
            generated_rows = out["generated_token_ids"]
            max_steps = max((len(row) for row in generated_rows), default=0)
            for step in range(max_steps):
                token_ids = [row[step] if step < len(row) else None for row in generated_rows]
                if return_dict:
                    texts = []
                    if tokenizer is not None:
                        texts = ["" if token_id is None else tokenizer.decode([token_id]) for token_id in token_ids]
                    yield {
                        "step": step,
                        "token_ids": token_ids,
                        "texts": texts,
                        "finished": [token_id is None or (eos_id is not None and token_id == eos_id) for token_id in token_ids],
                    }
                else:
                    yield token_ids
            return

        prompt_embeddings, _ = self._build_generation_embeddings(token_history, images)
        finished = False
        for step in range(max_new_tokens):
            x_input = prompt_embeddings[:, -self.config.block_size :]
            seq_len = x_input.size(1)
            pos = torch.arange(0, seq_len, device=x_input.device)
            freqs_cos = self.text_model.freqs_cos[pos]
            freqs_sin = self.text_model.freqs_sin[pos]

            hidden = self.text_model.transformer.drop(x_input)
            for block in self.text_model.transformer.h:
                hidden, _ = block(hidden, freqs_cos, freqs_sin, kv_cache=None)
            hidden = self.text_model.transformer.ln_f(hidden)
            logits = self.text_model.lm_head(hidden)[:, -1, :]
            next_token = self.text_model._sample_token(
                logits,
                token_history,
                temperature,
                top_k,
                top_p,
                min_p,
                repetition_penalty,
            )
            token_id = int(next_token.item())
            token_history = torch.cat([token_history, next_token], dim=1)
            next_emb = self.text_model.transformer.wte(next_token)
            prompt_embeddings = torch.cat([prompt_embeddings, next_emb], dim=1)
            if eos_id is not None and token_id == int(eos_id):
                finished = True
            if return_dict:
                yield {
                    "step": step,
                    "token_id": token_id,
                    "text": "" if tokenizer is None else tokenizer.decode([token_id]),
                    "finished": finished,
                }
            else:
                yield token_id
            if finished:
                break
