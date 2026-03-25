"""Multimodal tokenizer utilities."""
from __future__ import annotations

import torch

IMAGE_START_TOKEN = "<image>"
IMAGE_END_TOKEN = "</image>"


def add_multimodal_tokens(tokenizer, model=None):
    """Add multimodal tokens and optionally resize model embeddings."""
    special_tokens = [IMAGE_START_TOKEN, IMAGE_END_TOKEN]

    if hasattr(tokenizer, "special_map"):
        next_id = max(tokenizer.special_map.values()) + 1
        for token in special_tokens:
            if token not in tokenizer.special_map:
                tokenizer.special_map[token] = next_id
                tokenizer.id_to_special[next_id] = token
                next_id += 1
        if hasattr(tokenizer, "_vocab_size"):
            tokenizer._vocab_size = max(tokenizer._vocab_size, next_id)

    if model is not None and hasattr(model, "text_model"):
        emb = model.text_model.transformer.wte
        lm_head = model.text_model.lm_head
        current_vocab = emb.weight.size(0)
        new_vocab = getattr(tokenizer, "_vocab_size", current_vocab)

        if new_vocab > current_vocab:
            old_weight = emb.weight.data
            new_emb = torch.nn.Embedding(new_vocab, old_weight.size(1)).to(old_weight.device)
            new_emb.weight.data[:current_vocab] = old_weight
            new_emb.weight.data[current_vocab:].normal_(mean=0.0, std=0.02)
            model.text_model.transformer.wte = new_emb

            new_head = torch.nn.Linear(old_weight.size(1), new_vocab, bias=False).to(old_weight.device)
            new_head.weight.data[:current_vocab] = lm_head.weight.data
            new_head.weight.data[current_vocab:].normal_(mean=0.0, std=0.02)
            model.text_model.lm_head = new_head
            model.text_model.transformer.wte.weight = model.text_model.lm_head.weight
            model.text_model.config.vocab_size = new_vocab

    image_start_id = tokenizer.special_map[IMAGE_START_TOKEN]
    image_end_id = tokenizer.special_map[IMAGE_END_TOKEN]
    return tokenizer, model, image_start_id, image_end_id
