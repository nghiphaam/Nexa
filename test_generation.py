#!/usr/bin/env python3
"""Test generation with Nexa model."""
import torch

from nexa import Config, NexaModel, load_tokenizer

# Load model and tokenizer
config = Config()
model = NexaModel(config)
tokenizer = load_tokenizer()

# Generate text
prompt = "The fundamental theorem of calculus states that"
input_ids = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long)
output = model.generate(input_ids, max_new_tokens=50)
print(tokenizer.decode(output[0].tolist()))
