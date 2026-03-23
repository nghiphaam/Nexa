#!/usr/bin/env python3
"""Test generation with Nexa model."""
from nexa import NexaModel, Config, load_tokenizer

# Load model and tokenizer
config = Config()
model = NexaModel(config)
tokenizer = load_tokenizer()

# Generate text
prompt = "The fundamental theorem of calculus states that"
input_ids = tokenizer.encode(prompt).ids
output = model.generate(input_ids, max_new_tokens=50)
print(tokenizer.decode(output[0].tolist()))
