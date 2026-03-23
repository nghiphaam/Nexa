"""Generation utilities."""
import torch


def generate(model, input_ids, max_new_tokens=100, temperature=0.8, top_k=50,
             top_p=0.9, min_p=0.05, repetition_penalty=1.1):
    """Simple generation wrapper."""
    if isinstance(input_ids, list):
        input_ids = torch.tensor([input_ids], dtype=torch.long)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    return model.generate(
        input_ids, max_new_tokens=max_new_tokens, temperature=temperature,
        top_k=top_k, top_p=top_p, min_p=min_p, repetition_penalty=repetition_penalty,
    )
