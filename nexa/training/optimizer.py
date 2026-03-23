"""Optimizer configuration."""
import torch
from nexa.utils.device import is_cuda_device, is_rocm


def configure_optimizer(model, config, device):
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    n_decay = sum(p.numel() for p in decay_params)
    n_nodecay = sum(p.numel() for p in no_decay_params)
    print(f"Weight decay : {n_decay:,} params WITH decay, {n_nodecay:,} params WITHOUT")

    param_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    # Fused AdamW: CUDA only (not ROCm, not XLA)
    use_fused = is_cuda_device(device) and not is_rocm()
    try:
        optimizer = torch.optim.AdamW(param_groups, lr=config.lr, betas=(0.9, 0.95), fused=use_fused)
        fused_str = "Fused" if use_fused else "Standard"
    except (TypeError, RuntimeError):
        # Fallback if fused not supported
        optimizer = torch.optim.AdamW(param_groups, lr=config.lr, betas=(0.9, 0.95))
        fused_str = "Standard"

    return optimizer, fused_str


import math


def get_lr(it, config):
    """Linear warmup -> cosine decay -> floor."""
    if it < config.warmup_iters:
        return config.lr * (it + 1) / config.warmup_iters
    if it >= config.max_iters:
        return config.min_lr
    progress = (it - config.warmup_iters) / (config.max_iters - config.warmup_iters)
    return config.min_lr + 0.5 * (config.lr - config.min_lr) * (1.0 + math.cos(math.pi * progress))
