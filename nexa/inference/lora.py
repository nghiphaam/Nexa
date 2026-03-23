"""LoRA (Low-Rank Adaptation) for efficient fine-tuning."""
import torch
import torch.nn as nn
import math


class LoRALinear(nn.Module):
    def __init__(self, original_linear, rank=8, alpha=16, dropout=0.0):
        super().__init__()
        self.original = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # Freeze original weights
        self.original.weight.requires_grad_(False)
        if self.original.bias is not None:
            self.original.bias.requires_grad_(False)

    def forward(self, x):
        base_out = self.original(x)
        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        return base_out + lora_out


def apply_lora(model, rank=8, alpha=16, dropout=0.0, target_modules=None):
    """Apply LoRA to model's linear layers."""
    if target_modules is None:
        target_modules = ["wq", "wk", "wv", "c_proj", "w1", "w2", "w3"]

    count = 0
    for name, module in model.named_modules():
        for attr_name in target_modules:
            if hasattr(module, attr_name):
                original = getattr(module, attr_name)
                if isinstance(original, nn.Linear):
                    lora_layer = LoRALinear(original, rank=rank, alpha=alpha, dropout=dropout)
                    setattr(module, attr_name, lora_layer)
                    count += 1

    # FIX #7: Ensure LoRA params are float32 for stability
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.data = param.data.float()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA applied to {count} layers (rank={rank}, alpha={alpha})")
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
    return model


def merge_lora(model):
    """Merge LoRA weights back into base model."""
    count = 0
    for name, module in model.named_modules():
        for attr_name in list(dir(module)):
            try:
                sub = getattr(module, attr_name)
            except Exception:
                continue
            if isinstance(sub, LoRALinear):
                with torch.no_grad():
                    merge_weight = sub.lora_B @ sub.lora_A * sub.scaling
                    sub.original.weight.add_(merge_weight.to(sub.original.weight.dtype))
                setattr(module, attr_name, sub.original)
                count += 1

    print(f"LoRA merged: {count} layers")
    return model
