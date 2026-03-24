"""Dynamic token selection based on image complexity."""
import torch


def compute_image_complexity(image_features, batch_normalize=True):
    """
    Estimate complexity by feature variance.

    Args:
        image_features: [B, N, D]
        batch_normalize: Normalize by batch mean for stability
    """
    variance = image_features.var(dim=1).mean(dim=1)

    if batch_normalize:
        variance = variance / (variance.mean() + 1e-8)

    return variance.clamp(0, 1)


def select_num_tokens(image_complexity):
    """
    Simple image → 32 tokens
    Complex image → 64 tokens
    """
    if isinstance(image_complexity, torch.Tensor):
        return torch.where(image_complexity < 0.5, 32, 64)
    return 32 if image_complexity < 0.5 else 64
