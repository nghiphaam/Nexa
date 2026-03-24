"""Track image usage via attention or heuristic score."""
import torch


class ImageUsageTracker:
    def __init__(self, num_image_tokens=64):
        self.num_image_tokens = num_image_tokens

    @torch.no_grad()
    def compute_image_usage_score(self, model, image, input_ids):
        """
        Heuristic image usage score.
        Full attention tracking requires model changes to expose attentions.
        For v1.5, use gate alpha as proxy.
        """
        model_unwrapped = model.module if hasattr(model, 'module') else model

        if hasattr(model_unwrapped, 'encode_image'):
            _, gate_alpha = model_unwrapped.encode_image(image, training=False)
            return gate_alpha.mean().item()

        return 0.0
