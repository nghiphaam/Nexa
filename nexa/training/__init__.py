"""Nexa training utilities."""

from nexa.training.contrastive_loss import ContrastiveLoss
from nexa.training.collapse_detector import CollapseDetector, CollapseEarlyStopping
from nexa.training.curriculum_loader import CurriculumDataLoader
from nexa.training.image_usage_tracker import ImageUsageTracker

__all__ = [
    "ContrastiveLoss",
    "CollapseDetector",
    "CollapseEarlyStopping",
    "CurriculumDataLoader",
    "ImageUsageTracker",
    "MultimodalTrainer",
]


def __getattr__(name):
    if name == "MultimodalTrainer":
        from nexa.training.multimodal_trainer import MultimodalTrainer
        return MultimodalTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
