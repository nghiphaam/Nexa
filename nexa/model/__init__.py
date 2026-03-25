"""Nexa model architecture."""

from nexa.model.config import Config
from nexa.model.nexa_model import NexaModel

__all__ = ["Config", "NexaModel", "MultimodalModel"]


def __getattr__(name):
    if name == "MultimodalModel":
        from nexa.model.multimodal_model import MultimodalModel
        return MultimodalModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
