"""Nexa 1.6 language and multimodal models."""

__version__ = "1.6.0"
VERSION_NAME = "Nexa 1.6"

__all__ = ["Config", "NexaModel", "MultimodalModel", "load_tokenizer", "__version__", "VERSION_NAME"]


def __getattr__(name):
    if name == "Config":
        from nexa.model.config import Config
        return Config
    if name == "NexaModel":
        from nexa.model.nexa_model import NexaModel
        return NexaModel
    if name == "MultimodalModel":
        from nexa.model.multimodal_model import MultimodalModel
        return MultimodalModel
    if name == "load_tokenizer":
        from nexa.tokenizer.tokenizer import load_tokenizer
        return load_tokenizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
