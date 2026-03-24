"""Nexa language and multimodal models."""
__version__ = "1.5.0"

# Lazy imports to avoid issues when torch not installed
__all__ = ["Config", "NexaModel", "MultimodalModel", "load_tokenizer"]

def __getattr__(name):
    if name == "Config":
        from nexa.model.config import Config
        return Config
    elif name == "NexaModel":
        from nexa.model.nexa_model import NexaModel
        return NexaModel
    elif name == "MultimodalModel":
        from nexa.model.multimodal_model import MultimodalModel
        return MultimodalModel
    elif name == "load_tokenizer":
        from nexa.tokenizer.tokenizer import load_tokenizer
        return load_tokenizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

