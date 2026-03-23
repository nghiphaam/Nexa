"""Nexa - A 1B parameter language model."""
__version__ = "1.3.0"

# Lazy imports to avoid issues when torch not installed
__all__ = ["Config", "NexaModel", "load_tokenizer"]

def __getattr__(name):
    if name == "Config":
        from nexa.model.config import Config
        return Config
    elif name == "NexaModel":
        from nexa.model.nexa_model import NexaModel
        return NexaModel
    elif name == "load_tokenizer":
        from nexa.tokenizer.tokenizer import load_tokenizer
        return load_tokenizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
