"""Nexa language model package."""

__version__ = "2.0.0"
VERSION_NAME = "Nexa"

__all__ = [
    "Config",
    "NexaModel",
    "NexaTokenizer",
    "load_tokenizer",
    "LoadedCheckpoint",
    "load_checkpoint",
    "load_model",
    "normalize_config",
    "__version__",
    "VERSION_NAME",
]


def __getattr__(name):
    if name == "Config":
        from nexa.model.config import Config
        return Config
    if name == "NexaModel":
        from nexa.model.nexa_model import NexaModel
        return NexaModel
    if name == "NexaTokenizer":
        from nexa.tokenizer.tokenizer import NexaTokenizer
        return NexaTokenizer
    if name == "load_tokenizer":
        from nexa.tokenizer.tokenizer import load_tokenizer
        return load_tokenizer
    if name == "LoadedCheckpoint":
        from nexa.runtime import LoadedCheckpoint
        return LoadedCheckpoint
    if name == "load_checkpoint":
        from nexa.runtime import load_checkpoint
        return load_checkpoint
    if name == "load_model":
        from nexa.runtime import load_model
        return load_model
    if name == "normalize_config":
        from nexa.runtime import normalize_config
        return normalize_config
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
