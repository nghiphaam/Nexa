"""Nexa model package."""

__all__ = ["Config", "NexaModel"]


def __getattr__(name):
    if name == "Config":
        from nexa.model.config import Config

        return Config
    if name == "NexaModel":
        from nexa.model.nexa_model import NexaModel

        return NexaModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
