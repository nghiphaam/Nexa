"""Shared runtime helpers for Nexa."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any

from nexa.model.config import Config

CHECKPOINT_FORMAT_VERSION = 2
CHECKPOINT_FORMAT_NAME = "nexa-lm-v2"


@dataclass(slots=True)
class LoadedCheckpoint:
    model: Any
    config: Config
    tokenizer: Any
    checkpoint: dict[str, Any]
    meta: dict[str, Any]
    checkpoint_path: str

    def generate(self, inputs: Any, **kwargs: Any):
        kwargs.setdefault("tokenizer", self.tokenizer)
        return self.model.generate(inputs, **kwargs)


_CONFIG_FIELD_NAMES = {field.name for field in fields(Config)}


def _filter_config_kwargs(raw_values: dict[str, Any]) -> dict[str, Any]:
    return {name: value for name, value in raw_values.items() if name in _CONFIG_FIELD_NAMES}


def normalize_config(raw_config: Any) -> Config:
    if raw_config is None:
        return Config()
    if isinstance(raw_config, Config):
        return raw_config
    if isinstance(raw_config, dict):
        return Config(**_filter_config_kwargs(raw_config))
    if is_dataclass(raw_config):
        return Config(**_filter_config_kwargs(asdict(raw_config)))
    config_kwargs = {name: getattr(raw_config, name) for name in _CONFIG_FIELD_NAMES if hasattr(raw_config, name)}
    return Config(**config_kwargs)


def resolve_runtime_device(requested_device: str) -> str:
    from nexa.utils.device import auto_select_device, get_xla_device, safe_cuda_alloc, safe_xla_alloc

    if requested_device == "auto":
        return auto_select_device(prefer_cuda=True)
    if requested_device == "cuda":
        if not safe_cuda_alloc(0):
            raise RuntimeError("CUDA requested but not available")
        return "cuda:0"
    if requested_device in ("xla", "tpu"):
        if not safe_xla_alloc():
            raise RuntimeError("XLA/TPU requested but not available")
        get_xla_device()
        return "xla"
    return requested_device


def infer_model_id(checkpoint_path: str) -> str:
    return Path(checkpoint_path).stem or "nexa"


def build_checkpoint_meta(config: Config, **extra: Any) -> dict[str, Any]:
    meta = {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "checkpoint_format_name": CHECKPOINT_FORMAT_NAME,
        "public_surface": "lm-core",
        "model_family": "nexa",
        "model_type": "causal-lm",
        "tokenizer": "nexa",
        "vocab_size": config.vocab_size,
        "dtype": config.dtype,
        "block_size": config.block_size,
    }
    meta.update(extra)
    return meta


def load_checkpoint(
    checkpoint_path: str,
    device: str = "auto",
    *,
    eval_mode: bool = True,
) -> LoadedCheckpoint:
    import torch

    from nexa.model.nexa_model import NexaModel
    from nexa.tokenizer.tokenizer import EOS_TOKEN, PAD_TOKEN, load_tokenizer
    from nexa.utils.device import is_xla_device

    resolved_device = resolve_runtime_device(device)
    map_location = resolved_device if not is_xla_device(resolved_device) else "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)

    config = normalize_config(checkpoint.get("config"))
    config.device = resolved_device

    tokenizer = load_tokenizer()
    config.vocab_size = tokenizer.get_vocab_size()
    config.eos_id = tokenizer.token_to_id(EOS_TOKEN)
    config.pad_token_id = tokenizer.token_to_id(PAD_TOKEN)

    raw_state_dict = checkpoint["model"]
    legacy_prefixes = ("critic_", "memory_")
    filtered_state_dict = {
        name: value for name, value in raw_state_dict.items() if not name.startswith(legacy_prefixes)
    }

    model = NexaModel(config)
    missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
    dropped_legacy_keys = len(raw_state_dict) - len(filtered_state_dict)
    if missing or unexpected or dropped_legacy_keys:
        print(
            f"[warn] checkpoint loaded with missing={len(missing)} "
            f"unexpected={len(unexpected)} dropped_legacy={dropped_legacy_keys}"
        )
    model = model.to(resolved_device)
    if eval_mode:
        model.eval()

    meta = dict(checkpoint.get("meta") or {})
    meta.setdefault("checkpoint_format_version", 1)
    meta.setdefault("checkpoint_format_name", "legacy")
    meta.setdefault("public_surface", "legacy")
    meta.setdefault("model_family", "nexa")
    meta.setdefault("model_type", "causal-lm")

    return LoadedCheckpoint(
        model=model,
        config=config,
        tokenizer=tokenizer,
        checkpoint=checkpoint,
        meta=meta,
        checkpoint_path=checkpoint_path,
    )


def load_model(
    checkpoint_path: str,
    device: str = "auto",
    *,
    eval_mode: bool = True,
) -> tuple[Any, Any, Config]:
    loaded = load_checkpoint(checkpoint_path, device=device, eval_mode=eval_mode)
    return loaded.model, loaded.tokenizer, loaded.config
