"""Nexa tokenizer based on tiktoken BPE."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

DEFAULT_VOCAB_SIZE = 50261

EOS_TOKEN = "<|endoftext|>"
PAD_TOKEN = "<|pad|>"
SYS_TOKEN = "<|system|>"
USR_TOKEN = "<|user|>"
AST_TOKEN = "<|assistant|>"


@dataclass
class EncodingResult:
    ids: list[int]


class NexaTokenizer:
    """Wrap tiktoken BPE with a minimal HuggingFace-like interface."""

    def __init__(self):
        import tiktoken

        self.enc = tiktoken.get_encoding("gpt2")
        self._base_vocab_size = self.enc.n_vocab
        self._vocab_size = DEFAULT_VOCAB_SIZE
        self._eos_id = self.enc.eot_token
        self.supports_special_role_tokens = True

        self.special_map = {
            EOS_TOKEN: 50256,
            PAD_TOKEN: 50257,
            SYS_TOKEN: 50258,
            USR_TOKEN: 50259,
            AST_TOKEN: 50260,
        }
        self.id_to_special = {v: k for k, v in self.special_map.items()}
        pattern = "|".join(re.escape(k) for k in self.special_map.keys())
        self.special_pattern = re.compile(f"({pattern})")

    @property
    def eos_id(self) -> int:
        return self._eos_id

    @property
    def pad_id(self) -> int:
        return self.special_map[PAD_TOKEN]

    def get_vocab_size(self) -> int:
        return self._vocab_size

    def token_to_id(self, token: str) -> int:
        if token in self.special_map:
            return self.special_map[token]
        raise ValueError(f"Unknown token: {token}")

    def _normalize_text(self, text) -> str:
        if text is None:
            return ""
        if not isinstance(text, str):
            text = str(text)
        return text.replace("\r\n", "\n")

    def encode(self, text) -> EncodingResult:
        normalized = self._normalize_text(text)
        if not normalized:
            return EncodingResult(ids=[])
        parts = self.special_pattern.split(normalized)
        ids: list[int] = []
        for part in parts:
            if not part:
                continue
            if part in self.special_map:
                ids.append(self.special_map[part])
            else:
                ids.extend(self.enc.encode_ordinary(part))
        return EncodingResult(ids=ids)

    def encode_batch(self, texts: Iterable[str]) -> list[EncodingResult]:
        return [self.encode(text) for text in texts]

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        if ids is None:
            return ""
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        result: list[str] = []
        buf: list[int] = []
        for token_id in ids:
            token_id = int(token_id)
            if token_id in self.id_to_special:
                if buf:
                    result.append(self.enc.decode(buf))
                    buf = []
                if not skip_special_tokens:
                    result.append(self.id_to_special[token_id])
            elif 0 <= token_id < self._base_vocab_size:
                buf.append(token_id)
        if buf:
            result.append(self.enc.decode(buf))
        return "".join(result)

    def decode_batch(self, batch_ids, skip_special_tokens: bool = True) -> list[str]:
        if batch_ids is None:
            return []
        if hasattr(batch_ids, "tolist"):
            batch_ids = batch_ids.tolist()
        return [self.decode(ids, skip_special_tokens=skip_special_tokens) for ids in batch_ids]


_cached_tokenizer: dict[str, NexaTokenizer] = {}


def load_tokenizer() -> NexaTokenizer:
    global _cached_tokenizer
    if "nexa" not in _cached_tokenizer:
        print("Tokenizer: Nexa BPE (vocab=50,261)")
        _cached_tokenizer["nexa"] = NexaTokenizer()
    return _cached_tokenizer["nexa"]


def get_vocab_size() -> int:
    return DEFAULT_VOCAB_SIZE
