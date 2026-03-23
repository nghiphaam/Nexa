"""Nexa tokenizer based on tiktoken BPE."""
import re

DEFAULT_VOCAB_SIZE = 50261

EOS_TOKEN = "<|endoftext|>"
PAD_TOKEN = "<|pad|>"
SYS_TOKEN = "<|system|>"
USR_TOKEN = "<|user|>"
AST_TOKEN = "<|assistant|>"


class NexaTokenizer:
    """Wraps tiktoken BPE to match HuggingFace tokenizers interface.
    Expanded vocab to 50,261 to avoid collisions with GPT-2 merges.
    """

    def __init__(self):
        import tiktoken

        self.enc = tiktoken.get_encoding("gpt2")
        self._base_vocab_size = self.enc.n_vocab  # 50257
        self._vocab_size = 50261
        self._eos_id = self.enc.eot_token  # 50256
        self.supports_special_role_tokens = True

        self.special_map = {
            EOS_TOKEN: 50256,
            PAD_TOKEN: 50257,
            SYS_TOKEN: 50258,
            USR_TOKEN: 50259,
            AST_TOKEN: 50260,
        }
        self.id_to_special = {v: k for k, v in self.special_map.items()}

        # Regex for splitting by special tokens
        pattern = "|".join(re.escape(k) for k in self.special_map.keys())
        self.special_pattern = re.compile(f"({pattern})")

    def get_vocab_size(self):
        return self._vocab_size

    def token_to_id(self, token):
        if token in self.special_map:
            return self.special_map[token]
        raise ValueError(f"Unknown token: {token}")

    def encode(self, text):
        parts = self.special_pattern.split(text)
        ids = []
        for p in parts:
            if p in self.special_map:
                ids.append(self.special_map[p])
            elif p:
                ids.extend(self.enc.encode_ordinary(p))
        return type("E", (), {"ids": ids})()

    def encode_batch(self, texts):
        results = []
        for t in texts:
            if t and t.strip():
                results.append(self.encode(t))
            else:
                results.append(type("E", (), {"ids": []})())
        return results

    def decode(self, ids, skip_special_tokens=True):
        result = []
        buf = []
        for i in ids:
            if i in self.id_to_special:
                if buf:
                    result.append(self.enc.decode(buf))
                    buf = []
                if not skip_special_tokens:
                    result.append(self.id_to_special[i])
            elif i < self._base_vocab_size:
                buf.append(i)
        if buf:
            result.append(self.enc.decode(buf))
        return "".join(result)


_cached_tokenizer = {}


def load_tokenizer():
    """Load tokenizer — Nexa default."""
    global _cached_tokenizer
    if "nexa" not in _cached_tokenizer:
        print("Tokenizer: Nexa BPE (vocab=50,261)")
        _cached_tokenizer["nexa"] = NexaTokenizer()
    return _cached_tokenizer["nexa"]


def get_vocab_size():
    return 50261
