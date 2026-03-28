"""Configuration dataclass for the Nexa causal LM core."""
from dataclasses import dataclass


@dataclass(slots=True)
class Config:
    vocab_size: int = 50261
    eos_id: int | None = None
    pad_token_id: int | None = 50257
    block_size: int = 2048
    sliding_window: int | None = 2048
    n_embd: int = 2048
    n_head: int = 16
    n_kv_head: int = 4
    n_layer: int = 24
    dropout: float = 0.1

    gen_len: int = 200
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    min_p: float = 0.05
    repetition_penalty: float = 1.1
    enable_speculative: bool = False
    speculative_gamma: int = 4
    n_global_tokens: int = 16

    device: str = "cpu"
    dtype: str = "float32"

    def __post_init__(self):
        if self.n_head % self.n_kv_head != 0:
            raise ValueError(
                f"n_head ({self.n_head}) must be divisible by n_kv_head ({self.n_kv_head}) "
                f"for Grouped Query Attention. Got ratio: {self.n_head / self.n_kv_head}"
            )
        if self.n_embd <= 0 or self.n_head <= 0 or self.n_kv_head <= 0 or self.n_layer <= 0:
            raise ValueError("Model dimensions must be positive")
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.sliding_window is not None and self.sliding_window <= 0:
            raise ValueError("sliding_window must be positive when set")
        if self.gen_len <= 0:
            raise ValueError("gen_len must be positive")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in the interval (0, 1]")
        if not 0.0 <= self.min_p <= 1.0:
            raise ValueError("min_p must be in the interval [0, 1]")
        if self.repetition_penalty <= 0.0:
            raise ValueError("repetition_penalty must be > 0")
        if self.speculative_gamma <= 0:
            raise ValueError("speculative_gamma must be positive")
        if self.n_global_tokens < 0:
            raise ValueError("n_global_tokens must be >= 0")
