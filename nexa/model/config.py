"""Configuration dataclass for Nexa 1.6."""
from dataclasses import dataclass


@dataclass
class Config:
    # Data
    data_dir: str = "data"

    # Model core
    vocab_size: int = 50261
    eos_id: int | None = None
    pad_token_id: int | None = 0
    block_size: int = 2048
    sliding_window: int | None = 2048
    n_embd: int = 2048
    n_head: int = 16
    n_kv_head: int = 4
    n_layer: int = 24
    dropout: float = 0.1

    # Multimodal / fusion
    multimodal_hidden_size: int = 2304
    cross_modal_layers: int = 2
    cross_modal_heads: int = 8
    multimodal_dropout: float = 0.1
    num_image_tokens: int = 64
    vision_model_name: str = "google/siglip-base-patch16-224"

    # Training
    batch_size: int = 2
    grad_accum_steps: int = 16
    lr: float = 2e-4
    min_lr: float = 2e-5
    weight_decay: float = 0.1
    max_iters: int = 10000
    warmup_iters: int = 500
    grad_clip: float = 1.0
    use_grad_ckpt: bool = False
    use_fp16: bool = False
    enable_early_stopping: bool = True
    early_stopping_patience: int = 3
    collapse_threshold: float = 0.85
    collapse_patience: int = 3

    # Eval / logging
    eval_interval: int = 500
    eval_iters: int = 100
    log_interval: int = 50
    log_dir: str = "logs/nexa-1.6"
    show_progress: bool = True

    # Generation
    gen_len: int = 200
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    min_p: float = 0.05
    repetition_penalty: float = 1.1
    num_samples: int = 3
    return_text: bool = False
    enable_speculative: bool = False
    speculative_gamma: int = 4
    reasoning_loss_weight: float = 0.0
    critic_score_loss_weight: float = 0.0
    critic_rank_loss_weight: float = 1.0
    n_global_tokens: int = 16
    memory_state_scale: float = 0.0
    memory_train_dropout: float = 0.3
    speculative_disable_steps: int = 16

    # System
    device: str = "cpu"
    dtype: str = "float32"
    compile_model: bool = True
    compile_mode: str = "reduce-overhead"
    seed: int = 1337
    checkpoint_dir: str = "checkpoints"
    preset: str = "auto"

    def __post_init__(self):
        if self.n_head % self.n_kv_head != 0:
            raise ValueError(
                f"n_head ({self.n_head}) must be divisible by n_kv_head ({self.n_kv_head}) "
                f"for Grouped Query Attention. Got ratio: {self.n_head / self.n_kv_head}"
            )
        if self.n_embd <= 0 or self.n_head <= 0 or self.n_kv_head <= 0 or self.n_layer <= 0:
            raise ValueError("Model dimensions must be positive")
        if self.multimodal_hidden_size < self.n_embd:
            raise ValueError("multimodal_hidden_size must be >= n_embd")
        if self.cross_modal_layers < 0:
            raise ValueError("cross_modal_layers must be >= 0")
        if self.cross_modal_heads <= 0:
            raise ValueError("cross_modal_heads must be positive")
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")
        if self.n_embd % self.cross_modal_heads != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by cross_modal_heads ({self.cross_modal_heads})"
            )
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.grad_accum_steps <= 0:
            raise ValueError("grad_accum_steps must be positive")
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
