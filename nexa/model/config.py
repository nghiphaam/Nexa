"""Config dataclass for Nexa model."""
import math
from dataclasses import dataclass
import torch


@dataclass
class Config:
    # Data
    data_dir: str = "data"

    # Model - NEXA 1.2: Scaled to 1B parameters
    vocab_size: int = 50261
    eos_id: int = None
    block_size: int = 2048
    sliding_window: int = 2048
    n_embd: int = 2048
    n_head: int = 16
    n_kv_head: int = 4
    n_layer: int = 24
    dropout: float = 0.1

    # Training
    batch_size: int = 2
    grad_accum_steps: int = 16
    lr: float = 2e-4
    min_lr: float = 2e-5
    weight_decay: float = 0.1
    max_iters: int = 10000
    warmup_iters: int = 500
    grad_clip: float = 1.0

    # Eval / Log
    eval_interval: int = 500
    eval_iters: int = 100
    log_interval: int = 50

    # Generation
    gen_len: int = 200
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    min_p: float = 0.05
    repetition_penalty: float = 1.1
    num_samples: int = 3
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
    use_grad_ckpt: bool = False
    compile_model: bool = True
    compile_mode: str = "reduce-overhead"
    seed: int = 1337
    checkpoint_dir: str = "checkpoints"
    preset: str = "auto"

    def __post_init__(self):
        """Validate configuration after initialization."""
        # GQA validation: n_head must be divisible by n_kv_head
        if self.n_head % self.n_kv_head != 0:
            raise ValueError(
                f"n_head ({self.n_head}) must be divisible by n_kv_head ({self.n_kv_head}) "
                f"for Grouped Query Attention. Got ratio: {self.n_head / self.n_kv_head}"
            )

        # Ensure positive values
        if self.n_embd <= 0 or self.n_head <= 0 or self.n_kv_head <= 0 or self.n_layer <= 0:
            raise ValueError("Model dimensions must be positive")

        if self.block_size <= 0:
            raise ValueError("block_size must be positive")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
