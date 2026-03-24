# Nexa

Nexa is a compact transformer codebase for text generation, distributed training, and experimental multimodal image-text training.

## What is in this repo

### Text model
- `NexaModel` in [nexa/model/nexa_model.py](nexa/model/nexa_model.py)
- GQA attention, SwiGLU FFN, RMSNorm, RoPE, KV cache
- Sampling with top-k, top-p, min-p, repetition penalty
- Critic heads and memory-state injection hooks

### Training
- Main trainer in [nexa/training/trainer.py](nexa/training/trainer.py)
- CUDA / ROCm / TPU-aware data loading and training utilities
- DDP support for multi-GPU runs
- TPU/XLA-specific lightweight memmap sampling path

### Tokenization
- Base tokenizer exports in [nexa/tokenizer/__init__.py](nexa/tokenizer/__init__.py)
- Multimodal special-token helpers in [nexa/tokenizer/multimodal_tokenizer.py](nexa/tokenizer/multimodal_tokenizer.py)

### Multimodal image understanding (Nexa 1.5)
- `MultimodalModel` in [nexa/model/multimodal_model.py](nexa/model/multimodal_model.py)
- Frozen SigLIP vision encoder in [nexa/vision/vision_encoder.py](nexa/vision/vision_encoder.py)
- Vision projector, image-text gate, image dropout, patch selection, image preprocessing in [nexa/vision/](nexa/vision/)
- Multimodal trainer, contrastive loss, collapse detection, curriculum loader, and image usage tracking in [nexa/training/](nexa/training/)

## Current status

### Implemented
- Text-only training and generation
- Multi-GPU / TPU-aware training infrastructure
- Multimodal image-text training path
- Multimodal special-token utilities

### Not implemented yet
- Full multimodal autoregressive generation in `MultimodalModel.generate()`
  - Current behavior explicitly raises `NotImplementedError` when `images` are passed for generation.
  - Training and evaluation with `forward(..., images=...)` are implemented.

## Install

Use Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch numpy tokenizers transformers pillow tensorboard
```

For TPU/XLA or ROCm, install the matching PyTorch build for your environment.

## Quick start

### Import the package

```python
from nexa import Config, NexaModel, MultimodalModel, load_tokenizer
```

### Text model

```python
from nexa import Config, NexaModel, load_tokenizer

config = Config(device="cpu")
tokenizer = load_tokenizer()
config.vocab_size = tokenizer.get_vocab_size()
model = NexaModel(config)
```

### Multimodal model

```python
from nexa import Config, MultimodalModel
from nexa.tokenizer.multimodal_tokenizer import add_multimodal_tokens

config = Config(device="cpu")
model = MultimodalModel(config)
```

## Main package layout

- [nexa/](nexa/)
  - [nexa/model/](nexa/model/)
  - [nexa/tokenizer/](nexa/tokenizer/)
  - [nexa/training/](nexa/training/)
  - [nexa/vision/](nexa/vision/)
  - [nexa/utils/](nexa/utils/)

## Notes

- The README reflects the current package layout, not the older `lm.py` / `chat.py` structure.
- If you need production multimodal inference, that part still needs implementation on top of the current training stack.

## License

Apache 2.0. See [LICENSE](LICENSE).
