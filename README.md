# Nexa

Nexa is a compact transformer codebase for text generation, distributed training, and multimodal image-text understanding.

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
- Vision projector, image-text gate, image dropout, patch selection in [nexa/vision/](nexa/vision/)
- Multimodal trainer, contrastive loss, curriculum loader in [nexa/training/](nexa/training/)
- Complete autoregressive generation with image inputs

## Install

Use Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch numpy tokenizers transformers pillow tensorboard
```

For TPU/XLA or ROCm, install the matching PyTorch build for your environment.

## Quick start

### Training from scratch

```bash
# Prepare dataset
python pre_train.py \
  --dataset HuggingFaceFW/fineweb-edu \
  --dataset-config sample-350BT \
  --data-dir data \
  --max-samples 500000

# Train with auto-config
python train.py \
  --data-dir data \
  --device auto \
  --preset auto \
  --max-iters 5000 \
  --checkpoint-dir checkpoints
```

## Main package layout

- [nexa/](nexa/) - Core package
  - [nexa/model/](nexa/model/) - Model architectures (NexaModel, MultimodalModel)
  - [nexa/tokenizer/](nexa/tokenizer/) - Tokenization (BPE + multimodal tokens)
  - [nexa/training/](nexa/training/) - Training loop, optimizer, data loading
  - [nexa/vision/](nexa/vision/) - Vision encoder, projector, preprocessing
  - [nexa/utils/](nexa/utils/) - Device detection, distributed training utils
- [train.py](train.py) - Main training script
- [pre_train.py](pre_train.py) - Dataset preparation

## Model Architecture

**Nexa** (1.26B parameters):
- 24 layers, 2048 hidden dim, 16 Q heads, 4 KV heads (GQA 4:1)
- SwiGLU FFN, RMSNorm, RoPE positional encoding
- Sliding window attention with KV cache
- BPE tokenizer (50,261 vocab, GPT-2 compatible)

**Training features**:
- Mixed precision (bfloat16/float16)
- Gradient accumulation + checkpointing
- Auto-config for TPU/CUDA/ROCm
- Distributed training (DDP, XLA)

## License

Apache 2.0. See [LICENSE](LICENSE).
