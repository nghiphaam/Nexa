# Nexa 🌌

![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)

Nexa is an advanced, lightweight, open-source language model designed for high efficiency, flexible experimentation, and modern generative AI architectures. Grounded in a highly optimized codebase, Nexa brings modern architectural advancements alongside a full-stack reasoning engine.

## 🌟 Key Features

### 🏗 Architecture
- **Modern Transformer Backbone:** Utilizes Grouped-Query Attention (GQA), SwiGLU Feed-Forward Networks (FFN), RMSNorm, and Rotary Position Embeddings (RoPE).
- **Inference Optimizations:** Features an efficient KV Cache with precise temporal slot tracking, memory-state injections, and Speculative Decoding for accelerated local inference.

### 🧠 Reasoning & Agent Engine
- **Internal Reasoning Engine:** An autonomous reasoning loop built directly into the chat runtime that orchestrates a **Planner**, a **Critic model**, and multi-path thought exploration to robustly verify and improve its own answers.
- **Agent Tool Calling:** Embedded Python Sandbox mechanism allows Nexa to robustly execute `CALL: python(...)` queries, interact with tools, and cleanly route execution outputs directly back into the context window.
- **Long-term Vector Memory:** Intercepts, retrieves, and maintains cross-turn historical context intelligently seamlessly injecting hidden conversational details into the main token stream.

### 💿 Pre-training Pipeline
- **Smart Data Preparation:** `pre_train.py` manages end-to-end extraction from Hugging Face datasets.
- **Dynamic Streaming:** Uses system RAM heuristics to decide whether to map data fully in-memory or cleanly stream huge corpora directly to `train.bin` and `val.bin`.

---

## 📂 Project Structure

- `lm.py`: The core LLM framework. Houses model architecture (Transformer blocks, Memory state projection, Critic adapters), the deep learning training loop, and the generation CLI.
- `chat.py`: The full-fledge runtime environment. Controls conversational formatting, the reasoning engine, speculative tool calls, history sliding-window, and both CLI and Web (Gradio) user interfaces.
- `pre_train.py`: Standalone dataset downloader and binary mapping utility. Builds system-aware chunks and `dataset_manifest.json`.

---

## 🚀 Installation

Ensure you are using Python 3.10+ and set up your virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch numpy datasets huggingface_hub tiktoken gradio
```

*Note: If you use TPU/XLA or ROCm, ensure you install the mutually compatible `torch` build for your specific environment.*

---

## 🛠️ Usage Pipeline

### 1. Data Preparation

Nexa does not assume a default dataset. Use the standalone `pre_train.py` script to fetch, map, and serialize text corpora into optimized BPE tokenized binary files.

```bash
python pre_train.py --dataset HuggingFaceFW/fineweb-edu --data-dir data
```

**Advanced options:**
```bash
python pre_train.py \
  --dataset ptb_text_only \
  --dataset-config default \
  --data-dir data \
  --max-samples 1000000 \
  --force-stream
```

### 2. Pre-Training / Fine-tuning

Once `train.bin` and `val.bin` are securely written:

```bash
python lm.py --data-dir data
```

You can target specific pre-defined network volumes using `--preset`:
```bash
python lm.py --data-dir data --preset low   # Best for debugging / tiny scale
python lm.py --data-dir data --preset mid
python lm.py --data-dir data --preset high
```

### 3. Generation (Raw CLI)

Validate model generation capacity independently from the chat subsystem:

```bash
python lm.py --data-dir data --generate --prompt "The fundamental theorem of calculus states that"
```

### 4. Interactive Chat (Agent / Reasoning)

The chat runtime supports local persistent history, reasoning paths, and vector memory.
Point it to your `.pt` checkpoint.

**Terminal CLI Mode:**
```bash
python chat.py --checkpoint checkpoints/best.pt --cli
```

**Web UI Mode (Gradio):**
```bash
python chat.py --checkpoint checkpoints/best.pt
```

---

## 📝 Design Philosophy & Notes

- **Modularity:** Tokenization strictly leverages optimized BPE vocabulary format to standardize cross-platform testing without muddying the model logic.
- **Decoupled Processing:** `lm.py` is entirely separated from dataset formatting logic, strictly consuming MemMap arrays. This prevents memory leaks during training and standardizes benchmark ingestion.
- **Robust formatting:** Linting rules and architecture patterns strictly adhere to `ruff` and `black` Python integrations ensuring maximum readability.

## 📄 License

This repository is distributed under the **Apache License 2.0**. See the `LICENSE` file for more details.
