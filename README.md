# Nexa

Nexa is a causal language model codebase centered on the core model, tokenizer, checkpoint format, and minimal Python inference.

The public package surface is intentionally small:

- `Config`
- `NexaModel`
- `NexaTokenizer`
- `load_tokenizer()`
- `load_checkpoint()`
- `load_model()`

## Install

Use Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quick Start

Nexa does not ship pretrained weights in this repo. Load one of your checkpoints through the Python API:

```python
from nexa import load_checkpoint

loaded = load_checkpoint("checkpoints/best.pt", device="auto")
output = loaded.generate(
    "Explain grouped-query attention simply.",
    max_new_tokens=128,
    temperature=0.8,
    top_p=0.9,
    return_dict=True,
    include_prompt=False,
)
print(output["generated_texts"][0])
```

## Repo Layout

- `nexa/model/` transformer architecture and config
- `nexa/tokenizer/` tokenizer implementation
- `nexa/runtime.py` checkpoint loading helpers

## License

Apache License. See [LICENSE](LICENSE).
