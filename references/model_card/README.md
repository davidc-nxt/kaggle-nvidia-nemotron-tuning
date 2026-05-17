# Nemotron-3-Nano-30B-A3B-BF16 — Local Notes

This directory holds metadata about the base model. The actual weights (63 GB
uncompressed) are **not** stored locally — pull them inside your Kaggle training
notebook via `kagglehub.model_download(...)`.

## Quick facts

| Field    | Value                                                                    |
| -------- | ------------------------------------------------------------------------ |
| Slug     | `metric/nemotron-3-nano-30b-a3b-bf16/transformers/default`               |
| URL      | https://www.kaggle.com/models/metric/nemotron-3-nano-30b-a3b-bf16        |
| Author   | Kaggle Competition Metrics                                               |
| Version  | 1 (id 784907, published 2026-03-13)                                      |
| Size     | 63,174,970,906 bytes (~63 GB uncompressed)                               |
| License  | Other (see model description)                                            |
| Loadable | `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`      |
| dtype    | `torch.bfloat16` (BF16 weights)                                          |

## How to load (inside a Kaggle notebook)

```python
import kagglehub, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

path = kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")
model = AutoModelForCausalLM.from_pretrained(
    path,
    device_map="auto",
    trust_remote_code=True,
    dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
```

## Architectural notes

- "A3B" → Mixture-of-Experts with ~3B active parameters per token.
- Imports `mamba_ssm` → hybrid Mamba state-space + transformer layers.
- LoRA target modules in the official demo: `.*\.(in_proj|out_proj|up_proj|down_proj)$`
  - `in_proj`, `out_proj` → Mamba blocks
  - `up_proj`, `down_proj` → MLP blocks
- Custom modeling code lives in the model repo; `trust_remote_code=True` is required.

## Memory ballpark for LoRA training (rank 32)

| Setup                                | Approx. VRAM |
| ------------------------------------ | -----------: |
| BF16 model load                      | ~60 GB       |
| + activations / gradients / optimizer | +20–40 GB   |
| → recommended: single H100 80GB, or 2× L4 / 4× T4 with quantisation + sharding |

On Kaggle, the L4 x4 instance (Pro tier) or TPU VM are the realistic options for fine-tuning at native precision. T4 x2 will likely require 4-bit base + LoRA (QLoRA) and small batch size.
