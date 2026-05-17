# Competition Notes — NVIDIA Nemotron Model Reasoning Challenge

**URL:** https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge
**Host:** NVIDIA · **Prize pool:** $106,388 · **Status as of 2026-05-17:** 3,117 teams, top score **0.87**

## Task

Improve reasoning on a novel "Alice's Wonderland" benchmark by training a **LoRA adapter** for NVIDIA's open Nemotron-3-Nano-30B-A3B-BF16 model. Submission is a `submission.zip` containing the adapter.

Six puzzle families (~1,580 train rows each, 9,500 total):

| Category   | First line opener                                                       |
| ---------- | ----------------------------------------------------------------------- |
| `binary`   | secret bit manipulation rule transforms 8-bit binary numbers            |
| `gravity`  | gravitational constant has been secretly changed                        |
| `units`    | secret unit conversion is applied to measurements                       |
| `cipher`   | secret encryption rules are used on text                                |
| `numeral`  | numbers are secretly converted into a different numeral system          |
| `equation` | secret set of transformation rules is applied to equations              |

Each prompt provides ~8 input→output examples and asks for the output of one new input.

## Base model

- **Slug:** `metric/nemotron-3-nano-30b-a3b-bf16/transformers/default`
- **Size:** 30B total params, ~3B active (MoE, "A3B"), 63 GB uncompressed
- **Architecture:** Hybrid Mamba/Transformer (`mamba_ssm` is imported in the demo)
- **License:** Other (specified in model description)
- Load with `trust_remote_code=True`, `dtype=torch.bfloat16`

## Submission

You submit a LoRA adapter packaged as `submission.zip`. The grader loads the base model + your adapter into **vLLM** and runs inference with:

| Param                    | Value |
| ------------------------ | ----- |
| `max_lora_rank`          | 32    |
| `max_tokens`             | 7680  |
| `top_p`                  | 1.0   |
| `temperature`            | 0.0   |
| `max_num_seqs`           | 64    |
| `gpu_memory_utilization` | 0.85  |
| `max_model_len`          | 8192  |

The grader extracts the final answer from the last `\boxed{...}` in the generation; falls back to the last number. Correct = exact string match OR relative numeric tolerance ≤ 1e-2.

**Demo LoRA config** (from Ryan Holbrook's submission demo):

```python
LoraConfig(
    r=32, lora_alpha=16,
    target_modules=r".*\.(in_proj|out_proj|up_proj|down_proj)$",
    lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
)
```

`in_proj`/`out_proj` reach the Mamba blocks; `up_proj`/`down_proj` reach the MLPs.

## Rules to remember

- Adapter rank ≤ 32 enforced by the grader.
- `temperature=0.0` and `top_p=1.0` — no sampling diversity, deterministic decoding.
- Max generation length 7,680 tokens — plenty of room for chain-of-thought reasoning.
- **Public notebook + write-up required for prize eligibility.**
- Daily submission cap: 5 · Max team: 5.

## Key dates (UTC, 23:59)

| Date       | Event                                |
| ---------- | ------------------------------------ |
| 2026-03-16 | Start                                |
| 2026-04-09 | Midpoint cut-off                     |
| 2026-06-08 | Entry & team-merger deadline         |
| **2026-06-15** | **Final submission deadline**    |

## Encouraged frameworks

NVIDIA-provided recipes are optional. Free to use Hugging Face TRL, Unsloth, Axolotl, or similar. Only constraint: the submission zip must contain a working LoRA adapter (with `adapter_config.json`) for the published base model.

## Likely high-leverage directions

1. **Synthetic data generation.** All six puzzle families have known-generator structure — you can synthesise arbitrarily many train examples whose ground-truth answers are computable in code. This is almost certainly how strong teams scale.
2. **Curated CoT traces.** Distil reasoning traces from a stronger model that explicitly walk through the rule discovery, ending in `\boxed{...}`.
3. **Per-category fine-tuning vs single adapter.** Single adapter is the constraint, but you can mix the distribution heavily toward weak categories during training.
4. **Format adherence.** A non-trivial chunk of error mass on most LLMs is the model failing to emit a valid boxed answer. Train with high boxed-format compliance.
