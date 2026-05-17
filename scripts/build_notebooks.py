"""One-shot script that builds the starter notebooks.

Run from repo root:
    python scripts/build_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NB_DIR = REPO / "notebooks"
NB_DIR.mkdir(exist_ok=True)


def md(*lines: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _src(lines)}


def code(*lines: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _src(lines),
    }


def _src(lines: tuple[str, ...]) -> list[str]:
    text = "\n".join(lines)
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


def nb(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.13"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# -----------------------------------------------------------------------------
# 01_eda.ipynb
# -----------------------------------------------------------------------------
eda = nb([
    md(
        "# 01 · EDA — Alice's Wonderland reasoning puzzles",
        "",
        "Confirm the six puzzle categories, look at length distributions, sample a few",
        "puzzles per category to see what the model is expected to do.",
    ),
    code(
        "import sys, pathlib",
        "ROOT = pathlib.Path.cwd().parent if pathlib.Path.cwd().name == 'notebooks' else pathlib.Path.cwd()",
        "sys.path.insert(0, str(ROOT))",
        "",
        "import polars as pl",
        "from src.data import load_train, load_test",
        "",
        "train = load_train()",
        "test = load_test()",
        "print('train shape:', train.shape)",
        "print('test shape :', test.shape)",
        "train.head(3)",
    ),
    md("## Category distribution"),
    code(
        "train.group_by('category').agg(pl.len().alias('count')).sort('count', descending=True)",
    ),
    md("## Prompt and answer length distributions"),
    code(
        "stats = train.with_columns([",
        "    pl.col('prompt').str.len_chars().alias('prompt_chars'),",
        "    pl.col('answer').str.len_chars().alias('answer_chars'),",
        "])",
        "",
        "stats.group_by('category').agg([",
        "    pl.col('prompt_chars').mean().round(0).alias('prompt_mean'),",
        "    pl.col('prompt_chars').max().alias('prompt_max'),",
        "    pl.col('answer_chars').mean().round(1).alias('answer_mean'),",
        "    pl.col('answer_chars').min().alias('answer_min'),",
        "    pl.col('answer_chars').max().alias('answer_max'),",
        "]).sort('category')",
    ),
    md("## One sample per category"),
    code(
        "for cat in sorted(train['category'].unique()):",
        "    row = train.filter(pl.col('category') == cat).head(1).to_dicts()[0]",
        "    print('=' * 80)",
        "    print(f'CATEGORY: {cat}   id={row[\"id\"]}')",
        "    print('=' * 80)",
        "    print(row['prompt'])",
        "    print()",
        "    print(f'ANSWER: {row[\"answer\"]}')",
        "    print()",
    ),
    md("## Quick sanity check on the local metric"),
    code(
        "from src.metric import is_correct, extract_answer, score",
        "",
        "examples = [",
        "    ('The result is therefore \\\\boxed{42}.', '42', True),",
        "    ('I think the answer might be 41.999, so \\\\boxed{42.005}', '42', True),  # within 1e-2 rel tol",
        "    ('\\\\boxed{wrong}', 'right', False),",
        "    ('No box, just an answer of 7', '7', True),  # fallback to last number",
        "]",
        "for text, target, expected in examples:",
        "    got = is_correct(text, target)",
        "    print(f'extract={extract_answer(text)!r:>15}  target={target!r:>10}  ok={got}  expected={expected}')",
    ),
])


# -----------------------------------------------------------------------------
# 02_train_kaggle.ipynb — adapted from the official submission demo
# -----------------------------------------------------------------------------
train_nb = nb([
    md(
        "# 02 · Train LoRA on Nemotron-3-Nano-30B (Kaggle GPU)",
        "",
        "Upload this notebook to Kaggle, attach the competition dataset and the",
        "`metric/nemotron-3-nano-30b-a3b-bf16` model, then run top to bottom.",
        "",
        "Output: `/kaggle/working/submission.zip` containing the LoRA adapter.",
    ),
    md(
        "## 0. Environment",
        "",
        "Kaggle's notebook images already include PyTorch + Transformers. We install",
        "PEFT, accelerate, TRL, and bits/pieces the demo needs.",
    ),
    code(
        "%pip install -q -U peft accelerate trl datasets",
    ),
    md(
        "## 1. Load the prebuilt SFT dataset",
        "",
        "Expects a Kaggle Dataset attached to the notebook containing `sft_v1.parquet`",
        "(produced locally via `scripts/build_sft_data.py`). Adjust `SFT_PATH` to the",
        "exact mount point Kaggle gives you.",
    ),
    code(
        "import json",
        "import polars as pl",
        "from datasets import Dataset",
        "",
        "SFT_PATH = '/kaggle/input/wonderland-sft-v1/sft_v1.parquet'  # update if needed",
        "",
        "df = pl.read_parquet(SFT_PATH)",
        "print('total rows:', df.height)",
        "print(df.group_by(['category', 'source']).agg(pl.len().alias('n')).sort(['category', 'source']))",
    ),
    md("## 2. Build the HuggingFace Dataset (messages format)"),
    code(
        "rows = [{'messages': json.loads(m)} for m in df['messages'].to_list()]",
        "ds = Dataset.from_list(rows)",
        "print(ds)",
        "print('example messages:')",
        "for m in ds[0]['messages']:",
        "    print(f\"[{m['role']}]\", m['content'][:200])",
    ),
    md("## 3. Load Nemotron-3-Nano-30B + attach LoRA"),
    code(
        "import site",
        "# The competition's reference notebook includes a CUTLASS DSL helper required by the model.",
        "# Adjust the path if you copy this notebook outside the official Kaggle competition environment.",
        "cutlass_pkg_path = '/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script/nvidia_cutlass_dsl/python_packages/'",
        "site.addsitedir(cutlass_pkg_path)",
        "",
        "import kagglehub, torch",
        "from peft import LoraConfig, get_peft_model, TaskType",
        "from transformers import AutoModelForCausalLM, AutoTokenizer",
        "",
        "MODEL_PATH = kagglehub.model_download('metric/nemotron-3-nano-30b-a3b-bf16/transformers/default')",
        "OUTPUT_DIR = '/kaggle/working'",
        "LORA_RANK = 32  # grader enforces max_lora_rank=32",
        "",
        "tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)",
        "model = AutoModelForCausalLM.from_pretrained(",
        "    MODEL_PATH,",
        "    device_map='auto',",
        "    trust_remote_code=True,",
        "    dtype=torch.bfloat16,",
        ")",
        "",
        "lora_config = LoraConfig(",
        "    r=LORA_RANK,",
        "    lora_alpha=16,",
        "    target_modules=r'.*\\\\.(in_proj|out_proj|up_proj|down_proj)$',",
        "    lora_dropout=0.05,",
        "    bias='none',",
        "    task_type=TaskType.CAUSAL_LM,",
        ")",
        "model = get_peft_model(model, lora_config)",
        "model.print_trainable_parameters()",
    ),
    md(
        "## 4. SFT training",
        "",
        "Single-epoch SFT with TRL. Tune batch size / grad-accum / LR to fit your GPU.",
        "On L4 x4 the values below are a reasonable starting point; on T4 x2 drop",
        "`per_device_train_batch_size` to 1 and add 4-bit base loading.",
    ),
    code(
        "from trl import SFTTrainer, SFTConfig",
        "",
        "cfg = SFTConfig(",
        "    output_dir=OUTPUT_DIR,",
        "    per_device_train_batch_size=1,",
        "    gradient_accumulation_steps=16,",
        "    num_train_epochs=1,",
        "    learning_rate=2e-4,",
        "    bf16=True,",
        "    logging_steps=25,",
        "    save_strategy='no',",
        "    max_seq_length=2048,",
        "    packing=False,",
        ")",
        "",
        "trainer = SFTTrainer(",
        "    model=model,",
        "    args=cfg,",
        "    train_dataset=ds,",
        "    tokenizer=tokenizer,",
        ")",
        "trainer.train()",
    ),
    md("## 5. Save adapter and package submission.zip"),
    code(
        "model.save_pretrained(OUTPUT_DIR)",
        "import os, subprocess",
        "os.chdir(OUTPUT_DIR)",
        "subprocess.run('zip -m submission.zip adapter_config.json adapter_model.safetensors', shell=True, check=True)",
        "print('Wrote /kaggle/working/submission.zip')",
    ),
])


def write_nb(name: str, data: dict) -> None:
    out = NB_DIR / name
    out.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {out.relative_to(REPO)}  ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    write_nb("01_eda.ipynb", eda)
    write_nb("02_train_kaggle.ipynb", train_nb)
