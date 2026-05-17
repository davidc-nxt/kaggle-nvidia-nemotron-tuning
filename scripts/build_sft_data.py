"""Combine original train.csv + synthetic parquet and emit an SFT-ready file.

Output schema (parquet at data/processed/sft_v1.parquet):
    id, category, source ("orig" | "synth"), prompt, answer, completion, messages
where `messages` is a list[{role, content}] usable directly with TRL's SFTTrainer.

Usage:
    python scripts/generate_synthetic.py --per-category 5000          # mint synthetic first
    python scripts/build_sft_data.py                                  # combine + format
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl

from src.data import categorize, load_train
from src.prompts import SYSTEM_PROMPT
from src.reasoning import build_completion

REPO = Path(__file__).resolve().parents[1]
SYNTH_PATH = REPO / "data" / "synthetic" / "synth_v1.parquet"
OUT_PATH = REPO / "data" / "processed" / "sft_v1.parquet"


def _to_messages(prompt: str, completion: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": completion},
    ]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--synth", type=Path, default=SYNTH_PATH)
    p.add_argument("--out", type=Path, default=OUT_PATH)
    p.add_argument("--limit", type=int, default=0, help="Cap total rows (0 = unlimited)")
    args = p.parse_args()

    orig = load_train()
    orig = orig.with_columns(pl.lit("orig").alias("source"))

    frames = [orig.select(["id", "category", "source", "prompt", "answer"])]
    if args.synth.exists():
        synth = pl.read_parquet(args.synth).with_columns(pl.lit("synth").alias("source"))
        frames.append(synth.select(["id", "category", "source", "prompt", "answer"]))
        print(f"loaded {synth.height:,} synthetic rows from {args.synth.name}")
    else:
        print(f"NOTE: {args.synth} not found — run scripts/generate_synthetic.py first for synth data")

    df = pl.concat(frames, how="vertical_relaxed")
    if args.limit:
        df = df.head(args.limit)

    print(f"combined {df.height:,} rows; building completions...")
    completions = []
    for row in df.iter_rows(named=True):
        completions.append(build_completion(row["prompt"], row["answer"], category=row["category"]))

    df = df.with_columns(pl.Series("completion", completions))
    # JSON-encode messages column (Polars list[struct] is also fine but JSON keeps it portable).
    messages = [
        json.dumps(_to_messages(row["prompt"], row["completion"]), ensure_ascii=False)
        for row in df.iter_rows(named=True)
    ]
    df = df.with_columns(pl.Series("messages", messages))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(args.out)
    print(f"wrote {args.out.relative_to(args.out.parents[2])}  ({args.out.stat().st_size/1024/1024:.2f} MB)")
    print()
    print("category breakdown:")
    print(df.group_by(["category", "source"]).agg(pl.len().alias("n")).sort(["category", "source"]))
    print()
    print("sample completion (one per category):")
    for cat in sorted(set(df["category"].to_list())):
        ex = df.filter(pl.col("category") == cat).head(1).to_dicts()[0]
        print(f"\n--- {cat} (source={ex['source']}, id={ex['id']}) ---")
        print(ex["completion"][:400] + ("..." if len(ex["completion"]) > 400 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
