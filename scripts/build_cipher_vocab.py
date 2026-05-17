"""Build the closed-class cipher vocabulary from the training set.

Output: data/processed/cipher_vocab.json — an OrderedDict-ish mapping
{word -> training-frequency}, sorted by frequency descending.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl

from src.data import load_train

OUT = Path(__file__).resolve().parents[1] / "data" / "processed" / "cipher_vocab.json"


def main() -> int:
    train = load_train()
    cipher = train.filter(pl.col("category") == "cipher")

    words: Counter[str] = Counter()
    for prompt, answer in zip(cipher["prompt"].to_list(), cipher["answer"].to_list()):
        # Plain texts appear AFTER '->' in each example line, plus the answer.
        for line in prompt.splitlines():
            if "->" in line:
                _, p = line.split("->", 1)
                for w in p.strip().split():
                    if w.isalpha():
                        words[w.lower()] += 1
        for w in answer.split():
            if w.isalpha():
                words[w.lower()] += 1

    # Sort by frequency descending, write as dict (preserves insertion order in JSON).
    OUT.parent.mkdir(parents=True, exist_ok=True)
    ordered = dict(words.most_common())
    OUT.write_text(json.dumps(ordered, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(OUT.parents[2])} — {len(ordered)} words")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
