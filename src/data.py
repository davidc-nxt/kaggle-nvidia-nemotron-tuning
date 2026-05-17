from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"

# Each train prompt opens with "In Alice's Wonderland, ..." and one of these
# templates identifies the puzzle category. Stable across the released train set.
CATEGORY_PATTERNS: dict[str, re.Pattern[str]] = {
    "binary":   re.compile(r"secret bit manipulation rule"),
    "gravity":  re.compile(r"gravitational constant has been secretly changed"),
    "units":    re.compile(r"secret unit conversion"),
    "cipher":   re.compile(r"secret encryption rules"),
    "numeral":  re.compile(r"converted into a different numeral system|numeral system"),
    "equation": re.compile(r"transformation rules is applied to equation"),
}


def categorize(prompt: str) -> str:
    for name, pat in CATEGORY_PATTERNS.items():
        if pat.search(prompt):
            return name
    return "unknown"


def load_train(path: str | Path | None = None) -> pl.DataFrame:
    p = Path(path) if path else RAW_DIR / "train.csv"
    df = pl.read_csv(p)
    return df.with_columns(
        pl.col("prompt").map_elements(categorize, return_dtype=pl.Utf8).alias("category")
    )


def load_test(path: str | Path | None = None) -> pl.DataFrame:
    p = Path(path) if path else RAW_DIR / "test.csv"
    return pl.read_csv(p)


@dataclass(frozen=True)
class Example:
    id: str
    prompt: str
    answer: str | None
    category: str


def iter_examples(df: pl.DataFrame) -> Iterable[Example]:
    has_answer = "answer" in df.columns
    has_cat = "category" in df.columns
    for row in df.iter_rows(named=True):
        yield Example(
            id=row["id"],
            prompt=row["prompt"],
            answer=row["answer"] if has_answer else None,
            category=row["category"] if has_cat else categorize(row["prompt"]),
        )
