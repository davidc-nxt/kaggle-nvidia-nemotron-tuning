"""Mint a synthetic puzzle dataset and write it to data/synthetic/synth_v1.parquet.

The output schema matches train.csv (id, prompt, answer) plus a `category` column.
Run with --per-category N to control size. Default produces ~30k rows.

Usage:
    python scripts/generate_synthetic.py --per-category 5000 --seed 0
    python scripts/generate_synthetic.py --verify  # also self-check via solvers
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl

from src.metric import is_correct
from src.puzzles import generators, registry

OUT = Path(__file__).resolve().parents[1] / "data" / "synthetic" / "synth_v1.parquet"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--per-category", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--verify", action="store_true", help="Re-solve each row and assert correctness")
    p.add_argument(
        "--categories",
        nargs="+",
        default=list(generators.GENERATORS.keys()),
        choices=list(generators.GENERATORS.keys()),
    )
    args = p.parse_args()

    rng = random.Random(args.seed)
    rows = []
    t0 = perf_counter()
    for cat in args.categories:
        for _ in range(args.per_category):
            rows.append(generators.generate_one(cat, rng))
    elapsed = perf_counter() - t0
    print(f"generated {len(rows)} rows in {elapsed:.1f}s ({len(rows)/elapsed:,.0f} rows/s)")

    if args.verify:
        t0 = perf_counter()
        n_bad = 0
        for row in rows:
            try:
                pred = registry.solve(row["prompt"], category=row["category"])
            except Exception as e:
                pred = None
            if pred is None or not is_correct(pred, row["answer"]):
                n_bad += 1
                if n_bad <= 5:
                    print(f"  BAD  id={row['id']} cat={row['category']} truth={row['answer']!r} pred={pred!r}")
        print(f"verified: {len(rows) - n_bad}/{len(rows)} correct  ({perf_counter()-t0:.1f}s)")

    df = pl.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(args.out)
    print(f"wrote {args.out.relative_to(args.out.parents[2])}  ({args.out.stat().st_size/1024:.1f} KB)")
    print(df.group_by("category").agg(pl.len().alias("n")).sort("category"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
