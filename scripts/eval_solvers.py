"""Run all available solvers on train.csv and print per-category accuracy.

Usage:
    python scripts/eval_solvers.py [--sample N] [--show-failures K]
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl

from src.data import load_train
from src.metric import is_correct
from src.puzzles import registry


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=0, help="Evaluate on a random sample of this size (0 = full train)")
    p.add_argument("--show-failures", type=int, default=3, help="Print up to this many failures per category")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    df = load_train()
    if args.sample > 0:
        df = df.sample(args.sample, seed=args.seed)

    correct: dict[str, int] = defaultdict(int)
    total: dict[str, int] = defaultdict(int)
    skipped: dict[str, int] = defaultdict(int)
    errors: dict[str, list[tuple[str, str, str | None, str]]] = defaultdict(list)

    for row in df.iter_rows(named=True):
        cat = row["category"]
        total[cat] += 1
        if not registry.has_solver(cat):
            skipped[cat] += 1
            continue
        try:
            pred = registry.solve(row["prompt"], category=cat)
        except Exception as e:  # solver bug or unusual prompt shape
            errors[cat].append((row["id"], row["answer"], None, f"EXC: {e}"))
            continue
        if pred is None:
            skipped[cat] += 1
            continue
        if is_correct(pred, row["answer"]):
            correct[cat] += 1
        else:
            errors[cat].append((row["id"], row["answer"], pred, row["prompt"]))

    rows = []
    overall_solved = 0
    overall_total = 0
    for cat in sorted(total.keys()):
        n = total[cat]
        c = correct[cat]
        s = skipped[cat]
        e = len(errors[cat])
        acc = c / (n - s) if (n - s) else float("nan")
        rows.append({"category": cat, "n": n, "covered": n - s, "correct": c, "errors": e, "accuracy": round(acc, 4)})
        if n - s:
            overall_solved += c
            overall_total += (n - s)

    print(pl.DataFrame(rows).sort("category"))
    if overall_total:
        print(f"\nOverall coverage accuracy: {overall_solved}/{overall_total} = {overall_solved/overall_total:.4f}")

    if args.show_failures > 0:
        for cat in sorted(errors.keys()):
            if not errors[cat]:
                continue
            print(f"\n--- Failures in {cat} (showing up to {args.show_failures} of {len(errors[cat])}) ---")
            for rid, truth, pred, prompt_or_msg in errors[cat][: args.show_failures]:
                tail = prompt_or_msg.strip().split("\n")[-1] if pred is not None else prompt_or_msg
                print(f"  id={rid}  truth={truth!r}  pred={pred!r}   {tail}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
