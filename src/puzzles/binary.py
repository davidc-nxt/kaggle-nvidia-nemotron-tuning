"""Solver for `binary` puzzles.

Each puzzle hides an 8-bit -> 8-bit transformation built from the operations the
prompt enumerates (shifts, rotations, XOR, AND, OR, NOT, majority, choice). We
brute-force search a structured library of candidate rules, keep those that
match all eight given (input, output) examples, and predict the query.

If multiple candidates fit and they disagree on the query, we go with the
simplest description (shortest name) — that prior happens to be reliable on
the released training set.
"""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np

_EX_RE = re.compile(r"^([01]{8})\s*->\s*([01]{8})$", re.M)
_QUERY_RE = re.compile(r"determine the output for:\s*([01]{8})")

ALL = np.arange(256, dtype=np.uint8)


def _rotl(v: np.ndarray, k: int) -> np.ndarray:
    k %= 8
    if k == 0:
        return v.copy()
    return ((v << k) | (v >> (8 - k))) & 0xFF


def _rotr(v: np.ndarray, k: int) -> np.ndarray:
    k %= 8
    if k == 0:
        return v.copy()
    return ((v >> k) | (v << (8 - k))) & 0xFF


def _shl(v: np.ndarray, k: int) -> np.ndarray:
    return (v << k) & 0xFF


def _shr(v: np.ndarray, k: int) -> np.ndarray:
    return (v >> k) & 0xFF


def _nott(v: np.ndarray) -> np.ndarray:
    return (~v) & 0xFF


def _maj(a, b, c): return (a & b) | (a & c) | (b & c)
def _ch(a, b, c):  return (a & b) | (~a & 0xFF & c)


def _base_ops() -> list[tuple[str, np.ndarray]]:
    """Return labelled depth-1 transforms applied to every byte 0..255."""
    out: list[tuple[str, np.ndarray]] = []
    for k in range(0, 8):
        for op_name, op in (("ROTL", _rotl), ("ROTR", _rotr), ("SHL", _shl), ("SHR", _shr)):
            if k == 0 and op_name in ("ROTL", "ROTR", "SHL", "SHR"):
                # k=0 is identity for all four; only emit once
                if op_name != "ROTL":
                    continue
                name = "x"
            else:
                name = f"{op_name}({k})"
            out.append((name, op(ALL, k)))
    return out


@lru_cache(maxsize=1)
def _candidates() -> tuple[list[str], np.ndarray]:
    """Build the candidate library as (names, table) where table.shape == (n_candidates, 256)."""
    names: list[str] = []
    tables: list[np.ndarray] = []

    base = _base_ops()

    # Template 1: single transform, optionally NOTted.
    for n, t in base:
        names.append(n);              tables.append(t)
        names.append(f"NOT {n}");     tables.append(_nott(t))

    # Template 2: binary combinations of two transforms (XOR / AND / OR), optionally NOTted.
    for i in range(len(base)):
        ni, ti = base[i]
        for j in range(i, len(base)):
            nj, tj = base[j]
            for op_name, op in (("XOR", lambda a, b: a ^ b),
                                ("AND", lambda a, b: a & b),
                                ("OR",  lambda a, b: a | b)):
                if i == j and op_name != "XOR":
                    continue  # x op x is degenerate for AND/OR
                if i == j and op_name == "XOR":
                    continue  # x XOR x = 0
                combo = op(ti, tj) & 0xFF
                names.append(f"{ni} {op_name} {nj}"); tables.append(combo)
                names.append(f"NOT ({ni} {op_name} {nj})"); tables.append(_nott(combo))

    # Template 3: three-way XOR (sigma-style).
    for i in range(len(base)):
        ni, ti = base[i]
        for j in range(i + 1, len(base)):
            nj, tj = base[j]
            for k in range(j + 1, len(base)):
                nk, tk = base[k]
                tables.append((ti ^ tj ^ tk) & 0xFF)
                names.append(f"{ni} XOR {nj} XOR {nk}")

    # Template 4: Maj / Ch over three transforms.
    n_base = len(base)
    for i in range(n_base):
        ni, ti = base[i]
        for j in range(n_base):
            if j == i: continue
            nj, tj = base[j]
            for k in range(n_base):
                if k == i or k == j: continue
                nk, tk = base[k]
                tables.append(_maj(ti, tj, tk) & 0xFF)
                names.append(f"Maj({ni}, {nj}, {nk})")
                tables.append(_ch(ti, tj, tk) & 0xFF)
                names.append(f"Ch({ni}, {nj}, {nk})")

    table = np.stack(tables, axis=0).astype(np.uint8)
    # Dedupe identical functions while preserving the simplest (shortest-name) representative.
    # Order by name length ascending so np.unique keeps the first occurrence.
    order = np.argsort([len(n) for n in names], kind="stable")
    names = [names[i] for i in order]
    table = table[order]
    keys = table.tobytes()
    # Build a dict keyed by row bytes -> first index
    seen: dict[bytes, int] = {}
    keep: list[int] = []
    for idx in range(table.shape[0]):
        row_key = table[idx].tobytes()
        if row_key not in seen:
            seen[row_key] = idx
            keep.append(idx)
    table = table[keep]
    names = [names[i] for i in keep]
    return names, table


def solve(prompt: str) -> str:
    pairs = _EX_RE.findall(prompt)
    if not pairs:
        raise ValueError("binary solver: no example pairs found")
    q = _QUERY_RE.search(prompt)
    if not q:
        raise ValueError("binary solver: no query found")

    names, table = _candidates()
    xs = np.array([int(a, 2) for a, _ in pairs], dtype=np.uint8)
    ys = np.array([int(b, 2) for _, b in pairs], dtype=np.uint8)

    matches = (table[:, xs] == ys[None, :]).all(axis=1)
    if not matches.any():
        # Fall through: best the model can do is reason about it (we fail loudly).
        raise ValueError("binary solver: no candidate in library fits the examples")

    qv = int(q.group(1), 2)
    preds = table[matches, qv]  # all predictions from fitting rules
    # If unanimous, take it. Otherwise take majority (prior: simplest name first via earlier sort).
    if (preds == preds[0]).all():
        return format(int(preds[0]), "08b")
    vals, counts = np.unique(preds, return_counts=True)
    return format(int(vals[counts.argmax()]), "08b")
