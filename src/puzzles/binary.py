"""Solver for `binary` puzzles.

Each puzzle hides an 8-bit -> 8-bit transformation built from the operations the
prompt enumerates (shifts, rotations, XOR, AND, OR, NOT, majority, choice). We
brute-force search a structured library of candidate rules, keep those that
match all eight given (input, output) examples, and predict the query.

If multiple candidates fit and they disagree on the query, we use a simple
prior: prefer the simplest description (shortest name), unless a strong
majority of the fitting candidates clusters on a different prediction. The
training set is biased toward "simple" rules, which makes shortest-name a good
first guess; the majority guard catches cases where the shortest fitting rule
is an artefact of the library (e.g. a depth-1 op that happens to match the
examples by accident in a richer truth).
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
    n_base = len(base)

    # Template 1: single transform, optionally NOTted.
    for n, t in base:
        names.append(n);              tables.append(t)
        names.append(f"NOT {n}");     tables.append(_nott(t))

    # Template 2: binary combinations of two transforms (XOR / AND / OR), optionally NOTted.
    for i in range(n_base):
        ni, ti = base[i]
        for j in range(i, n_base):
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
    for i in range(n_base):
        ni, ti = base[i]
        for j in range(i + 1, n_base):
            nj, tj = base[j]
            for k in range(j + 1, n_base):
                nk, tk = base[k]
                tables.append((ti ^ tj ^ tk) & 0xFF)
                names.append(f"{ni} XOR {nj} XOR {nk}")

    # Template 3b: three-way AND / OR.
    for i in range(n_base):
        ni, ti = base[i]
        for j in range(i + 1, n_base):
            nj, tj = base[j]
            for k in range(j + 1, n_base):
                nk, tk = base[k]
                tables.append((ti & tj & tk) & 0xFF)
                names.append(f"{ni} AND {nj} AND {nk}")
                tables.append((ti | tj | tk) & 0xFF)
                names.append(f"{ni} OR {nj} OR {nk}")

    # Template 3c: mixed depth-2 boolean (a OP1 b) OP2 c — captures rules where
    # a different operator binds the third input.
    for i in range(n_base):
        ni, ti = base[i]
        for j in range(n_base):
            if j == i: continue
            nj, tj = base[j]
            for k in range(n_base):
                if k == i or k == j: continue
                nk, tk = base[k]
                tables.append(((ti ^ tj) & tk) & 0xFF)
                names.append(f"({ni} XOR {nj}) AND {nk}")
                tables.append(((ti ^ tj) | tk) & 0xFF)
                names.append(f"({ni} XOR {nj}) OR {nk}")
                tables.append(((ti & tj) ^ tk) & 0xFF)
                names.append(f"({ni} AND {nj}) XOR {nk}")
                tables.append(((ti | tj) ^ tk) & 0xFF)
                names.append(f"({ni} OR {nj}) XOR {nk}")
                tables.append(((ti & tj) | tk) & 0xFF)
                names.append(f"({ni} AND {nj}) OR {nk}")
                tables.append(((ti | tj) & tk) & 0xFF)
                names.append(f"({ni} OR {nj}) AND {nk}")

    # Template 4: Maj / Ch over three transforms, with NOT variants and one
    # NOT'd argument.
    for i in range(n_base):
        ni, ti = base[i]
        for j in range(n_base):
            if j == i: continue
            nj, tj = base[j]
            for k in range(n_base):
                if k == i or k == j: continue
                nk, tk = base[k]
                m = _maj(ti, tj, tk) & 0xFF
                c = _ch(ti, tj, tk) & 0xFF
                tables.append(m);           names.append(f"Maj({ni}, {nj}, {nk})")
                tables.append(c);           names.append(f"Ch({ni}, {nj}, {nk})")
                tables.append(_nott(m));    names.append(f"NOT Maj({ni}, {nj}, {nk})")
                tables.append(_nott(c));    names.append(f"NOT Ch({ni}, {nj}, {nk})")
                nti = _nott(ti)
                ntj = _nott(tj)
                ntk = _nott(tk)
                tables.append(_maj(nti, tj, tk) & 0xFF)
                names.append(f"Maj(NOT {ni}, {nj}, {nk})")
                tables.append(_ch(nti, tj, tk) & 0xFF)
                names.append(f"Ch(NOT {ni}, {nj}, {nk})")
                tables.append(_ch(ti, ntj, tk) & 0xFF)
                names.append(f"Ch({ni}, NOT {nj}, {nk})")
                tables.append(_ch(ti, tj, ntk) & 0xFF)
                names.append(f"Ch({ni}, {nj}, NOT {nk})")

    # Template 5: constant byte (c for c in 0..255). Cheap and useful both
    # as a direct match and as a XOR mask in the pair-XOR fallback below.
    for c in range(256):
        tables.append(np.full(256, c, dtype=np.uint8))
        names.append(f"CONST({c})")

    # Template 6: depth-3 (a OP b) OP c where each of a, b, c can independently
    # be a base transform or its NOT. This is the single biggest expansion of
    # the library — it captures rules like "(NOT ROTL(3) XOR SHL(2)) OR x" that
    # the simpler templates above can't reach. We dedupe inners up front to keep
    # the candidate count manageable (~94k unique depth-3 functions after
    # dedup against the simpler templates).
    full: list[tuple[str, np.ndarray]] = list(base) + [
        (f"NOT {n}", _nott(t)) for n, t in base
    ]
    inner_uniq: dict[bytes, tuple[str, str, str, np.ndarray]] = {}
    for i in range(len(full)):
        for j in range(i + 1, len(full)):
            for opn, op in (
                ("XOR", np.bitwise_xor),
                ("AND", np.bitwise_and),
                ("OR", np.bitwise_or),
            ):
                t = op(full[i][1], full[j][1]) & 0xFF
                key = t.tobytes()
                if key not in inner_uniq:
                    inner_uniq[key] = (full[i][0], opn, full[j][0], t)

    for ni, opn_inner, nj, ti in inner_uniq.values():
        for nk, tk in full:
            for opn_outer, op_outer in (
                ("XOR", np.bitwise_xor),
                ("AND", np.bitwise_and),
                ("OR", np.bitwise_or),
            ):
                t = op_outer(ti, tk) & 0xFF
                names.append(f"({ni} {opn_inner} {nj}) {opn_outer} {nk}")
                tables.append(t)

    table = np.stack(tables, axis=0).astype(np.uint8)
    # Dedupe identical functions while preserving the simplest (shortest-name) representative.
    # Order by name length ascending so the first occurrence is the simplest.
    order = np.argsort([len(n) for n in names], kind="stable")
    names = [names[i] for i in order]
    table = table[order]
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


def _pair_xor_predict(table: np.ndarray, xs: np.ndarray, ys: np.ndarray, qv: int) -> int | None:
    """Runtime fallback: if no single library function fits, look for a pair
    ``(f_i, f_j)`` in the library such that ``f_i(x) XOR f_j(x) == y`` on every
    example. The XOR-of-two-functions family includes everything reachable as
    ``f(x) XOR const`` (since constants are in the library), and any rule that
    decomposes as the XOR of two transforms we already enumerate.

    Implementation trick: build a hash from row bytes ``f(xs).tobytes()`` to a
    candidate index, then for each candidate ``i`` look up the bytes of the
    residual ``f_i(xs) XOR ys`` to find ``j``. This is O(n) not O(n^2).
    """
    cand_at_xs = table[:, xs]
    by_bytes: dict[bytes, int] = {}
    for i in range(cand_at_xs.shape[0]):
        k = cand_at_xs[i].tobytes()
        # Keep the simplest (first-encountered, names sorted by length in _candidates).
        if k not in by_bytes:
            by_bytes[k] = i

    preds: list[int] = []
    ys_bytes_arr = ys
    for i in range(cand_at_xs.shape[0]):
        residual = (cand_at_xs[i] ^ ys_bytes_arr).tobytes()
        j = by_bytes.get(residual)
        if j is None or j == i:
            continue
        preds.append(int(table[i, qv] ^ table[j, qv]))

    if not preds:
        return None
    # Majority-vote across all valid pairs. Predictions tend to cluster on the
    # right answer because the bulk of fitting pairs are paraphrases of the
    # same underlying rule.
    vals, counts = np.unique(np.asarray(preds, dtype=np.uint8), return_counts=True)
    return int(vals[counts.argmax()])


def _affine_gf2_predict(xs: np.ndarray, ys: np.ndarray, qv: int) -> int | None:
    """Per-output-bit affine fit over GF(2): solve for each output bit
    ``y_i = XOR_j a_{ij} x_j  XOR  b_i`` independently. If every bit's system is
    consistent, return the byte predicted at ``qv``; otherwise None.

    Acts as a last-ditch fallback when neither the static library nor the
    pair-XOR search produces a hit. Catches rules that decompose as a bit
    permutation + per-bit NOT plus a fixed XOR mask but happen not to be in the
    enumerated templates.
    """
    n = xs.shape[0]
    # Build per-example feature matrix: 8 input bits + bias column.
    X = np.zeros((n, 9), dtype=np.uint8)
    for k in range(n):
        v = int(xs[k])
        for j in range(8):
            X[k, j] = (v >> j) & 1
        X[k, 8] = 1
    q_vec = np.zeros(9, dtype=np.uint8)
    for j in range(8):
        q_vec[j] = (qv >> j) & 1
    q_vec[8] = 1

    pred = 0
    for out_bit in range(8):
        t = np.array([(int(y) >> out_bit) & 1 for y in ys], dtype=np.uint8)
        # Augmented matrix [X | t].
        M = np.concatenate([X, t[:, None]], axis=1)
        rows = M.shape[0]
        row = 0
        pivot_cols: list[int] = []
        for c in range(9):
            pr = -1
            for r in range(row, rows):
                if M[r, c]:
                    pr = r
                    break
            if pr < 0:
                continue
            if pr != row:
                M[[row, pr]] = M[[pr, row]]
            for r in range(rows):
                if r != row and M[r, c]:
                    M[r] ^= M[row]
            pivot_cols.append(c)
            row += 1
        # Consistency: any zero-row with non-zero rhs => no solution.
        for r in range(row, rows):
            if M[r, 9]:
                return None
        # Particular solution: free variables set to 0.
        w = np.zeros(9, dtype=np.uint8)
        for pr_idx, pc in enumerate(pivot_cols):
            w[pc] = M[pr_idx, 9]
        bit = int(np.bitwise_xor.reduce(w & q_vec) & 1)
        pred |= bit << out_bit
    return pred


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
    qv = int(q.group(1), 2)

    matches = (table[:, xs] == ys[None, :]).all(axis=1)
    if matches.any():
        idx_list = np.where(matches)[0]
        preds = table[idx_list, qv]
        if (preds == preds[0]).all():
            return format(int(preds[0]), "08b")
        # Mixed predictions: prefer a strong majority (>= 70% of fits) if there
        # are at least three fits; otherwise default to the simplest rule
        # (first index = shortest name, by construction in `_candidates`).
        vals, counts = np.unique(preds, return_counts=True)
        max_count = int(counts.max())
        n_total = int(preds.shape[0])
        if max_count * 10 >= n_total * 7 and n_total >= 3:
            return format(int(vals[counts.argmax()]), "08b")
        return format(int(preds[0]), "08b")

    # Runtime fallback 1: XOR of two library functions.
    pair = _pair_xor_predict(table, xs, ys, qv)
    if pair is not None:
        return format(pair, "08b")

    # Runtime fallback 2: per-output-bit affine GF(2) fit.
    aff = _affine_gf2_predict(xs, ys, qv)
    if aff is not None:
        return format(aff, "08b")

    # Truly nothing fits: surface a loud error so the caller can route to the
    # model-only path.
    raise ValueError("binary solver: no candidate in library fits the examples")
