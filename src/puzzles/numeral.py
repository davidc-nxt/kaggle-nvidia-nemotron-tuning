"""Solver for `numeral` puzzles — Roman numerals 1-99 in every observed sample."""

from __future__ import annotations

import re

_QUERY_RE = re.compile(r"write the number\s+(-?\d+)\s+in the Wonderland numeral system", re.I)

_ROMAN_PAIRS: tuple[tuple[int, str], ...] = (
    (1000, "M"), (900, "CM"),
    (500, "D"),  (400, "CD"),
    (100, "C"),  (90, "XC"),
    (50, "L"),   (40, "XL"),
    (10, "X"),   (9, "IX"),
    (5, "V"),    (4, "IV"),
    (1, "I"),
)


def to_roman(n: int) -> str:
    if n <= 0:
        raise ValueError(f"Roman numerals don't represent {n}")
    out = []
    for value, sym in _ROMAN_PAIRS:
        while n >= value:
            out.append(sym)
            n -= value
    return "".join(out)


def solve(prompt: str) -> str:
    m = _QUERY_RE.search(prompt)
    if not m:
        raise ValueError("numeral solver: could not locate query number in prompt")
    return to_roman(int(m.group(1)))
