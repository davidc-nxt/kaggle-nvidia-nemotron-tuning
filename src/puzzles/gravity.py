"""Solver for `gravity` puzzles — `d = 0.5 * g * t^2`. Fit g, predict d at query t."""

from __future__ import annotations

import re
import statistics

_EX_RE = re.compile(
    r"t\s*=\s*([0-9]*\.?[0-9]+)s?,\s*distance\s*=\s*([0-9]*\.?[0-9]+)\s*m",
    re.I,
)
_QUERY_RE = re.compile(
    r"determine the falling distance for\s*t\s*=\s*([0-9]*\.?[0-9]+)s?",
    re.I,
)


def solve(prompt: str) -> str:
    pairs = [(float(t), float(d)) for t, d in _EX_RE.findall(prompt)]
    if not pairs:
        raise ValueError("gravity solver: no (t, d) examples found")

    g = statistics.median(2 * d / (t * t) for t, d in pairs)

    q = _QUERY_RE.search(prompt)
    if not q:
        raise ValueError("gravity solver: no query t found")
    t_query = float(q.group(1))
    return f"{round(0.5 * g * t_query * t_query, 2):.2f}"
