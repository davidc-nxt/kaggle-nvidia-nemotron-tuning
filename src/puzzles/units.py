"""Solver for `units` puzzles — `y = k * x`, fit k from examples, predict for query x.

Answers are rounded to 2 decimals in the training set.
"""

from __future__ import annotations

import re
import statistics

_EX_RE = re.compile(r"([0-9]*\.?[0-9]+)\s*m\s*becomes\s*([0-9]*\.?[0-9]+)", re.I)
_QUERY_RE = re.compile(r"convert the following measurement:\s*([0-9]*\.?[0-9]+)\s*m", re.I)


def solve(prompt: str) -> str:
    pairs = [(float(a), float(b)) for a, b in _EX_RE.findall(prompt)]
    if not pairs:
        raise ValueError("units solver: no example pairs found")

    # Median ratio is robust to any single noisy example. Mean works equally
    # well on the released train rows, but median costs nothing extra.
    k = statistics.median(b / a for a, b in pairs)

    q = _QUERY_RE.search(prompt)
    if not q:
        raise ValueError("units solver: no query measurement found")
    x = float(q.group(1))
    return f"{round(k * x, 2):.2f}"
