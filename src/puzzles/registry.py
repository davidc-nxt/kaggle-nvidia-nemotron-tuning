"""Dispatch a prompt to the right per-category solver."""

from __future__ import annotations

from typing import Callable

from src.data import categorize
from src.puzzles import binary, gravity, numeral, units

Solver = Callable[[str], str]

_SOLVERS: dict[str, Solver] = {
    "numeral": numeral.solve,
    "units": units.solve,
    "gravity": gravity.solve,
    "binary": binary.solve,
}


def has_solver(category: str) -> bool:
    return category in _SOLVERS


def solve(prompt: str, category: str | None = None) -> str | None:
    cat = category or categorize(prompt)
    solver = _SOLVERS.get(cat)
    if solver is None:
        return None
    return solver(prompt)
