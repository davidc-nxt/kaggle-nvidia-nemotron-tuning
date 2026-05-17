"""Local re-implementation of the NVIDIA Nemotron Metric for offline scoring.

Mirrors the rules described on the competition Evaluation page:
- Prefer the content inside the LAST `\\boxed{...}` in the generated text.
- Fall back to the last number found.
- Correct if the extracted answer matches the ground truth either as an exact
  string OR within a relative numeric tolerance of 1e-2.
"""

from __future__ import annotations

import re
from typing import Sequence

_BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")

REL_TOL = 1e-2


def extract_boxed(text: str) -> str | None:
    matches = _BOXED_RE.findall(text)
    return matches[-1].strip() if matches else None


def extract_last_number(text: str) -> str | None:
    nums = _NUM_RE.findall(text)
    return nums[-1] if nums else None


def extract_answer(text: str) -> str:
    """Pull the final answer the metric would score, following the documented heuristic."""
    boxed = extract_boxed(text)
    if boxed is not None:
        return boxed
    n = extract_last_number(text)
    return n if n is not None else text.strip()


def _try_float(s: str) -> float | None:
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def is_correct(prediction: str, target: str) -> bool:
    pred = extract_answer(prediction).strip()
    tgt = target.strip()

    if pred == tgt:
        return True

    p_num = _try_float(pred)
    t_num = _try_float(tgt)
    if p_num is not None and t_num is not None:
        if t_num == 0.0:
            return abs(p_num) <= REL_TOL
        return abs(p_num - t_num) / abs(t_num) <= REL_TOL

    return False


def score(predictions: Sequence[str], targets: Sequence[str]) -> float:
    assert len(predictions) == len(targets), "prediction / target length mismatch"
    if not predictions:
        return 0.0
    return sum(is_correct(p, t) for p, t in zip(predictions, targets)) / len(predictions)
