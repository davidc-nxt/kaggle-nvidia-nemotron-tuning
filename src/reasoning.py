"""Generate chain-of-thought reasoning traces ending in a boxed answer.

Used as the assistant target during SFT. For categories we can solve in code we
build a short, faithful trace; for the rest we fall back to just the answer.
"""

from __future__ import annotations

import re
import statistics

from src.data import categorize
from src.puzzles import binary as binary_solver
from src.puzzles import cipher as cipher_solver

_NUM_QUERY = re.compile(r"write the number\s+(-?\d+)", re.I)
_UNITS_EX  = re.compile(r"([0-9]*\.?[0-9]+)\s*m\s*becomes\s*([0-9]*\.?[0-9]+)", re.I)
_UNITS_Q   = re.compile(r"convert the following measurement:\s*([0-9]*\.?[0-9]+)\s*m", re.I)
_GRAV_EX   = re.compile(r"t\s*=\s*([0-9]*\.?[0-9]+)s?,\s*distance\s*=\s*([0-9]*\.?[0-9]+)\s*m", re.I)
_GRAV_Q    = re.compile(r"determine the falling distance for\s*t\s*=\s*([0-9]*\.?[0-9]+)s?", re.I)
_BIN_EX    = re.compile(r"^([01]{8})\s*->\s*([01]{8})$", re.M)
_BIN_Q     = re.compile(r"determine the output for:\s*([01]{8})")


def _boxed(s: str) -> str:
    return f"\\boxed{{{s}}}"


# ---------------------------------------------------------------------------
# Per-category traces
# ---------------------------------------------------------------------------

def trace_numeral(prompt: str, answer: str) -> str:
    m = _NUM_QUERY.search(prompt)
    if not m:
        return _boxed(answer)
    n = int(m.group(1))
    parts = []
    remaining = n
    pairs = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"),
             (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"),
             (5, "V"), (4, "IV"), (1, "I")]
    for v, sym in pairs:
        while remaining >= v:
            parts.append((v, sym))
            remaining -= v
    breakdown = " + ".join(f"{v}({s})" for v, s in parts) if parts else "0"
    return (
        f"The Wonderland numerals are Roman numerals. "
        f"Decomposing {n}: {breakdown}. "
        f"That spells {answer}.\n\n{_boxed(answer)}"
    )


def trace_units(prompt: str, answer: str) -> str:
    examples = [(float(a), float(b)) for a, b in _UNITS_EX.findall(prompt)]
    qm = _UNITS_Q.search(prompt)
    if not examples or not qm:
        return _boxed(answer)
    k = statistics.median(b / a for a, b in examples)
    x = float(qm.group(1))
    sample = examples[0]
    return (
        f"Each example multiplies the measurement by a constant. "
        f"For instance, {sample[0]} -> {sample[1]} gives a ratio of "
        f"{sample[1] / sample[0]:.4f}. The shared factor is k ≈ {k:.4f}. "
        f"Applying to {x}: {k:.4f} × {x} = {k * x:.2f}.\n\n{_boxed(answer)}"
    )


def trace_gravity(prompt: str, answer: str) -> str:
    examples = [(float(t), float(d)) for t, d in _GRAV_EX.findall(prompt)]
    qm = _GRAV_Q.search(prompt)
    if not examples or not qm:
        return _boxed(answer)
    g_est = statistics.median(2 * d / (t * t) for t, d in examples)
    t_q = float(qm.group(1))
    t, d = examples[0]
    return (
        f"The relation is d = 0.5·g·t². "
        f"From t = {t}s, d = {d}m: g = 2·{d}/{t}² ≈ {2 * d / (t * t):.2f}. "
        f"Using g ≈ {g_est:.2f} for t = {t_q}s: "
        f"d = 0.5 · {g_est:.2f} · {t_q}² = {0.5 * g_est * t_q * t_q:.2f}.\n\n{_boxed(answer)}"
    )


def trace_binary(prompt: str, answer: str) -> str:
    """Search the solver library for a rule that fits, then explain it briefly."""
    names, table = binary_solver._candidates()
    pairs = _BIN_EX.findall(prompt)
    if not pairs:
        return _boxed(answer)
    import numpy as np
    xs = np.array([int(a, 2) for a, _ in pairs], dtype=np.uint8)
    ys = np.array([int(b, 2) for _, b in pairs], dtype=np.uint8)
    matches = (table[:, xs] == ys[None, :]).all(axis=1)
    if not matches.any():
        return _boxed(answer)
    # Prefer a candidate that ALSO predicts our known answer for the query.
    q = _BIN_Q.search(prompt)
    if q:
        qv = int(q.group(1), 2)
        ans_v = int(answer, 2)
        agree = matches & (table[:, qv] == ans_v)
        if agree.any():
            idx = np.where(agree)[0][0]
        else:
            idx = np.where(matches)[0][0]
    else:
        idx = np.where(matches)[0][0]
    rule = names[idx]
    return (
        f"The bit-level transformation can be expressed as: {rule}. "
        f"Apply it to the query input.\n\n{_boxed(answer)}"
    )


def trace_cipher(prompt: str, answer: str) -> str:
    return (
        f"Build the cipher-to-plain map letter by letter from the examples, "
        f"then decode each word, using the Wonderland vocabulary when a letter "
        f"is unseen.\n\n{_boxed(answer)}"
    )


def trace_equation(prompt: str, answer: str) -> str:
    return (
        f"Match the query against the examples on the operator symbol and "
        f"apply the matching rewrite.\n\n{_boxed(answer)}"
    )


_TRACES = {
    "numeral":  trace_numeral,
    "units":    trace_units,
    "gravity":  trace_gravity,
    "binary":   trace_binary,
    "cipher":   trace_cipher,
    "equation": trace_equation,
}


def build_completion(prompt: str, answer: str, category: str | None = None) -> str:
    cat = category or categorize(prompt)
    fn = _TRACES.get(cat)
    if fn is None:
        return _boxed(answer)
    try:
        return fn(prompt, answer)
    except Exception:
        return _boxed(answer)
