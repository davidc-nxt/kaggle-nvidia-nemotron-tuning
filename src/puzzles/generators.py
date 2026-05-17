"""Synthetic puzzle generators — one per category we can solve.

Each generator produces a (prompt, answer, category) tuple in the same format as
the original train rows. Random sampling is deterministic given a seed so we can
mint reproducible datasets.

Equation is not yet supported (its rule structure is still under investigation).
"""

from __future__ import annotations

import hashlib
import json
import random
import string
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.puzzles import binary as binary_solver
from src.puzzles.numeral import to_roman

_REPO = Path(__file__).resolve().parents[2]
_VOCAB_PATH = _REPO / "data" / "processed" / "cipher_vocab.json"


def _new_id(*parts: object) -> str:
    h = hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()
    return h[:8]


# ---------------------------------------------------------------------------
# Numeral
# ---------------------------------------------------------------------------

def gen_numeral(rng: random.Random) -> tuple[str, str, str]:
    nums = rng.sample(range(1, 100), k=rng.randint(4, 5))
    query = nums.pop()
    body = "\n".join(f"{n} -> {to_roman(n)}" for n in nums)
    prompt = (
        "In Alice's Wonderland, numbers are secretly converted into a different "
        "numeral system. Some examples are given below:\n"
        f"{body}\n"
        f"Now, write the number {query} in the Wonderland numeral system."
    )
    return prompt, to_roman(query), "numeral"


# ---------------------------------------------------------------------------
# Units — y = k * x
# ---------------------------------------------------------------------------

def gen_units(rng: random.Random) -> tuple[str, str, str]:
    # Observed k range in training spans roughly [0.4, 2.5]. Sample widely so the
    # model sees both shrinking and stretching conversions.
    k = round(rng.uniform(0.4, 2.5), 4)
    xs = [round(rng.uniform(2.0, 50.0), 2) for _ in range(rng.randint(3, 5))]
    query = round(rng.uniform(2.0, 50.0), 2)

    lines = []
    for x in xs:
        y = round(k * x, 2)
        lines.append(f"{x:g} m becomes {y:.2f}")
    prompt = (
        "In Alice's Wonderland, a secret unit conversion is applied to measurements. "
        "For example:\n"
        + "\n".join(lines)
        + f"\nNow, convert the following measurement: {query:g} m"
    )
    return prompt, f"{round(k * query, 2):.2f}", "units"


# ---------------------------------------------------------------------------
# Gravity — d = 0.5 * g * t^2
# ---------------------------------------------------------------------------

def gen_gravity(rng: random.Random) -> tuple[str, str, str]:
    g = round(rng.uniform(5.0, 25.0), 3)
    ts = sorted({round(rng.uniform(1.0, 5.0), 2) for _ in range(rng.randint(3, 6))})
    query = round(rng.uniform(1.0, 5.0), 2)

    lines = [f"For t = {t}s, distance = {round(0.5 * g * t * t, 2):.2f} m" for t in ts]
    prompt = (
        "In Alice's Wonderland, the gravitational constant has been secretly changed. "
        "Here are some example observations:\n"
        + "\n".join(lines)
        + f"\nNow, determine the falling distance for t = {query}s given d = 0.5*g*t^2."
    )
    return prompt, f"{round(0.5 * g * query * query, 2):.2f}", "gravity"


# ---------------------------------------------------------------------------
# Binary — random rule from the solver's library
# ---------------------------------------------------------------------------

def gen_binary(rng: random.Random) -> tuple[str, str, str]:
    names, table = binary_solver._candidates()
    idx = rng.randrange(len(names))
    rule_table = table[idx]  # shape (256,)

    inputs = rng.sample(range(256), 8)
    pairs = [(x, int(rule_table[x])) for x in inputs]
    # query must NOT collide with example inputs and must be a fresh draw
    used = {x for x, _ in pairs}
    while True:
        q = rng.randrange(256)
        if q not in used:
            break
    expected = int(rule_table[q])

    def fmt(v: int) -> str:
        return format(v, "08b")

    body = "\n".join(f"{fmt(x)} -> {fmt(y)}" for x, y in pairs)
    prompt = (
        "In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers. "
        "The transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT, "
        "and possibly majority or choice functions.\n\n"
        "Here are some examples of input -> output:\n"
        f"{body}\n\nNow, determine the output for: {fmt(q)}"
    )
    return prompt, fmt(expected), "binary"


# ---------------------------------------------------------------------------
# Cipher — random alphabet permutation, sentences from the closed-class vocab
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _cipher_vocab() -> list[str]:
    with _VOCAB_PATH.open() as f:
        return list(json.load(f).keys())


def _random_perm(rng: random.Random) -> dict[str, str]:
    """Random derangement-like permutation. Doesn't require pure derangement; the
    training distribution sometimes fixes a few letters."""
    letters = list(string.ascii_lowercase)
    shuffled = letters[:]
    rng.shuffle(shuffled)
    return dict(zip(letters, shuffled))


def gen_cipher(rng: random.Random) -> tuple[str, str, str]:
    vocab = _cipher_vocab()
    plain_to_cipher = _random_perm(rng)  # plain -> cipher
    cipher_to_plain = {c: p for p, c in plain_to_cipher.items()}

    def encrypt(text: str) -> str:
        return "".join(plain_to_cipher.get(ch, ch) for ch in text)

    # 4-6 example sentences plus one query, each 2-5 vocab words
    n_examples = rng.randint(4, 6)
    sentences = []
    for _ in range(n_examples + 1):
        n_words = rng.randint(2, 5)
        words = [rng.choice(vocab) for _ in range(n_words)]
        sentences.append(" ".join(words))
    query_plain = sentences[-1]
    examples = sentences[:-1]

    body = "\n".join(f"{encrypt(s)} -> {s}" for s in examples)
    prompt = (
        "In Alice's Wonderland, secret encryption rules are used on text. Here are some examples:\n"
        f"{body}\n"
        f"Now, decrypt the following text: {encrypt(query_plain)}"
    )
    return prompt, query_plain, "cipher"


# ---------------------------------------------------------------------------
# Registry + driver
# ---------------------------------------------------------------------------

GENERATORS = {
    "numeral":  gen_numeral,
    "units":    gen_units,
    "gravity":  gen_gravity,
    "binary":   gen_binary,
    "cipher":   gen_cipher,
}


def generate_one(category: str, rng: random.Random) -> dict[str, str]:
    fn = GENERATORS[category]
    prompt, answer, cat = fn(rng)
    return {
        "id": _new_id(category, prompt, answer),
        "prompt": prompt,
        "answer": answer,
        "category": cat,
    }
