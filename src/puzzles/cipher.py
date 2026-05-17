"""Solver for `cipher` puzzles.

Each puzzle hides a random substitution cipher cipher_letter -> plain_letter.
We deduce as much of the mapping as we can from the in-prompt examples, then
fill the remaining letters by matching the query words against a tight,
closed-class vocabulary (77 words) extracted from the training set.

The vocabulary lives at `data/processed/cipher_vocab.json` and is built once
by `scripts/build_cipher_vocab.py`.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_VOCAB_PATH = _REPO / "data" / "processed" / "cipher_vocab.json"

_EX_RE = re.compile(r"^([a-z ]+)\s*->\s*([a-z ]+)$", re.M | re.I)
_QUERY_RE = re.compile(r"decrypt the following text:\s*(.+?)\s*$", re.I | re.M)


@lru_cache(maxsize=1)
def _vocab() -> list[str]:
    with _VOCAB_PATH.open() as f:
        return list(json.load(f).keys())


def _deduce_mapping(pairs: list[tuple[str, str]]) -> dict[str, str]:
    m: dict[str, str] = {}
    for c, p in pairs:
        for ci, pi in zip(c, p):
            if ci == " " or pi == " ":
                continue
            if ci in m and m[ci] != pi:
                raise ValueError("cipher solver: contradictory example mapping")
            m[ci] = pi
    return m


def _candidates_for_word(cw: str, mapping: dict[str, str], vocab: list[str]) -> list[tuple[str, dict[str, str]]]:
    """Vocabulary words consistent with the current mapping when applied to `cw`."""
    plain_inverse = {v: k for k, v in mapping.items()}  # plain -> cipher we already trust
    out: list[tuple[str, dict[str, str]]] = []
    for vw in vocab:
        if len(vw) != len(cw):
            continue
        new_letters: dict[str, str] = {}
        ok = True
        for ci, pi in zip(cw, vw):
            if ci in mapping:
                if mapping[ci] != pi:
                    ok = False
                    break
            else:
                # New mapping ci -> pi proposed. Validate against:
                #  (a) any prior decision in this word
                #  (b) one-to-one constraint vs. existing inverse mapping
                if ci in new_letters and new_letters[ci] != pi:
                    ok = False
                    break
                if pi in plain_inverse and plain_inverse[pi] != ci:
                    ok = False
                    break
                if pi in new_letters.values() and pi not in [new_letters.get(ci, None)]:
                    # another cipher letter in THIS word already proposes the same plain target
                    if any(np == pi and nc != ci for nc, np in new_letters.items()):
                        ok = False
                        break
                new_letters[ci] = pi
        if ok:
            out.append((vw, new_letters))
    return out


def solve(prompt: str) -> str:
    pair_matches = _EX_RE.findall(prompt)
    pairs = [(c.strip(), p.strip()) for c, p in pair_matches if "->" not in p]  # cheap sanity
    if not pairs:
        # Fall back to less strict line-by-line parse.
        pairs = []
        block = prompt.split("examples:", 1)[1] if "examples:" in prompt else prompt
        block = block.split("Now,", 1)[0]
        for line in block.splitlines():
            if "->" in line:
                c, p = line.split("->", 1)
                pairs.append((c.strip(), p.strip()))
    if not pairs:
        raise ValueError("cipher solver: no example pairs found")

    qm = _QUERY_RE.search(prompt)
    if not qm:
        raise ValueError("cipher solver: no decrypt query found")
    query = qm.group(1).strip()

    mapping = _deduce_mapping(pairs)
    vocab = _vocab()

    words = query.split()
    decoded: list[str | None] = [None] * len(words)

    # Greedy multi-pass: resolve any word that has a unique candidate, propagate, repeat.
    progress = True
    while progress:
        progress = False
        for i, cw in enumerate(words):
            if decoded[i] is not None:
                continue
            cands = _candidates_for_word(cw, mapping, vocab)
            # Filter to UNIQUE plain decoding (multiple vocab entries can match if they imply same plain)
            unique_decodings = {vw for vw, _ in cands}
            if len(unique_decodings) == 1:
                vw, new = cands[0]
                decoded[i] = vw
                for ci, pi in new.items():
                    if ci not in mapping:
                        mapping[ci] = pi
                progress = True

    # Anything still unresolved: pick the most common vocab word that fits.
    for i, cw in enumerate(words):
        if decoded[i] is not None:
            continue
        cands = _candidates_for_word(cw, mapping, vocab)
        if cands:
            decoded[i] = cands[0][0]  # vocab is roughly frequency-sorted
        else:
            decoded[i] = "".join(mapping.get(c, c) for c in cw)

    return " ".join(d or "" for d in decoded)
