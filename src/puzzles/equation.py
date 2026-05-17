"""Solver for `equation` puzzles in Alice's Wonderland.

Each puzzle gives several examples of the form `<2 chars><op><2 chars> = <output>`.
The hidden transformation rule depends on the operator and (sometimes) on the
specific characters. There is no single closed-form rule across the corpus -- the
training data exposes many distinct families.

Strategy: enumerate a large library of candidate rules covering common
families (real arithmetic, reverse-arithmetic, concat permutations, digit-wise
operations, operator-prefix/suffix variants for both numeric and symbolic
inputs). For each puzzle, pick the first rule consistent with every example
that shares the query's operator (falling back to all examples when the query
op never appears in the demonstrations).
"""

from __future__ import annotations

import re
from itertools import product
from typing import Callable

_QUERY_RE = re.compile(r"^\s*Now,\s*determine the result for:\s*(.{5})\s*$", re.M | re.I)
_EXAMPLE_RE = re.compile(r"^(.{5}) = (.+?)\s*$", re.M)

ALPHABET = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}'
SYM_TO_IDX = {c: i for i, c in enumerate(ALPHABET)}
IDX_TO_SYM = {i: c for i, c in enumerate(ALPHABET)}


# ----------------------------------------------------------------------------
# Prompt parsing
# ----------------------------------------------------------------------------

def parse_prompt(prompt: str) -> tuple[list[tuple[str, str, str, str]], tuple[str, str, str] | None]:
    """Return (examples, query). Each example is (a, op, b, output).

    Skips the query line via the dedicated regex so it doesn't pollute examples.
    """
    qm = _QUERY_RE.search(prompt)
    query: tuple[str, str, str] | None = None
    if qm:
        s = qm.group(1)
        query = (s[:2], s[2], s[3:5])

    examples: list[tuple[str, str, str, str]] = []
    for line in prompt.split("\n"):
        if _QUERY_RE.match(line):
            continue
        m = _EXAMPLE_RE.match(line)
        if not m:
            continue
        lhs = m.group(1)
        out = m.group(2)
        examples.append((lhs[:2], lhs[2], lhs[3:5], out))
    return examples, query


# ----------------------------------------------------------------------------
# Rule helpers
# ----------------------------------------------------------------------------

Rule = Callable[[str, str, str], "str | None"]


def _arith_fn(op_name: str):
    if op_name == "+":
        return lambda a, b: a + b
    if op_name == "-":
        return lambda a, b: a - b
    if op_name == "*":
        return lambda a, b: a * b
    if op_name == "absub":
        return lambda a, b: abs(a - b)
    raise ValueError(op_name)


def _num_arith(op_name: str, *, in_rev: bool = False, swap: bool = False,
               out_rev: bool = False, shift: int = 0, pad: int = 0,
               sign_op: bool = False, prefix: str | None = None,
               suffix: str | None = None) -> Rule:
    op_fn = _arith_fn(op_name)

    def fn(a_s: str, op_sym: str, b_s: str) -> str | None:
        if not (a_s.isdigit() and b_s.isdigit()):
            return None
        a = int(a_s[::-1]) if in_rev else int(a_s)
        b = int(b_s[::-1]) if in_rev else int(b_s)
        if swap:
            a, b = b, a
        v = op_fn(a, b) + shift
        if v < 0:
            inner = f"{-v:0{pad}d}" if pad else str(-v)
            if out_rev:
                inner = inner[::-1]
            sign = op_sym if sign_op else "-"
            s = sign + inner
        else:
            s = f"{v:0{pad}d}" if pad else str(v)
            if out_rev:
                s = s[::-1]
        if prefix == "op":
            s = op_sym + s
        elif suffix == "op":
            s = s + op_sym
        return s

    return fn


def _sym_arith(op_name: str, *, in_rev: bool = False, swap: bool = False,
               out_rev: bool = False, shift: int = 0, pad: int = 0,
               sign_op: bool = False, prefix: str | None = None,
               suffix: str | None = None) -> Rule:
    op_fn = _arith_fn(op_name)

    def to_sym_str(v: int, width: int) -> str:
        if v == 0:
            s = IDX_TO_SYM[0]
        else:
            chars = []
            while v > 0:
                chars.append(IDX_TO_SYM[v % 31])
                v //= 31
            s = "".join(reversed(chars))
        if width and len(s) < width:
            s = IDX_TO_SYM[0] * (width - len(s)) + s
        return s

    def fn(a_s: str, op_sym: str, b_s: str) -> str | None:
        if not (all(c in SYM_TO_IDX for c in a_s) and all(c in SYM_TO_IDX for c in b_s)):
            return None
        aa = a_s[::-1] if in_rev else a_s
        bb = b_s[::-1] if in_rev else b_s
        a = 0
        for c in aa:
            a = a * 31 + SYM_TO_IDX[c]
        b = 0
        for c in bb:
            b = b * 31 + SYM_TO_IDX[c]
        if swap:
            a, b = b, a
        v = op_fn(a, b) + shift
        if v < 0:
            inner = to_sym_str(-v, pad)
            if out_rev:
                inner = inner[::-1]
            sign = op_sym if sign_op else "-"
            s = sign + inner
        else:
            s = to_sym_str(v, pad)
            if out_rev:
                s = s[::-1]
        if prefix == "op":
            s = op_sym + s
        elif suffix == "op":
            s = s + op_sym
        return s

    return fn


def _concat(*parts: str, prefix: str | None = None, suffix: str | None = None) -> Rule:
    """Concatenate pieces selected from {a, b, ra, rb, op, a0, a1, b0, b1}."""

    def fn(a: str, op: str, b: str) -> str | None:
        out_parts: list[str] = []
        for p in parts:
            if p == "a":
                out_parts.append(a)
            elif p == "b":
                out_parts.append(b)
            elif p == "ra":
                out_parts.append(a[::-1])
            elif p == "rb":
                out_parts.append(b[::-1])
            elif p == "op":
                out_parts.append(op)
            elif p == "a0":
                out_parts.append(a[0])
            elif p == "a1":
                out_parts.append(a[1])
            elif p == "b0":
                out_parts.append(b[0])
            elif p == "b1":
                out_parts.append(b[1])
        s = "".join(out_parts)
        if prefix == "op":
            s = op + s
        elif suffix == "op":
            s = s + op
        return s

    return fn


def _digitwise_num(op_name: str, *, swap: bool = False, out_rev: bool = False,
                   prefix: str | None = None, suffix: str | None = None) -> Rule:
    def fn(a: str, op: str, b: str) -> str | None:
        if not (a.isdigit() and b.isdigit()):
            return None
        aa, bb = (b, a) if swap else (a, b)
        out = []
        for x, y in zip(aa, bb):
            ix, iy = int(x), int(y)
            if op_name == "+":
                v = (ix + iy) % 10
            elif op_name == "-":
                v = (ix - iy) % 10
            elif op_name == "*":
                v = (ix * iy) % 10
            else:
                return None
            out.append(str(v))
        s = "".join(out)
        if out_rev:
            s = s[::-1]
        if prefix == "op":
            s = op + s
        elif suffix == "op":
            s = s + op
        return s

    return fn


def _digitwise_sym(op_name: str, *, swap: bool = False, out_rev: bool = False,
                   prefix: str | None = None, suffix: str | None = None) -> Rule:
    def fn(a: str, op: str, b: str) -> str | None:
        if not (all(c in SYM_TO_IDX for c in a) and all(c in SYM_TO_IDX for c in b)):
            return None
        aa, bb = (b, a) if swap else (a, b)
        out = []
        for x, y in zip(aa, bb):
            ix, iy = SYM_TO_IDX[x], SYM_TO_IDX[y]
            if op_name == "+":
                v = (ix + iy) % 31
            elif op_name == "-":
                v = (ix - iy) % 31
            elif op_name == "*":
                v = (ix * iy) % 31
            else:
                return None
            out.append(IDX_TO_SYM[v])
        s = "".join(out)
        if out_rev:
            s = s[::-1]
        if prefix == "op":
            s = op + s
        elif suffix == "op":
            s = s + op
        return s

    return fn


# ----------------------------------------------------------------------------
# Rule library
# ----------------------------------------------------------------------------

def _build_rules() -> list[tuple[str, Rule]]:
    rules: list[tuple[str, Rule]] = []

    # Numeric arithmetic (broad coverage). Order matters when several rules tie:
    # in_rev=True, out_rev=True (the "reverse-both" pattern) is the most common
    # generalizing variant in training, so list it first within each op.
    # absub comes before signed `-` because examples rarely demonstrate negatives,
    # and absolute-difference is the more common Wonderland rule.
    for op_name in ("+", "absub", "-", "*"):
        # The (True, True) "reverse-both" pattern is the most common Wonderland
        # rule; list it before the direct (False, False) variant.
        for in_rev, out_rev in ((True, True), (False, False), (True, False), (False, True)):
            for swap in (False, True):
                for shift in (0, 1, -1):
                    # pad=0 first (no padding) since that's the default Wonderland format;
                    # padding rules tie with their unpadded equivalents on 2-digit cases.
                    for pad in (0, 2, 3, 4):
                        for sign_op in (False, True):
                            # Most puzzles use the operator symbol when negative AND when prefixing.
                            for prefix in (None, "op"):
                                for suffix in (None, "op"):
                                    if prefix and suffix:
                                        continue
                                    name = (
                                        f"num|{op_name}|i{in_rev}|sw{swap}|o{out_rev}|"
                                        f"s{shift}|p{pad}|so{sign_op}|pre{prefix}|suf{suffix}"
                                    )
                                    rules.append((name, _num_arith(
                                        op_name, in_rev=in_rev, swap=swap,
                                        out_rev=out_rev, shift=shift, pad=pad,
                                        sign_op=sign_op, prefix=prefix, suffix=suffix)))

    # Symbolic arithmetic
    for op_name in ("+", "-", "*", "absub"):
        for in_rev in (False, True):
            for swap in (False, True):
                for out_rev in (False, True):
                    for pad in (0, 2):
                        for sign_op in (False, True):
                            for prefix in (None, "op"):
                                for suffix in (None, "op"):
                                    if prefix and suffix:
                                        continue
                                    name = (
                                        f"sym|{op_name}|i{in_rev}|sw{swap}|o{out_rev}|"
                                        f"p{pad}|so{sign_op}|pre{prefix}|suf{suffix}"
                                    )
                                    rules.append((name, _sym_arith(
                                        op_name, in_rev=in_rev, swap=swap,
                                        out_rev=out_rev, pad=pad,
                                        sign_op=sign_op, prefix=prefix, suffix=suffix)))

    # Concat permutations -- single/double/triple piece templates
    parts_opts_full = ("a", "b", "ra", "rb")
    parts_opts_chars = ("a0", "a1", "b0", "b1")

    # 2-piece full templates
    for p1 in parts_opts_full:
        for p2 in parts_opts_full:
            for prefix in (None, "op"):
                for suffix in (None, "op"):
                    if prefix and suffix:
                        continue
                    rules.append((f"c|{p1}{p2}|pre{prefix}|suf{suffix}",
                                  _concat(p1, p2, prefix=prefix, suffix=suffix)))

    # 1-piece full templates
    for p in parts_opts_full:
        for prefix in (None, "op"):
            for suffix in (None, "op"):
                if prefix and suffix:
                    continue
                rules.append((f"c|{p}|pre{prefix}|suf{suffix}",
                              _concat(p, prefix=prefix, suffix=suffix)))

    # 2-3-4 character templates from positions
    from itertools import product
    for L in (1, 2, 3, 4):
        for combo in product(parts_opts_chars, repeat=L):
            for prefix in (None, "op"):
                for suffix in (None, "op"):
                    if prefix and suffix:
                        continue
                    rules.append((f"cc|{','.join(combo)}|pre{prefix}|suf{suffix}",
                                  _concat(*combo, prefix=prefix, suffix=suffix)))

    # Mixed templates: combine full-piece (a, b, ra, rb) with single chars
    for L in (3,):
        for combo in product(parts_opts_full + parts_opts_chars, repeat=L):
            for prefix in (None, "op"):
                for suffix in (None, "op"):
                    if prefix and suffix:
                        continue
                    rules.append((f"mx|{','.join(combo)}|pre{prefix}|suf{suffix}",
                                  _concat(*combo, prefix=prefix, suffix=suffix)))

    # Digit-wise numeric
    for op_name in ("+", "-", "*"):
        for swap in (False, True):
            for out_rev in (False, True):
                for prefix in (None, "op"):
                    for suffix in (None, "op"):
                        if prefix and suffix:
                            continue
                        rules.append((f"dwn|{op_name}|sw{swap}|o{out_rev}|pre{prefix}|suf{suffix}",
                                      _digitwise_num(op_name, swap=swap, out_rev=out_rev,
                                                     prefix=prefix, suffix=suffix)))

    # Digit-wise symbolic
    for op_name in ("+", "-", "*"):
        for swap in (False, True):
            for out_rev in (False, True):
                for prefix in (None, "op"):
                    for suffix in (None, "op"):
                        if prefix and suffix:
                            continue
                        rules.append((f"dws|{op_name}|sw{swap}|o{out_rev}|pre{prefix}|suf{suffix}",
                                      _digitwise_sym(op_name, swap=swap, out_rev=out_rev,
                                                     prefix=prefix, suffix=suffix)))

    return rules


_RULES_CACHE: list[tuple[str, Rule]] | None = None


def _rules() -> list[tuple[str, Rule]]:
    global _RULES_CACHE
    if _RULES_CACHE is None:
        _RULES_CACHE = _build_rules()
    return _RULES_CACHE


# ----------------------------------------------------------------------------
# Main solver
# ----------------------------------------------------------------------------

def _fit_rule(examples: list[tuple[str, str, str, str]]) -> tuple[str, Rule] | None:
    """Return the first rule that maps every example correctly."""
    if not examples:
        return None
    for name, fn in _rules():
        try:
            if all(fn(a, op, b) == r for a, op, b, r in examples):
                return (name, fn)
        except Exception:
            continue
    return None


# ----------------------------------------------------------------------------
# Positional-template fallback. Covers same-length-output puzzles whose answer
# is some permutation (with repetition) of the five input characters, with an
# optional reversal and an optional bracketing operator. The main rule library
# already covers most of these but the explicit template search is cheap and
# catches a few edge cases the named-piece concatenations miss.
# ----------------------------------------------------------------------------

_POSITIONAL_TEMPLATES: list[tuple[tuple[int, ...], bool, bool, bool]] = []
for _L in (1, 2, 3, 4, 5):
    for _p in product(range(5), repeat=_L):
        for _rev in (False, True):
            for _op_pre in (False, True):
                for _op_post in (False, True):
                    if _op_pre and _op_post:
                        continue
                    _POSITIONAL_TEMPLATES.append((_p, _rev, _op_pre, _op_post))


def _apply_positional(template, lhs5: str) -> str | None:
    positions, rev, op_pre, op_post = template
    if len(lhs5) != 5:
        return None
    s = "".join(lhs5[i] for i in positions)
    if rev:
        s = s[::-1]
    op_char = lhs5[2]
    if op_pre:
        s = op_char + s
    elif op_post:
        s = s + op_char
    return s


def _fit_positional(op_examples):
    for template in _POSITIONAL_TEMPLATES:
        ok = True
        for lhs5, out in op_examples:
            if _apply_positional(template, lhs5) != out:
                ok = False
                break
        if ok:
            return template
    return None


# ----------------------------------------------------------------------------
# Per-output-position rules (copy / shift / constant) in either alphabet.
# Used when neither the main library nor the positional templates fit; recovers
# puzzles whose rule shifts characters by a constant in their native alphabet.
# ----------------------------------------------------------------------------

def _char_idx(c: str, digits: bool) -> int | None:
    if digits:
        return int(c) if c.isdigit() else None
    return SYM_TO_IDX.get(c)


def _char_from_idx(i: int, digits: bool, base: int) -> str:
    if digits:
        return str(i % 10)
    return IDX_TO_SYM[i % base]


def _find_per_position(op_examples, digits: bool, base: int):
    if not op_examples:
        return None
    lengths = {len(out) for _, out in op_examples}
    if len(lengths) != 1:
        return None
    L = next(iter(lengths))
    if L == 0 or L > 5:
        return None
    rules = []
    for i in range(L):
        opts = []
        if all(out[i] == op_examples[0][1][i] for _, out in op_examples):
            opts.append(("const", op_examples[0][1][i]))
        for p in range(5):
            if all(inp[p] == out[i] for inp, out in op_examples):
                opts.append(("copy", p))
        for p in range(5):
            shifts = []
            ok = True
            for inp, out in op_examples:
                ii = _char_idx(inp[p], digits)
                oi = _char_idx(out[i], digits)
                if ii is None or oi is None:
                    ok = False
                    break
                shifts.append((oi - ii) % base)
            if ok and len(set(shifts)) == 1:
                opts.append(("shift", p, shifts[0]))
        if not opts:
            return None

        def _key(opt):
            if opt[0] == "copy":
                return (0, 0)
            if opt[0] == "const":
                return (1, 0)
            if opt[0] == "shift":
                return (2, min(opt[2], base - opt[2]))
            return (10, 0)

        opts.sort(key=_key)
        rules.append(opts[0])
    return rules


def _apply_per_position(rules, lhs5: str, digits: bool, base: int) -> str | None:
    out: list[str] = []
    for r in rules:
        tag = r[0]
        if tag == "copy":
            if r[1] >= len(lhs5):
                return None
            out.append(lhs5[r[1]])
        elif tag == "shift":
            ii = _char_idx(lhs5[r[1]], digits)
            if ii is None:
                return None
            out.append(_char_from_idx((ii + r[2]) % base, digits, base))
        elif tag == "const":
            out.append(r[1])
        else:
            return None
    return "".join(out)


def _is_digit_context(op_examples, query5: str) -> bool:
    for lhs5, out in op_examples:
        if not (len(lhs5) == 5 and lhs5[0:2].isdigit() and lhs5[3:5].isdigit()):
            return False
        if not all(c.isdigit() or c == "-" for c in out):
            return False
    if not (len(query5) == 5 and query5[0:2].isdigit() and query5[3:5].isdigit()):
        return False
    return True


# ----------------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------------

def solve(prompt: str) -> str:
    examples, query = parse_prompt(prompt)
    if query is None:
        raise ValueError("equation solver: no query found")
    a_q, op_q, b_q = query

    by_op: dict[str, list[tuple[str, str, str, str]]] = {}
    for a, op, b, r in examples:
        by_op.setdefault(op, []).append((a, op, b, r))

    # 1) Prefer rules that fit the query operator's examples.
    main_matched = False
    if op_q in by_op:
        fit = _fit_rule(by_op[op_q])
        if fit is not None:
            main_matched = True
            _, fn = fit
            out = fn(a_q, op_q, b_q)
            if out is not None:
                return _maybe_box(out)

    # 2) Otherwise try fitting on all examples (uniform-rule puzzles).
    if not main_matched:
        fit = _fit_rule(examples)
        if fit is not None:
            _, fn = fit
            out = fn(a_q, op_q, b_q)
            if out is not None:
                return _maybe_box(out)

    # 3) Positional-template fallback on same-op examples.
    if op_q in by_op:
        op_examples = [(a + op + b, r) for a, op, b, r in by_op[op_q]]
        tmpl = _fit_positional(op_examples)
        if tmpl is not None:
            pred = _apply_positional(tmpl, a_q + op_q + b_q)
            if pred is not None:
                return _maybe_box(pred)

        # 4) Per-output-position copy/shift/constant fallback.
        digits = _is_digit_context(op_examples, a_q + op_q + b_q)
        base = 10 if digits else 31
        pp = _find_per_position(op_examples, digits, base)
        if pp is not None:
            pred = _apply_per_position(pp, a_q + op_q + b_q, digits, base)
            if pred is not None:
                ok = True
                for lhs5, out in op_examples:
                    if _apply_per_position(pp, lhs5, digits, base) != out:
                        ok = False
                        break
                if ok:
                    return _maybe_box(pred)

    # 5) Last-resort fallback so we always emit something.
    return _maybe_box(a_q + b_q)


def _maybe_box(answer: str) -> str:
    """Wrap the answer in ``\\boxed{...}`` when the result contains characters that
    would otherwise confuse the metric's number/string extractor.

    The competition metric prefers ``\\boxed{...}`` over the "last number" fallback,
    but its boxed regex only allows a single level of brace nesting. If the answer
    itself contains an unmatched ``{`` or ``}`` we leave it raw -- in that case the
    string is already non-numeric and the metric falls through to a direct compare.
    """
    if "{" in answer or "}" in answer:
        return answer
    # Wrap whenever any digit is present together with non-digit content -- otherwise
    # the last-number fallback would silently drop the suffix (e.g. "17/" -> "17").
    has_digit = any(c.isdigit() for c in answer)
    has_other = any(not (c.isdigit() or c == "-") for c in answer)
    if has_digit and has_other:
        return "\\boxed{" + answer + "}"
    return answer
