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


def _num_arith_mod(op_name: str, mod_n: int, *, in_rev: bool = False,
                   swap: bool = False, out_rev: bool = False, pad: int = 0,
                   prefix: str | None = None, suffix: str | None = None) -> Rule:
    """Arithmetic in base 10 with the result reduced modulo ``mod_n``.

    Useful for puzzles like ``80:32 = 48`` where the rule is ``(a - b) mod
    100`` — equivalent to absolute difference for positive results, but
    wraps negatives into the [0, mod_n) range instead of emitting a
    ``-`` sign.
    """
    op_fn = _arith_fn(op_name)

    def fn(a_s: str, op_sym: str, b_s: str) -> str | None:
        if not (a_s.isdigit() and b_s.isdigit()):
            return None
        a = int(a_s[::-1]) if in_rev else int(a_s)
        b = int(b_s[::-1]) if in_rev else int(b_s)
        if swap:
            a, b = b, a
        v = op_fn(a, b) % mod_n
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
        if len(a) != len(b):
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
            elif op_name == "absub":
                v = abs(ix - iy) % 10
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
        if len(a) != len(b):
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
            elif op_name == "absub":
                v = abs(ix - iy) % 31
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


def _digitwise_pairs_num(spec: tuple, *, out_rev: bool = False,
                          prefix: str | None = None, suffix: str | None = None) -> Rule:
    """Per-output-position rule for digit context.

    ``spec`` is a tuple of per-position descriptors. Each descriptor is one of::

        ("copy", p)              - copy input[p]
        ("op_arith", p1, p2, k)  - arithmetic combination of input[p1] and input[p2]

    where p in 0..3 indexes a/b digits as [a0, a1, b0, b1] and op_arith is one of
    "+", "-", "absub", "*", with output digit = (op(x, y) + k) % 10.
    """
    def fn(a: str, op: str, b: str) -> str | None:
        if not (a.isdigit() and b.isdigit() and len(a) == 2 and len(b) == 2):
            return None
        chars = [a[0], a[1], b[0], b[1]]
        out = []
        for desc in spec:
            if desc[0] == "copy":
                out.append(chars[desc[1]])
            else:
                arith, p1, p2, k = desc
                ix = int(chars[p1])
                iy = int(chars[p2])
                if arith == "+":
                    v = ix + iy
                elif arith == "-":
                    v = ix - iy
                elif arith == "absub":
                    v = abs(ix - iy)
                elif arith == "*":
                    v = ix * iy
                else:
                    return None
                out.append(str((v + k) % 10))
        s = "".join(out)
        if out_rev:
            s = s[::-1]
        if prefix == "op":
            s = op + s
        elif suffix == "op":
            s = s + op
        return s

    return fn


def _slice_concat(*spec: tuple, prefix: str | None = None, suffix: str | None = None) -> Rule:
    """Output is concatenation of slices/transformations of a and b.

    Each entry in ``spec`` is one of:
        ("a", start, stop)   - a[start:stop]
        ("b", start, stop)   - b[start:stop]
        ("ra", start, stop)  - a[::-1][start:stop]
        ("rb", start, stop)  - b[::-1][start:stop]
    """
    def fn(a: str, op: str, b: str) -> str | None:
        parts = []
        for desc in spec:
            kind, start, stop = desc
            if kind == "a":
                parts.append(a[start:stop])
            elif kind == "b":
                parts.append(b[start:stop])
            elif kind == "ra":
                parts.append(a[::-1][start:stop])
            elif kind == "rb":
                parts.append(b[::-1][start:stop])
            else:
                return None
        s = "".join(parts)
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

    # Digit-wise numeric (now includes absub variant)
    for op_name in ("+", "-", "absub", "*"):
        for swap in (False, True):
            for out_rev in (False, True):
                for prefix in (None, "op"):
                    for suffix in (None, "op"):
                        if prefix and suffix:
                            continue
                        rules.append((f"dwn|{op_name}|sw{swap}|o{out_rev}|pre{prefix}|suf{suffix}",
                                      _digitwise_num(op_name, swap=swap, out_rev=out_rev,
                                                     prefix=prefix, suffix=suffix)))

    # Digit-wise symbolic (now includes absub variant)
    for op_name in ("+", "-", "absub", "*"):
        for swap in (False, True):
            for out_rev in (False, True):
                for prefix in (None, "op"):
                    for suffix in (None, "op"):
                        if prefix and suffix:
                            continue
                        rules.append((f"dws|{op_name}|sw{swap}|o{out_rev}|pre{prefix}|suf{suffix}",
                                      _digitwise_sym(op_name, swap=swap, out_rev=out_rev,
                                                     prefix=prefix, suffix=suffix)))

    # Modular arithmetic — useful when the rule is ``(a op b) mod N`` and the
    # query produces a value that wraps. Most common is mod 100 for 2-digit
    # outputs (e.g. ``a - b`` written as the last two digits of (a-b) mod 100).
    for op_name in ("-", "+", "absub", "*"):
        for mod_n in (100, 10, 1000):
            for in_rev, out_rev in ((False, False), (True, True)):
                for swap in (False, True):
                    for pad in (0, 2, 3):
                        for prefix in (None, "op"):
                            for suffix in (None, "op"):
                                if prefix and suffix:
                                    continue
                                name = (
                                    f"nmod|{op_name}|m{mod_n}|i{in_rev}|sw{swap}|"
                                    f"o{out_rev}|p{pad}|pre{prefix}|suf{suffix}"
                                )
                                rules.append((name, _num_arith_mod(
                                    op_name, mod_n, in_rev=in_rev, swap=swap,
                                    out_rev=out_rev, pad=pad,
                                    prefix=prefix, suffix=suffix)))

    # Final reorder: stable sort by an empirically-derived rule priority.
    rules.sort(key=_rule_priority)
    return rules


def _rule_priority(item: tuple[str, Rule]) -> tuple:
    """Stable-sort key for ranking rules by empirical precision on training.

    The key prefers (in order):
      - concat templates with high precision (c|ab, c|ba, cc|*, mx|*)
      - numeric arithmetic with reversed-both mirror (iTrue/oTrue), low shift,
        no padding, no swap, no prefix/suffix
      - digit-wise numeric (broad coverage)
      - symbolic arithmetic
      - digit-wise symbolic
    Within each family, the most precise variants come first.
    """
    name, _ = item
    parts = name.split("|")
    family = parts[0]
    if family == "num":
        # arith: + and * are highest precision in train
        op_arith = parts[1]
        op_order = {"*": 0, "+": 1, "-": 2, "absub": 3}.get(op_arith, 9)
        # in_rev/out_rev mirror is the dominant pattern
        ir, or_ = parts[2], parts[4]
        iror_order = {
            ("iTrue", "oTrue"): 0,
            ("iFalse", "oFalse"): 1,
            ("iTrue", "oFalse"): 2,
            ("iFalse", "oTrue"): 3,
        }.get((ir, or_), 9)
        # shift: 0 first, then 1, -1
        try:
            shift = int(parts[5][1:])
        except (ValueError, IndexError):
            shift = 9
        shift_order = {0: 0, 1: 1, -1: 2}.get(shift, 9)
        # pad: 0 first
        try:
            pad = int(parts[6][1:])
        except (ValueError, IndexError):
            pad = 9
        pad_order = {0: 0, 2: 1, 3: 2, 4: 3}.get(pad, 9)
        # swap: False first
        swap_order = 0 if parts[3] == "swFalse" else 1
        # sign_op: depends on op_arith. For - / absub, op-as-sign is more common.
        sign_op = parts[7] == "soTrue"
        if op_arith in ("-", "absub"):
            sign_order = 0 if sign_op else 1
        else:
            sign_order = 0 if not sign_op else 1
        # prefix/suffix: None first
        ps_order = 0 if (parts[8] == "preNone" and parts[9] == "sufNone") else 1
        return (1, iror_order, op_order, shift_order, swap_order, pad_order, ps_order, sign_order)
    if family == "c":
        # c|ab and c|ba are extremely high precision; other concat templates
        # come slightly later but still before sym arith.
        sub = parts[1]
        sub_order = 0 if sub in ("ab", "ba") else 1
        ps_order = 0 if (parts[2] == "preNone" and parts[3] == "sufNone") else 1
        return (0, sub_order, ps_order)
    if family in ("cc", "mx"):
        ps_order = 0 if (parts[2] == "preNone" and parts[3] == "sufNone") else 1
        return (2, 0, ps_order)
    if family == "sym":
        op_arith = parts[1]
        op_order = {"*": 0, "+": 1, "-": 2, "absub": 3}.get(op_arith, 9)
        ir, or_ = parts[2], parts[4]
        iror_order = {
            ("iTrue", "oTrue"): 0,
            ("iFalse", "oFalse"): 1,
            ("iTrue", "oFalse"): 2,
            ("iFalse", "oTrue"): 3,
        }.get((ir, or_), 9)
        swap_order = 0 if parts[3] == "swFalse" else 1
        try:
            pad = int(parts[5][1:])
        except (ValueError, IndexError):
            pad = 9
        pad_order = {0: 0, 2: 1}.get(pad, 9)
        sign_op = parts[6] == "soTrue"
        if op_arith in ("-", "absub"):
            sign_order = 0 if sign_op else 1
        else:
            sign_order = 0 if not sign_op else 1
        ps_order = 0 if (parts[7] == "preNone" and parts[8] == "sufNone") else 1
        return (3, iror_order, op_order, swap_order, pad_order, sign_order, ps_order)
    if family == "dwn":
        op_arith = parts[1]
        op_order = {"+": 0, "-": 1, "absub": 2, "*": 3}.get(op_arith, 9)
        swap_order = 0 if parts[2] == "swFalse" else 1
        out_rev_order = 0 if parts[3] == "oFalse" else 1
        ps_order = 0 if (parts[4] == "preNone" and parts[5] == "sufNone") else 1
        return (4, op_order, swap_order, out_rev_order, ps_order)
    if family == "dws":
        op_arith = parts[1]
        op_order = {"+": 0, "-": 1, "absub": 2, "*": 3}.get(op_arith, 9)
        swap_order = 0 if parts[2] == "swFalse" else 1
        out_rev_order = 0 if parts[3] == "oFalse" else 1
        ps_order = 0 if (parts[4] == "preNone" and parts[5] == "sufNone") else 1
        return (5, op_order, swap_order, out_rev_order, ps_order)
    if family == "nmod":
        # Modular arithmetic: prefer ``a - b mod 100`` (the absdiff-style
        # rule) before the others. Mirror reverse variants come after.
        op_arith = parts[1]
        op_order = {"-": 0, "absub": 1, "+": 2, "*": 3}.get(op_arith, 9)
        try:
            mod_n = int(parts[2][1:])
        except (ValueError, IndexError):
            mod_n = 9999
        mod_order = {100: 0, 10: 1, 1000: 2}.get(mod_n, 9)
        ir, or_ = parts[3], parts[5]
        iror_order = {("iFalse", "oFalse"): 0, ("iTrue", "oTrue"): 1}.get((ir, or_), 9)
        swap_order = 0 if parts[4] == "swFalse" else 1
        try:
            pad = int(parts[6][1:])
        except (ValueError, IndexError):
            pad = 9
        pad_order = {0: 0, 2: 1, 3: 2}.get(pad, 9)
        ps_order = 0 if (parts[7] == "preNone" and parts[8] == "sufNone") else 1
        # Modular rules slot between num arithmetic with iror_order<=2 and the
        # less-common num variants (iror_order=3 or unknown). This way the
        # dominant ``num|*|iTrue|oTrue|s0`` rule still wins, but modular
        # arithmetic is preferred over weirder num variants.
        return (1, 4, op_order, mod_order, iror_order, swap_order, pad_order, ps_order)
    return (9, 0)


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


def _fit_rule_for_query(
    examples: list[tuple[str, str, str, str]],
    query: tuple[str, str, str],
) -> str | None:
    """Find a rule that fits the examples and return its query prediction.

    Equivalent to ``_fit_rule(examples)`` plus applying the rule to the
    query. Returns the first valid prediction (the rule list is ordered by
    empirical priority in :func:`_rule_priority`).
    """
    if not examples:
        return None
    a_q, op_q, b_q = query
    for name, fn in _rules():
        try:
            if not all(fn(a, op, b) == r for a, op, b, r in examples):
                continue
            pred = fn(a_q, op_q, b_q)
            if pred is not None:
                return pred
        except Exception:
            continue
    return None


# ----------------------------------------------------------------------------
# Cross-operator template fitting.
#
# When the query's operator was never demonstrated, we cannot derive its rule
# from the prompt alone. But many puzzles use rules that share a structural
# template across operators (e.g. "reverse both, do arithmetic, reverse output"
# with the operator character determining which arithmetic). When this is the
# case we can predict for the query op by reusing the template with an
# arithmetic guess.
#
# Templates are derived from rule names: each non-trivial arithmetic rule has
# a name like ``num|<arith>|<knobs>``. We replace ``<arith>`` with a wildcard
# and look for a wildcard template that has at least one matching rule for
# every demonstrated operator. For the query op we try every arithmetic that
# is plausible under the template; the first prediction that's well-typed is
# returned.
# ----------------------------------------------------------------------------

# Built lazily so we don't pay the cost on import.
_TEMPLATE_INDEX: dict[str, list[tuple[str, str, Rule]]] | None = None


def _template_key(name: str) -> str | None:
    """Map ``<family>|<arith>|<knobs>`` to a wildcard template.

    For arithmetic families (``num``/``sym``/``dwn``/``dws``) we erase the
    arithmetic name AND the shift knob — both can differ across operators in
    the same puzzle while everything else stays constant.
    """
    parts = name.split("|")
    if parts[0] in ("num", "sym", "dwn", "dws") and len(parts) > 2:
        # Replace arith with wildcard. For num/sym we also wildcard the shift,
        # which is a per-op tuning parameter we want to let vary.
        parts[1] = "*"
        if parts[0] in ("num", "sym"):
            for i, p in enumerate(parts):
                if p.startswith("s") and len(p) > 1 and (p[1:].lstrip("-")).isdigit():
                    parts[i] = "s*"
                    break
        return "|".join(parts)
    return None


def _build_template_index() -> dict[str, list[tuple[str, str, Rule]]]:
    idx: dict[str, list[tuple[str, str, Rule]]] = {}
    for name, fn in _rules():
        key = _template_key(name)
        if key is None:
            continue
        arith = name.split("|")[1]
        idx.setdefault(key, []).append((arith, name, fn))
    return idx


def _templates() -> dict[str, list[tuple[str, str, Rule]]]:
    global _TEMPLATE_INDEX
    if _TEMPLATE_INDEX is None:
        _TEMPLATE_INDEX = _build_template_index()
    return _TEMPLATE_INDEX


# Order in which we try arithmetics for the query op. ``absub`` and ``+`` are
# the most common Wonderland defaults so they sit at the front; ``-`` and
# ``*`` follow.
_ARITH_PRIORITY = ("absub", "+", "*", "-")


def _fit_template_constrained(
    by_op: dict[str, list[tuple[str, str, str, str]]],
    query: tuple[str, str, str],
) -> str | None:
    """Pick a rule for the query op whose structural template is shared with
    rules fitting every other demonstrated operator.

    Useful when ``op_q`` has only one or two same-op demonstrations and the
    main library's first-match rule is overfit to those few examples. By
    requiring the rule's template to "extend" to the other operators in the
    puzzle, we bias toward a rule that's structurally consistent across the
    whole puzzle's operator zoo.

    Returns a predicted answer string or None.
    """
    a_q, op_q, b_q = query
    if op_q not in by_op or len(by_op) < 2:
        return None
    op_q_examples = by_op[op_q]
    templates = _templates()
    # Walk rules in priority order so the first matching template-consistent
    # rule wins.
    for name, fn in _rules():
        try:
            if not all(fn(a, op, b) == r for a, op, b, r in op_q_examples):
                continue
        except Exception:
            continue
        tmpl = _template_key(name)
        if tmpl is None:
            continue
        candidates = templates.get(tmpl, [])
        # Check every other operator has a rule under this template.
        valid = True
        for op_char, exs in by_op.items():
            if op_char == op_q:
                continue
            fits = False
            for _, _, fn2 in candidates:
                try:
                    if all(fn2(a, op, b) == r for a, op, b, r in exs):
                        fits = True
                        break
                except Exception:
                    pass
            if not fits:
                valid = False
                break
        if not valid:
            continue
        try:
            pred = fn(a_q, op_q, b_q)
            if pred is not None:
                return pred
        except Exception:
            continue
    return None


def _fit_cross_op_template(by_op: dict[str, list[tuple[str, str, str, str]]],
                           query: tuple[str, str, str]) -> str | None:
    """Find a structural template that fits every demonstrated operator, then
    aggregate predictions across templates and arithmetics.

    For each template that has a matching arithmetic for every demonstrated
    operator, we generate a prediction for the query operator using:
      - the query op char's arithmetic symbol (if op_q is ``+``/``-``/``*``)
      - otherwise ``absub`` as the default Wonderland guess

    The final prediction is the most-voted across all templates. This is the
    best we can do without an oracle for the query op's arithmetic.
    """
    a_q, op_q, b_q = query
    if not by_op:
        return None
    from collections import Counter

    op_to_arith_symbol = {"+": "+", "-": "-", "*": "*"}
    pred_votes: Counter = Counter()
    templates = _templates()
    for tmpl, candidates in templates.items():
        # Each operator must have at least one matching arithmetic under tmpl.
        op_arith_options: dict[str, list[tuple[str, Rule]]] = {}
        ok = True
        for op_char, exs in by_op.items():
            matches: list[tuple[str, Rule]] = []
            for arith, _, fn in candidates:
                try:
                    if all(fn(a, op, b) == r for a, op, b, r in exs):
                        matches.append((arith, fn))
                except Exception:
                    pass
            if not matches:
                ok = False
                break
            op_arith_options[op_char] = matches
        if not ok:
            continue

        # Decide the arithmetic to use for the query op. Default to the most
        # plausible single guess (matches op_q if it's a standard symbol;
        # otherwise ``absub``).
        if op_q in op_to_arith_symbol:
            chosen_arith = op_to_arith_symbol[op_q]
        else:
            # Fall back to a common arithmetic if one exists.
            all_demo_ariths = [
                {a for a, _ in opts} for opts in op_arith_options.values()
            ]
            common = (
                set.intersection(*all_demo_ariths) if all_demo_ariths else set()
            )
            if common:
                # Pick the highest-priority arith from the common set.
                chosen_arith = next(
                    (a for a in _ARITH_PRIORITY if a in common), "absub"
                )
            else:
                # No common arith — guess ``absub`` (Wonderland default).
                chosen_arith = "absub"

        for cand_arith, cand_name, fn in candidates:
            if cand_arith != chosen_arith:
                continue
            try:
                pred = fn(a_q, op_q, b_q)
            except Exception:
                pred = None
            if pred is not None:
                pred_votes[pred] += 1
                break

    if not pred_votes:
        return None
    return pred_votes.most_common(1)[0][0]


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

    # 1) When op_q has only one or two same-op demonstrations, the standard
    # first-match rule is highly ambiguous. Prefer a rule whose structural
    # template is shared with rules fitting every other demonstrated operator
    # — this picks a rule that's "consistent across the puzzle".
    if op_q in by_op and len(by_op) > 1 and len(by_op[op_q]) <= 2:
        out = _fit_template_constrained(by_op, query)
        if out is not None:
            return _maybe_box(out)

    # 2) Prefer rules that fit the query operator's examples.
    main_matched = False
    if op_q in by_op:
        out = _fit_rule_for_query(by_op[op_q], query)
        if out is not None:
            main_matched = True
            return _maybe_box(out)

    # 3) Otherwise try fitting on all examples (uniform-rule puzzles).
    if not main_matched:
        out = _fit_rule_for_query(examples, query)
        if out is not None:
            return _maybe_box(out)

        # 3b) Cross-operator template fitting: when the query op was never
        # demonstrated, look for a structural template (across all ops) and
        # instantiate it for the query op with a plausible arithmetic guess.
        if op_q not in by_op:
            cross = _fit_cross_op_template(by_op, query)
            if cross is not None:
                return _maybe_box(cross)

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
