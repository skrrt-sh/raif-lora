"""Differential equivalence: the pure-Python `raif_decode` must produce the same
JSON value as the canonical TS decoder for *every* input — not just the corpus.

Each test is a property over generated inputs: `py_decode(x) ≡ js_decode(x)`
(same ok/fail flag, and equal value on success). Inputs are generated
independently of how the Python port reads — round-tripped from random JSON
objects through the real TS encoder, then degraded by mutations that target each
repair branch. The point is to surface divergences a hand-written corpus would
miss; we assert *agreement with the oracle*, never a pre-imagined output, so the
test can't bake in the author's misreadings.

Skips when `bun` / the `raif-standard` checkout is absent.

Run:  python src/test_raif_differential.py   (or pytest)
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _raif_oracle as oracle  # noqa: E402
from raif_decode import decode  # noqa: E402

try:  # under pytest, skip the whole module cleanly when the TS toolchain is absent
    import pytest

    pytestmark = pytest.mark.skipif(
        not oracle.available(), reason="bun / raif-standard prototype not available"
    )
except ImportError:
    pass


# ─── Differential assertion ─────────────────────────────────────────────────


def assert_agrees(raifs: list[str], label: str, schema=None) -> None:
    """Decode every input in both engines; raise on the first divergence."""
    js = oracle.js_decode(raifs, schema=schema)
    assert len(js) == len(raifs), (
        f"[{label}] oracle output length {len(js)} != input count {len(raifs)} "
        f"— results would be silently truncated by zip"
    )
    for i, (raif, b) in enumerate(zip(raifs, js, strict=True)):
        p = decode(raif, schema) if schema is not None else decode(raif)
        if p["ok"] != b["ok"]:
            raise AssertionError(
                f"[{label} #{i}] ok-flag: py={p['ok']} js={b['ok']}\n"
                f"  input={raif!r}\n  py_error={p.get('error')!r}"
            )
        if b["ok"] and not oracle.values_equal(p["value"], b["value"]):
            raise AssertionError(
                f"[{label} #{i}] value divergence\n  input={raif!r}\n"
                f"  py ={p['value']!r}\n  js ={b['value']!r}"
            )


def assert_agrees_pairs(pairs: list[tuple], label: str) -> None:
    """Like `assert_agrees`, but each item carries its own schema:
    `pairs = [(raif, schema|None), ...]`. One oracle call for all pairs."""
    js = oracle.js_decode_pairs(pairs)
    assert len(js) == len(pairs), (
        f"[{label}] oracle output length {len(js)} != pair count {len(pairs)} "
        f"— results would be silently truncated by zip"
    )
    for i, ((raif, schema), b) in enumerate(zip(pairs, js, strict=True)):
        p = decode(raif, schema) if schema is not None else decode(raif)
        if p["ok"] != b["ok"]:
            raise AssertionError(
                f"[{label} #{i}] ok-flag: py={p['ok']} js={b['ok']}\n"
                f"  input={raif!r}\n  schema={schema!r}\n  py_error={p.get('error')!r}"
            )
        if b["ok"] and not oracle.values_equal(p["value"], b["value"]):
            raise AssertionError(
                f"[{label} #{i}] value divergence\n  input={raif!r}\n  schema={schema!r}\n"
                f"  py ={p['value']!r}\n  js ={b['value']!r}"
            )


def encode_objects(objs: list, profile: str) -> list[str]:
    """Encode via the oracle, dropping objects the encoder legitimately rejects
    (non-finite numbers, `<<<`/`>>>` in keys)."""
    enc = oracle.js_encode(objs, profile=profile)
    return [e["raif"] for e in enc if e["ok"]]


# ─── Generators ─────────────────────────────────────────────────────────────

# Strings that hit the encoder's wrap-condition branches (and so the decoder's
# unwrap / inference branches): literals-as-strings, delimiters, separators,
# whitespace, newlines, markers, fences.
TRICKY_STRINGS = [
    "",
    " ",
    "  x  ",
    "trailing ",
    " leading",
    "\t",
    "tab\there",
    "a\nb",
    "a\nb\nc",
    "line\n>>>\nmore",
    "\r",
    "a\r\nb",
    ">>>",
    "<<<",
    "<<<x>>>",
    "a>>>b",
    "x=<<<",
    "<<<a,>",
    "true",
    "false",
    "null",
    "123",
    "-0",
    "1.5",
    "1e5",
    "01",
    "1e999",
    "[]",
    "{}",
    "[",
    "]",
    "{a=1}",
    "a,b,c",
    "a:b",
    "key=val",
    "a[0]",
    "ts 14:02:33",
    "emoji 🎉",
    "<raif>",
    "</raif>",
    "```",
    "<|raif_start|>",
    "héllo",
    "naïve",
    "  ",
    "a.b.c",
]

# Key characters the encoder must round-trip (everything except <<< / >>>, which
# it rejects). Includes the path-structural chars that force key wrapping.
KEY_CHARS = "abcXYZ09 .[]=:,{}@-_é🎉"


def rng_string(rng: random.Random) -> str:
    """Random string biased toward encoder wrap-trigger shapes."""
    if rng.random() < 0.55:
        return rng.choice(TRICKY_STRINGS)
    n = rng.randint(0, 8)
    return "".join(rng.choice("ab cZ9.\n>,=:{}[]🎉") for _ in range(n))


def rng_key(rng: random.Random) -> str:
    """Random object key (occasionally empty); never contains the `<<<`/`>>>` delimiters the encoder rejects."""
    if rng.random() < 0.15:
        return ""  # empty key — encoder wraps it
    n = rng.randint(1, 6)
    k = "".join(rng.choice(KEY_CHARS) for _ in range(n))
    # The encoder throws on keys containing the delimiters; keep them out.
    return k.replace("<<<", "x").replace(">>>", "x")


def rng_primitive(rng: random.Random):
    """Random JSON primitive (null/bool/int/float/string)."""
    r = rng.random()
    if r < 0.18:
        return None
    if r < 0.34:
        return rng.choice([True, False])
    if r < 0.55:
        return rng.choice([0, 1, -1, 42, -7, 1000000, 2, 256])
    if r < 0.68:
        return rng.choice([0.5, -1.5, 3.14159, 1e5, -0.0, 100.25])
    return rng_string(rng)


def rng_value(rng: random.Random, depth: int):
    """Random JSON value to the given depth, biased toward table/inline/array-eligible arrays."""
    if depth <= 0 or rng.random() < 0.45:
        return rng_primitive(rng)
    r = rng.random()
    if r < 0.4:
        # Array: bias toward homogeneous-object arrays so table/inline/literal
        # emission modes get exercised.
        n = rng.randint(0, 4)
        if rng.random() < 0.45 and n >= 2:
            cols = [rng_key(rng) for _ in range(rng.randint(1, 3))]
            cols = list(dict.fromkeys(c for c in cols if c))  # unique, non-empty
            if cols:
                return [{c: rng_primitive(rng) for c in cols} for _ in range(n)]
        return [rng_value(rng, depth - 1) for _ in range(n)]
    # Nested object
    return rng_object(rng, depth - 1)


def rng_object(rng: random.Random, depth: int) -> dict:
    """Random JSON object to the given depth."""
    n = rng.randint(0, 5)
    obj: dict = {}
    for _ in range(n):
        obj[rng_key(rng)] = rng_value(rng, depth)
    return obj


def gen_objects(seed: int, count: int, depth: int) -> list[dict]:
    """Deterministic list of random objects for a seed."""
    rng = random.Random(seed)
    return [rng_object(rng, depth) for _ in range(count)]


# ─── Tests (vertical slices; each adds one behavior) ─────────────────────────


def test_tracer_flat_objects_canonical():
    """Tracer bullet: random flat primitive objects survive encode→decode
    identically in both engines."""
    rng = random.Random(1)
    objs = [
        {rng_key(rng): rng_primitive(rng) for _ in range(rng.randint(0, 6))}
        for _ in range(400)
    ]
    raifs = encode_objects(objs, "canonical")
    assert_agrees(raifs, "flat-canonical")


def test_rich_objects_both_profiles():
    """Nested objects, arrays (table/inline/literal modes), and multiline string
    blocks round-trip identically — in both the canonical and generation
    emission profiles (the model is trained on the generation profile)."""
    objs = gen_objects(seed=2, count=1500, depth=4)
    for profile in ("canonical", "generation"):
        raifs = encode_objects(objs, profile)
        assert_agrees(raifs, f"rich-{profile}")


def test_rich_objects_with_markers():
    """The same rich corpus, framed in `<raif>`/`</raif>` mode markers — the
    decoder's marker-stripping pre-pass must not change the value."""
    objs = gen_objects(seed=3, count=600, depth=4)
    enc = oracle.js_encode(objs, profile="generation", markers=True)
    raifs = [e["raif"] for e in enc if e["ok"]]
    assert_agrees(raifs, "markers")


def _valid_corpus(
    seed: int, count: int, depth: int, profile: str, markers=False
) -> list[str]:
    """Encode random objects via the oracle and keep the valid RAIF strings."""
    objs = gen_objects(seed, count, depth)
    enc = oracle.js_encode(objs, profile=profile, markers=markers)
    return [e["raif"] for e in enc if e["ok"]]


def test_truncation_fuzzing():
    """Cut valid RAIF at every line boundary and at random byte offsets — the
    truncation-recovery branches (unterminated block/array closed at EOF,
    missing close marker, sparse-array rejection) must agree with the oracle."""
    rng = random.Random(4)
    base = _valid_corpus(4, 500, 4, "generation", markers=True)
    base += _valid_corpus(40, 500, 4, "canonical")
    mutated: list[str] = []
    for raif in base:
        # every line-boundary prefix
        lines = raif.split("\n")
        for k in range(1, len(lines)):
            mutated.append("\n".join(lines[:k]))
        # a couple of random byte cuts
        if len(raif) > 2:
            for _ in range(2):
                mutated.append(raif[: rng.randint(1, len(raif) - 1)])
    # de-dup and cap to keep the bun batch reasonable
    mutated = list(dict.fromkeys(mutated))
    rng.shuffle(mutated)
    assert_agrees(mutated[:6000], "truncation")


def test_surface_wrapping_mutations():
    """Markdown fences, mode markers (incl. opener-only = truncation), and CRLF
    line endings are surface pre-passes — wrapping valid RAIF in them must not
    change the decoded value, and must match the oracle on the truncation
    signatures too."""
    base = _valid_corpus(5, 800, 3, "generation")
    out: list[str] = []
    for raif in base:
        out.append(f"```\n{raif}\n```")
        out.append(f"```raif\n{raif}\n```")
        out.append(f"<raif>\n{raif}\n</raif>")
        out.append(f"<|raif_start|>\n{raif}\n<|raif_end|>")
        out.append(f"<raif>\n{raif}")  # opener only → missing_close_marker
        out.append(raif.replace("\n", "\r\n"))  # CRLF
        out.append(f"  \n{raif}\n  ")  # leading/trailing whitespace
    assert_agrees(out, "surface-wrapping")


import re as _re  # noqa: E402


def _mutate_line_structural(raif: str, rng: random.Random) -> str:
    """Apply one blind line-level corruption (sep coerce, dup, delimiter/nonce mutate, garbage, shuffle, indent)."""
    lines = raif.split("\n")
    if not lines:
        return raif
    op = rng.choice(
        ["coerce_sep", "dup", "delim_shrink", "nonce", "garbage", "shuffle", "indent"]
    )
    if op == "coerce_sep":
        i = rng.randrange(len(lines))
        lines[i] = lines[i].replace("=", ":", 1)
    elif op == "dup":
        i = rng.randrange(len(lines))
        lines.insert(i, lines[i])
    elif op == "delim_shrink":
        sub = rng.choice([("<<<", "<<"), ("<<<", "<"), (">>>", ">>"), (">>>", ">")])
        raif2 = raif.replace(*sub, 1) if rng.random() < 0.5 else raif.replace(*sub)
        return raif2
    elif op == "nonce":
        # mutate hex right after a <<< opener or >>> closer
        return _re.sub(
            r"(<<<|>>>)([0-9a-fA-F]*)",
            lambda m: (
                m.group(1)
                + (
                    "".join(
                        rng.choice("0123456789abcdef") for _ in range(rng.randint(0, 4))
                    )
                )
            ),
            raif,
            count=1,
        )
    elif op == "garbage":
        i = rng.randrange(len(lines) + 1)
        lines.insert(i, rng_string(rng).replace("\n", " "))
    elif op == "shuffle":
        rng.shuffle(lines)
    elif op == "indent":
        i = rng.randrange(len(lines))
        lines[i] = "  " + lines[i]
    return "\n".join(lines)


def test_structural_line_mutations():
    """Blind line-level corruption of valid RAIF — separator coercion, repeated
    keys, delimiter-count and nonce mutation, garbage-line insertion, line
    reordering, indentation. Outcomes aren't predicted; the property is that
    the Python decoder agrees with the oracle (both accept-and-equal, or both
    reject)."""
    rng = random.Random(6)
    base = _valid_corpus(6, 1200, 4, "generation") + _valid_corpus(
        60, 800, 4, "canonical"
    )
    out: list[str] = []
    for raif in base:
        for _ in range(3):
            out.append(_mutate_line_structural(raif, rng))
    out = list(dict.fromkeys(out))
    rng.shuffle(out)
    assert_agrees(out[:6000], "structural-mutations")


def test_brace_and_relaxed_forms():
    """JSON-style multi-line nested braces (the small-model fallback the
    flattening pre-pass targets) and relaxed `<<`/`<` openers."""
    rng = random.Random(7)
    out: list[str] = []

    def emit(obj: dict, indent: int) -> list[str]:
        """Emit JSON-style multi-line brace text for an object."""
        pad = "  " * indent
        lines: list[str] = []
        for k, v in obj.items():
            if isinstance(v, dict) and v:
                lines.append(f"{pad}{k}={{")
                lines.extend(emit(v, indent + 1))
                lines.append(f"{pad}}}")
            elif (
                isinstance(v, list)
                and v
                and all(not isinstance(x, (dict, list)) for x in v)
            ):
                lines.append(f"{pad}{k}=[")
                for x in v:
                    lines.append(f"{pad}  {x if not isinstance(x, str) else x}")
                lines.append(f"{pad}]")
            else:
                lines.append(
                    f"{pad}{k}={v if not isinstance(v, (dict, list)) else 'x'}"
                )
        return lines

    for obj in gen_objects(seed=7, count=1200, depth=4):
        out.append("\n".join(emit(obj, 0)))
    # relaxed openers
    for _ in range(400):
        nonce = "".join(
            rng.choice("0123456789abcdef") for _ in range(rng.randint(0, 3))
        )
        body = "\n".join(
            rng_string(rng).replace("\n", " ") for _ in range(rng.randint(0, 3))
        )
        op = rng.choice(["<<", "<"])
        cl = rng.choice([">>>", ">>", ">"])
        out.append(f"k={op}{nonce}\n{body}\n{cl}{nonce}")
    out = [s for s in dict.fromkeys(out) if s]
    assert_agrees(out, "brace-relaxed")


def test_encode_gate_unencodable_keys():
    """Regression for the encode round-trip gate: TS `decode` re-encodes the
    assembled value and propagates the throw, so a *bare* key segment that
    contains `<<<`/`>>>` makes decode fail even though parsing succeeded. Found
    by `test_structural_line_mutations`; pinned here deterministically. Mixed
    with clean controls (keys with `<<`/`<`/`>` but no full delimiter, which
    must still decode)."""
    cases = [
        # — assembled key contains >>> → reject (encode gate) —
        "<<>>>=1",
        "a>>>b=2",
        "x.y>>>=3",
        "rows::>>>a\nrows[0]=1",
        "<<>>>.<<<>>>:s=ok",  # the shape the fuzzer surfaced
        # — assembled key contains <<< via a non-opener spelling → reject —
        "k=v\nweird>>>key=2",
        # — clean controls: partial delimiters in keys are encodable → accept —
        "<<a=1",
        "a<b=2",
        "a>b=3",
        ">>a=4",
        "plain.key=5",
        "<<<wrapped key>>>=6",  # wrapped → unwraps to clean key
        # — value-side delimiters are always fine (only keys gate) —
        "a=x>>>y\nb=<<<z>>>",
    ]
    assert_agrees(cases, "encode-gate")


# — schema-typed decode —


def _simple_key(rng: random.Random) -> str:
    """Short lowercase key for schema-friendly objects."""
    return "".join(rng.choice("abcdefgh") for _ in range(rng.randint(1, 4)))


def _typed_value(rng: random.Random, depth: int):
    """Random value for a schema-derivable object (homogeneous arrays, one nesting level)."""
    r = rng.random()
    if r < 0.18:
        return None
    if r < 0.34:
        return rng.choice([True, False])
    if r < 0.5:
        return rng.choice([0, 1, -3, 42, 256])
    if r < 0.62:
        return rng.choice([0.5, -1.5, 3.25])
    if r < 0.78:
        return rng.choice(
            ["hi", "a value", "", "with,comma", "x:y", "tab\tx", "emoji🎉"]
        )
    if depth <= 0:
        return rng.choice(["leaf", 7, True])
    if r < 0.88:
        # homogeneous array (primitives or primitive-objects)
        n = rng.randint(0, 3)
        if rng.random() < 0.5 and n >= 2:
            cols = list(
                dict.fromkeys(_simple_key(rng) for _ in range(rng.randint(1, 2)))
            )
            return [{c: _typed_prim(rng) for c in cols} for _ in range(n)]
        return [_typed_prim(rng) for _ in range(n)]
    # nested object
    return {
        _simple_key(rng): _typed_value(rng, depth - 1) for _ in range(rng.randint(0, 3))
    }


def _typed_prim(rng: random.Random):
    """Random primitive cell for typed objects."""
    return rng.choice([None, True, False, 0, 5, -2, 1.5, "hi", "x,y", ""])


def _gen_typed_object(rng: random.Random) -> dict:
    """Random object with simple keys, used to derive a matching schema."""
    keys = list(dict.fromkeys(_simple_key(rng) for _ in range(rng.randint(1, 5))))
    return {k: _typed_value(rng, 2) for k in keys}


def _elem_type(v) -> str:
    """Schema type letter (s/n/b/o) for an array element."""
    if isinstance(v, bool):
        return "b"
    if isinstance(v, (int, float)):
        return "n"
    if isinstance(v, str):
        return "s"
    return "o"


def _derive_schema(path: str, v, lines: list[str]) -> None:
    """Emit schema declaration lines matching a typed object's structure."""
    if v is None:
        lines.append(f"{path}:s?")
    elif isinstance(v, bool):
        lines.append(f"{path}:b")
    elif isinstance(v, (int, float)):
        lines.append(f"{path}:n")
    elif isinstance(v, str):
        lines.append(f"{path}:s")
    elif isinstance(v, list):
        if not v:
            lines.append(f"{path}:o?")
        elif isinstance(v[0], dict):
            for kk in v[0]:
                _derive_schema(f"{path}[].{kk}", v[0][kk], lines)
        else:
            lines.append(f"{path}[]:{_elem_type(v[0])}")
    elif isinstance(v, dict):
        if not v:
            lines.append(f"{path}:o?")
        else:
            for kk in v:
                _derive_schema(f"{path}.{kk}", v[kk], lines)


def _perturb_schema(decl: str, rng: random.Random) -> str:
    """Mutate a derived schema (flip type, drop/optional/add field, open node) to exercise accept/reject branches."""
    lines = [ln for ln in decl.split("\n") if ln]
    if not lines:
        return decl
    op = rng.choice(["fliptype", "drop", "optional_all", "add_required", "to_open"])
    if op == "fliptype":
        i = rng.randrange(len(lines))
        for a, b in (("s", "n"), ("n", "s"), ("b", "s"), ("s", "b")):
            if lines[i].rstrip("?").endswith(":" + a):
                lines[i] = lines[i].replace(":" + a, ":" + b)
                break
    elif op == "drop" and len(lines) > 1:
        del lines[rng.randrange(len(lines))]
    elif op == "optional_all":
        lines = [ln if ln.endswith("?") else ln + "?" for ln in lines]
    elif op == "add_required":
        lines.append("zzz_absent:s")
    elif op == "to_open":
        i = rng.randrange(len(lines))
        lines[i] = lines[i].split(":")[0] + ":o?"
    return "\n".join(lines)


def test_schema_typed_decode():
    """Schema-typed decode (ADR-0019): generate a typed object, encode it, and
    decode the RAIF under (a) the exactly-derived schema and (b) perturbed
    schemas (wrong type, dropped field → unknown-field, all-optional, extra
    required → missing, open node). Each is decoded with the schema in both
    engines and must agree."""
    rng = random.Random(8)
    objs = [_gen_typed_object(rng) for _ in range(1800)]
    pairs: list[tuple] = []
    for profile in ("generation", "canonical"):
        enc = oracle.js_encode(objs, profile=profile)
        assert len(enc) == len(objs), "oracle encode count != object count"
        for obj, e in zip(objs, enc, strict=True):
            if not e["ok"]:
                continue
            raif = e["raif"]
            lines: list[str] = []
            for k, v in obj.items():
                _derive_schema(k, v, lines)
            base = "\n".join(lines)
            for sch in (base, _perturb_schema(base, rng), _perturb_schema(base, rng)):
                pairs.append((raif, sch))
    assert_agrees_pairs(pairs, "schema")


def test_pure_garbage():
    """Random character soup the encoder would never emit. Most of it should be
    rejected by both engines; whatever one accepts, the other must accept with
    the same value. Catches divergence at the parser's accept/reject boundary."""
    rng = random.Random(9)
    # An alphabet dense in structurally-significant characters.
    # Includes Unicode classes that differ between JS and Python: exotic
    # whitespace (NEL, FS, BOM, NBSP), a non-ASCII digit, and an astral char.
    alpha = "ab12 .=:,{}[]<>\n\t/-_\r🎉é\x85\x1c\ufeff\xa0١𐍊"
    out: list[str] = []
    for _ in range(8000):
        n = rng.randint(0, 40)
        out.append("".join(rng.choice(alpha) for _ in range(n)))
    # Plus dense delimiter/separator runs that stress the openers and splitters.
    tokens = [
        "<<<",
        ">>>",
        "<<",
        ">>",
        "=",
        "::",
        ":s=",
        ":n=",
        "=[",
        "={",
        "]",
        "}",
        "\n",
        "a",
        "1",
        ",",
        "[0]",
        "<raif>",
        "</raif>",
        "```",
    ]
    for _ in range(4000):
        n = rng.randint(1, 12)
        out.append("".join(rng.choice(tokens) for _ in range(n)))
    out = list(dict.fromkeys(out))
    assert_agrees(out, "garbage")


def test_line_terminator_semantics():
    """Regression for JS-vs-Python `.` semantics: an interior CR / U+2028 /
    U+2029 is a line terminator to JS regex `.` (so `^(.+)=[$` etc. won't span
    it) but not to Python's. A line carrying one before an opener tail must
    decode as a plain leaf, not as an array/brace/multiline opener. Found by
    `test_pure_garbage`; pinned here."""
    cases = [
        "x\r=[",  # not an array opener → value "["
        "x\r={",  # not a brace opener
        "k\r=<<<",  # not a multiline opener → value "<<<"
        "k\r=<<",  # not a relaxed opener
        "a.b\u2028c=[",  # U+2028 interior
        "a.b\u2029c=[",  # U+2029 interior
        "é{/,éb.é🎉>\r=[",  # the exact string the fuzzer surfaced
        # controls: a *trailing* CR is structural and IS stripped, so these ARE
        # openers (rows/closer follow) — must still match the oracle.
        "xs=[\r\n1\r\n2\r\n]",
        "doc=<<<\r\nbody\r\n>>>",
    ]
    assert_agrees(cases, "line-terminators")


def test_unicode_whitespace_and_numbers():
    """Regressions from the Codex audit. JS and Python disagree on (1) the
    whitespace set used by trims/blank-checks — Python strips NEL/C0-separators
    but not the BOM, JS the reverse; (2) `\\d` — Python matches Unicode digits,
    JS is ASCII-only; (3) number semantics — JS doubles lose precision and
    reject overflow, Python ints don't. All must now agree with the oracle."""
    cases = [
        # (1) whitespace set — leading/trailing
        "\x85a=1",
        "\x1ca=1",
        "\x1da=1",
        "\x1ea=1",
        "\x1fa=1",  # py-only ws → keep as data
        "﻿a=1",
        "a=1﻿",  # BOM: js-only ws
        "\xa0a=1",
        " a=1 ",
        "\ta=1\t",  # shared ws
        "\x85",
        "﻿",
        "   ",  # whole-doc blank variants
        # (2) non-ASCII digits must NOT parse as numbers / indices
        "a=1١",
        "a:n=1١",
        "a[1١]=x",
        "ts::c\nts[1١]=9",
        "a=١",
        # (3) double precision + overflow
        "a=9007199254740993",
        "a=9007199254740992",
        "a=1e308",
        "a=" + "9" * 400,
        "a=1e309",
        "a:n=" + "9" * 400,  # overflow → both reject
        "a=-0",
        "a=0",
        "a=00",
        "a=1.0",
        "a=1e0",
        # combined: BOM-led fence, exotic ws around leaves
        "﻿```\na=1\nb=2\n```",
        "\x85a=1\x85\nb=2",
    ]
    assert_agrees(cases, "unicode-number")


def test_repeated_key_detail_utf16_order():
    """Finding 4 (repairs, which the oracle comparison doesn't cover): the
    `repeated_keys_indexed` detail is sorted in JS `Array.sort` order = UTF-16
    code units, NOT Python code points. For an astral key (😀, surrogate lead
    0xD83D) vs a BMP key (U+E000) the two orders differ; pin the UTF-16 one."""
    r = decode("😀=1\n😀=2\n=3\n=4")
    detail = next(
        x["detail"] for x in r["repairs"] if x["kind"] == "repeated_keys_indexed"
    )
    # UTF-16: 😀 (0xD83D…) sorts before  (0xE000); code points would reverse it.
    assert detail == "😀,", [hex(ord(c)) for c in detail]


if __name__ == "__main__":
    if not oracle.available():
        print("SKIP: bun / raif-standard not available.")
        raise SystemExit(0)
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL  {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} differential properties hold.")
    raise SystemExit(1 if failed else 0)
