"""Pure-Python RAIF v0.4 decoder — a faithful port of `decode` / `decodeLenient`
from `raif-standard/prototype/src/raif.ts`.

Why this exists: the model emits RAIF, not JSON, so every consumer needs one
decode step at the output boundary. The canonical decoder is the TypeScript
`decode()` in `raif-standard`; the eval shells out to it through `bun` so its
fidelity numbers come from ground truth. This module is the drop-in for Python
harnesses that don't want a `bun` subprocess in the hot path.

Scope: the *decoder* only. The encoder (canonical re-emission) is not ported —
`decode` projects RAIF to a JSON value and never needs it. `fix`/`validate`,
which do re-encode, are therefore out of scope here. Everything the decoder
relies on — the full repair pipeline (markdown fences, mode markers, CRLF,
multi-line brace flattening, nonce-block recovery, relaxed openers, array
literals, tables, inline objects, repeated-key indexing, sparse-array
rejection) and optional schema-typed decoding — is ported.

Parity with the canonical decoder is pinned by `src/test_raif_decode.py`, which
runs both decoders over the real corpus and compares.

Public API mirrors the TS result shapes so this is a drop-in replacement:

    >>> from raif_decode import decode
    >>> decode("a=1\\nb=hi")
    {'ok': True, 'value': {'a': 1, 'b': 'hi'}, 'repairs': []}

`decode_lenient` is the per-leaf recovery variant (never raises; bad leaves are
skipped and reported), matching `decodeLenient`.
"""

from __future__ import annotations

import math
import re
from typing import Any

__all__ = [
    "decode",
    "decode_lenient",
    "parse_schema",
    "RaifSchema",
    "SchemaNode",
    "RaifError",
]

OPEN = "<<<"
CLOSE = ">>>"


class RaifError(Exception):
    """Decode error. `decode` catches these and returns `{ok: False, ...}`;
    they surface raw from the internal helpers, exactly like the TS throws."""


def _strip_cr(line: str) -> str:
    """Strip a single trailing carriage return (mirrors TS `stripCR`)."""
    return line[:-1] if line.endswith("\r") else line


# ECMAScript WhiteSpace + LineTerminator code points — the set JS `String.trim`
# and regex `\s` use. Python's `str.strip()` / `\s` use Unicode whitespace,
# which DIFFERS in both directions: Python strips NEL (U+0085) and the C0
# separators (U+001C–U+001F) that JS leaves as data, and JS strips the BOM
# (U+FEFF) that Python leaves as data. Decode structure trims and blank-line
# tests must use this set, not Python's, to match the canonical decoder.
_JS_WS = "".join(
    map(
        chr,
        [
            0x09,
            0x0A,
            0x0B,
            0x0C,
            0x0D,
            0x20,
            0xA0,
            0x1680,
            0x2000,
            0x2001,
            0x2002,
            0x2003,
            0x2004,
            0x2005,
            0x2006,
            0x2007,
            0x2008,
            0x2009,
            0x200A,
            0x2028,
            0x2029,
            0x202F,
            0x205F,
            0x3000,
            0xFEFF,
        ],
    )
)
# Same set as a regex character-class body (for patterns that used `\s`).
_WS = "[\t\n\x0b\x0c\r \xa0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000\ufeff]"


def _js_strip(s: str) -> str:
    """`str.strip` over the ECMAScript whitespace set (mirrors JS `.trim()` /
    `text.replace(/^\\s+|\\s+$/g, "")`)."""
    return s.strip(_JS_WS)


# JSON number grammar (spec §3). Shared by inference and schema-typed decode.
# `re.ASCII` so `\d` matches ONLY ASCII 0-9 — JS `\d` is ASCII-only, but Python's
# default `\d` also accepts Unicode decimal digits (e.g. ARABIC-INDIC U+0661),
# which would let `1١` slip through as a number.
NUMBER_RE = re.compile(r"^-?(0|[1-9]\d*)(\.\d+)?([eE][+-]?\d+)?$", re.ASCII)


def _parse_number(raw: str) -> Any:
    """Mirror JS `JSON.parse` over a NUMBER_RE-matching string. JS numbers are
    IEEE-754 doubles, so every value is routed through `float` to reproduce JS's
    precision loss (e.g. `9007199254740993` → `…992`) and overflow rejection
    (`>` double range → error, as TS's `Number.isFinite` guard does). Integral
    finite results are returned as `int` to match JS's canonical form
    (`JSON.stringify(5.0) === "5"`); fractional results stay `float`."""
    f = float(raw)
    if not math.isfinite(f):
        raise RaifError(f"number out of double range: {raw}")
    if f.is_integer():
        return int(f)
    return f


# ─── Decoder entry points ───────────────────────────────────────────────────


def decode(text: str, schema: Any | None = None) -> dict:
    """RAIF → JSON. Returns `{"ok": True, "value", "repairs"}` on success or
    `{"ok": False, "error", "repairs"}` on an unrepairable input. `repairs` is
    a list of `{"kind", "detail"?}` dicts — the same shape the TS decoder emits
    and what the eval counts. `schema` is an optional schema declaration string
    or a parsed schema (`parse_schema(...)`). `text` is the raw RAIF string."""
    repairs: list[dict] = []
    try:
        root = _pipeline(text, repairs, _to_schema(schema))
        return {"ok": True, "value": root, "repairs": repairs}
    except RaifError as e:
        return {"ok": False, "error": str(e), "repairs": repairs}


def decode_lenient(text: str, schema: Any | None = None) -> dict:
    """Per-leaf recovery. Never raises: bad leaves are skipped and reported in
    `errors`; every leaf that parses lands in `value`. Returns
    `{"value", "errors", "repairs", "truncated"}`. `truncated` is True when the
    input carries a truncation signature (open `<raif>` with no closer, or a
    block/array closed at EOF). `text` is the raw RAIF string."""
    repairs: list[dict] = []
    errors: list[dict] = []
    text = _prepare_text(text, repairs)
    leaves = _parse_leaves(text, repairs, errors)
    leaves = _repair_repeated_keys(leaves, repairs)
    value = _assemble(leaves, repairs, errors, _to_schema(schema))
    truncated = any(r["kind"] in _TRUNCATION_REPAIRS for r in repairs)
    return {
        "value": value,
        "errors": errors,
        "repairs": repairs,
        "truncated": truncated,
    }


_TRUNCATION_REPAIRS = {
    "missing_close_marker",
    "unterminated_block_closed_at_eof",
    "unterminated_array_closed_at_eof",
}


def _pipeline(text: str, repairs: list[dict], schema: SchemaNode | None) -> dict:
    """Strict decode pipeline (prepare -> parse -> repair keys -> assemble), then the encode round-trip throw-gate. Returns the JSON value."""
    text = _prepare_text(text, repairs)
    leaves = _parse_leaves(text, repairs, None)
    leaves = _repair_repeated_keys(leaves, repairs)
    root = _assemble(leaves, repairs, None, schema)
    # TS `decode` runs through `fixInternal`, which computes `encode(json)` for
    # the canonical form and *propagates any throw* — so a value the encoder
    # rejects fails decode even though the canonical string is discarded. The
    # only encoder rejection reachable from a decode-produced value is a dict
    # key containing `<<<`/`>>>` (the root is always an object and all numbers
    # are already finite). `decodeLenient` does NOT re-encode, so this gate is
    # applied only here, on the strict path.
    _check_encodable(root)
    return root


def _check_encodable(value: Any) -> None:
    """Mirror the throw behavior of the encoder's `needsKeyQuoting` over the
    assembled value. Traversal order (code-point-sorted keys, pre-order DFS)
    matches the encoder's `walk`, so the surfaced key matches too."""
    if isinstance(value, dict):
        for k in sorted(
            value.keys()
        ):  # Python str order == TS compareUtf8 (code points)
            if OPEN in k or CLOSE in k:
                raise RaifError(
                    f"key contains <<< or >>> which is unsupported in this prototype: {k}"
                )
            _check_encodable(value[k])
    elif isinstance(value, list):
        for v in value:
            _check_encodable(v)


# ─── Surface pre-passes (prepare_text) ──────────────────────────────────────

_FENCE_RE = re.compile(rf"\A{_WS}*```[a-zA-Z]*\n(.*?)\n```{_WS}*\Z", re.S)
_MODE_OPEN_EDGE_RE = re.compile(rf"^{_WS}*(?:<\|raif_start\|>|<raif>)(?:\n|$)")
_MODE_CLOSE_EDGE_RE = re.compile(rf"(?:^|\n)(?:<\|raif_end\|>|</raif>){_WS}*$")
_BRACE_TAIL_RE = re.compile(rf"=\{{{_WS}*$", re.M)


def _prepare_text(text: str, repairs: list[dict]) -> str:
    """Run the surface pre-passes (fence strip, CRLF normalize, mode-marker strip, JS-trim, brace flatten) before leaf parsing."""
    stripped = _strip_markdown_fences(text)
    if stripped != text:
        repairs.append({"kind": "markdown_stripped"})
        text = stripped
    lines = text.split("\n")
    if len(lines) > 1 and all(line.endswith("\r") for line in lines[:-1]):
        repairs.append({"kind": "line_endings_normalized"})
        text = "\n".join(_strip_cr(line) for line in lines)
    text = _strip_mode_markers(text, repairs)
    text = _js_strip(text)
    text = _flatten_multiline_braces(text, repairs)
    return text


def _strip_markdown_fences(text: str) -> str:
    """Unwrap a single fenced code block if the whole text is one ```...``` block."""
    m = _FENCE_RE.match(text)
    return m.group(1) if m else text


def _strip_mode_markers(text: str, repairs: list[dict]) -> str:
    """Strip edge `<raif>`/`</raif>` (or special-token) framing; record the truncation signal when only the opener is present."""
    opened = bool(_MODE_OPEN_EDGE_RE.search(text))
    closed = bool(_MODE_CLOSE_EDGE_RE.search(text))
    if not opened and not closed:
        return text
    repairs.append({"kind": "mode_markers_stripped"})
    if opened and not closed:
        repairs.append({"kind": "missing_close_marker"})
    text = _MODE_OPEN_EDGE_RE.sub("", text, count=1)
    text = _MODE_CLOSE_EDGE_RE.sub("", text, count=1)
    return text


# ─── Multi-line brace flattening (JSON-style nested objects → path leaves) ───

# JS regex `.` (no `s` flag) matches any char EXCEPT the ECMAScript line
# terminators LF, CR, LS, PS. Python `.` excludes only `\n`, so a bare `.`
# would wrongly span an interior `\r` (or LS/PS) inside a line — lines are
# split on `\n` only). `_NN` is the JS-equivalent "any non-line-terminator".
_NN = r"[^\n\r\u2028\u2029]"

_NONCE_OPENER_RE = re.compile(rf"^({_NN}*?)=<<<([0-9a-fA-F]*)$")
_ARRAY_OPENER_RE = re.compile(rf"^({_NN}+)=\[$")
_BRACE_OPENER_LINE = re.compile(rf"^({_NN}+)=\{{$")


def _flatten_multiline_braces(text: str, repairs: list[dict]) -> str:
    """Rewrite JSON-style multi-line `key={ ... }` blocks into path-mode leaves (the small-model fallback the spec doesn't define)."""
    if not _BRACE_TAIL_RE.search(text):
        return text
    lines = text.split("\n")
    out_lines, changed = _flatten_brace_block(lines, 0, len(lines), "")
    if changed:
        repairs.append({"kind": "multiline_braces_flattened"})
    return "\n".join(out_lines)


def _multiline_block_end(lines: list[str], idx: int, end: int, nonce: str) -> int:
    """Line index of the `>>>NONCE` closer for the block opened at `idx`, or `end` on truncation."""
    for j in range(idx + 1, end):
        if _strip_cr(lines[j]) == f"{CLOSE}{nonce}":
            return j
    return end


def _flatten_brace_block(
    lines: list[str], start: int, end: int, prefix: str
) -> tuple[list[str], bool]:
    """Recursively flatten one balanced brace region, prefixing nested keys; passes array-literal and multiline blocks through verbatim."""
    out: list[str] = []
    changed = False
    arr_depth = 0
    i = start
    while i < end:
        raw = lines[i]
        trimmed = _js_strip(_strip_cr(raw))

        if arr_depth > 0:
            out.append(raw)
            if trimmed == "]":
                arr_depth -= 1
            elif _ARRAY_OPENER_RE.match(trimmed):
                arr_depth += 1
            i += 1
            continue

        ml_open = _NONCE_OPENER_RE.match(trimmed)
        if ml_open:
            block_end = _multiline_block_end(lines, i, end, ml_open.group(2))
            out.append(f"{prefix}.{trimmed}" if prefix else raw)
            for k in range(i + 1, min(block_end + 1, end)):
                out.append(lines[k])
            i = min(block_end + 1, end)
            continue

        brace_open = _BRACE_OPENER_LINE.match(trimmed)
        if brace_open:
            inner_key = _js_strip(brace_open.group(1))
            full_key = f"{prefix}.{inner_key}" if prefix else inner_key
            depth = 1
            inner_arr = 0
            j = i + 1
            while j < end:
                t = _js_strip(_strip_cr(lines[j]))
                t_open = _NONCE_OPENER_RE.match(t) if inner_arr == 0 else None
                if inner_arr > 0:
                    if t == "]":
                        inner_arr -= 1
                    elif _ARRAY_OPENER_RE.match(t):
                        inner_arr += 1
                elif t_open:
                    j = _multiline_block_end(lines, j, end, t_open.group(2))
                    if j >= end:
                        break
                elif t == "}":
                    depth -= 1
                    if depth == 0:
                        break
                elif _BRACE_OPENER_LINE.match(t):
                    depth += 1
                elif _ARRAY_OPENER_RE.match(t):
                    inner_arr = 1
                j += 1
            if depth != 0:
                out.append(raw)
                i += 1
                continue
            inner_lines, _ = _flatten_brace_block(lines, i + 1, j, full_key)
            out.extend(inner_lines)
            changed = True
            i = j + 1
            continue

        arr_open = _ARRAY_OPENER_RE.match(trimmed)
        if arr_open:
            out.append(f"{prefix}.{trimmed}" if prefix else raw)
            arr_depth = 1
            i += 1
            continue

        if len(trimmed) == 0:
            out.append(raw)
        elif prefix:
            out.append(f"{prefix}.{trimmed}")
        else:
            out.append(raw)
        i += 1
    return out, changed


# ─── Leaf parsing ───────────────────────────────────────────────────────────

# A leaf body is a dict with a "kind" tag:
#   {"kind": "bare", "raw": str}
#   {"kind": "typed", "type": "s"|"n"|"b"|"t", "raw": str}
#   {"kind": "multiline", "raw": str}
#   {"kind": "table_header", "cols": list[str]}
#   {"kind": "table_row", "cells": list[str]}
#   {"kind": "array_literal", "rows": list[str]}
# A parsed leaf is {"key": str, "body": <body>}.

_RELAXED_OPENER_RE = re.compile(rf"^({_NN}*?)=(<{{1,2}})([0-9a-fA-F]*)$")
_LEAF_LIKE = re.compile(r"^[^=:]*(=|::|:[sntb]=)")
_RELAXED_CLOSER_RE = re.compile(r"^(>{1,3})([0-9a-fA-F]*)$")
_CLOSER_CANDIDATE_RE = re.compile(r"^>>>[0-9a-fA-F]*$")


def _parse_leaves(
    text: str, repairs: list[dict], lenient_errors: list[dict] | None
) -> list[dict]:
    """Parse prepared text into leaves (bare/typed/multiline/table/array), running the truncation/nonce/delimiter recovery ladders. Collects per-leaf errors when `lenient_errors` is given, else raises."""
    lines = text.split("\n")
    leaves: list[dict] = []
    table_cols: dict[str, list[str]] = {}
    cr_repaired = [False]

    def structural(raw: str) -> str:
        """Strip a trailing CR from a structural line, recording the line-ending repair once."""
        s = _strip_cr(raw)
        if s != raw and not cr_repaired[0]:
            cr_repaired[0] = True
            repairs.append({"kind": "line_endings_normalized"})
        return s

    i = 0
    n = len(lines)
    while i < n:
        line = structural(lines[i])
        if _js_strip(line) == "":
            i += 1
            continue

        arr_opener = _ARRAY_OPENER_RE.match(line)
        if arr_opener:
            key = arr_opener.group(1)
            rows: list[str] = []
            j = i + 1
            closed = False
            while j < n:
                row_line = structural(lines[j])
                if row_line == "]":
                    closed = True
                    break
                if _js_strip(row_line) != "":
                    rows.append(row_line)
                j += 1
            if not closed:
                repairs.append(
                    {
                        "kind": "unterminated_array_closed_at_eof",
                        "detail": f"key '{key}'",
                    }
                )
            leaves.append({"key": key, "body": {"kind": "array_literal", "rows": rows}})
            i = j + 1 if closed else j
            continue

        opener = _NONCE_OPENER_RE.match(line)
        if opener:
            key = opener.group(1)
            nonce = opener.group(2)
            closer = f"{CLOSE}{nonce}"
            content: list[str] = []
            j = i + 1
            while j < n and _strip_cr(lines[j]) != closer:
                content.append(lines[j])
                j += 1
            if j < n:
                structural(lines[j])  # record CR repair if the closer carried one
                leaves.append(
                    {
                        "key": key,
                        "body": {"kind": "multiline", "raw": "\n".join(content)},
                    }
                )
                i = j + 1
                continue
            # No exact closer — recovery ladder.
            candidates = _closer_candidates(lines, i + 1)
            recovered: int | None = None
            if len(candidates) == 1:
                recovered = candidates[0]
            elif len(candidates) > 1 and len(nonce) > 0:
                same_len = [
                    k
                    for k in candidates
                    if len(_strip_cr(lines[k])) == len(CLOSE) + len(nonce)
                ]
                if len(same_len) == 1:
                    recovered = same_len[0]
            if recovered is not None:
                repairs.append(
                    {"kind": "mismatched_nonce_recovered", "detail": f"nonce {nonce}"}
                )
                body = [lines[k] for k in range(i + 1, recovered)]
                leaves.append(
                    {"key": key, "body": {"kind": "multiline", "raw": "\n".join(body)}}
                )
                i = recovered + 1
                continue
            relaxed = _find_relaxed_closer(lines, i, nonce)
            if relaxed is not None:
                repairs.append(
                    {
                        "kind": "delimiter_count_repaired",
                        "detail": f"opener=<×3 closer=>×{relaxed[1]}",
                    }
                )
                body = [lines[k] for k in range(i + 1, relaxed[0])]
                leaves.append(
                    {"key": key, "body": {"kind": "multiline", "raw": "\n".join(body)}}
                )
                i = relaxed[0] + 1
                continue
            if len(candidates) == 0:
                repairs.append(
                    {
                        "kind": "unterminated_block_closed_at_eof",
                        "detail": f"key '{key}'",
                    }
                )
                leaves.append(
                    {
                        "key": key,
                        "body": {"kind": "multiline", "raw": "\n".join(content)},
                    }
                )
                i = n
                continue
            msg = (
                f"unterminated multiline block at line {i + 1} "
                f"(nonce {nonce or '<none>'}): ambiguous closers"
            )
            if lenient_errors is None:
                raise RaifError(msg)
            lenient_errors.append({"line": i + 1, "key": key, "error": msg})
            i += 1
            continue

        relaxed_opener = _RELAXED_OPENER_RE.match(line)
        if relaxed_opener:
            opener_count = len(relaxed_opener.group(2))
            found = _find_relaxed_closer(lines, i, relaxed_opener.group(3))
            if found is not None:
                between = lines[i + 1 : found[0]]
                looks_like_content = any(not _LEAF_LIKE.match(bl) for bl in between)
                if looks_like_content or len(between) == 0:
                    repairs.append(
                        {
                            "kind": "delimiter_count_repaired",
                            "detail": f"opener=<×{opener_count} closer=>×{found[1]}",
                        }
                    )
                    leaves.append(
                        {
                            "key": relaxed_opener.group(1),
                            "body": {"kind": "multiline", "raw": "\n".join(between)},
                        }
                    )
                    i = found[0] + 1
                    continue

        try:
            leaves.append(_parse_single_line_leaf(line, i + 1, table_cols, repairs))
        except RaifError as e:
            if lenient_errors is None:
                raise
            lenient_errors.append({"line": i + 1, "error": str(e)})
        i += 1
    return leaves


def _find_relaxed_closer(
    lines: list[str], opener_idx: int, nonce: str
) -> tuple[int, int] | None:
    """Find a downstream `>{1,3}NONCE` closer for a relaxed/short opener; returns (line index, closer length) or None."""
    for j in range(opener_idx + 1, len(lines)):
        cm = _RELAXED_CLOSER_RE.match(_strip_cr(lines[j]))
        if cm and cm.group(2) == nonce:
            return (j, len(cm.group(1)))
    return None


def _closer_candidates(lines: list[str], frm: int) -> list[int]:
    """Indices of downstream lines shaped like a `>>>hex` closer (mismatched-nonce recovery candidates)."""
    return [
        k
        for k in range(frm, len(lines))
        if _CLOSER_CANDIDATE_RE.match(_strip_cr(lines[k]))
    ]


_TYPED_RE = re.compile(r"^([sntb])=(.*)$", re.S)
_INDEXED_KEY_RE = re.compile(rf"^({_NN}+)\[\d+\]$", re.ASCII)


def _parse_single_line_leaf(
    line: str, line_num: int, table_cols: dict[str, list[str]], repairs: list[dict]
) -> dict:
    """Parse one single-line leaf into key + body, locating the first top-level separator (`=`/`:`/`::`)."""
    i = 0
    sep_index = -1
    sep_char = ""
    is_double_colon = False
    length = len(line)
    while i < length:
        if line.startswith(OPEN, i):
            end = line.find(CLOSE, i + len(OPEN))
            if end == -1:
                break
            i = end + len(CLOSE)
            continue
        c = line[i]
        if c == "=":
            sep_index = i
            sep_char = "="
            break
        if c == ":":
            if i + 1 < length and line[i + 1] == ":":
                sep_index = i
                sep_char = ":"
                is_double_colon = True
            else:
                sep_index = i
                sep_char = ":"
            break
        i += 1
    if sep_index == -1:
        raise RaifError(f"no separator in leaf at line {line_num}: {line}")
    key = line[:sep_index]
    rest = line[sep_index + 2 :] if is_double_colon else line[sep_index + 1 :]

    if is_double_colon:
        cols = _split_top_level_commas(rest)
        table_cols[key] = cols
        return {"key": key, "body": {"kind": "table_header", "cols": cols}}

    if sep_char == ":":
        typed = _TYPED_RE.match(rest)
        if typed:
            return {
                "key": key,
                "body": {
                    "kind": "typed",
                    "type": typed.group(1),
                    "raw": typed.group(2),
                },
            }
        # Stray `:` used as the KV separator — coerce when unambiguous (the
        # suffix does not look like a typed-leaf prefix).
        repairs.append(
            {"kind": "separator_coerced", "detail": f"':' → '=' at line {line_num}"}
        )
        table_row = _INDEXED_KEY_RE.match(key)
        if table_row and table_row.group(1) in table_cols:
            cells = _split_top_level_commas(rest)
            return {"key": key, "body": {"kind": "table_row", "cells": cells}}
        return {"key": key, "body": {"kind": "bare", "raw": rest}}

    # sep_char == "=". Table-mode row when key is `prefix[N]` with a header.
    table_row = _INDEXED_KEY_RE.match(key)
    if table_row and table_row.group(1) in table_cols:
        cells = _split_top_level_commas(rest)
        return {"key": key, "body": {"kind": "table_row", "cells": cells}}

    return {"key": key, "body": {"kind": "bare", "raw": rest}}


def _split_top_level_commas(s: str) -> list[str]:
    """Split on commas at top level, skipping `<<<...>>>` ranges and `{...}` nesting."""
    out: list[str] = []
    start = 0
    i = 0
    brace_depth = 0
    length = len(s)
    while i < length:
        if s.startswith(OPEN, i):
            end = s.find(CLOSE, i + len(OPEN))
            if end == -1:
                break
            i = end + len(CLOSE)
            continue
        c = s[i]
        if c == "{":
            brace_depth += 1
            i += 1
            continue
        if c == "}":
            if brace_depth > 0:
                brace_depth -= 1
            i += 1
            continue
        if c == "," and brace_depth == 0:
            out.append(s[start:i])
            start = i + 1
        i += 1
    out.append(s[start:])
    return out


# ─── Repeated-key auto-indexing (TIER 2-B) ──────────────────────────────────


def _repair_repeated_keys(leaves: list[dict], repairs: list[dict]) -> list[dict]:
    """TIER 2-B: rewrite repeated same-scope keys to `key[0]`, `key[1]`, ... when no indexed/table form already claims the prefix."""
    key_count: dict[str, int] = {}
    indexed_prefixes: set[str] = set()
    table_header_keys: set[str] = set()
    for leaf in leaves:
        if leaf["body"]["kind"] == "table_header":
            table_header_keys.add(leaf["key"])
            continue
        m = _INDEXED_KEY_RE.match(leaf["key"])
        if m:
            indexed_prefixes.add(m.group(1))
        key_count[leaf["key"]] = key_count.get(leaf["key"], 0) + 1
    repeated_keys: set[str] = set()
    for key, count in key_count.items():
        if count < 2:
            continue
        if key in indexed_prefixes:
            continue
        if key in table_header_keys:
            continue
        repeated_keys.add(key)
    if not repeated_keys:
        return leaves
    index_counter: dict[str, int] = {}
    out: list[dict] = []
    for leaf in leaves:
        if leaf["key"] not in repeated_keys:
            out.append(leaf)
            continue
        idx = index_counter.get(leaf["key"], 0)
        index_counter[leaf["key"]] = idx + 1
        out.append({"key": f"{leaf['key']}[{idx}]", "body": leaf["body"]})
    repairs.append(
        {
            "kind": "repeated_keys_indexed",
            "detail": ",".join(
                sorted(repeated_keys, key=lambda s: s.encode("utf-16-be"))
            ),
        }
    )
    return out


# ─── Assembly ───────────────────────────────────────────────────────────────


class _Missing:
    """Sentinel for an array slot not yet filled (sparse-array detection)."""

    _instance = None

    def __new__(cls):
        """Return the shared MISSING singleton."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        """Readable sentinel repr."""
        return "<MISSING>"


MISSING = _Missing()

_TABLE_ROW_RE = re.compile(rf"^({_NN}+)\[(\d+)\]$", re.ASCII)


def _assemble(
    leaves: list[dict],
    repairs: list[dict] | None,
    lenient_errors: list[dict] | None,
    schema: SchemaNode | None,
) -> dict:
    """Build the JSON value from parsed leaves: resolve paths (schema-aware), decode bodies, insert, then reject sparse arrays / check required fields."""
    root: dict = {}
    table_cols_by_key: dict[str, list[str]] = {}

    def resolve_leaf(key: str) -> tuple[list, SchemaNode | None]:
        """Resolve a leaf key to (path segments, schema node), with pathological-flat-key recovery."""
        path = _parse_path(key)
        if schema is None:
            return path, None
        node = _resolve_node(schema, path)
        if node is not None:
            return path, node
        if len(path) > 1:
            literal = schema.children.get(key) if schema.children else None
            if literal is not None:
                if repairs is not None:
                    repairs.append({"kind": "pathological_key_resolved", "detail": key})
                return [{"kind": "key", "name": key}], literal
        raise RaifError(f"schema: unknown field '{key}'")

    for leaf in leaves:
        try:
            body = leaf["body"]
            kind = body["kind"]
            if kind == "table_header":
                table_cols_by_key[leaf["key"]] = body["cols"]
                continue
            if kind == "array_literal":
                path, node = resolve_leaf(leaf["key"])
                if (
                    node is not None
                    and node is not OPEN_NODE
                    and node.element is None
                    and node.type != "o"
                ):
                    raise RaifError(f"schema: '{leaf['key']}' is not an array")
                if node is OPEN_NODE or (node is not None and node.type == "o"):
                    elem_node = OPEN_NODE
                else:
                    elem_node = node.element if node is not None else None
                elements = [
                    _decode_bare_value(row, repairs, elem_node) for row in body["rows"]
                ]
                _insert(root, path, elements)
                continue
            if kind == "table_row":
                m = _TABLE_ROW_RE.match(leaf["key"])
                if not m:
                    raise RaifError(f"table row key not indexable: {leaf['key']}")
                cols = table_cols_by_key.get(m.group(1))
                if cols is None:
                    raise RaifError(f"table row before header for prefix: {m.group(1)}")
                cells = body["cells"]
                if len(cells) != len(cols):
                    raise RaifError(
                        f"table row column count mismatch at {leaf['key']}: "
                        f"expected {len(cols)}, got {len(cells)}"
                    )
                path, node = resolve_leaf(leaf["key"])
                row_obj: dict = {}
                for idx, c in enumerate(cols):
                    cell_node = _child_node(node, c)
                    if (
                        schema is not None
                        and node is not None
                        and node is not OPEN_NODE
                        and cell_node is None
                    ):
                        raise RaifError(
                            f"schema: unknown column '{c}' at '{leaf['key']}'"
                        )
                    row_obj[c] = _decode_bare_value(cells[idx], repairs, cell_node)
                _insert(root, path, row_obj)
                continue
            path, node = resolve_leaf(leaf["key"])
            value = _decode_body(body, repairs, node)
            _insert(root, path, value)
        except RaifError as e:
            if lenient_errors is None:
                raise
            lenient_errors.append({"key": leaf["key"], "error": str(e)})

    if lenient_errors is not None:
        _prune_sparse_arrays(root, lenient_errors, "")
    else:
        _validate_no_missing(root, "")
    if schema is not None:
        missing: list[str] = []
        _check_required(schema, root, "", missing)
        if missing:
            err = f"schema: missing required field(s): {', '.join(missing)}"
            if lenient_errors is None:
                raise RaifError(err)
            for m in missing:
                lenient_errors.append(
                    {"key": m, "error": "schema: missing required field"}
                )
    return root


def _prune_sparse_arrays(obj: dict, errors: list[dict], path: str) -> None:
    """Lenient mode: drop any subtree containing a sparse array and record what was dropped."""
    for k in list(obj.keys()):
        v = obj[k]
        p = k if path == "" else f"{path}.{k}"
        if isinstance(v, list):
            if _contains_missing(v):
                del obj[k]
                errors.append(
                    {"key": p, "error": f"sparse array under '{p}' — subtree dropped"}
                )
        elif v is not None and isinstance(v, dict):
            _prune_sparse_arrays(v, errors, p)


def _contains_missing(node: Any) -> bool:
    """True if the value tree contains an unfilled array slot (the MISSING sentinel)."""
    if node is MISSING:
        return True
    if isinstance(node, list):
        return any(_contains_missing(x) for x in node)
    if node is not None and isinstance(node, dict):
        return any(_contains_missing(x) for x in node.values())
    return False


# ─── Value decoding ─────────────────────────────────────────────────────────


def _decode_bare_value(
    raw: str, repairs: list[dict] | None, node: SchemaNode | None
) -> Any:
    """Decode a bare value (leaf RHS or cell) by schema type when given, else by value-shape inference."""
    if node is None or node is OPEN_NODE:
        return _decode_inferred(raw, repairs)
    if raw == "null" and node.optional:
        return None
    return _decode_schema_typed(raw, node, repairs)


def _decode_inferred(raw: str, repairs: list[dict] | None) -> Any:
    """Infer a bare value with no schema: unwrap `<<<...>>>`, JSON literals, numbers, inline objects, else a bare string."""
    if (
        raw.startswith(OPEN)
        and raw.endswith(CLOSE)
        and len(raw) >= len(OPEN) + len(CLOSE)
    ):
        return raw[len(OPEN) : len(raw) - len(CLOSE)]
    if raw == "true":
        return True
    if raw == "false":
        return False
    if raw == "null":
        return None
    if raw == "[]":
        return []
    if raw == "{}":
        return {}
    if NUMBER_RE.match(raw):
        return _parse_number(raw)
    if raw.startswith("{") and raw.endswith("}"):
        obj = _try_parse_inline_object(raw, repairs, None)
        if obj is not None:
            return obj
    return raw


def _decode_schema_typed(raw: str, node: SchemaNode, repairs: list[dict] | None) -> Any:
    """Decode a value against its schema node (ADR-0019): `s`/`t` verbatim, `n`/`b` must parse, `o`/structured fall through to inference."""
    wrapped = (
        raw.startswith(OPEN)
        and raw.endswith(CLOSE)
        and len(raw) >= len(OPEN) + len(CLOSE)
    )
    inner = raw[len(OPEN) : len(raw) - len(CLOSE)] if wrapped else raw
    t = node.type
    if t in ("s", "t"):
        return inner
    if t == "n":
        if not NUMBER_RE.match(inner):
            raise RaifError(f"schema: expected number, got '{raw}'")
        return _parse_number(inner)
    if t == "b":
        if inner == "true":
            return True
        if inner == "false":
            return False
        raise RaifError(f"schema: expected boolean, got '{raw}'")
    if t == "o":
        if node.children is None:
            return _decode_inferred(raw, repairs)
        # fall through to structured handling
    if node.element is not None:
        if not wrapped and raw == "[]":
            return []
        raise RaifError(f"schema: expected array, got '{raw}'")
    if node.children is not None:
        if not wrapped and raw == "{}":
            return {}
        if not wrapped and raw.startswith("{") and raw.endswith("}"):
            obj = _try_parse_inline_object(raw, repairs, node)
            if obj is not None:
                return obj
        raise RaifError(f"schema: expected object, got '{raw}'")
    return _decode_inferred(raw, repairs)


def _try_parse_inline_object(
    s: str, repairs: list[dict] | None, node: SchemaNode | None
) -> dict | None:
    """Parse `{k=v,...}` into a dict, or None if it doesn't match the inline-object grammar; records the nested-flatten repair."""
    if s == "{}":
        return {}
    if not s.startswith("{") or not s.endswith("}") or len(s) < 2:
        return None
    inner = s[1:-1]
    if len(inner) == 0:
        return None
    pairs = _split_top_level_commas(inner)
    out: dict = {}
    saw_nested = False
    for pair in pairs:
        eq = _find_top_level_char(pair, "=")
        if eq == -1:
            return None
        raw_key = pair[:eq]
        if (
            raw_key.startswith(OPEN)
            and raw_key.endswith(CLOSE)
            and len(raw_key) >= len(OPEN) + len(CLOSE)
        ):
            key = raw_key[len(OPEN) : len(raw_key) - len(CLOSE)]
        else:
            key = raw_key
        if len(key) == 0:
            return None
        if key in out:
            return None
        cell_node = _child_node(node, key)
        if (
            node is not None
            and node is not OPEN_NODE
            and node.children is not None
            and cell_node is None
        ):
            raise RaifError(f"schema: unknown field '{key}' in inline object")
        value = _decode_bare_value(pair[eq + 1 :], repairs, cell_node)
        if value is not None and isinstance(value, (dict, list)):
            saw_nested = True
        out[key] = value
    if saw_nested and repairs is not None:
        repairs.append({"kind": "nested_inline_flattened"})
    return out


def _find_top_level_char(s: str, ch: str) -> int:
    """First top-level index of `ch`, skipping `<<<...>>>` ranges; -1 if none."""
    i = 0
    length = len(s)
    while i < length:
        if s.startswith(OPEN, i):
            end = s.find(CLOSE, i + len(OPEN))
            if end == -1:
                return -1
            i = end + len(CLOSE)
            continue
        if s[i] == ch:
            return i
        i += 1
    return -1


# ─── Path parsing and insertion ─────────────────────────────────────────────

_INDEX_RE = re.compile(r"^(0|[1-9]\d*)$", re.ASCII)


def _parse_path(key: str) -> list[dict]:
    """Parse a leaf key into path segments (dotted keys, `<<<...>>>` quoted segments, `[N]` indices), strictly."""
    segs: list[dict] = []
    i = 0
    length = len(key)
    while i < length:
        if key.startswith(OPEN, i):
            end = key.find(CLOSE, i + len(OPEN))
            if end == -1:
                raise RaifError(f"unterminated quoted key segment in: {key}")
            segs.append({"kind": "key", "name": key[i + len(OPEN) : end]})
            i = _consume_segment_boundary(key, end + len(CLOSE))
            continue
        if key[i] == "[":
            end = key.find("]", i + 1)
            if end == -1:
                raise RaifError(f"unterminated index segment in: {key}")
            raw_idx = key[i + 1 : end]
            if not _INDEX_RE.match(raw_idx):
                raise RaifError(f"bad index '{raw_idx}' in: {key}")
            segs.append({"kind": "index", "idx": int(raw_idx)})
            i = _consume_segment_boundary(key, end + 1)
            continue
        j = i
        while j < length and key[j] != "." and key[j] != "[":
            j += 1
        if j == i:
            raise RaifError(f"empty path segment in: {key}")
        segs.append({"kind": "key", "name": key[i:j]})
        i = _consume_segment_boundary(key, j) if j < length and key[j] == "." else j
    if len(segs) == 0:
        raise RaifError("empty path")
    return segs


def _consume_segment_boundary(key: str, pos: int) -> int:
    """Validate and step past a key-path segment boundary (`.`/`[`/end)."""
    if pos >= len(key) or key[pos] == "[":
        return pos
    if key[pos] == ".":
        if pos + 1 >= len(key):
            raise RaifError(f"trailing '.' in path: {key}")
        return pos + 1
    raise RaifError(f"malformed path after segment in: {key}")


def _decode_body(
    body: dict, repairs: list[dict] | None, node: SchemaNode | None
) -> Any:
    """Decode a parsed leaf body (multiline/typed/bare) to a JSON value, honoring the schema node when present."""
    kind = body["kind"]
    if kind == "multiline":
        if (
            node is not None
            and node is not OPEN_NODE
            and node.type not in ("s", "t", "o")
        ):
            raise RaifError(
                f"schema: multiline block where {_expected_kind(node)} expected"
            )
        return body["raw"]
    if kind == "typed":
        if node is not None and node is not OPEN_NODE:
            return _decode_schema_typed(body["raw"], node, repairs)
        if body["type"] in ("s", "t"):
            return _unwrap_delim(body["raw"])
        if body["type"] == "n":
            if not NUMBER_RE.match(body["raw"]):
                raise RaifError(f"bad number: {body['raw']}")
            return _parse_number(body["raw"])
        if body["raw"] == "true":
            return True
        if body["raw"] == "false":
            return False
        raise RaifError(f"bad boolean: {body['raw']}")
    if kind == "bare":
        return _decode_bare_value(body["raw"], repairs, node)
    raise RaifError(f"internal: _decode_body called on {kind}")


def _expected_kind(node: SchemaNode) -> str:
    """Human-readable expected kind of a schema node (array/object/type) for error messages."""
    if node.element is not None:
        return "array"
    if node.children is not None:
        return "object"
    return node.type or "value"


def _unwrap_delim(raw: str) -> str:
    """Strip one `<<<...>>>` wrapper if present, else return the string unchanged."""
    if (
        raw.startswith(OPEN)
        and raw.endswith(CLOSE)
        and len(raw) >= len(OPEN) + len(CLOSE)
    ):
        return raw[len(OPEN) : len(raw) - len(CLOSE)]
    return raw


def _insert(root: dict, path: list[dict], value: Any) -> None:
    """Insert `value` at `path` into root, creating objects/arrays and detecting path collisions; pads arrays with MISSING."""
    cursor: Any = root
    for i in range(len(path) - 1):
        seg = path[i]
        nxt = path[i + 1]
        child_init: Any = [] if nxt["kind"] == "index" else {}
        if seg["kind"] == "key":
            if isinstance(cursor, list):
                raise RaifError(
                    f"path collision: list expected dict-key '{seg['name']}'"
                )
            existing = cursor.get(seg["name"], None) if seg["name"] in cursor else None
            if seg["name"] not in cursor:
                cursor[seg["name"]] = child_init
                cursor = child_init
            elif (
                isinstance(existing, list)
                and nxt["kind"] == "index"
                or (
                    isinstance(existing, dict)
                    and existing is not None
                    and nxt["kind"] == "key"
                )
            ):
                cursor = existing
            else:
                raise RaifError(f"path collision at '{seg['name']}'")
        else:
            if not isinstance(cursor, list):
                raise RaifError(
                    f"path collision: dict expected list-index {seg['idx']}"
                )
            while len(cursor) <= seg["idx"]:
                cursor.append(MISSING)
            existing = cursor[seg["idx"]]
            if existing is MISSING:
                cursor[seg["idx"]] = child_init
                cursor = child_init
            elif (
                isinstance(existing, list)
                and nxt["kind"] == "index"
                or isinstance(existing, dict)
                and nxt["kind"] == "key"
            ):
                cursor = existing
            else:
                raise RaifError(f"path collision at index {seg['idx']}")
    last = path[-1]
    if last["kind"] == "key":
        if isinstance(cursor, list):
            raise RaifError(f"path collision: list expected dict-key '{last['name']}'")
        if last["name"] in cursor:
            raise RaifError(f"path collision: '{last['name']}' already exists")
        cursor[last["name"]] = value
    else:
        if not isinstance(cursor, list):
            raise RaifError(f"path collision: dict expected list-index {last['idx']}")
        while len(cursor) <= last["idx"]:
            cursor.append(MISSING)
        if cursor[last["idx"]] is not MISSING:
            raise RaifError(f"path collision: index {last['idx']} already exists")
        cursor[last["idx"]] = value


def _validate_no_missing(node: Any, path: str) -> None:
    """Raise on any sparse-array slot (MISSING) left in the tree -- RAIF rejects sparse arrays."""
    if isinstance(node, list):
        for i, v in enumerate(node):
            if v is MISSING:
                raise RaifError(
                    f"sparse array at {path}[{i}] — RAIF rejects sparse arrays"
                )
            _validate_no_missing(v, f"{path}[{i}]")
    elif node is not None and isinstance(node, dict):
        for k, v in node.items():
            _validate_no_missing(v, k if path == "" else f"{path}.{k}")


# ─── Schema (optional; ADR-0016 / ADR-0019) ─────────────────────────────────


class SchemaNode:
    __slots__ = ("type", "optional", "element", "children")

    def __init__(self, optional: bool = False):
        """Initialize the node/schema container."""
        self.type: str | None = None
        self.optional: bool = optional
        self.element: SchemaNode | None = None
        self.children: dict[str, SchemaNode] | None = None


# Sentinel: a node declared `:o` accepts arbitrary children with inferred types.
OPEN_NODE = SchemaNode(optional=True)


class RaifSchema:
    def __init__(self, root: SchemaNode):
        """Initialize the node/schema container."""
        self.root = root


_SCHEMA_TYPE_RE = re.compile(r"^([sntbo])(\?)?$")
_SCHEMA_TAG_RE = re.compile(r"</?schema>")


def parse_schema(decl: str) -> RaifSchema:
    """Parse a `<schema>`-block declaration into a `RaifSchema`. Accepts the
    block with or without the wrapper tags (they're stripped)."""
    root = SchemaNode()
    root.children = {}
    lines = _SCHEMA_TAG_RE.sub("", decl).split("\n")
    for idx, raw_line in enumerate(lines):
        line = _js_strip(_strip_cr(raw_line))
        if len(line) == 0:
            continue

        def at(msg: str) -> RaifError:
            """Build a schema-line error carrying the 1-based line number."""
            return RaifError(f"schema line {idx + 1}: {msg}")

        sep = _find_top_level_char(line, ":")
        if sep == -1:
            raise at(f"missing ':' in '{line}'")
        tm = _SCHEMA_TYPE_RE.match(_js_strip(line[sep + 1 :]))
        if not tm:
            raise at(f"bad type '{_js_strip(line[sep + 1 :])}'")
        segs = _parse_schema_path(_js_strip(line[:sep]), at)
        node = root
        field = root
        for seg in segs:
            if node.children is None:
                node.children = {}
            child = node.children.get(seg["name"])
            if child is None:
                child = SchemaNode()
                node.children[seg["name"]] = child
            field = child
            node = child
            for _ in range(seg["arrays"]):
                if node.element is None:
                    node.element = SchemaNode()
                node = node.element
        node.type = tm.group(1)
        if tm.group(2):
            field.optional = True
    return RaifSchema(root)


def _parse_schema_path(path: str, at) -> list[dict]:
    """Parse a schema declaration path into {name, arrays} segments."""
    segs: list[dict] = []
    i = 0
    length = len(path)
    while i < length:
        if path.startswith(OPEN, i):
            end = path.find(CLOSE, i + len(OPEN))
            if end == -1:
                raise at(f"unterminated <<< in '{path}'")
            name = path[i + len(OPEN) : end]
            i = end + len(CLOSE)
        else:
            j = i
            while j < length and path[j] != "." and path[j] != "[":
                j += 1
            name = path[i:j]
            i = j
        if len(name) == 0:
            raise at(f"empty segment in '{path}'")
        arrays = 0
        while path.startswith("[]", i):
            arrays += 1
            i += 2
        if i < length:
            if path[i] != ".":
                raise at(f"malformed path '{path}'")
            i += 1
            if i == length:
                raise at(f"trailing '.' in '{path}'")
        segs.append({"name": name, "arrays": arrays})
    if len(segs) == 0:
        raise at("empty path")
    return segs


def _to_schema(schema: Any | None) -> SchemaNode | None:
    """Coerce a schema argument (declaration string / RaifSchema / SchemaNode / None) to a root SchemaNode."""
    if schema is None:
        return None
    if isinstance(schema, str):
        return parse_schema(schema).root
    if isinstance(schema, RaifSchema):
        return schema.root
    if isinstance(schema, SchemaNode):
        return schema
    raise RaifError("schema must be a declaration string, RaifSchema, or SchemaNode")


def _resolve_node(node: SchemaNode | None, segs: list[dict]) -> SchemaNode | None:
    """Walk a parsed leaf path through the schema; returns the node, OPEN_NODE under `:o`, or None if not admitted."""
    cur = node
    for seg in segs:
        if cur is None:
            return None
        if cur is OPEN_NODE:
            return OPEN_NODE
        if seg["kind"] == "key":
            nxt = cur.children.get(seg["name"]) if cur.children else None
        else:
            nxt = cur.element
        if nxt is None and cur.type == "o":
            return OPEN_NODE
        cur = nxt
    return cur


def _child_node(node: SchemaNode | None, key: str) -> SchemaNode | None:
    """Schema node for a named cell (inline-object pair or table column)."""
    if node is None:
        return None
    if node is OPEN_NODE:
        return OPEN_NODE
    child = node.children.get(key) if node.children else None
    if child is None and node.type == "o":
        return OPEN_NODE
    return child


def _check_required(
    node: SchemaNode, value: Any, path: str, missing: list[str]
) -> None:
    """After assembly, collect every non-optional declared field that is absent."""
    if value is None or node is OPEN_NODE:
        return
    if node.children and isinstance(value, dict) and not isinstance(value, list):
        for k, child in node.children.items():
            p = k if path == "" else f"{path}.{k}"
            v = value[k] if k in value else None
            if k not in value:
                if not child.optional:
                    missing.append(p)
                continue
            _check_required(child, v, p, missing)
    if node.element and isinstance(value, list):
        for i, v in enumerate(value):
            _check_required(node.element, v, f"{path}[{i}]", missing)
