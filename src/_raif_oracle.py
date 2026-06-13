"""Bun bridge to the canonical TS RAIF encoder/decoder — the oracle for the
differential tests in `test_raif_differential.py`.

`encode` lets the tests generate *valid* RAIF from random JSON objects; `decode`
is ground truth the Python port must match. Both are batched through one `bun`
process per call so a fuzzing run stays fast.

Not a test module itself (leading underscore keeps pytest from collecting it).
"""

from __future__ import annotations

from raif_bun import available, run_bridge

__all__ = ["available", "js_encode", "js_decode", "js_decode_pairs", "values_equal"]

_BRIDGE = """
import { encode, decode } from "./src/raif.ts";
const spec = JSON.parse(await Bun.file(process.env.RAIF_BRIDGE_INPUT).text());
let out;
if (spec.op === "encode") {
  out = spec.items.map((o) => {
    try { return { ok: true, raif: encode(o, { profile: spec.profile, markers: spec.markers }) }; }
    catch (e) { return { ok: false, error: String(e) }; }
  });
} else if (spec.op === "decode") {
  out = spec.items.map((s) => {
    try {
      const r = spec.schema !== null ? decode(s, spec.schema) : decode(s);
      return r && r.ok ? { ok: true, value: r.value } : { ok: false };
    } catch (e) { return { ok: false }; }
  });
} else {
  // decode_pairs: each item is [raif, schema|null]
  out = spec.items.map(([s, schema]) => {
    try {
      const r = schema !== null ? decode(s, schema) : decode(s);
      return r && r.ok ? { ok: true, value: r.value } : { ok: false };
    } catch (e) { return { ok: false }; }
  });
}
process.stdout.write(JSON.stringify(out));
"""


def _run(spec: dict) -> list[dict]:
    """Run the bun bridge over one spec dict and return the parsed JSON result list."""
    return run_bridge(_BRIDGE, spec, timeout=max(120, len(spec["items"]) // 50 + 60))


def js_encode(objs: list, profile: str = "canonical", markers: bool = False) -> list[dict]:
    """Encode JSON objects → `[{ok, raif} | {ok: False, error}]`."""
    if not objs:
        return []
    return _run({"op": "encode", "profile": profile, "markers": markers, "items": objs})


def js_decode(raifs: list[str], schema=None) -> list[dict]:
    """Decode RAIF strings → `[{ok, value} | {ok: False}]` (ground truth)."""
    if not raifs:
        return []
    return _run({"op": "decode", "schema": schema, "items": raifs})


def js_decode_pairs(pairs: list[tuple]) -> list[dict]:
    """Decode `[(raif, schema|None), ...]` with a per-item schema, in one call."""
    if not pairs:
        return []
    return _run({"op": "decode_pairs", "items": [[r, s] for r, s in pairs]})


def values_equal(a, b) -> bool:
    """Deep equality with JS-number tolerance: int 5 and float 5.0 compare
    equal (the two decoders may land on either for the same numeric value, and
    integers beyond 2^53 coincide once both are read as doubles)."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return False
        return all(values_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(values_equal(x, y) for x, y in zip(a, b, strict=True))
    return a == b
