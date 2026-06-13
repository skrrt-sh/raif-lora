"""Sanity checks for generated dataset files (run after make_data.sh).

Asserts:
  1. eval_holdout.jsonl contains ONLY the held-out shapes (plan §3.4).
  2. train.jsonl and valid.jsonl contain NONE of the held-out shapes.
  3. valid.jsonl is stratified: every non-held-out shape appears, with
     counts within ±1 of each other.
  4. Every instruct-task prompt contains every primitive leaf value of its
     completion (strings/bools/null via json.dumps, numbers via str()).
     Translate prompts get the same check (the minified JSON guarantees it).

Usage: uv run python src/check_data.py [data_dir]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Canonical held-out shape list (plan §3.4) — the single source of truth.
# make_data.sh derives its default --holdout-shapes from this via `python -c`,
# so the generator and this validator can never drift apart.
HOLDOUT_SHAPES = frozenset({
    "multiline_body", "pathological_keys", "large_table",
    "deep_array_literal", "flat_inline_object",
})


def load(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def primitive_leaves(value) -> list:
    out = []
    if value is None or not isinstance(value, (dict, list)):
        out.append(value)
    elif isinstance(value, list):
        for el in value:
            out.extend(primitive_leaves(el))
    else:
        for v in value.values():
            out.extend(primitive_leaves(v))
    return out


def fmt(v) -> str:
    # Matches the generator: strings JSON-quoted, numbers via String(v),
    # booleans/null as literals. ensure_ascii=False mirrors JS JSON.stringify,
    # which leaves non-ASCII characters literal (real datasets carry unicode;
    # the default ensure_ascii=True would escape it and false-fail containment).
    if isinstance(v, bool) or v is None or isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    return str(v)  # int/float — same shortest-roundtrip repr as JS String(v)


def main() -> int:
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")
    train = load(data_dir / "train.jsonl")
    valid = load(data_dir / "valid.jsonl")
    holdout = load(data_dir / "eval_holdout.jsonl")
    failures = 0

    # 1 & 2: holdout containment.
    holdout_shapes = {ex["meta"]["shape"] for ex in holdout}
    if not holdout_shapes <= HOLDOUT_SHAPES:
        print(f"FAIL: eval_holdout.jsonl has non-holdout shapes: {holdout_shapes - HOLDOUT_SHAPES}")
        failures += 1
    if holdout_shapes != HOLDOUT_SHAPES:
        print(f"WARN: holdout file missing shapes: {HOLDOUT_SHAPES - holdout_shapes}")
    for name, exs in (("train", train), ("valid", valid)):
        leaked = {ex["meta"]["shape"] for ex in exs} & HOLDOUT_SHAPES
        if leaked:
            print(f"FAIL: {name}.jsonl contains held-out shapes: {leaked}")
            failures += 1

    # 3: valid stratification.
    counts: dict[str, int] = {}
    for ex in valid:
        counts[ex["meta"]["shape"]] = counts.get(ex["meta"]["shape"], 0) + 1
    train_shapes = {ex["meta"]["shape"] for ex in train}
    missing = train_shapes - set(counts)
    if missing:
        print(f"FAIL: valid.jsonl missing shapes: {missing}")
        failures += 1
    if counts and max(counts.values()) - min(counts.values()) > 1:
        print(f"FAIL: valid.jsonl not evenly stratified: {counts}")
        failures += 1
    print(f"valid.jsonl per-shape counts: {counts}")

    # 4: prompt contains every primitive leaf value.
    bad = 0
    checked = 0
    for ex in train + valid + holdout:
        prompt = ex["messages"][0]["content"]
        source = ex["meta"]["source"]
        checked += 1
        missing_leaves = [
            fmt(leaf) for leaf in primitive_leaves(source) if fmt(leaf) not in prompt
        ]
        if missing_leaves:
            bad += 1
            print(f"FAIL: {ex['meta']['shape']} seed={ex['meta']['variation_seed']} "
                  f"task={ex['meta']['task']}: {len(missing_leaves)} leaf/leaves not in "
                  f"prompt: {', '.join(repr(m) for m in missing_leaves)}")
    if bad:
        failures += 1
    print(f"prompt↔completion leaf containment: {checked - bad}/{checked} examples OK")

    print("ALL CHECKS PASSED" if failures == 0 else f"{failures} CHECK(S) FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
