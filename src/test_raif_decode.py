"""Parity test: the pure-Python `raif_decode` must agree with the canonical
TypeScript decoder (`raif-standard/prototype/src/raif.ts`) on every input.

The TS `decode()` is ground truth — the same decoder the eval scores against.
This test runs both over (a) the entire real RAIF corpus (the assistant turns in
`data/*.jsonl`) and (b) a battery of crafted inputs that exercise the repair
pipeline (markdown fences, mode markers, CRLF, brace flattening, truncated
blocks, mismatched nonces, relaxed delimiters, tables, inline objects, repeated
keys, schema-typed decode). For each input we require the Python and TS decoders
to agree on the ok/fail flag and, on success, on the decoded JSON value.

Skips (does not fail) when `bun` or the sibling `raif-standard` checkout is
absent, so it stays runnable in environments without the TS toolchain.

Run:  uv run python src/test_raif_decode.py   (or: pytest src/test_raif_decode.py)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _raif_oracle import values_equal  # noqa: E402
from eval_core import batch_decode  # noqa: E402
from raif_bun import available  # noqa: E402
from raif_decode import decode  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Crafted inputs that hit repair branches the corpus may not cover. Each is a
# raw RAIF string the Python and TS decoders must agree on (ok flag + value).
CRAFTED: list[str] = [
    # — primitives & inference —
    "a=1\nb=2.5\nc=true\nd=false\ne=null\nf=hello",
    "n=-0\nz=0\nbig=9007199254740993\nexp=1e3\nfrac=10.50",
    "s=<<<wrapped value>>>\nlit=<<<true>>>\nnum=<<<42>>>",
    "empty_arr=[]\nempty_obj={}\nbracket=<<<[>>>",
    # — typed leaves —
    "name:s=null\npri:n=2\nok:b=true\ntext:t=verbatim : value",
    # — paths & indices —
    "user.profile.name=Ann\nuser.profile.age=30\nuser.tags[0]=a\nuser.tags[1]=b",
    "matrix[0][0]=1\nmatrix[0][1]=2\nmatrix[1][0]=3\nmatrix[1][1]=4",
    # — tables —
    "rows::id,name,active\nrows[0]=1,Ann,true\nrows[1]=2,Bob,false",
    "rows::id,note\nrows[0]=1,<<<has, comma>>>\nrows[1]=2,null",
    # — inline objects (incl. nested → flatten repair) —
    "o={a=1,b=hi,c=true}",
    "o={user={id=7,name=Ann},active=true}",
    "wrapk={<<<a.b>>>=1,plain=2}",
    # — array literals —
    "xs=[\n1\n2\n3\n]",
    "people=[\n{id=1,name=Ann}\n{id=2,name=Bob}\n]",
    "mixed=[\nhello\n<<<[>>>\n<<<]>>>\n42\ntrue\n]",
    # — multiline blocks —
    "doc=<<<\nline one\nline two\n>>>",
    "doc=<<<dead\n>>>\nactual closer below\n>>>dead",
    # — repair: markdown fence —
    "```\na=1\nb=2\n```",
    "```raif\na=1\nb=2\n```",
    # — repair: mode markers —
    "<raif>\na=1\nb=2\n</raif>",
    "<|raif_start|>\na=1\n<|raif_end|>",
    # — repair: missing close marker (truncation) —
    "<raif>\na=1\nb=2",
    # — repair: CRLF —
    "a=1\r\nb=2\r\nc=3",
    # — repair: multi-line brace flattening —
    "a={\nb={\nc=1\n}\nd=2\n}",
    "outer={\nlist=[\n1\n2\n]\ninner={\nx=9\n}\n}",
    # — repair: truncated array literal —
    "xs=[\n1\n2",
    # — repair: truncated multiline block —
    "doc=<<<\nline one\nline two",
    # — repair: mismatched nonce (single candidate) —
    "doc=<<<aaaa\ncontent\n>>>bbbb",
    # — repair: relaxed delimiters —
    "doc=<<\ncontent line\n>>",
    # — repair: repeated keys → auto-index —
    "tag=red\ntag=green\ntag=blue",
    # — values with separators that must stay bare —
    "ts=14:02:33\nrange=1,2,3\nurl=http://x/y?a=b\npath=a[0]",
    # — error cases (both must reject) —
    "no_separator_here",
    "a[0]=1\na[2]=3",  # sparse array → reject
    "k=1\nk.sub=2",  # path collision
    "bad=[01]",  # not actually an opener; bare value
    # — empty / whitespace —
    "",
    "   ",
    "\n\n",
]


def load_corpus() -> list[str]:
    """Load every assistant-turn RAIF string from the dataset jsonl files."""
    raifs: list[str] = []
    for name in ("valid.jsonl", "eval_holdout.jsonl", "train.jsonl"):
        p = DATA_DIR / name
        if not p.exists():
            continue
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)
                raifs.append(ex["messages"][1]["content"])
    return raifs


def run_parity(raifs: list[str], label: str) -> int:
    """Returns the number of mismatches (0 == parity)."""
    bun = batch_decode(raifs)
    assert len(bun) == len(raifs), (
        f"[{label}] bun output length {len(bun)} != input count {len(raifs)} "
        f"— parity mismatches would be undercounted by zip"
    )
    mismatches = 0
    for i, (raif, b) in enumerate(zip(raifs, bun, strict=True)):
        p = decode(raif)
        if p["ok"] != b["ok"]:
            mismatches += 1
            print(f"  [{label} #{i}] OK-FLAG mismatch: py={p['ok']} bun={b['ok']}")
            print(f"      input: {raif[:160]!r}")
            if not p["ok"]:
                print(f"      py error: {p.get('error')}")
            continue
        if b["ok"] and not values_equal(p["value"], b["value"]):
            mismatches += 1
            print(f"  [{label} #{i}] VALUE mismatch")
            print(f"      input: {raif[:160]!r}")
            print(f"      py:  {json.dumps(p['value'], ensure_ascii=False)[:240]}")
            print(f"      bun: {json.dumps(b['value'], ensure_ascii=False)[:240]}")
    return mismatches


def main() -> int:
    """Run crafted + corpus parity against the canonical decoder; return the process exit code."""
    if not available():
        print("SKIP: bun and/or raif-standard prototype not available — parity not checked.")
        return 0

    total_mismatch = 0

    print(f"── parity: {len(CRAFTED)} crafted repair-trigger cases ──")
    total_mismatch += run_parity(CRAFTED, "crafted")

    corpus = load_corpus()
    if corpus:
        print(f"── parity: {len(corpus)} real corpus RAIF strings ──")
        total_mismatch += run_parity(corpus, "corpus")
    else:
        print("── no corpus found under data/ — skipping corpus parity ──")

    n = len(CRAFTED) + len(corpus)
    if total_mismatch == 0:
        print(f"\nPASS — Python decoder matches canonical TS decoder on all {n} inputs.")
        return 0
    print(f"\nFAIL — {total_mismatch}/{n} inputs diverged.")
    return 1


def test_parity_with_canonical_decoder():
    """pytest entry point. Skips when the TS toolchain is unavailable; fails
    on any Python/TS divergence over crafted cases + the real corpus."""
    if not available():
        import pytest

        pytest.skip("bun and/or raif-standard prototype not available")
    corpus = load_corpus()
    mismatches = run_parity(CRAFTED, "crafted")
    if corpus:
        mismatches += run_parity(corpus, "corpus")
    assert mismatches == 0, f"{mismatches} input(s) diverged from the canonical decoder"


if __name__ == "__main__":
    raise SystemExit(main())
