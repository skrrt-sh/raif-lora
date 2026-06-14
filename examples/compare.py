#!/usr/bin/env python3
"""Base Llama vs. the RAIF LoRA — same prompts, side by side.

Runs a batch of structured-output tasks through two models on the same base
weights: the stock Llama-3.2-3B-Instruct, and the same model with the RAIF
adapter. For each it measures the two things that matter for machine consumption:

  - parse:    did the output turn into the intended value without hand-holding?
              (base → strip markdown/prose, JSON.parse; RAIF → raif.decode)
  - fidelity: was that value byte-identical to the source object?
  - tokens:   how many output tokens did it cost (same tokenizer for both)?

The point isn't that base Llama "can't" emit structure — it can — it's that it
wraps JSON in prose/markdown you have to claw back, and spends more tokens doing
it. RAIF decodes deterministically and is leaner.

Usage:
    uv run python examples/setup_adapter.py    # one-time
    uv run python examples/compare.py [--n 12]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import raif_models as M

REPO = M.REPO

BASE_TMPL = "Return this data as compact JSON, nothing else:\n{json}"
RAIF_TMPL = "Rewrite this JSON payload as RAIF:\n{json}"

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def canon(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def stratify(examples: list[dict], n: int) -> list[dict]:
    """Round-robin one example per shape so the sample spans the shape vocabulary
    instead of stacking on whichever shape comes first in the file."""
    by_shape: dict[str, list[dict]] = {}
    for ex in examples:
        by_shape.setdefault(ex["meta"].get("shape", "?"), []).append(ex)
    shapes = sorted(by_shape)
    if not shapes or n <= 0:
        return []
    out: list[dict] = []
    i = 0
    while len(out) < min(n, len(examples)):
        bucket = by_shape[shapes[i % len(shapes)]]
        idx = i // len(shapes)
        if idx < len(bucket):
            out.append(bucket[idx])
        i += 1
    return out


def extract_json(text: str):
    """Best-effort recovery of a JSON value from a chatty base-model reply:
    prefer a fenced block, else the first balanced {...}/[...] span."""
    m = _FENCE_RE.search(text)
    candidates = [m.group(1)] if m else []
    # first object/array span as a fallback
    for opener, closer in (("{", "}"), ("[", "]")):
        i, j = text.find(opener), text.rfind(closer)
        if 0 <= i < j:
            candidates.append(text[i : j + 1])
    candidates.append(text.strip())
    for c in candidates:
        try:
            return json.loads(c), True
        except (json.JSONDecodeError, ValueError):
            continue
    return None, False


def n_tokens(tok, text: str) -> int:
    return len(tok.encode(text))


def load_two(model: str):
    """Load base weights twice — once plain, once with the RAIF adapter.
    (load_adapters mutates in place, so we need two instances.) Returns the
    spec too, for the think-strip flag."""
    s = M.spec(model)
    if not (s["mlx_dir"] / "adapters.safetensors").exists():
        sys.exit(f"Adapter for {model!r} not built. "
                 f"Run: uv run python examples/setup_adapter.py --model {model}")
    from mlx_lm import load
    from mlx_lm.utils import load_adapters

    base, tok = load(M.base_path(s))
    raif, _ = load(M.base_path(s))
    raif = load_adapters(raif, str(s["mlx_dir"]))
    base.eval()
    raif.eval()
    return base, raif, tok, s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=M.DEFAULT_MODEL, choices=list(M.MODELS),
                    help="which RAIF model to compare against its own base")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--data", default="valid",
                    help="'valid', 'holdout', or a path to a .jsonl eval file")
    args = ap.parse_args()

    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler
    from raif import decode

    print(f"Loading {args.model} base + RAIF (two instances)...", file=sys.stderr)
    base, raif, tok, s = load_two(args.model)
    greedy = make_sampler(temp=0.0)

    def gen(model, user: str) -> str:
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": user}],
            add_generation_prompt=True, tokenize=False)
        out = generate(model, tok, prompt=prompt, max_tokens=1024,
                       sampler=greedy, verbose=False)
        return M.strip_think(s, out).strip()

    data_map = {"valid": REPO / "data" / "valid.jsonl",
                "holdout": REPO / "data" / "eval_holdout.jsonl"}
    path = data_map.get(args.data, Path(args.data))
    if not path.exists():
        sys.exit(f"No eval data at {path}")
    print(f"Data: {path.name}", file=sys.stderr)
    examples = [json.loads(line) for line in path.open() if line.strip()]
    examples = [e for e in examples if e["meta"].get("task") == "translate"]
    examples = stratify(examples, args.n)
    if not examples:
        sys.exit(f"No 'translate' examples to compare in {path}")

    agg = {"base": {"parse": 0, "fid": 0, "tok": 0},
           "raif": {"parse": 0, "fid": 0, "tok": 0}}
    n = 0
    for ex in examples:
        source = decode(ex["messages"][1]["content"])  # gold RAIF -> the value
        if not source["ok"]:
            continue
        src = source["value"]
        src_json = json.dumps(src, separators=(",", ":"))
        n += 1

        b_out = gen(base, BASE_TMPL.format(json=src_json))
        b_val, b_ok = extract_json(b_out)
        b_fid = b_ok and canon(b_val) == canon(src)

        r_out = gen(raif, RAIF_TMPL.format(json=src_json))
        r_res = decode(r_out)
        r_ok = bool(r_res.get("ok"))
        r_fid = r_ok and canon(r_res["value"]) == canon(src)

        agg["base"]["parse"] += b_ok
        agg["base"]["fid"] += b_fid
        agg["base"]["tok"] += n_tokens(tok, b_out)
        agg["raif"]["parse"] += r_ok
        agg["raif"]["fid"] += r_fid
        agg["raif"]["tok"] += n_tokens(tok, r_out)

        print(f"\n{'='*64}\n{ex['meta']['shape']}  (source {len(src_json)} B JSON)")
        print(f"  base : parse {'✓' if b_ok else '✗'}  fidelity "
              f"{'✓' if b_fid else '✗'}  {n_tokens(tok, b_out):>4} tok")
        print(f"  raif : parse {'✓' if r_ok else '✗'}  fidelity "
              f"{'✓' if r_fid else '✗'}  {n_tokens(tok, r_out):>4} tok")
        if not b_fid:
            print(f"    base raw: {b_out[:120]!r}")

    print(f"\n{'='*64}\nSUMMARY over {n} structured-output tasks")
    if n == 0:
        sys.exit("No decodable gold RAIF examples; nothing to compare.")
    print("  (base is *coached*: 'compact JSON, nothing else' — its best case)")
    for label in ("base", "raif"):
        a = agg[label]
        print(f"  {label:5s}: parse {a['parse']}/{n}  fidelity {a['fid']}/{n}  "
              f"avg {a['tok']/n:.0f} output tokens")
    if agg["base"]["tok"]:
        ratio = agg["raif"]["tok"] / agg["base"]["tok"]
        print(f"\n  RAIF uses {ratio:.2f}× the output tokens of coached base JSON "
              f"({100*(1-ratio):+.0f}%).")

    # The qualitative half: how base Llama behaves *uncoached* — the thing you'd
    # actually have to parse if you hadn't fine-tuned. Same model, plain ask.
    print(f"\n{'='*64}\nHOW BASE LLAMA BEHAVES UNCOACHED (same prompt the LoRA was trained on)")
    demo = '{"user":"ada","tasks":["write","test","ship"],"done":false,"count":3}'
    nat = gen(base, RAIF_TMPL.format(json=demo))
    print(f"  prompt: Rewrite this JSON payload as RAIF: {demo}")
    print(f"  base output ({n_tokens(tok, nat)} tok):\n    "
          + nat[:400].replace("\n", "\n    "))
    print("  → it doesn't know RAIF; it hallucinates/echoes and wraps in prose.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
