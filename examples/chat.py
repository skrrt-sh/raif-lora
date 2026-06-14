#!/usr/bin/env python3
"""Talk to a RAIF LoRA in your terminal — locally, on Apple Silicon.

Pick a model with --model (llama-3b | qwen-0.5b | qwen-4b). It doesn't emit JSON;
it emits RAIF, a token-leaner serialization. The "decode at the output boundary"
is the one step every consumer needs: we run `raif.decode()` (the published
`raif-format` package) and get an ordinary JSON value back, plus a repair pass
that recovers truncated/malformed output.

Usage:
    uv run python examples/setup_adapter.py --model qwen-0.5b   # one-time
    uv run python examples/chat.py --model qwen-0.5b            # interactive REPL
    uv run python examples/chat.py --model qwen-0.5b --selftest # round-trip eval
    echo '{"a":1,"b":[2,3]}' | uv run python examples/chat.py   # one-shot from stdin

In the REPL, paste a JSON object to translate it to RAIF (the model's main job),
or type any instruction. Each turn prints the raw RAIF, the decoded JSON, and the
byte savings vs. minified JSON. (--selftest needs the eval corpus; see README.)

Requires:  uv pip install raif-format   (the codec; mlx-lm is already a dep)
"""

from __future__ import annotations

import argparse
import json
import sys

import raif_models as M

TRANSLATE_TMPL = "Rewrite this JSON payload as RAIF:\n{json}"


def make_generate(model: str):
    """Load base + RAIF adapter for `model` and return a greedy generate(msg)
    callable plus the spec (for the think-strip flag)."""
    s = M.spec(model)
    if not (s["mlx_dir"] / "adapters.safetensors").exists():
        sys.exit(f"Adapter for {model!r} not built.\n"
                 f"Run:  uv run python examples/setup_adapter.py --model {model}")
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.utils import load_adapters

    print(f"Loading {s['label']} (MLX)...", file=sys.stderr)
    mdl, tok = load(M.base_path(s))
    mdl = load_adapters(mdl, str(s["mlx_dir"]))
    mdl.eval()
    greedy = make_sampler(temp=0.0)

    def gen(user_msg: str, max_tokens: int = 1024) -> str:
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            add_generation_prompt=True, tokenize=False)
        out = generate(mdl, tok, prompt=prompt, max_tokens=max_tokens,
                       sampler=greedy, verbose=False)
        return M.strip_think(s, out).strip()

    return gen, tok


def as_translate_prompt(text: str) -> str:
    """If `text` is a JSON value, wrap it in the translate instruction; else
    pass it through verbatim (lets you try instruct-style prompts too)."""
    try:
        json.loads(text)
        return TRANSLATE_TMPL.format(json=text)
    except (json.JSONDecodeError, ValueError):
        return text


def show_turn(raif_text: str) -> None:
    """Decode one RAIF output and print RAIF + JSON + token savings."""
    from raif import decode

    print("\n\033[90m── RAIF (model output) ──\033[0m")
    print(raif_text)

    result = decode(raif_text)
    print("\n\033[90m── decode() -> JSON ──\033[0m")
    if result["ok"]:
        value = result["value"]
        print(json.dumps(value, indent=2, ensure_ascii=False))
        repairs = result.get("repairs") or []
        if repairs:
            print(f"\033[33m(recovered via {len(repairs)} repair(s))\033[0m")
        json_bytes = len(json.dumps(value, separators=(",", ":")).encode())
        raif_bytes = len(raif_text.encode())
        if json_bytes:
            delta = 100 * (raif_bytes - json_bytes) / json_bytes
            print(f"\033[90m{raif_bytes} B RAIF vs {json_bytes} B JSON "
                  f"({delta:+.0f}%)\033[0m")
    else:
        print(f"\033[31mdecode failed: {result.get('error')}\033[0m")


def selftest(gen, n: int = 12) -> int:
    """Round-trip n held-out examples; print parse/fidelity. Exit nonzero if low."""
    from raif import decode

    path = M.REPO / "data" / "valid.jsonl"
    if not path.exists():
        sys.exit(f"No eval data at {path} (run src/make_data.sh).")
    examples = [json.loads(line) for line in path.open() if line.strip()][:n]
    n_parse = n_fid = 0
    for i, ex in enumerate(examples):
        user, gold = ex["messages"][0]["content"], ex["messages"][1]["content"]
        out = gen(user)
        r, gr = decode(out), decode(gold)
        ok = r.get("ok")
        fid = bool(ok and gr.get("ok")
                   and json.dumps(r["value"], sort_keys=True)
                   == json.dumps(gr["value"], sort_keys=True))
        n_parse += bool(ok)
        n_fid += fid
        print(f"[{i:2d}] {ex['meta']['shape']:26s} "
              f"parse {'✓' if ok else '✗'}  fidelity {'✓' if fid else '✗'}")
    print(f"\nparse {n_parse}/{len(examples)}  fidelity {n_fid}/{len(examples)}")
    return 0 if n_fid >= 0.8 * len(examples) else 1


def repl(gen) -> None:
    print("Paste JSON to translate, or type an instruction. Ctrl-D or 'exit' to quit.\n")
    while True:
        try:
            line = input("\033[36myou ›\033[0m ").strip()
        except EOFError:
            print()
            return
        if line in {"exit", "quit"}:
            return
        if not line:
            continue
        show_turn(gen(as_translate_prompt(line)))
        print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=M.DEFAULT_MODEL, choices=list(M.MODELS),
                    help="which RAIF model to load")
    ap.add_argument("--selftest", action="store_true",
                    help="round-trip held-out examples and report fidelity")
    ap.add_argument("--n", type=int, default=12, help="examples for --selftest")
    args = ap.parse_args()

    gen, _ = make_generate(args.model)

    if args.selftest:
        return selftest(gen, args.n)
    if not sys.stdin.isatty():               # piped input -> one-shot
        data = sys.stdin.read().strip()
        if data:
            show_turn(gen(as_translate_prompt(data)))
        return 0
    repl(gen)
    return 0


if __name__ == "__main__":
    sys.exit(main())
