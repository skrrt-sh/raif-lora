"""Smoke-eval the LoRA adapter — CUDA / unsloth path.

The mirror of ../src/eval_smoke.py for adapters trained with train_unsloth.py.
It reuses the SAME scoring meter and bun decoder (eval_core) as the MLX eval,
so numbers from the two stacks are directly comparable. The only difference is
how the model is loaded and how a token is generated.

Run from the raif-lora root:

    python cuda/eval_cuda.py --adapter ./adapters-cuda/warm --n 13
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the shared, framework-free meter importable from ../src.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from eval_core import (  # noqa: E402
    eval_group, load_examples, sample_examples,
    evaluate_gate, print_gate, write_results_json,
)

VALID_FILE = Path("./data/valid.jsonl")
HOLDOUT_FILE = Path("./data/eval_holdout.jsonl")
N_SAMPLES = 10


def make_hf_generate():
    """Return a generate(model, tok, prompt, max_tokens, verbose) matching the
    callable eval_group expects. Greedy/deterministic, to match the MLX eval."""
    import torch

    def generate(model, tok, prompt, max_tokens, verbose=False):
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return tok.decode(new_tokens, skip_special_tokens=True)

    return generate


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", required=True,
                   help="LoRA adapter dir saved by train_unsloth.py")
    p.add_argument("--n", type=int, default=N_SAMPLES,
                   help=f"examples to sample per group (default {N_SAMPLES})")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for example sampling (default 0)")
    p.add_argument("--max-seq", type=int, default=2048,
                   help="max sequence length for loading (default 2048)")
    p.add_argument("--valid", type=Path, default=VALID_FILE)
    p.add_argument("--holdout", type=Path, default=HOLDOUT_FILE)
    p.add_argument("--out", type=Path, default=None,
                   help="write full results JSON here (per-example rows + summary + gate)")
    p.add_argument("--gate", default=None,
                   choices=["smoke", "warm", "mid", "full"],
                   help="check this stage's ITERATION_PLAN gate and print PASS/FAIL; "
                        "exit nonzero on FAIL")
    args = p.parse_args()

    from unsloth import FastLanguageModel

    print(f"Loading base + adapter from {args.adapter}...")
    model, tok = FastLanguageModel.from_pretrained(
        model_name=args.adapter,     # adapter dir; unsloth resolves the base
        max_seq_length=args.max_seq,
        dtype=None,
        load_in_4bit=False,
    )
    FastLanguageModel.for_inference(model)

    generate = make_hf_generate()
    groups = [
        ("valid (in-training shapes)", args.valid),
        ("holdout (withheld shapes)", args.holdout),
    ]
    results = []
    for name, path in groups:
        examples = sample_examples(load_examples(path), args.n, args.seed)
        results.append((name, eval_group(name, examples, model, tok, generate)))

    print("── summary ──")
    for name, stats in results:
        if stats is None or stats["n"] == 0:
            print(f"{name:30s} (no scored examples)")
            continue
        print(f"{name:30s} parse {stats['parse']}/{stats['n']} "
              f"({100*stats['parse']/stats['n']:.0f}%)  "
              f"fidelity {stats['fidelity']}/{stats['n']} "
              f"({100*stats['fidelity']/stats['n']:.0f}%)  "
              f"skipped {stats['skipped']}")

    valid_stats = results[0][1]
    holdout_stats = results[1][1]
    gate = None
    if args.gate:
        gate = evaluate_gate(args.gate, valid_stats, holdout_stats)
        print()
        print_gate(gate)

    if args.out:
        write_results_json(args.out, {
            "stack": "cuda",
            "adapter": str(args.adapter),
            "n_per_group": args.n,
            "seed": args.seed,
            "groups": {name: stats for name, stats in results},
            "gate": gate,
        })

    if gate and gate.get("passed") is False:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
