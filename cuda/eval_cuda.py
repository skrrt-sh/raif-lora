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
from eval_core import add_common_eval_args, run_eval  # noqa: E402


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
    p.add_argument("--max-seq", type=int, default=2048,
                   help="max sequence length for loading (default 2048)")
    add_common_eval_args(p)
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
    return run_eval(args, model, tok, generate, stack="cuda",
                    extra_payload={"adapter": str(args.adapter)})


if __name__ == "__main__":
    raise SystemExit(main())
