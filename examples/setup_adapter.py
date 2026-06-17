#!/usr/bin/env python3
"""Make a published RAIF adapter runnable on the local MLX stack.

Why this exists: the published adapters were trained with PEFT/unsloth, so their
weights are in PEFT (torch) format. This repo's local runtime is MLX, which uses a
*different* adapter layout. So we convert the PEFT LoRA into an MLX adapter, once.

The conversion is mechanical and lossless (see raif_models.convert_to_mlx):
  - key rename:  base_model.model.model.layers.N.MOD.PROJ.lora_{A,B}.weight
                 ->                model.layers.N.MOD.PROJ.lora_{a,b}
  - transpose:   PEFT lora_A (rank, in)  -> MLX lora_a (in, rank)
                 PEFT lora_B (out, rank)  -> MLX lora_b (rank, out)
  - scale:       scale = lora_alpha / r = 64 / 32 = 2.0

Run:
    uv run python examples/setup_adapter.py                  # default: llama-3b
    uv run python examples/setup_adapter.py --model qwen-0.5b
    uv run python examples/setup_adapter.py --model all

Then:  uv run python examples/chat.py --model qwen-0.5b
"""

from __future__ import annotations

import argparse

import raif_models as M


def setup(model: str) -> None:
    s = M.spec(model)
    print(f"\n[{model}] {s['label']}")
    peft = M.resolve_peft_dir(s)
    layers = M.convert_to_mlx(peft, s["mlx_dir"])
    print(f"  wrote {s['mlx_dir'].relative_to(M.REPO)}  ({layers} layers, scale 2.0)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--model",
        default=M.DEFAULT_MODEL,
        choices=[*M.MODELS, "all"],
        help="which adapter to convert",
    )
    args = ap.parse_args()

    targets = list(M.MODELS) if args.model == "all" else [args.model]
    for m in targets:
        setup(m)
    print("\nDone. Next:")
    print(f"  uv run python examples/chat.py --model {targets[0]} --selftest")
    print(f"  uv run python examples/chat.py --model {targets[0]}")
