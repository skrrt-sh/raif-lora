"""Smoke-eval the LoRA adapter (MLX path).

Loads base model + adapter, generates RAIF for held-out prompts from TWO
groups, scored and reported separately:

  - valid.jsonl        — stratified eval split of the in-training shapes
  - eval_holdout.jsonl — shapes withheld from training entirely (plan §3.4)

Scoring per example (parse + byte-fidelity) and all decoding go through the
shared, framework-free meter in `eval_core` — the same code the CUDA/unsloth
eval (`cuda/eval_cuda.py`) uses, so the two stacks' numbers are comparable.

This script only adds the MLX-specific pieces: loading the base model +
adapter via mlx_lm, and resolving a checkpoint label to a concrete dir.

Not a replacement for the full harness — that's the next step once smoke
training looks promising. This script answers one question:
'did the model learn to emit RAIF at all?'
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from eval_core import add_common_eval_args, run_eval

MODEL_PATH = Path("./models/llama-3.2-3b-instruct-bf16")


def select_adapter_checkpoint(adapter_dir: Path, label: str) -> Path:
    """Map a label ('latest' or an iter number) to a concrete adapter dir.

    mlx-lm's load_adapters reads `adapters.safetensors` from the dir. To
    eval a specific checkpoint, copy that checkpoint into a temp sibling
    dir alongside the same adapter_config.json.
    """
    import shutil
    if label == "latest":
        return adapter_dir
    src = adapter_dir / f"{int(label):07d}_adapters.safetensors"
    if not src.exists():
        raise FileNotFoundError(f"no checkpoint at iter {label}: {src}")
    dst = adapter_dir.parent / f"{adapter_dir.name}-iter{int(label):04d}"
    dst.mkdir(exist_ok=True)
    shutil.copy(src, dst / "adapters.safetensors")
    shutil.copy(adapter_dir / "adapter_config.json", dst / "adapter_config.json")
    return dst


def main() -> int:
    """Load the MLX base model + adapter checkpoint, then run the shared eval driver."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", required=True,
                   help="path to adapter dir (e.g. ./adapters/llama-3-3b-raif-sft-warm)")
    p.add_argument("--checkpoint", default="latest",
                   help="'latest' or an iter number like '1500'")
    add_common_eval_args(p)
    args = p.parse_args()

    # Heavy imports deferred so `--help` works without model deps loaded.
    from mlx_lm import load, generate
    from mlx_lm.utils import load_adapters

    print(f"Loading base model from {MODEL_PATH}...")
    model, tok = load(str(MODEL_PATH))
    adapter_dir = select_adapter_checkpoint(Path(args.adapter), args.checkpoint)
    print(f"Loading adapter from {adapter_dir}...")
    model = load_adapters(model, str(adapter_dir))
    model.eval()

    return run_eval(args, model, tok, generate, stack="mlx",
                    extra_payload={"adapter": str(args.adapter),
                                   "checkpoint": args.checkpoint})


if __name__ == "__main__":
    sys.exit(main())
