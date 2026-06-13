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

from eval_core import (
    eval_group, load_examples, sample_examples,
    evaluate_gate, print_gate, write_results_json,
)

MODEL_PATH = Path("./models/llama-3.2-3b-instruct-bf16")
VALID_FILE = Path("./data/valid.jsonl")
HOLDOUT_FILE = Path("./data/eval_holdout.jsonl")

N_SAMPLES = 10


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
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", required=True,
                   help="path to adapter dir (e.g. ./adapters/llama-3-3b-raif-sft-warm)")
    p.add_argument("--checkpoint", default="latest",
                   help="'latest' or an iter number like '1500'")
    p.add_argument("--n", type=int, default=N_SAMPLES,
                   help=f"examples to sample per group (default {N_SAMPLES})")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for example sampling (default 0)")
    p.add_argument("--valid", type=Path, default=VALID_FILE,
                   help=f"in-training-shape eval file (default {VALID_FILE})")
    p.add_argument("--holdout", type=Path, default=HOLDOUT_FILE,
                   help=f"held-out-shape eval file (default {HOLDOUT_FILE})")
    p.add_argument("--out", type=Path, default=None,
                   help="write full results JSON here (per-example rows + summary + gate)")
    p.add_argument("--gate", default=None,
                   choices=["smoke", "warm", "mid", "full"],
                   help="check this stage's ITERATION_PLAN gate and print PASS/FAIL; "
                        "exit nonzero on FAIL")
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
            "stack": "mlx",
            "adapter": str(args.adapter),
            "checkpoint": args.checkpoint,
            "n_per_group": args.n,
            "seed": args.seed,
            "groups": {name: stats for name, stats in results},
            "gate": gate,
        })

    if gate and gate.get("passed") is False:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
