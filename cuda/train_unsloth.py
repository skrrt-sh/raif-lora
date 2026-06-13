"""RAIF LoRA trainer — CUDA / unsloth path (for the RTX 5070 Ti, Blackwell).

This is the CUDA mirror of the MLX configs in ../configs/*.yaml. It does NOT
replace them — the MLX flow (`uv run mlx_lm.lora --config ...`) stays the
working path on the Mac. This script lets the same smoke→warm→full ladder run
on an NVIDIA GPU, ~3-4× faster, using the *same* data/ JSONL and the same
stage gates from ../ITERATION_PLAN.md.

Run from the raif-lora root so paths resolve:

    python cuda/train_unsloth.py --stage smoke
    python cuda/train_unsloth.py --stage warm
    python cuda/train_unsloth.py --stage full     # 2048 seq; see --micro-batch if OOM

Then score with the matching eval (same meter as the MLX eval):

    python cuda/eval_cuda.py --adapter ./adapters-cuda/<stage>

Hyperparameter parity with the MLX configs, and the deliberate divergences,
are documented in cuda/README.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Stage ladder — mirrors ../configs/llama-3-3b-sft-{smoke,warm,full}.yaml.
# `iters` is the MLX micro-batch count (each = one batch of `batch`); HF counts
# OPTIMIZER steps, so max_steps = iters // grad_accum keeps examples-seen (and
# therefore epochs) identical to the MLX runs. alpha = rank * scale(2.0).
STAGES = {
    "smoke": dict(iters=300, rank=16, alpha=32, num_layers=16, max_seq=1024,
                  batch=4, grad_accum=1, save_every=100),
    "warm":  dict(iters=1500, rank=32, alpha=64, num_layers=16, max_seq=1024,
                  batch=4, grad_accum=1, save_every=250),
    "full":  dict(iters=7000, rank=32, alpha=64, num_layers=-1, max_seq=2048,
                  batch=4, grad_accum=4, save_every=500),
}

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

# Llama-3.2 chat-template part markers — used to mask the prompt tokens so loss
# falls only on the assistant's RAIF emission (the MLX `mask_prompt: true`).
INSTRUCTION_PART = "<|start_header_id|>user<|end_header_id|>\n\n"
RESPONSE_PART = "<|start_header_id|>assistant<|end_header_id|>\n\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", required=True, choices=sorted(STAGES),
                   help="which rung of the ladder to run")
    p.add_argument("--model", default="unsloth/Llama-3.2-3B-Instruct",
                   help="base model (HF id or local path). unsloth's mirror is "
                        "ungated and matches meta's weights.")
    p.add_argument("--data", type=Path, default=Path("./data"),
                   help="dir with train.jsonl + valid.jsonl (default ./data)")
    p.add_argument("--out", type=Path, default=None,
                   help="adapter output dir (default ./adapters-cuda/<stage>)")
    p.add_argument("--micro-batch", type=int, default=None,
                   help="override per-device batch (lower it + raise grad-accum "
                        "to fit 16 GB on the full run; effective batch is kept)")
    p.add_argument("--optim", default="adamw_8bit",
                   help="optimizer (adamw_8bit needs bitsandbytes w/ Blackwell "
                        "support; use adamw_torch if it errors on sm_120)")
    p.add_argument("--seed", type=int, default=0, help="MLX-parity seed (default 0)")
    return p.parse_args()


def build_text_field(tok):
    """Map {messages,...} -> {"text": <full chat-template string>} for SFT."""
    def fmt(batch):
        return {"text": [tok.apply_chat_template(m, tokenize=False)
                         for m in batch["messages"]]}
    return fmt


def main() -> int:
    args = parse_args()
    cfg = STAGES[args.stage]
    out = args.out or Path(f"./adapters-cuda/{args.stage}")

    # Keep effective batch constant if the user lowers the micro-batch to fit VRAM.
    micro = args.micro_batch or cfg["batch"]
    grad_accum = cfg["grad_accum"] * (cfg["batch"] // micro if micro else 1)
    grad_accum = max(1, grad_accum)
    # max_steps counts optimizer steps; iters counts micro-batches (MLX). Divide
    # by the FULL accumulation so examples-seen == the MLX run's.
    max_steps = max(1, cfg["iters"] // grad_accum)

    print(f"[stage={args.stage}] base={args.model} seq={cfg['max_seq']} "
          f"micro_batch={micro} grad_accum={grad_accum} max_steps={max_steps} "
          f"rank={cfg['rank']} alpha={cfg['alpha']} num_layers={cfg['num_layers']}")

    # Heavy imports deferred so --help works without the CUDA stack installed.
    import torch
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    model, tok = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=cfg["max_seq"],
        dtype=None,            # auto: bf16 on Blackwell
        load_in_4bit=False,    # 3B in bf16 fits 16 GB; keep full precision
    )

    # last-N-layers parity with MLX `num_layers` (full run = all layers).
    n_layers = model.config.num_hidden_layers
    layers_to_transform = (
        None if cfg["num_layers"] < 0
        else list(range(max(0, n_layers - cfg["num_layers"]), n_layers))
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["rank"],
        target_modules=TARGET_MODULES,
        lora_alpha=cfg["alpha"],
        lora_dropout=0,         # 0 keeps unsloth's fused fast path (MLX used 0.05)
        bias="none",
        use_gradient_checkpointing="unsloth",   # 16 GB headroom for 2048 seq
        random_state=args.seed,
        max_seq_length=cfg["max_seq"],
        layers_to_transform=layers_to_transform,
    )

    ds = load_dataset("json", data_files={
        "train": str(args.data / "train.jsonl"),
        "valid": str(args.data / "valid.jsonl"),
    })
    fmt = build_text_field(tok)
    train_ds = ds["train"].map(fmt, batched=True, remove_columns=ds["train"].column_names)
    eval_ds = ds["valid"].map(fmt, batched=True, remove_columns=ds["valid"].column_names)

    # bf16 only where the GPU supports it (Blackwell/Ampere+); else fp16.
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    # trl >=0.2x: SFTConfig seq arg is `max_length` (was `max_seq_length`),
    # SFTTrainer takes `processing_class` (was `tokenizer`). See requirements-cloud.txt pins.
    sft = SFTConfig(
        output_dir=str(out),
        dataset_text_field="text",
        max_length=cfg["max_seq"],
        per_device_train_batch_size=micro,
        gradient_accumulation_steps=grad_accum,
        max_steps=max_steps,
        learning_rate=2e-4,
        optim=args.optim,
        lr_scheduler_type="constant",   # MLX used a flat 2e-4, no decay/warmup
        warmup_steps=0,
        logging_steps=max(1, max_steps // 30),
        eval_strategy="steps",
        eval_steps=max(1, max_steps // 5),
        per_device_eval_batch_size=micro,
        save_strategy="steps",
        save_steps=max(1, cfg["save_every"] // grad_accum),
        save_total_limit=3,
        seed=args.seed,
        report_to="none",
        bf16=use_bf16,
        fp16=not use_bf16,
    )

    trainer = SFTTrainer(model=model, processing_class=tok,
                         train_dataset=train_ds, eval_dataset=eval_ds, args=sft)
    # Mask everything but the assistant turn — the MLX `mask_prompt: true`.
    trainer = train_on_responses_only(
        trainer, instruction_part=INSTRUCTION_PART, response_part=RESPONSE_PART)

    trainer.train()

    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tok.save_pretrained(str(out))
    print(f"\nSaved LoRA adapter to {out}")
    print(f"Eval it with:  python cuda/eval_cuda.py --adapter {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
