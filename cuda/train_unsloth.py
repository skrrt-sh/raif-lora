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
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

# Stage ladder — mirrors ../configs/llama-3-3b-sft-{smoke,warm,full}.yaml.
# `iters` is the MLX micro-batch count (each = one batch of `batch`); HF counts
# OPTIMIZER steps, so max_steps = iters // grad_accum keeps examples-seen (and
# therefore epochs) identical to the MLX runs. alpha = rank * scale(2.0).
STAGES = {
    "smoke": dict(
        iters=300,
        rank=16,
        alpha=32,
        num_layers=16,
        max_seq=1024,
        batch=4,
        grad_accum=1,
        save_every=100,
    ),
    "warm": dict(
        iters=1500,
        rank=32,
        alpha=64,
        num_layers=16,
        max_seq=1024,
        batch=4,
        grad_accum=1,
        save_every=250,
    ),
    "full": dict(
        iters=7000,
        rank=32,
        alpha=64,
        num_layers=-1,
        max_seq=2048,
        batch=4,
        grad_accum=4,
        save_every=500,
    ),
}

TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# Chat-template part markers — used to mask the prompt tokens so loss falls only
# on the assistant's RAIF emission (the MLX `mask_prompt: true`). These are
# template-family specific: Llama-3.x uses header-id markers, Qwen2.5 (and other
# ChatML bases) use <|im_start|>role\n. `chat_markers()` picks the right pair from
# the base-model id; --instruction-part/--response-part override it explicitly.
CHAT_MARKERS = {
    "llama": (
        "<|start_header_id|>user<|end_header_id|>\n\n",
        "<|start_header_id|>assistant<|end_header_id|>\n\n",
    ),
    "qwen": ("<|im_start|>user\n", "<|im_start|>assistant\n"),
}


def chat_markers(model_id: str) -> tuple[str, str]:
    """Pick (instruction_part, response_part) for the base model's chat template.
    Defaults to Llama; switches to ChatML for Qwen/ChatML-family bases."""
    mid = model_id.lower()
    if "qwen" in mid or "chatml" in mid:
        return CHAT_MARKERS["qwen"]
    return CHAT_MARKERS["llama"]


def parse_args() -> argparse.Namespace:
    """Parse the trainer CLI (stage, base model, data dir, hyperparam overrides)."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--stage",
        required=True,
        choices=sorted(STAGES),
        help="which rung of the ladder to run",
    )
    p.add_argument(
        "--model",
        default="unsloth/Llama-3.2-3B-Instruct",
        help="base model (HF id or local path). unsloth's mirror is "
        "ungated and matches meta's weights.",
    )
    p.add_argument(
        "--data",
        type=Path,
        default=Path("./data"),
        help="dir with train.jsonl + valid.jsonl (default ./data)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="adapter output dir (default ./adapters-cuda/<stage>)",
    )
    p.add_argument(
        "--micro-batch",
        type=int,
        default=None,
        help="override per-device batch (lower it + raise grad-accum "
        "to fit 16 GB on the full run; effective batch is kept)",
    )
    p.add_argument(
        "--optim",
        default="adamw_8bit",
        help="optimizer (adamw_8bit needs bitsandbytes w/ Blackwell "
        "support; use adamw_torch if it errors on sm_120)",
    )
    p.add_argument("--seed", type=int, default=0, help="MLX-parity seed (default 0)")
    p.add_argument(
        "--iters",
        type=int,
        default=None,
        help="override the stage's micro-batch count (MLX 'iters'); raise it to "
        "train more epochs over a larger dataset. examples-seen scales linearly.",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="learning rate (default 2e-4; lower e.g. 1e-4 to curb overfitting)",
    )
    p.add_argument(
        "--lora-dropout",
        type=float,
        default=0.0,
        help="LoRA dropout (default 0 keeps unsloth's fused fast path; >0 "
        "regularizes but disables the fused kernels, so training is slower)",
    )
    p.add_argument(
        "--instruction-part",
        default=None,
        help="chat-template marker that precedes the user turn (overrides "
        "the auto-detected Llama/Qwen default; for prompt masking)",
    )
    p.add_argument(
        "--response-part",
        default=None,
        help="chat-template marker that precedes the assistant turn "
        "(overrides the auto-detected Llama/Qwen default)",
    )
    p.add_argument(
        "--export-tar",
        action="store_true",
        help="after saving, tar the adapter dir to <out>.tgz for pulling "
        "off the pod (/workspace is wiped on terminate)",
    )
    return p.parse_args()


def count_jsonl(path: Path) -> int:
    """Count non-blank lines (examples) in a JSONL file."""
    with path.open() as f:
        return sum(1 for line in f if line.strip())


def validate_data(data_dir: Path) -> dict:
    """Fail fast (before the heavy CUDA imports) if the data is missing or empty.
    Returns {train, valid} example counts for the run record."""
    counts = {}
    for name in ("train.jsonl", "valid.jsonl"):
        path = data_dir / name
        if not path.exists():
            raise SystemExit(
                f"✗ missing data file: {path}\n"
                f"  generate it first:  bash src/make_data.sh <smoke|warm|full>"
            )
        n = count_jsonl(path)
        if n == 0:
            raise SystemExit(f"✗ empty data file: {path}")
        counts[name.split(".")[0]] = n
    return counts


def git_sha() -> str | None:
    """Short HEAD commit SHA for the run record, or None outside a git checkout."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


# Qwen3 chat templates inject an empty `<think>\n\n</think>\n\n` block right after
# the `<|im_start|>assistant\n` marker. We deliberately KEEP it in the training
# target: stripping it removes the buffer between the marker and the content, which
# breaks train_on_responses_only's token-level match of the response marker (every
# example masks to all -100 and gets dropped → num_samples=0). So the model learns
# to emit the empty think block then RAIF; the block is stripped at the decode
# boundary instead (eval_core.strip_think_prefix). It's a fixed ~4-token overhead
# and the standard way Qwen3 output is consumed.
def build_text_field(tok):
    """Map {messages,...} -> {"text": <full chat-template string>} for SFT."""

    def fmt(batch):
        """Render each example's messages to the chat-template `text` field."""
        return {
            "text": [
                tok.apply_chat_template(m, tokenize=False) for m in batch["messages"]
            ]
        }

    return fmt


def main() -> int:
    """Run one stage of the LoRA ladder: load, train, save adapter + run record."""
    args = parse_args()
    cfg = dict(STAGES[args.stage])
    if args.iters is not None:
        cfg["iters"] = args.iters
    out = args.out or Path(f"./adapters-cuda/{args.stage}")

    # Keep effective batch constant if the user lowers the micro-batch to fit VRAM.
    micro = args.micro_batch or cfg["batch"]
    grad_accum = cfg["grad_accum"] * (cfg["batch"] // micro if micro else 1)
    grad_accum = max(1, grad_accum)
    # max_steps counts optimizer steps; iters counts micro-batches (MLX). Divide
    # by the FULL accumulation so examples-seen == the MLX run's.
    max_steps = max(1, cfg["iters"] // grad_accum)

    # Fail fast on missing/empty data BEFORE the multi-second CUDA imports and
    # the model download — cheap to catch, expensive to discover 10 minutes in.
    data_counts = validate_data(args.data)

    print(
        f"[stage={args.stage}] base={args.model} seq={cfg['max_seq']} "
        f"micro_batch={micro} grad_accum={grad_accum} max_steps={max_steps} "
        f"rank={cfg['rank']} alpha={cfg['alpha']} num_layers={cfg['num_layers']}"
    )
    examples_seen = max_steps * micro * grad_accum
    print(
        f"[data] train={data_counts['train']} valid={data_counts['valid']}  "
        f"examples-seen≈{examples_seen} "
        f"(~{examples_seen / data_counts['train']:.2f} epochs)"
    )

    # Heavy imports deferred so --help works without the CUDA stack installed.
    import torch
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only

    model, tok = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=cfg["max_seq"],
        dtype=None,  # auto: bf16 on Blackwell
        load_in_4bit=False,  # 3B in bf16 fits 16 GB; keep full precision
    )

    # last-N-layers parity with MLX `num_layers` (full run = all layers).
    n_layers = model.config.num_hidden_layers
    layers_to_transform = (
        None
        if cfg["num_layers"] < 0
        else list(range(max(0, n_layers - cfg["num_layers"]), n_layers))
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["rank"],
        target_modules=TARGET_MODULES,
        lora_alpha=cfg["alpha"],
        lora_dropout=args.lora_dropout,  # 0 keeps unsloth's fused fast path; >0 regularizes
        bias="none",
        use_gradient_checkpointing="unsloth",  # 16 GB headroom for 2048 seq
        random_state=args.seed,
        max_seq_length=cfg["max_seq"],
        layers_to_transform=layers_to_transform,
    )

    # Read only the `messages` field. The JSONL also carries a `meta` blob whose
    # nested fields have mixed types across rows (e.g. meta.source.label is a number
    # in some rows, a bool in others), which makes HF's pyarrow JSON reader fail with
    # "Column changed from number to boolean". Training never uses meta, so load the
    # lines by hand and keep messages only — sidestepping pyarrow schema inference.
    def load_messages(path: Path):
        """Load a JSONL file into a Dataset of {messages} only (drops the meta blob)."""
        rows = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append({"messages": json.loads(line)["messages"]})
        return Dataset.from_list(rows)

    fmt = build_text_field(tok)
    # NOTE: do NOT pass load_from_cache_file=False here — it interacts badly with
    # trl's downstream SFT processing and yields an empty train set (num_samples=0).
    # The rendered `text` depends on the chat template, and the map cache is keyed on
    # (data fingerprint, fmt hash, tokenizer), so it already invalidates when the base
    # model or fmt changes. If you edit fmt's logic without changing its signature,
    # clear the cache once:  rm -rf "$HF_HOME/datasets"  (or ~/.cache/huggingface/datasets)
    train_ds = load_messages(args.data / "train.jsonl").map(
        fmt, batched=True, remove_columns=["messages"]
    )
    eval_ds = load_messages(args.data / "valid.jsonl").map(
        fmt, batched=True, remove_columns=["messages"]
    )

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
        learning_rate=args.lr,
        optim=args.optim,
        lr_scheduler_type="constant",  # MLX used a flat 2e-4, no decay/warmup
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

    trainer = SFTTrainer(
        model=model,
        processing_class=tok,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=sft,
    )
    # Mask everything but the assistant turn — the MLX `mask_prompt: true`.
    # Markers are template-family specific (Llama header-id vs Qwen/ChatML);
    # auto-detect from the base id, with explicit CLI overrides.
    auto_instr, auto_resp = chat_markers(args.model)
    instruction_part = args.instruction_part or auto_instr
    response_part = args.response_part or auto_resp
    print(
        f"[mask] instruction_part={instruction_part!r} response_part={response_part!r}"
    )
    trainer = train_on_responses_only(
        trainer, instruction_part=instruction_part, response_part=response_part
    )

    t0 = time.time()
    train_result = trainer.train()
    train_secs = time.time() - t0

    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tok.save_pretrained(str(out))

    # ── Run record (so a saved adapter is self-describing & reproducible) ──
    # Pull the final train loss and the last eval loss out of the trainer's
    # step log, and persist the whole curve for later plotting / regression.
    log_history = list(getattr(trainer.state, "log_history", []))
    final_train_loss = next(
        (e["loss"] for e in reversed(log_history) if "loss" in e), None
    )
    final_eval_loss = next(
        (e["eval_loss"] for e in reversed(log_history) if "eval_loss" in e), None
    )
    run_meta = {
        "stage": args.stage,
        "base_model": args.model,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "seed": args.seed,
        "hyperparams": {
            "iters": cfg["iters"],
            "rank": cfg["rank"],
            "alpha": cfg["alpha"],
            "num_layers": cfg["num_layers"],
            "target_modules": TARGET_MODULES,
            "max_seq": cfg["max_seq"],
            "micro_batch": micro,
            "grad_accum": grad_accum,
            "max_steps": max_steps,
            "learning_rate": args.lr,
            "lr_scheduler": "constant",
            "optim": args.optim,
            "lora_dropout": args.lora_dropout,
            "instruction_part": instruction_part,
            "response_part": response_part,
        },
        "data": {
            **data_counts,
            "examples_seen": examples_seen,
            "epochs": round(examples_seen / data_counts["train"], 3),
            "dir": str(args.data),
        },
        "result": {
            "train_seconds": round(train_secs, 1),
            "final_train_loss": final_train_loss,
            "final_eval_loss": final_eval_loss,
            "train_runtime_metrics": getattr(train_result, "metrics", None),
        },
    }
    (out / "run_meta.json").write_text(json.dumps(run_meta, indent=2))
    (out / "train_log.json").write_text(json.dumps(log_history, indent=2))

    print(f"\nSaved LoRA adapter to {out}")
    print(
        f"  run_meta.json  — stage/hyperparams/data/result "
        f"(train {train_secs / 60:.1f} min, "
        f"loss {final_train_loss} → eval {final_eval_loss})"
    )
    print(f"  train_log.json — {len(log_history)} step records")

    if args.export_tar:
        tar_path = out.with_suffix(".tgz")
        subprocess.run(
            ["tar", "czf", str(tar_path), "-C", str(out.parent), out.name], check=True
        )
        print(f"  exported {tar_path}  (pull with: runpodctl send {tar_path})")

    print(
        f"\nEval it with:  python cuda/eval_cuda.py --adapter {out} "
        f"--gate {args.stage} --out {out}/eval.json"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
