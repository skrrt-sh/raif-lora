# raif-lora

LoRA fine-tune for the RAIF v0.5 wire format. Apple Silicon (MLX) variant of the [fine_tune_plan](../raif-standard/docs/fine_tune_plan.md).

The plan as written targets CUDA/axolotl. This repo executes the same recipe with `mlx-lm` on Apple Silicon, which is the only ML stack that uses Metal + unified memory natively. On a 128 GB M4 Max we can hold a 3B base in bf16 + activations + optimizer state without sharding.

## Layout

```
raif-lora/
├── pyproject.toml             ← uv-managed; mlx, mlx-lm, hf-hub
├── configs/
│   ├── llama-3-3b-sft-smoke.yaml   ← 30-min smoke run
│   └── llama-3-3b-sft.yaml         ← full acceptance run
├── src/
│   ├── make_data.sh                ← shells out to ../raif-standard/prototype/src/dataset.ts
│   └── eval_smoke.py               ← parse/fidelity scoring on a few generations
├── data/                           ← train.jsonl / valid.jsonl (gitignored)
├── grammars/                       ← raif.gbnf (hand-written from spec)
├── adapters/                       ← LoRA weights (gitignored)
└── logs/                           ← training logs (gitignored)
```

## Day-zero setup

```sh
# 1. Python env
cd raif-lora
uv sync

# 2. llama.cpp with Metal (for GBNF testing + GGUF inference)
brew install llama.cpp        # ships with Metal on Apple Silicon

# 3. Smoke dataset
src/make_data.sh smoke        # writes data/train.jsonl + data/valid.jsonl

# 4. Smoke train (≈30 min on M4 Max 128 GB, 3B bf16)
uv run mlx_lm.lora --config configs/llama-3-3b-sft-smoke.yaml

# 5. Eyeball
uv run python src/eval_smoke.py
```

## Decisions that diverge from the v0.5 plan

| Plan says | Here we do | Why |
|---|---|---|
| axolotl / unsloth on CUDA | `mlx-lm` on Metal | axolotl has no MPS backend; unsloth is CUDA-only. mlx-lm has first-class LoRA. |
| A100/H100, ~3–5 hr full SFT | M4 Max 128 GB, expect 8–12 hr full SFT | Slower per-step but ample memory; we can use larger micro-batches. |
| Chat template via axolotl `chat_template: llama3` | mlx-lm reads `{"messages":[...]}` JSONL and applies the model's HF chat template automatically | Same wire format; framework difference. |
| bf16 + flash_attention | bf16 unified memory; mlx implements fused attention by default | No flag needed on MLX. |

## Acceptance gate

Same as plan §1 — parse ≥ 98%, fidelity ≥ 95%, token Δ ≤ −8%, no held-out regression.
