# RAIF LoRA — CUDA / unsloth port (RTX 5070 Ti)

This folder is the **NVIDIA mirror** of the MLX training flow. It exists so the
same smoke→warm→full ladder can run on a CUDA GPU (~3–4× faster than MLX on the
Mac, no battery worry) while the original MLX path stays fully intact.

**Nothing here replaces the MLX flow.** `uv run mlx_lm.lora --config
../configs/*.yaml` still works on the Mac exactly as before. The eval meter is
*shared* (`../src/eval_core.py`), so numbers from both stacks are comparable.

## What's here

| file | role |
|---|---|
| `train_unsloth.py` | stage-parametrized trainer; mirrors `../configs/llama-3-3b-sft-{smoke,warm,full}.yaml` |
| `eval_cuda.py` | parse/fidelity eval; HF generation + the shared `eval_core` meter + bun decoder |
| `requirements.txt` | pinned Blackwell (sm_120) stack |

## Prerequisites

1. **Blackwell-capable stack** — see the header of `requirements.txt`. The
   sanity check must report capability `(12, 0)`:
   ```sh
   python -c "import torch; print(torch.__version__, torch.cuda.get_device_capability())"
   ```
2. **bun + the prototype decoder** — the eval scores RAIF by shelling out to the
   canonical TS decoder, identical to the MLX eval. On the CUDA box:
   ```sh
   cd ../raif-standard/prototype && bun install   # bun must be on PATH
   ```
   (`eval_core.py` resolves the prototype dir relative to the repo, so once it's
   present no config is needed.)
3. **Data** — uses the same `../data/{train,valid,eval_holdout}.jsonl`. Copy the
   `raif-lora/data/` dir to the CUDA machine, or run on the same checkout.

## Run the ladder

From the **raif-lora root** (so `./data` and `./adapters-cuda` resolve):

```sh
# stage 1 — smoke (~2-3 min on a 5070 Ti)
python cuda/train_unsloth.py --stage smoke
python cuda/eval_cuda.py     --adapter ./adapters-cuda/smoke --n 13

# stage 2 — warm (~8-12 min)
python cuda/train_unsloth.py --stage warm
python cuda/eval_cuda.py     --adapter ./adapters-cuda/warm --n 13

# stage 4 — full (~2-4 hr; 2048 seq)
python cuda/train_unsloth.py --stage full
python cuda/eval_cuda.py     --adapter ./adapters-cuda/full --n 13
```

Stage gates are the same as `../ITERATION_PLAN.md` (e.g. warm: valid fidelity
≥ 75% AND holdout > smoke's). The MLX smoke run already hit **69% valid
fidelity / 23% holdout** at 300 iters — that's the bar the CUDA smoke should
roughly reproduce, validating the port before you trust the longer runs.

### If you hit a 16 GB OOM on the full run (2048 seq)

Lower the micro-batch; the script raises grad-accum to keep the effective batch
(and therefore examples-seen / epochs) identical:

```sh
python cuda/train_unsloth.py --stage full --micro-batch 2
```

### If `adamw_8bit` errors on Blackwell

bitsandbytes may lag on a brand-new arch. Fall back to the full-precision
optimizer (LoRA optimizer states are tiny, so memory cost is negligible):

```sh
python cuda/train_unsloth.py --stage warm --optim adamw_torch
```

## Hyperparameter parity with the MLX configs

Held identical so the runs are comparable:

| knob | value | matches MLX |
|---|---|---|
| base weights | `unsloth/Llama-3.2-3B-Instruct` | same weights as `mlx-community/...-bf16` |
| rank / alpha | 16/32 (smoke), 32/64 (warm, full) | `rank` + `scale 2.0` |
| target modules | q,k,v,o,gate,up,down | mlx-lm auto-target |
| last-N layers | 16 (smoke/warm), all (full) | `num_layers` via `layers_to_transform` |
| LR / schedule | 2e-4 constant, no warmup | flat 2e-4 |
| seq length | 1024 / 1024 / 2048 | `max_seq_length` |
| effective batch | 4 / 4 / 16 | `batch_size × grad_accumulation_steps` |
| examples seen | `max_steps = iters // grad_accum` | MLX `iters` are micro-batches |
| prompt masking | `train_on_responses_only` | `mask_prompt: true` |
| seed | 0 | `seed: 0` |

**Deliberate divergences** (chosen to keep unsloth's speed/memory wins; each is
negligible for a 3B LoRA, but noted so a number gap isn't a mystery):

- **LoRA dropout 0** (MLX used 0.05). Non-zero dropout disables unsloth's fused
  LoRA kernels. The regularization difference on a 3B LoRA is minor.
- **Gradient checkpointing ON** (`"unsloth"`; MLX had it off on a 128 GB host).
  Needed for headroom at 2048 seq on 16 GB. Trades a little speed for memory,
  no effect on the result.
- **Optimizer `adamw_8bit`** by default (MLX used 32-bit adamw). LoRA optimizer
  state is tiny, so the precision difference is immaterial; switch with
  `--optim adamw_torch` if you want exact parity or hit a bnb/Blackwell issue.
