# raif-lora

![base](https://img.shields.io/badge/base-Llama--3.2--3B-blue)
![stacks](https://img.shields.io/badge/stacks-MLX%20%2B%20unsloth-black)
![smoke](https://img.shields.io/badge/smoke%20fidelity-69%25-brightgreen)
![target](https://img.shields.io/badge/gate-98%25%20parse%20%2F%2095%25%20fid-orange)

LoRA fine-tune that teaches **Llama-3.2-3B** to natively emit
[RAIF](https://github.com/skrrt-sh/raif-standard) instead of JSON — bringing
RAIF's token savings and truncation recovery to cheap/local inference. RAIF is
fluent on ~20B+ models but marginal below 8B; this closes that gap.

Two interchangeable training stacks, same data, same eval meter:

- **MLX** (`configs/` + `src/`) — Apple Silicon, path of record.
- **unsloth/CUDA** (`cuda/`) — NVIDIA mirror, hyperparameter parity, ~3–4× faster (built for RTX 5070 Ti / Blackwell).

## Result so far

A 300-iter smoke run hits **69% fidelity** on in-training shapes (23% held-out),
up from 6%/0% on the old data — the prior "0% fidelity" wall was a data artifact
(no values in prompts), not a recipe limit. Remaining misses are the
newline-bounded-delimiter shapes. Full ramp and gates: [`ITERATION_PLAN.md`](./ITERATION_PLAN.md).

## Features

- **Two stacks, one ladder** — identical smoke→warm→full stages + gates on MLX or NVIDIA; comparable numbers.
- **Trustworthy meter** — parse/fidelity is pinned by oracle tests (`src/test_eval_smoke.py`) and decodes through RAIF's *real* canonical decoder, not a reimplementation.
- **Gated ramp** — 50 → 300 → 1500 → 7000 iters, each with a go/no-go fidelity gate, so long runs never start until short ones earn it.
- **Reproducible data** — synthetic RAIF data regenerates from the spec corpus; nothing hand-labeled.

## Usage

> Clone [`raif-standard`](https://github.com/skrrt-sh/raif-standard) as a sibling
> and run `bun install` in its `prototype/` — the eval shells out to its decoder.

**Apple Silicon (MLX):**

```sh
uv sync
src/make_data.sh smoke
uv run python src/test_eval_smoke.py     # meter green first
uv run mlx_lm.lora --config configs/llama-3-3b-sft-smoke.yaml
uv run python src/eval_smoke.py --adapter ./adapters/llama-3-3b-raif-sft-smoke --n 13
```

**NVIDIA (unsloth):**

```sh
pip install -r cuda/requirements.txt     # Blackwell notes in cuda/README.md
python cuda/train_unsloth.py --stage smoke
python cuda/eval_cuda.py     --adapter ./adapters-cuda/smoke --n 13
```

Then climb the ladder: `--stage warm`, `--stage full`. Don't advance a stage
until its gate in [`ITERATION_PLAN.md`](./ITERATION_PLAN.md) passes.

## Layout

```
ITERATION_PLAN.md   staged ramp + gates       NOTES.md   loss curves, findings
configs/            MLX LoRA configs          cuda/      unsloth mirror + setup
src/eval_core.py    shared parse/fidelity meter (framework-free)
src/test_eval_smoke.py   oracle tests         grammars/  raif.gbnf + lint
```

`data/ · adapters/ · models/ · logs/` are gitignored — regenerate with
`src/make_data.sh`.

## Acceptance gate

Per the v0.5 plan: **parse ≥ 98%, fidelity ≥ 95%, token Δ ≤ −8%, no held-out
regression.** Base locked to Llama-3.2-3B until the gate clears.
