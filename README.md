# raif-lora

Teach a small open model to **natively emit [RAIF](https://github.com/skrrt-sh/raif-standard)**
instead of JSON for tool calls and structured output — so you get RAIF's token
savings and truncation recovery without paying for a frontier model.

This is the v0.5 fine-tune workstream for RAIF. It LoRA-tunes
**Llama-3.2-3B-Instruct** on synthetic RAIF data and scores it with a shared
parse/fidelity meter. Two interchangeable training stacks ship here:

- **MLX** (`configs/` + `src/`) — Apple Silicon, the working path of record.
- **unsloth/CUDA** (`cuda/`) — an NVIDIA mirror with hyperparameter parity,
  ~3–4× faster wallclock (built for an RTX 5070 Ti / Blackwell).

Both read the same data and are scored by the same meter, so their numbers are
directly comparable.

## Why fine-tune at all?

RAIF is fluent out-of-the-box on ~20B+ models, but marginal below 8B (a model
knows JSON cold; it only knows RAIF from a few in-prompt examples). A LoRA fixes
that for the 1B–8B class: the goal is a 3B that emits RAIF as reliably as it
emits JSON, making RAIF practical for cheap/local inference. Design rationale is
[ADR-0017](../raif-standard/docs/adr/0017-fine-tune-integration-philosophy.md).

## Headline result so far

The earlier "0% fidelity" wall turned out to be a *data* artifact (training
prompts contained no values to condition on), not a recipe or capacity limit.
On regenerated data with values in the prompts, a **300-iter smoke run** already
hits:

| group | parse | fidelity |
|---|---:|---:|
| in-training shapes | 77% | **69%** |
| held-out shapes | 77% | 23% |

vs **6%/0%** on the old data at the same budget. The remaining parse misses are
all the newline-bounded-delimiter shapes (multi-line bodies, array literals) —
a known class that more iterations or GBNF-constrained decoding addresses. Full
ramp, gates, and numbers are in [`ITERATION_PLAN.md`](./ITERATION_PLAN.md).

## Killer features

- **Two stacks, one ladder.** Identical smoke→warm→full stages and eval gates on
  MLX (Mac) or unsloth (NVIDIA). Pick by hardware; numbers stay comparable.
- **A trustworthy meter.** Parse/fidelity scoring is pinned by oracle tests
  (`src/test_eval_smoke.py`): a perfect echo scores 100%, value corruption fails
  fidelity only, garbage fails parse, and repair-assisted parses are counted
  separately so the gate can't be silently softened. The eval decodes through
  RAIF's *real* canonical decoder, not a reimplementation.
- **Gated, gradual scaling.** A pipeline-check (50 iters) → smoke (300) → warm
  (1500) → full (7000) ramp, each with a go/no-go fidelity gate, so a long run
  never starts until a short one earns it.
- **Reproducible data.** Synthetic RAIF training data regenerates from the spec
  corpus via one script; nothing is hand-labeled.

## Layout

```
raif-lora/
├── ITERATION_PLAN.md           ← the staged ramp + eval gates (start here)
├── NOTES.md                    ← run-of-record: loss curves, findings
├── configs/                    ← MLX LoRA configs: smoke / warm / full
├── src/
│   ├── make_data.sh            ← regenerates data via ../raif-standard prototype
│   ├── eval_core.py            ← shared, framework-free parse/fidelity meter
│   ├── eval_smoke.py           ← MLX eval (loads base+adapter via mlx-lm)
│   ├── test_eval_smoke.py      ← oracle tests pinning the meter
│   └── check_data.py           ← data containment / leakage / stratification checks
├── cuda/                       ← NVIDIA mirror (unsloth)
│   ├── train_unsloth.py        ← stage-parametrized trainer; parity with configs/
│   ├── eval_cuda.py            ← CUDA eval; same meter as the MLX path
│   ├── requirements.txt        ← pinned Blackwell (sm_120) stack
│   └── README.md               ← setup, parity table, OOM/bitsandbytes fallbacks
├── grammars/                   ← raif.gbnf + a built-in lint (GBNF-constrained decode)
├── data/                       ← train/valid/holdout JSONL (gitignored; regenerate)
└── adapters/ · models/ · logs/ ← gitignored (large binaries / artifacts)
```

> **Sibling-checkout dependency:** the eval shells out to RAIF's canonical
> decoder at `../raif-standard/prototype`. Clone
> [`raif-standard`](https://github.com/skrrt-sh/raif-standard) next to this repo
> and run `bun install` in its `prototype/` once.

## How to use it

### Apple Silicon (MLX)

```sh
cd raif-lora
uv sync                                   # mlx, mlx-lm, hf-hub
src/make_data.sh smoke                    # regenerate data/*.jsonl

uv run python src/test_eval_smoke.py      # meter must be green first
uv run python src/check_data.py           # data integrity

uv run mlx_lm.lora --config configs/llama-3-3b-sft-smoke.yaml
uv run python src/eval_smoke.py --adapter ./adapters/llama-3-3b-raif-sft-smoke --n 13
```

### NVIDIA (unsloth, e.g. RTX 5070 Ti)

```sh
pip install -r cuda/requirements.txt      # see cuda/README.md for Blackwell notes
python -c "import torch; print(torch.cuda.get_device_capability())"   # expect (12, 0)

python cuda/train_unsloth.py --stage smoke
python cuda/eval_cuda.py     --adapter ./adapters-cuda/smoke --n 13
```

Both then climb the same ladder: `--stage warm`, then `--stage full`. Each stage
has a fidelity gate in [`ITERATION_PLAN.md`](./ITERATION_PLAN.md); don't advance
until it passes.

## Acceptance gate (full run → ship)

Per the v0.5 plan §1: **parse ≥ 98%, fidelity ≥ 95%, token Δ ≤ −8% vs JSON, no
held-out regression.** Base model is locked to Llama-3.2-3B-Instruct for the
first ship; other bases are deferred until this gate clears.
