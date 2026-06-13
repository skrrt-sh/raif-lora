# RAIF LoRA — iteration-scaling plan (pinned 2026-06-13)

Goal: ramp training iterations gradually, with a go/no-go eval gate between
each stage, so we never spend battery/time on a longer run until the shorter
one has earned it. Plan-of-record for hyperparams is
`../raif-standard/docs/fine_tune_plan.md`; this file is the *operational* ramp.

## Why a ramp (not just "run the full config")

The full SFT is 7,000 iters ≈ 12 hr on this M4 Max. The two prior failure
modes were both cheap to catch early:

1. **0% fidelity** (warm run, old data) — a *data* artifact: prompts had no
   values to learn from. Caught at 300 iters on regenerated data (fidelity
   69%, see below). A 12-hr run would have reproduced 0% at full cost.
2. **Eval-meter trust** — the score itself was suspect. Now pinned by
   `src/test_eval_smoke.py` (perfect→100, corruption→fidelity 0, garbage→
   parse 0, colon-refusal→parse via repair). Run it before trusting any gate.

## The meter (run first, every time)

```sh
cd raif-lora
uv run python src/test_eval_smoke.py     # 4 oracle tests — must be OK
uv run python src/check_data.py          # data containment/leakage/stratify
```

## Dataset (v0.5 overhaul — 2026-06-13)

The generator (`../raif-standard/prototype/src/dataset.ts`) gained three
in-training **mechanism-carrier** shapes, fixing a hole where some wire-format
mechanisms were taught *only* by held-out shapes (so the model never learned
them, and the holdout measured "never saw it" instead of generalization):

| mechanism | held-out (test) | new in-training carrier |
|---|---|---|
| multiline `<<<…>>>` block | `multiline_body` | `record_with_note` |
| `<<<key>>>` key wrapping | `pathological_keys` | `dotted_paths` |
| bracket array under nesting | `deep_array_literal` | `nested_event_log` |

The held-out set is unchanged, so old numbers stay comparable; the carriers
add coverage, not leakage (`check_data.py` still passes: 2100/2100 containment,
stratified, zero holdout leakage). The pre-overhaul `smoke` adapter parses the
13 original shapes at the same 69% but fails all three new shapes — they need a
fresh train run to pay off. **Regenerate, then re-climb from smoke.**

## Gates are now machine-checked

`eval_{cuda,smoke}.py --gate <stage>` checks the numeric gate below and exits
nonzero on FAIL; `--out results.json` persists per-example rows + the verdict.
On CUDA, `bash cuda/run_stage.sh <stage>` runs meter→data→train→eval+gate→export
as one command and refuses to advertise the next rung until the gate passes.

## Stage gates

Each stage's gate must pass before starting the next. "valid" = in-training
shapes (`data/valid.jsonl`); "holdout" = shapes withheld from training
(`data/eval_holdout.jsonl`, the harder generalization test).

| Stage | iters | rank | data | ~time | Gate to advance |
|---|---:|---:|---|---:|---|
| 0 pipeline-check | 50 | 16 | current | ~1 min | adapter saves; eval runs; loss ↓. **DONE** |
| 1 smoke | 300 | 16 | current 1235/65/500 | ~4 min | valid fidelity ≥ 50%. **DONE — 69%** |
| 2 warm | 1500 | 32 | current | ~17 min | valid fidelity ≥ 75% AND holdout fidelity > smoke's 23% |
| 3 mid | 3500 | 32 | regen @ ~300 var | ~40 min | valid parse ≥ 95% AND multi-line shapes start parsing |
| 4 full | 7000 | 32 | regen @ 500 var + adversarial | ~12 hr (on charger) | acceptance gate (below) |

Run a stage:
```sh
uv run mlx_lm.lora --config configs/llama-3-3b-sft-<stage>.yaml \
  --adapter-path ./adapters/<name> 2>&1 | tee logs/<name>.log
uv run python src/eval_smoke.py --adapter ./adapters/<name> --n 13
```

## Two stacks, same ladder & gates

The ladder runs on either stack — pick by hardware. **The MLX path is the
working one of record**; the CUDA path is a faster mirror (see `cuda/README.md`).

| | MLX (Mac, M4 Max) | CUDA (RTX 5070 Ti, Blackwell) |
|---|---|---|
| train | `uv run mlx_lm.lora --config configs/<stage>.yaml` | `python cuda/train_unsloth.py --stage <stage>` |
| eval | `uv run python src/eval_smoke.py --adapter ...` | `python cuda/eval_cuda.py --adapter ...` |
| full run wallclock | ~12 hr | ~2–4 hr |
| eval meter | shared `src/eval_core.py` (bun decoder) — identical, comparable numbers |

Hyperparams are held in parity across both (rank/alpha, target modules,
last-N-layers, LR, seq, effective batch, prompt masking, seed); the few
deliberate CUDA divergences are listed in `cuda/README.md`.

## Acceptance gate (stage 4 → ship, plan §1)

- parse ≥ 98%
- fidelity ≥ 95%
- ≤ 0.92× JSON tokens
- no held-out regression

## Pinned results

### Stage 0 — pipeline-check (50 iters, rank 16, regenerated data)
Train loss 4.10→0.24, val 1.76→0.83, peak 21.9 GB, ~1 min. Adapter saved,
eval ran clean. Pipeline end-to-end verified on the regenerated data.
valid parse 100% / fidelity 10%; holdout parse 100% / fidelity 10%
(50 iters is too few to fit values — expected).

### Stage 1 — smoke (300 iters, rank 16, regenerated data) — HEADLINE
Train loss → 0.022, val 1.76 → 0.068. Peak 27.4 GB.

| group | parse | fidelity |
|---|---:|---:|
| valid (in-training shapes) | 10/13 (77%) | **9/13 (69%)** |
| holdout (withheld shapes)  | 10/13 (77%) | 3/13 (23%) |

**The 0%-fidelity wall is gone.** Old data: smoke 6%, warm 0%. Regenerated
data (values-in-prompts, per ADR-0018/0019): **69% in-training fidelity at
the same 300 iters / rank 16.** This is the single most important signal in
the workstream — the recipe was never the problem, the data was.

Parse failures (both groups) are exactly the newline-bounded-delimiter
shapes: `multiline_body`, `deep_array_literal`, `wide_heterogeneous_array`,
`json_heavy`, `literal_strings`. Same persistent class NOTES flagged at the
warm run — they need more iters/data on the multi-line shapes, or
GBNF-constrained decoding (`grammars/raif.gbnf`), not a recipe change.

Holdout fidelity (23%) lagging valid (69%) is expected this early —
generalization to unseen shapes is what stages 2–4 buy.

## Next action (resume here)

Stage 2 — warm (1500 iters, rank 32). The long-pending re-run, now de-risked:
```sh
cd raif-lora
uv run python src/test_eval_smoke.py && uv run python src/check_data.py
uv run mlx_lm.lora --config configs/llama-3-3b-sft-warm.yaml 2>&1 | tee logs/warm-v042.log
uv run python src/eval_smoke.py --adapter ./adapters/llama-3-3b-raif-sft-warm --n 13
```
Gate: valid fidelity ≥ 75% AND holdout fidelity > 23%. ~17 min GPU — start
on charger (was at 60% battery / discharging when stage 1 finished).
