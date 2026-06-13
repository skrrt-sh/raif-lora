# RAIF LoRA v0.5 — the recipe that cleared the acceptance gate

Pinned 2026-06-13. This is the exact, reproducible recipe for the adapter that
hit **100% parse / 100% fidelity** on both in-training and held-out shapes and
cleared the full v0.5 acceptance gate. Machine-readable copies of every run's
config live in each adapter's `run_meta.json`; this file is the human summary.

## Result ladder (augmented data)

| stage | iters | lr | dropout | valid parse | valid fid | holdout parse | holdout fid |
|---|---:|---:|---:|---:|---:|---:|---:|
| old baseline (synthetic-only) | 300 | 2e-4 | 0 | 77% | 69% | 77% | 23% |
| smoke-aug | 300 | 2e-4 | 0 | 94% | 75% | 94% | 38% |
| warm-aug | 1500 | 2e-4 | 0 | 94% | 81% | 100% | 62% |
| full-aug | 12000 | 2e-4 | 0 | 100% | 94% | 88% | 81% |
| **full-reg (WINNER)** | **12000** | **1e-4** | **0.05** | **100%** | **100%** | **100%** | **100%** |

Token side of acceptance: **−14% vs minified JSON** (0.86× ≤ the 0.92× bar), from `bun bench` in the prototype.

The two levers that mattered:
1. **Data** — fixing mechanism coverage + adding real tool-call data (below) took holdout fidelity 23%→81%.
2. **Regularization** — lowering LR 2e-4→1e-4 and adding LoRA dropout 0.05 (vs the over-fit `full-aug`, eval_loss 0.011) recovered holdout parse 88%→100% and lifted everything to 100% (eval_loss 0.0056).

## Base model

`unsloth/Llama-3.2-3B-Instruct` (ungated mirror of Meta's weights). The trained
artifact is a LoRA adapter, not merged weights.

## Dataset recipe (the augmented corpus)

Two sources, merged. Holdout stays **synthetic-only** (it's a shape-generalization probe).

**A. Synthetic** (`raif-standard/prototype/src/dataset.ts`, `full` preset = 500 var/shape)
- 16 in-training shapes incl. three **mechanism-carrier** shapes added in this
  workstream so each hard wire-format mechanism is learned in-distribution
  (previously some were taught ONLY by a held-out shape → never learned):
  - `record_with_note` — multiline `<<<…>>>` block (held twin: `multiline_body`)
  - `dotted_paths` — `<<<key>>>` wrapping (held twin: `pathological_keys`)
  - `nested_event_log` — bracket array under nesting (held twin: `deep_array_literal`)
- 5 held-out shapes: `multiline_body, pathological_keys, large_table, deep_array_literal, flat_inline_object`
- completions encoded with the **generation profile** (ADR-0019); ~50/50 translate/instruct; `<schema>` block per the existing rules.

**B. Real** (`raif-standard/prototype/src/ingest_glaive.ts`)
- Source: **glaiveai/glaive-function-calling-v2** (Apache-2.0, ungated).
- Extract each tool-call `arguments` object; keep ONLY objects that round-trip
  losslessly through the canonical codec (0 failures observed); dedupe by
  canonical JSON; render through the SAME `renderExample` path as synthetic.
- 83,673 raw args → 10,702 unique → all used (`--max 0`).

**Merged:** train 18,277 (≈58% real) / valid 425 / holdout 2,500. `check_data.py` clean (21,202/21,202 leaf-containment, balanced stratification, no holdout leakage).

### Reproduce the data

```sh
# one-time: download the Glaive dataset from the LFS CDN (datasets-server rate-limits big pulls)
curl -sL https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2/resolve/main/glaive-function-calling-v2.json \
  -o /tmp/glaive_full.json
# build synthetic(full) + real, merge:
cd raif-lora
REAL_MAX=0 bash src/make_data_augmented.sh full /tmp/glaive_full.json
uv run python src/check_data.py     # must say ALL CHECKS PASSED
```

## Winning training config

Stage `full` with the regularization overrides:

```sh
cd raif-lora                      # on a CUDA box (this run: RunPod A40, 46 GB)
export PATH="$HOME/.bun/bin:$PATH" HF_HOME="/workspace/.cache/huggingface/"
python cuda/train_unsloth.py --stage full \
  --iters 12000 --lr 1e-4 --lora-dropout 0.05 \
  --out ./adapters-cuda/full-reg --export-tar
python cuda/eval_cuda.py --adapter ./adapters-cuda/full-reg --n 64 \
  --gate full --out ./adapters-cuda/full-reg/eval.json
```

Effective hyperparameters (from `run_meta.json`):

| knob | value |
|---|---|
| iters (MLX micro-batches) | 12000 → max_steps 3000 (grad_accum 4) |
| examples seen / epochs | 48,000 / **2.63** over 18,277 train |
| learning rate / schedule | **1e-4**, constant, no warmup |
| LoRA dropout | **0.05** (disables unsloth fused kernels → ~1.4× slower) |
| rank / alpha | 32 / 64 |
| target modules | q,k,v,o,gate,up,down |
| layers | all (`num_layers=-1`) |
| seq length | 2048 |
| micro batch / grad accum | 4 / 4 (eff. batch 16) |
| optimizer | adamw_8bit |
| prompt masking | train_on_responses_only (loss on assistant turn only) |
| seed | 0 |
| wall time (A40) | 87.8 min |
| final train / eval loss | 0.0041 / 0.0056 |

## Provenance / licensing

- Base weights: **Llama 3.2 Community License** — applies to the base and any
  derivative (the LoRA, and anything served by merging it). Include Meta's
  license + "Built with Llama" attribution when distributing.
- Real training data: **glaiveai/glaive-function-calling-v2, Apache-2.0** —
  attribute Glaive AI.
- Synthetic data + RAIF format: this repo / raif-standard.

## Where the artifacts are

- Adapter tarballs (gitignored): `raif-lora/adapters-cuda/{full-aug,full-reg}.tgz`
- Per-stage metrics: `raif-lora/logs/pod-{smoke,warm,full}-aug/`, `logs/pod-full-reg/`
  (`eval.json` = per-example rows + gate; `run_meta.json` = the config above).
