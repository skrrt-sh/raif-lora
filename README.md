<p align="center">
  <img src="assets/banner.jpg" alt="RAIF" width="640">
</p>

<h1 align="center">raif-lora</h1>

<p align="center"><strong>A LoRA fine-tune that teaches Llama-3.2-3B to emit RAIF instead of JSON</strong></p>

<p align="center">
  Brings <a href="https://github.com/skrrt-sh/raif-standard">RAIF</a>'s token savings and<br>
  truncation recovery to small, local, and self-hosted inference.
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="License: Apache-2.0"></a>
  <img src="https://img.shields.io/badge/base-Llama--3.2--3B-blue" alt="Base: Llama-3.2-3B">
  <img src="https://img.shields.io/badge/acceptance%20gate-PASS-brightgreen" alt="Acceptance gate: PASS">
  <a href="https://huggingface.co/skrrt-sh/raif-llama-3.2-3b-lora"><img src="https://img.shields.io/badge/model-Hugging%20Face-ffb000" alt="Model on Hugging Face"></a>
</p>

---

RAIF is fluent on large (~20B+) models but marginal below 8B — small models haven't
seen the format and fall back to malformed JSON. This adapter closes that gap: it
teaches **Llama-3.2-3B** to emit RAIF natively, so the format's token savings and
self-repair reach the model tier people actually run locally.

The trained artifact is a LoRA adapter (~195 MB), published on Hugging Face:

> **[skrrt-sh/raif-llama-3.2-3b-lora](https://huggingface.co/skrrt-sh/raif-llama-3.2-3b-lora)**

## Results

`parse` = output decodes; `fidelity` = byte-exact JSON round-trip. The published
adapter (`full-reg`) clears all four gate criteria, evaluated at n=64:

| group | parse | fidelity |
|---|---:|---:|
| valid (held-out split of in-training shapes) | **100%** | **100%** |
| holdout (shapes withheld from training entirely) | **100%** | **95%** |

Token cost: **−14% vs minified JSON**, inside the −8% acceptance bar.

### How it got there

Two levers moved the numbers, tracked stage by stage:

| stage | lr | dropout | valid fid | holdout fid |
|---|---:|---:|---:|---:|
| baseline (synthetic-only) | 2e-4 | 0 | 69% | 23% |
| + real-data augmentation | 2e-4 | 0 | 94% | 81% |
| + regularization (**published**) | 1e-4 | 0.05 | **100%** | **95–100%** |

1. **Data** — fixing mechanism coverage in the synthetic corpus and adding real
   tool-call arguments from [glaive-function-calling-v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2)
   took holdout fidelity from 23% to 81%.
2. **Regularization** — lowering the learning rate and adding LoRA dropout fixed
   the mild over-fit (held-out parse had dipped to 88%) and lifted everything to
   the gate.

The exact winning configuration, hyperparameters, and reproduction commands are in
[**`RECIPE.md`**](./RECIPE.md).

## Use the adapter

Load the base model and the adapter:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("unsloth/Llama-3.2-3B-Instruct")
tok = AutoTokenizer.from_pretrained("skrrt-sh/raif-llama-3.2-3b-lora")
model = PeftModel.from_pretrained(base, "skrrt-sh/raif-llama-3.2-3b-lora")
```

The model emits **RAIF, not JSON** — so the one step every consumer needs is a
decode at the output boundary. RAIF is a deterministic codec, not something a
harness has to be taught: run `decode()` and you get a JSON value back, plus a
repair pass that recovers truncated or malformed output that plain JSON can't.

This repo ships a dependency-free Python decoder so you don't need a `bun`
subprocess in the hot path:

```python
from raif_decode import decode  # pure stdlib; no torch/bun (src/ on path)

raif = generate(model, tok, prompt)   # whatever your generation call returns
result = decode(raif)                  # {"ok", "value"/"error", "repairs"}
if result["ok"]:
    data = result["value"]             # ← ordinary JSON; feed it downstream
```

`decode_lenient()` is the per-leaf-recovery variant for agent runtimes that
re-ask the model for only the broken fields. Both mirror the canonical
TypeScript decoder in [`raif-standard`](https://github.com/skrrt-sh/raif-standard).
That equivalence is enforced two ways: `src/test_raif_decode.py` pins parity over
the full corpus (21k+ strings), and `src/test_raif_differential.py` fuzzes both
decoders against each other — random objects round-tripped through the real TS
encoder, then degraded by mutations targeting every repair branch (truncation,
fences, markers, CRLF, nonce/delimiter, brace-flatten, schema-typed, pure
garbage) — asserting `decode_py(x) ≡ decode_ts(x)` for every `x`.

## Training stacks

Two interchangeable stacks share the same data and the same eval meter. The
published adapter was trained on the CUDA stack.

| stack | location | hardware | notes |
|---|---|---|---|
| unsloth / CUDA | `cuda/` | NVIDIA | produced the published adapter; ~3–4× faster |
| MLX | `configs/` + `src/` | Apple Silicon | hyperparameter parity, comparable numbers |

## Reproduce

> Clone [`raif-standard`](https://github.com/skrrt-sh/raif-standard) as a sibling
> and run `bun install` in its `prototype/` — the eval shells out to RAIF's real
> canonical decoder, not a reimplementation.

**NVIDIA (unsloth):**

```sh
pip install -r cuda/requirements.txt   # Blackwell notes in cuda/README.md
python cuda/train_unsloth.py --stage smoke
python cuda/eval_cuda.py --adapter ./adapters-cuda/smoke --n 13
```

**Apple Silicon (MLX):**

```sh
uv sync
src/make_data.sh smoke
uv run python src/test_eval_smoke.py   # meter green first
uv run mlx_lm.lora --config configs/llama-3-3b-sft-smoke.yaml
```

Then climb the ladder (`--stage warm`, `--stage full`). Don't advance a stage until
its gate in [`ITERATION_PLAN.md`](./ITERATION_PLAN.md) passes. For the exact
gate-clearing run, follow [`RECIPE.md`](./RECIPE.md).

## Acceptance gate

Per the v0.5 plan: **parse ≥ 98%, fidelity ≥ 95%, token delta ≤ −8%, no held-out
regression.** Base locked to Llama-3.2-3B for the first ship.

## Project layout

```
RECIPE.md                the gate-clearing run: config, ladder, reproduction
ITERATION_PLAN.md        staged ramp (smoke → warm → full) + go/no-go gates
cuda/                    unsloth/NVIDIA stack (+ push_to_hub.py)
configs/ · src/          MLX stack and shared tooling
src/eval_core.py         framework-free parse/fidelity meter
src/test_eval_smoke.py   oracle tests that pin the meter
src/raif_decode.py       pure-Python RAIF→JSON decoder (the consumer boundary)
src/test_raif_decode.py  corpus parity test vs the canonical TS decoder
src/test_raif_differential.py  differential fuzz: py decoder ≡ TS decoder
grammars/                raif.gbnf + lint
```

`data/ · adapters/ · models/ · logs/` are gitignored; regenerate data with
`src/make_data.sh` (or `src/make_data_augmented.sh` for the real-data corpus).

## License and attribution

- **Code and tooling in this repo:** [Apache-2.0](LICENSE).
- **The trained adapter** is a derivative of Llama 3.2 — the **Llama 3.2 Community
  License** applies ("Built with Llama").
- **Training data** includes `glaiveai/glaive-function-calling-v2` (Apache-2.0) —
  attribute Glaive AI.
