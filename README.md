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

Token cost: **−14% vs minified JSON**, inside the −8% acceptance bar — and
[benched across tokenizers](https://github.com/skrrt-sh/raif-standard/tree/main/benchmarks).

### How it got there

Two levers moved the numbers, tracked stage by stage:

| stage | lr | dropout | valid fid | holdout fid |
|---|---:|---:|---:|---:|
| baseline (synthetic-only) | 2e-4 | 0 | 69% | 23% |
| + real-data augmentation | 2e-4 | 0 | 94% | 81% |
| + regularization (**published**) | 1e-4 | 0.05 | **100%** | **95%** |

1. **Data** — fixing mechanism coverage in the synthetic corpus and adding real
   tool-call arguments from [glaive-function-calling-v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2)
   took holdout fidelity from 23% to 81%.
2. **Regularization** — lowering the learning rate and adding LoRA dropout fixed
   the mild over-fit (held-out parse had dipped to 88%) and lifted everything to
   the gate.

The exact winning configuration, hyperparameters, and reproduction commands are in
[**`RECIPE.md`**](./RECIPE.md).

## A deployable agent model: Qwen3-4B

The recipe also ports up to **Qwen3-4B-Instruct-2507** — a non-thinking instruct
model that runs on ~14 GB VRAM, the sweet spot for real self-hosted agents.

> **[skrrt-sh/raif-qwen3-4b-lora](https://huggingface.co/skrrt-sh/raif-qwen3-4b-lora)** · Apache-2.0 base

| group | parse | fidelity |
|---|---:|---:|
| valid (in-training shapes) | 97% | **95%** |
| holdout (withheld shapes) | **98%** | **95%** |

Trained on the same carrier-augmented corpus (12k iters / 2.56 epochs). It clears
the gate on **three of four metrics with margin** — including **holdout fidelity
95%** (the generalization test) and `multiline_body` recovering to 85%. The fourth,
valid-parse at 96.9% (97% rounded), misses the 98% bar by a single example: two of the longest
`tabular_report` tables (17+ rows) overran the eval's 384-token generation cap and
were truncated mid-row — an eval-budget artifact, not a model error (the cap is now
1024). Qwen3 emits an empty `<think></think>` block before its answer; this repo
strips it at the decode boundary (`eval_core.strip_think_prefix`), the standard way
Qwen3 output is consumed.

## Going smaller: Qwen2.5-0.5B

How far down does this push? We ported the same recipe to **Qwen2.5-0.5B-Instruct**
— a model 6× smaller, normally too weak for rigid structured output — to see what a
tiny, local model can do. Same pipeline, same data, same eval meter; only the base
model and its chat-template markers change.

> **[skrrt-sh/raif-qwen2.5-0.5b-lora](https://huggingface.co/skrrt-sh/raif-qwen2.5-0.5b-lora)** · Apache-2.0 base

| group | parse | fidelity |
|---|---:|---:|
| valid (in-training shapes) | 97% | 92% |
| holdout (withheld shapes) | 97% | 81% |

It does **not** clear the full gate (95% holdout fidelity), but it emits valid RAIF
97% of the time and is byte-exact on the realistic cases (100% on real tool-call
arguments). Two findings came out of pushing it:

- **The hardest shapes are a coverage problem, not a capacity wall.** The held-out
  `large_table` shape started at 8% fidelity; adding one in-training
  *mechanism-carrier* that teaches the `::` table form (which the corpus had never
  taught directly) lifted it to **77%** — the 0.5B learns any single mechanism once
  it's shown one in-distribution.
- **At 0.5B, mechanisms compete for capacity.** Teaching the table form *traded off*
  ~20 points of the multiline-block mechanism — a frontier the 3B never hits (it
  holds every mechanism at 100% simultaneously). Clearing the gate from here means
  more capacity (e.g. Qwen2.5-1.5B), not more data.

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

Install the official decoder — [`raif-format`](https://pypi.org/project/raif-format/)
on PyPI (and [`raif-format`](https://www.npmjs.com/package/raif-format) on npm for
JS/TS). It's pure-stdlib with zero dependencies, so there's no `bun` subprocess in
the hot path and nothing to clone:

```sh
pip install raif-format        # or: uv add raif-format
```

```python
from raif import decode        # installs as `raif-format`, imports as `raif`

raif_text = generate(model, tok, prompt)   # whatever your generation call returns
result = decode(raif_text)                 # {"ok", "value"/"error", "repairs"}
if result["ok"]:
    data = result["value"]                 # ← ordinary JSON; feed it downstream
```

`decode_lenient()` is the per-leaf-recovery variant for agent runtimes that
re-ask the model for only the broken fields. `raif-format` is the canonical codec
from [`raif-standard`](https://github.com/skrrt-sh/raif-standard), kept byte-for-byte
identical across Python and TypeScript by a shared conformance corpus.

### Run it locally

[**`examples/`**](./examples) has runnable end-to-end demos on Apple Silicon (MLX),
for any of the three published adapters (`--model llama-3b | qwen-0.5b | qwen-4b`,
everything pulled from the Hub): a terminal chat (`chat.py`), an OpenAI-compatible
server (`serve.sh`), a Vercel **AI SDK** client (`ai-sdk/`), and a head-to-head
**base-vs-RAIF** comparison (`compare.py`). A one-time `setup_adapter.py` converts
the published PEFT adapter to MLX format — the local MLX runtime needs a different
layout than the torch/PEFT weights above. See [`examples/README.md`](./examples/README.md).

> **Note:** for its own offline eval this repo still vendors a standalone copy of
> the decoder at `src/raif_decode.py`, pinned to the canonical TS decoder by
> `src/test_raif_decode.py` (parity over 21k+ strings) and
> `src/test_raif_differential.py` (fuzzes `decode_py(x) ≡ decode_ts(x)` across
> every repair branch). New consumers should use the published `raif-format`
> package above.

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
examples/                runnable local demos (chat, server, AI SDK, base-vs-RAIF)
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
