# Examples — run a RAIF LoRA locally

Drive any of the published RAIF fine-tunes on this machine (Apple Silicon / MLX)
and decode its output with the published
[`raif-format`](https://www.npmjs.com/package/raif-format) codec. Four ways in:

1. **`chat.py`** — talk to it in your terminal.
2. **`serve.sh`** — expose it over an OpenAI-compatible HTTP API.
3. **`ai-sdk/`** — drive that API from the Vercel AI SDK.
4. **`compare.py`** — base model vs. RAIF adapter, head to head.

The model emits **RAIF, not JSON**. RAIF is a token-leaner serialization; the one
step every consumer needs is a `decode()` at the output boundary, which returns a
plain JSON value (plus a repair pass for truncated/malformed output). These
examples show exactly that boundary.

## Choose a model

Pass `--model` to any script (default `llama-3b`). Everything is pulled from the
Hub — you don't need any local training artifacts.

| `--model` | base (downloaded) | adapter | size | notes |
|---|---|---|---|---|
| `llama-3b` | `mlx-community/Llama-3.2-3B-Instruct-bf16` | [`raif-llama-3.2-3b-lora`](https://huggingface.co/skrrt-sh/raif-llama-3.2-3b-lora) | ~6 GB | flagship — 100% valid fidelity |
| `qwen-0.5b` | `mlx-community/Qwen2.5-0.5B-Instruct-bf16` | [`raif-qwen2.5-0.5b-lora`](https://huggingface.co/skrrt-sh/raif-qwen2.5-0.5b-lora) | ~1 GB | tiny & fast — ~98% valid fidelity |
| `qwen-4b` | `mlx-community/Qwen3-4B-Instruct-2507-bf16` | [`raif-qwen3-4b-lora`](https://huggingface.co/skrrt-sh/raif-qwen3-4b-lora) | ~8 GB | deployable agent model |

## One-time setup

Published adapters are in **PEFT (torch) format**; the local runtime is **MLX**,
which uses a different adapter layout. `setup_adapter.py` converts a PEFT LoRA to
MLX once — a mechanical, lossless rename + transpose (`scale = alpha/rank = 2.0`,
see the script). It uses the local `adapters-cuda/*` copy if present, else
downloads the published LoRA from the Hub.

```sh
uv sync                                              # MLX runtime (already a dep)
uv pip install raif-format                           # the codec
uv run python examples/setup_adapter.py --model qwen-0.5b   # or llama-3b / qwen-4b / all
```

Verify it round-trips held-out examples:

```sh
uv run python examples/chat.py --model qwen-0.5b --selftest
```

> `--selftest` and `compare.py` read the eval corpus at `data/*.jsonl`, which is
> gitignored — regenerate it with `src/make_data.sh` (needs the `raif-standard`
> sibling repo; see the root README). The chat, server, and AI SDK demos don't
> need it.

## 1. Terminal chat

```sh
uv run python examples/chat.py --model qwen-0.5b
```

Paste a JSON object to translate it to RAIF (the model's main job), or type any
instruction. Each turn prints the raw RAIF, the decoded JSON, and the byte savings
vs. minified JSON. One-shot mode works too:

```sh
echo '{"user":"ada","tasks":["write","test","ship"],"done":false,"count":3}' \
  | uv run python examples/chat.py --model qwen-0.5b
```

```text
── RAIF (model output) ──
count=3
done=false
user=ada
tasks=[
write
test
ship
]
── decode() -> JSON ──
{ "count": 3, "done": false, "user": "ada", "tasks": ["write","test","ship"] }
53 B RAIF vs 69 B JSON (-23%)
```

## 2. OpenAI-compatible server

```sh
MODEL=qwen-0.5b examples/serve.sh      # -> http://127.0.0.1:8899/v1
```

It prints the adapter path to send in the request body:

```sh
curl -s http://127.0.0.1:8899/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"default_model","adapters":"<path printed by serve.sh>","temperature":0,
       "messages":[{"role":"user","content":"Rewrite this JSON payload as RAIF:\n{\"a\":1,\"b\":[2,3]}"}]}'
```

> **`adapters` in the body?** mlx-lm 0.31.x has a bug where the CLI
> `--adapter-path` isn't applied per request, so the client passes the adapter
> path in the request body. Harmless once the upstream bug is fixed.

## 3. Vercel AI SDK

With the server from step 2 running:

```sh
cd examples/ai-sdk
npm install
RAIF_ADAPTER="$(cd ../.. && pwd)/adapters/qwen-0.5b-mlx" npm run demo   # match serve.sh's model
```

`demo.ts` points `@ai-sdk/openai-compatible` at the local server and decodes the
result with `raif-format`. The AI SDK doesn't care that the model is local — it's
just another OpenAI-compatible endpoint. Any framework that speaks OpenAI (the
OpenAI SDK, LangChain, an agent loop, …) works the same way: **call the endpoint,
then `decode()` the output.**

## 4. Base model vs. RAIF — `compare.py`

```sh
uv run python examples/compare.py --model llama-3b --n 12               # in-training shapes
uv run python examples/compare.py --model llama-3b --data holdout --n 30 # withheld shapes
```

Runs the same structured-output tasks through the **stock** base model and the
**same model + RAIF adapter**, on identical weights, measuring parse / fidelity /
output tokens with one tokenizer. Base is given its *best case* — explicitly
coached with "return compact JSON, nothing else". A representative `llama-3b` run
at ≤3 examples/shape (51 in-training, 15 withheld):

| split | model | parse | fidelity | avg out-tokens |
|---|---|---:|---:|---:|
| valid (in-training shapes) | base (coached JSON) | 50/51 | **69%** | 49 |
| valid | RAIF adapter | 49/51 | **88%** | 47 |
| holdout (withheld shapes) | base (coached JSON) | 14/15 | **53%** | 126 |
| holdout | RAIF adapter | 15/15 | **93%** | 108 |

Takeaways:

- **Fidelity is where the adapter wins.** Both models almost always emit *something*
  parseable, but base is only byte-exact 69% / 53% of the time vs the adapter's
  88% / 93%. The base failures are systematic and land exactly on the shapes RAIF
  was trained to preserve — `numeric_string_ambiguity` (it coerces `"42"`→`42`),
  `dotted_paths`, `deep_nesting`, `pathological_keys`, `deep_array_literal`.
- **The gap is widest on small models.** Run `--model qwen-0.5b` and base
  Qwen2.5-0.5B is both *less accurate and more verbose* — the LoRA makes it leaner
  **and** correct. The smaller the base, the more the fine-tune buys you.
- **The token win tracks complexity** — modest on small payloads (−4 to −6%),
  reaching **−14%** on the larger withheld shapes. The ecosystem headline is the
  real-world figure (**~10%** on actual function-call data); −14% is the
  table-heavy eval-corpus aggregate.
- **Uncoached, base models are unusable for this.** Given the exact training prompt
  (`Rewrite this JSON payload as RAIF: …`) the base doesn't know RAIF — it
  hallucinates and wraps the answer in prose you'd have to scrape. The adapter is
  what turns the format into a deterministic, decode-at-the-boundary contract.

`compare.py` prints this plus a per-example and per-shape breakdown; small samples
are noisy, so bump `--n` to confirm.

## Files

| File | What |
|------|------|
| `raif_models.py` | Model registry (base + adapter per `--model`) and the PEFT→MLX converter |
| `setup_adapter.py` | Convert a published PEFT LoRA → MLX adapter (one-time) |
| `chat.py` | Terminal REPL / one-shot / `--selftest` |
| `serve.sh` | Launch the MLX OpenAI-compatible server |
| `ai-sdk/demo.ts` | Drive the server from the Vercel AI SDK + decode |
| `compare.py` | Base model vs. RAIF adapter, head-to-head |
