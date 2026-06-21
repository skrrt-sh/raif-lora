# raif-vllm

One vLLM plugin for transparent RAIF token savings. Install it and existing
OpenAI clients get RAIF on `tools` and `response_format` — **no proxy, no client
changes, no vLLM fork**. The fine-tuned model emits compact RAIF-G; the plugin
decodes it to JSON at the request/response boundary.

## Install

`raif-vllm` is not yet on PyPI; install it from the repo (it pulls
[`raif-format`](https://pypi.org/project/raif-format/) `>=0.6` from PyPI
automatically):

```sh
pip install "raif-vllm @ git+https://github.com/skrrt-sh/raif-lora.git#subdirectory=packages/vllm"
# or, from a checkout:  pip install -e packages/vllm
```

vLLM itself is provided by the serving host (it pins CUDA/torch); target
**`vllm>=0.19,<0.20`** — v0.19 is the last CUDA-12 vLLM and carries the hooks the
plugin needs. `pip install "raif-vllm[vllm]"` pulls a compatible engine for local
experiments.

## Serve

```sh
VLLM_PLUGINS=raif vllm serve unsloth/Llama-3.2-3B-Instruct \
  --enable-lora --lora-modules raif=skrrt-sh/raif-llama-3.2-3b-lora \
  --max-lora-rank 32 --max-model-len 8192 \
  --chat-template raif_llama32.jinja \
  --reasoning-parser raif \
  --enable-auto-tool-choice --tool-call-parser raif
```

- `VLLM_PLUGINS=raif` runs the entry point, which registers the `raif` reasoning +
  tool parsers **and** installs the `render_chat` inject hook (the seam that adds
  the compact `<schema>` cue before chat-templating).
- `--tool-call-parser raif` decodes the tools path into `tool_calls`;
  `--reasoning-parser raif` decodes the `response_format` path into
  `message.content`.
- `--chat-template raif_llama32.jinja` (in `cuda/cloud/`) is load-bearing: it
  renders messages only and ignores the `tools` variable, so the served prompt
  matches training. Without it the LoRA echoes the verbose OpenAI tool-def JSON.

## What a plain OpenAI client gets

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

# tools -> JSON tool_calls
client.chat.completions.create(model="raif", tools=[...], tool_choice="auto",
    messages=[{"role": "user", "content": "Weather in Oslo?"}])

# response_format -> JSON content (use non-streaming — see below)
client.chat.completions.create(model="raif",
    response_format={"type": "json_schema", "json_schema": {...}},
    messages=[{"role": "user", "content": "..."}])
```

| OpenAI path | Behavior |
|---|---|
| plain chat | passthrough, untouched |
| `tools` | RAIF-G → JSON `tool_calls` (streaming + non-streaming) |
| `response_format` (`json_schema` / `json_object`) | RAIF-G → JSON `message.content` |
| plain chat **streaming** | passthrough |

### Known limitation: streaming `response_format`

Streaming a `response_format` request is **not decoded** — the client receives raw
RAIF-G. (vLLM's streaming seam passes the parser no schema, and the shared
`is_reasoning_end` flag must stay `True` so the *tools* streaming path keeps
working.) **Use non-streaming `response_format` for structured output** — it
decodes fully. Tool-call streaming is unaffected. See
[`docs/vllm_e2e_results.md`](../../docs/vllm_e2e_results.md).

## More

- End-to-end GPU smoke: `cuda/cloud/serve_smoke_v019.sh` + `examples/smoke_plugin.py`.
- Serving guide + the chat-template fix: `docs/vllm_tool_calling.md`.
- RunPod runbook: `docs/runpod_testing.md`.
