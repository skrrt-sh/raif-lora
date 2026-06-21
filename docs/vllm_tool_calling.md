# RAIF tool calling on vLLM

How a RAIF-fine-tuned Llama serves OpenAI `tool_calls` over vLLM, the
chat-template pitfall that silently breaks it, and the one-line fix.

## Overview

The model (`unsloth/Llama-3.2-3B-Instruct` + LoRA
`skrrt-sh/raif-llama-3.2-3b-lora`, rank 32) does not emit OpenAI tool-call JSON.
It emits **RAIF-G** — RAIF's compact serialization. The `RaifToolParser`
(the tool-call path of the installable `raif-vllm` plugin —
`packages/vllm/raif_vllm/tool_parser.py`, registered under the name `raif`)
converts that to OpenAI `tool_calls` at the request and response boundaries.
End to end:

1. **`adjust_request`** runs before generation. It resolves the single tool the
   output will be parsed against (named `tool_choice`, or a lone tool under
   `auto`), builds the compact `<schema>` cue from that tool's
   `function.parameters`, and appends it to the last user message. It also sets
   `request.skip_special_tokens = False` so RAIF-G terminators
   (`</raif>`, `<|raif_end|>`) survive to the parser.
2. **The model** generates RAIF-G arguments, e.g. `city=Oslo\nunit=celsius`.
3. **`extract_tool_calls`** (non-streaming) / **`extract_tool_calls_streaming`**
   (streaming, coarse tier: name first, whole arguments object once the RAIF-G
   block terminates) decode the RAIF-G via `raif.decode` and assemble OpenAI
   `tool_calls`. Decode fails closed — an unrepairable output yields no tool
   call rather than malformed arguments.

The decode/encode logic is pure (no vLLM import) and unit-tested; the shim only
wires vLLM types to it. Package tests: 57 passed, 3 skipped (the 3 skips are
`importorskip` shims that run only where vLLM is installed). The tool path was
first proven against real vLLM 0.11 types; the live e2e (`serve_smoke_v019.sh`)
now runs on vLLM 0.19.

## The prompt contract

The LoRA was trained on exactly one user-turn shape — request plus an optional
compact schema cue, nothing else:

```
<request>

<schema>
<declaration>
</schema>
```

Citations from `raif-standard/docs/fine_tune_plan.md`:

- **§3.1** — each training example's user content is
  `"<request_template>\n\n<schema>\n<schema_declaration>\n</schema>"`.
- **§3.2** — the schema declaration uses RAIF-native compact syntax (grammar
  below); it is a prompt cue only and is never emitted by the model.
- **line 209** — "Prompts skip the spec block and few-shot examples — pass only
  the request + optional `<schema>` declaration." No system tool block, no
  OpenAI tool-definition JSON.

Training rendered messages with `tokenizer.apply_chat_template(messages)` and
**no `tools=` argument**, so the base template's tool branch never fired during
training. The model has never seen a verbose OpenAI tool definition in its
prompt.

### Schema declaration grammar (§3.2)

| Form | Meaning |
|---|---|
| `name:s` | string |
| `name:n` | number |
| `name:b` | boolean |
| `name:t` | multiline text (string with embedded newlines) |
| `name[]:s` | array of strings (any type code after `[]:`) |
| `name[]:o` | array of objects (heterogeneous; type implied) |
| `name.sub:b` | nested path |
| `name:s?` / `name?:s` | optional (field may be absent) |

`build_schema_block` / `tool_to_schema` produce exactly this from a tool's
`function.parameters`; types that can't be represented are reported in
`degraded_fields`, not emitted as a wrong type.

## The pitfall

When a client calls with `tools=[...]` and `tool_choice=auto`, vLLM's chat
template renders the **OpenAI tool-definition JSON into the prompt** because
`tools` is set — on top of the plugin's `<schema>` cue. The model now sees a
prompt it never saw in training, so it **echoes the tool definition back** as the
arguments instead of producing RAIF-G arguments.

Observed on RunPod A40, vLLM 0.11.0. The server returns a structurally valid
tool call with `name=get_weather`, but the arguments are the schema echoed back:

```json
{"function":{"function":{"name":"get_weather","parameters":{...}}}}
```

Send the **same prompt and the same `<schema>` cue without vLLM's tool template**
(the raw path), and the model emits correct RAIF-G:

```
city=Oslo
unit=celsius
```

which `raif.decode` turns into:

```json
{"city":"Oslo","unit":"celsius"}
```

Root cause: prompt pollution, not a decoder or plugin bug. The LoRA was trained
only on the bare `<schema>` cue; the extra tool-def JSON throws it off.

## The fix

Serve vLLM with a **custom `--chat-template` (`raif_llama32.jinja`) that renders
only the messages and ignores the `tools` variable.** The prompt the model then
receives is exactly `<request>\n\n<schema>\n...\n</schema>` (the plugin's
`adjust_request` already injected the `<schema>` cue) — exact training parity.

Crucially, **keep `request.tools` set**. Do not null it in `adjust_request`:

- vLLM's `--enable-auto-tool-choice` gating reads `request.tools`.
- The plugin's own `extract_tool_calls` / `extract_tool_calls_streaming` read
  `request.tools` (via `resolve_tool`) to decide which tool to parse against.

Only the **template** drops the tool rendering; the request object is untouched.

### Rejected alternatives

- **Retrain on the verbose tool-def JSON.** Throws away RAIF's input-token
  savings — the entire point of the compact `<schema>` cue. The model would
  again carry the full OpenAI tool definition in every prompt.
- **Null `request.tools` in `adjust_request`.** Breaks vLLM's tool-choice gating
  *and* the plugin's own extraction, both of which read `request.tools`. The
  server would stop routing to the parser and/or emit no tool calls.

## How to verify

Use vLLM's `/tokenize` endpoint to check the rendered prompt the model actually
receives:

- **Parity check.** The rendered prompt for a tool-call request must contain the
  `<schema>` cue and must **not** contain the verbose OpenAI tool-definition JSON
  (no `"parameters"`/`"properties"` tool-def blob, no tool-spec system block).
- **No-tools equivalence.** Rendering a no-tools request through the custom
  `raif_llama32.jinja` must equal rendering it through the base Llama-3.2
  template — the custom template only suppresses the tool branch and is otherwise
  identical.

End to end, the decoded arguments for the `get_weather` example should be
`{"city":"Oslo","unit":"celsius"}`, not the echoed schema.

## Serving command

The tool path now ships inside the single installable `raif-vllm` plugin, loaded
through vLLM's `general_plugins` entry point — no `--tool-parser-plugin FILE`.
Install the plugin (and the `raif-format` lib it depends on), then activate it
with `VLLM_PLUGINS=raif`:

```sh
# editable from the two sibling working trees (or, once released: pip install raif-format raif-vllm)
python3.12 -m pip install -e raif-standard/packages/py -e raif-lora/packages/vllm

VLLM_PLUGINS=raif python3.12 -m vllm.entrypoints.openai.api_server \
  --model unsloth/Llama-3.2-3B-Instruct \
  --enable-lora --lora-modules raif=skrrt-sh/raif-llama-3.2-3b-lora \
  --max-lora-rank 32 --max-model-len 8192 --enforce-eager \
  --chat-template /path/to/raif-lora/cuda/cloud/raif_llama32.jinja \
  --reasoning-parser raif \
  --enable-auto-tool-choice --tool-call-parser raif
```

`VLLM_PLUGINS=raif` runs the entry point, which registers the `raif` reasoning +
tool parsers **and** installs the `render_chat` inject hook. `--tool-call-parser
raif` selects the tool path; `--reasoning-parser raif` lights up the
`response_format` decode path. The `--chat-template raif_llama32.jinja` is still
the load-bearing flag for tool parity — it is what makes the served prompt match
training (renders messages only, ignores the `tools` variable). The end-to-end
driver is `cuda/cloud/serve_smoke_v019.sh`.

### Version pins (RunPod A40, driver CUDA 12.9)

| Pin | Why |
|---|---|
| `vllm>=0.19,<0.20` | v0.19.0 is the **last CUDA-12 vLLM** and carries every hook the plugin needs (reasoning-parser content decode + the `render_chat` inject seam). Newer vLLM ships cu130 torch and crashes on a CUDA-12 driver with "NVIDIA driver too old, found version 12090". |
| `fastapi==0.115.6` | pins starlette `<1.0`; vLLM's prometheus instrumentator breaks on starlette 1.x (`'_IncludedRouter' object has no attribute 'path'`), so the server never becomes healthy. |
| `python3.12` | the stock RunPod image's real interpreter with torch/vllm; plain `python3` is an empty 3.10. |

## Follow-ups

- **Streaming terminator — resolved (negative).** Confirmed on vLLM 0.19: the LoRA
  emits **no** RAIF-G terminator (`</raif>` / `<|raif_end|>`). Streaming tool calls
  still work (the tool parser decodes RAIF-G incrementally), but streaming
  `response_format` is **not** decoded — see `vllm_e2e_results.md` for the root
  cause (the `is_reasoning_end` gate) and the known-limitation note. Use
  non-streaming `response_format`.
- The MLX serving path (`raif-lora/examples/serve.sh`) is separate from this vLLM
  path and out of scope here.
