# vLLM End-to-End Results: RAIF-G Tool-Call Parser

Status: **Plugin shim tests PASS (17/17). Live smoke FAILED (prompt-format mismatch). Fix applied via custom `--chat-template`.**

This document is the test-results artifact for the GPU end-to-end (e2e) run of the
`raif-lora` vLLM `ToolParser` plugin against a live model on RunPod. It records the
working stack, the observed failure, the root-cause analysis, and the fix.

## What was run

The e2e run served the fine-tuned RAIF-G model behind an OpenAI-compatible vLLM
endpoint and exercised the tool-call path through the `raif` parser plugin.

- **Hardware:** RunPod A40 (NVIDIA driver CUDA 12.9 / "12090").
- **Model:** base `unsloth/Llama-3.2-3B-Instruct` + LoRA
  `skrrt-sh/raif-llama-3.2-3b-lora` (rank 32).
- **Serving:** vLLM OpenAI server with the `raif` `ToolParser` plugin
  (`src/raif_vllm.py`), `--enable-auto-tool-choice`, and the LoRA loaded as an
  adapter.
- **Decoder:** `raif.decode` from the `raif-format` package (RAIF-G → arguments dict).
- **Client probe:** a chat-completions request with `tools=[get_weather]` and
  `tool_choice="auto"`, plus a parallel raw-completion probe that bypasses vLLM's
  tool template.

## Final working stack / version matrix

| component    | working pin           | notes                                            |
| ------------ | --------------------- | ------------------------------------------------ |
| vLLM         | `0.11.0`              | torch 2.8.0+cu128, runs on the A40 driver 12.9   |
| torch        | `2.8.0+cu128`         | bundled with vLLM 0.11.0; CUDA 12.8 build         |
| transformers | `4.57.x` (`>=4.56,<5`)| keeps `all_special_tokens_extended` for vLLM 0.11 |
| python       | `python3.12`          | the image's real interpreter with torch/vLLM      |

### Why the stock image fails, and how the pins resolve it

The stock RunPod image (`runpod/pytorch:...-cu1290`) ships **torch 2.9 / cu130 +
transformers 5.x**. Two independent failures result:

1. **Latest vLLM (0.23.0) crashes on the A40.** Its torch requires a CUDA-13
   driver, but the A40 host driver is CUDA 12.9:

   ```
   NVIDIA driver too old, found version 12090
   ```

   Pinning `vllm==0.11.0` pulls `torch 2.8.0+cu128`, a CUDA-12.8 build that runs
   correctly on driver 12.9.

2. **transformers 5.x breaks vLLM 0.11 tokenizer init.** transformers 5.x removed
   `all_special_tokens_extended`, which vLLM 0.11's tokenizer initialization calls:

   ```
   TokenizersBackend has no attribute all_special_tokens_extended
   ```

   Pinning `transformers>=4.56,<5` (resolves to 4.57.x) restores the attribute.

Additional environment note: in the image, plain `python3` is an empty 3.10; the
interpreter that actually has torch/vLLM is `python3.12`. All commands below use
`python3.12`.

## Results

### Plugin shim tests: 17/17 PASS

The plugin shim test suite passes **17/17 against real vLLM 0.11 types**. This
includes the 3 `importorskip` tests that skip on Mac (no vLLM there) and run on the
GPU host where vLLM 0.11 is installed. The decoder is exercised and is correct.

### Live smoke: FAILED (tool-call path)

With the client passing `tools=[...]` and `tool_choice="auto"`, the server returned
a syntactically valid JSON tool call with the correct function name
(`get_weather`), but the **arguments were the tool schema echoed back** rather than
extracted values:

```json
{"function": {"function": {"name": "get_weather", "parameters": {...}}}}
```

### Raw-path proof: model emits correct RAIF

Using the same prompt plus the exact `<schema>` cue the plugin injects, but
**without** vLLM's tool template, the model emitted correct RAIF-G:

```
city=Oslo
unit=celsius
```

which `raif.decode` turns into:

```json
{"city": "Oslo", "unit": "celsius"}
```

The cue injected by the plugin's `adjust_request` was:

```
<schema>
city:s
unit:s
</schema>
```

The raw path and the tool-call path differ in exactly one way — whether vLLM also
rendered the OpenAI tool-definition JSON into the prompt — which isolates the cause.

## Root-cause analysis

The gap is a **prompt-format mismatch**, not a parser or decoder defect.

- **Training format (bare cue).** Per `docs/fine_tune_plan.md`, each training
  example's user turn is `"<request>\n\n<schema>\n<declaration>\n</schema>"`
  (§3.1), using the compact RAIF-native schema grammar (§3.2:
  `name:s` / `name:n` / `name:b` / `name:t`, `name[]:s`, `name.sub:b`, `name:s?`).
  Line 209: "Prompts skip the spec block and few-shot examples — pass only the
  request + optional `<schema>` declaration." Training rendered messages with
  `tokenizer.apply_chat_template(messages)` and **no** `tools=` argument, so the
  base template's tool branch never fired during training.

- **Tool-call path (double injection).** When the client passes `tools=[...]`,
  vLLM's chat template renders the verbose OpenAI tool-definition JSON into the
  prompt **on top of** the plugin's `<schema>` cue. The LoRA never saw the verbose
  tool defs, so it mimics/echoes them instead of producing RAIF arguments — hence
  the `{"function":{"function":{...}}}` schema echo.

- **Conclusion.** The raw-path proof (`city=Oslo\nunit=celsius` →
  `{"city":"Oslo","unit":"celsius"}`) plus 17/17 shim tests show the parser and
  decoder are correct. The model only misbehaves when the prompt deviates from
  training parity by carrying the extra tool-definition JSON.

## Fix applied

Serve vLLM with a **custom `--chat-template`** that renders only the messages and
ignores the `tools` variable. The model then receives exactly
`"<request>\n\n<schema>\n...\n</schema>"` (the plugin's `adjust_request` already
injects the `<schema>` cue into the last user message) — exact training parity.

Design constraints honored by this fix:

- **Keep `request.tools` set (do NOT null it).** Both vLLM's
  `--enable-auto-tool-choice` gating and the plugin's `extract_tool_calls` read
  `request.tools`. Only the *template* must drop the tool rendering; nulling
  `request.tools` in `adjust_request` would break both gates.
- **Do not retrain on verbose tool defs.** That would discard RAIF's input-token
  savings, which the compact `<schema>` cue exists to provide.

See `docs/vllm_tool_calling.md` for the full template and serving wiring.

## Reproduce

Pin the working stack:

```bash
python3.12 -m pip install "vllm==0.11.0" "transformers>=4.56,<5"
```

Serve base + LoRA with the `raif` parser and the custom (tools-ignoring) template:

```bash
python3.12 -m vllm.entrypoints.openai.api_server \
  --model unsloth/Llama-3.2-3B-Instruct \
  --enable-lora --lora-modules raif=skrrt-sh/raif-llama-3.2-3b-lora \
  --enable-auto-tool-choice --tool-call-parser raif \
  --tool-parser-plugin src/raif_vllm.py \
  --chat-template cuda/cloud/raif_llama32.jinja
```

Version pins (summary): `vllm==0.11.0`, `torch 2.8.0+cu128`,
`transformers>=4.56,<5` (4.57.x), `python3.12`. Do not use the stock
`cu1290` image's torch 2.9 / cu130 + transformers 5.x — it triggers the
"driver too old (12090)" crash with vLLM 0.23.0 and the
`all_special_tokens_extended` tokenizer error.
