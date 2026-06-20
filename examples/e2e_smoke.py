"""End-to-end smoke test of the RAIF vLLM tool-call plugin.

Run against a `vllm serve ... --tool-call-parser raif` endpoint:

    python examples/e2e_smoke.py --base-url http://localhost:8000/v1 --model raif

The model emits RAIF-G; the plugin must hand the OpenAI client a *JSON* tool
call. Non-streaming correctness is the hard gate. Streaming is reported (it also
diagnoses whether the adapter emits the `</raif>` terminator the coarse streamer
needs). `completion_tokens` is printed as the RAIF-G wire cost on the live path.

Before the assertions we also run a *prompt-parity guard*: we ask the server's
/tokenize endpoint to render the exact chat prompt (request + tools) and check
that the custom chat template stripped the verbose OpenAI tool-definition JSON
and left only the plugin's compact `<schema>` cue. That is the whole point of
the --chat-template fix: the LoRA was trained on the bare <schema> cue, so the
rendered prompt must NOT echo the tool defs back. The guard only WARNs on tool
markers (so a regressed template is loud but not fatal) and degrades gracefully
if /tokenize is unavailable on older servers.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

from openai import OpenAI

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city", "unit"],
            },
        },
    }
]
PROMPT = "What's the weather in Oslo in celsius?"


def _tokenize_base(base_url: str) -> str:
    """Server root for the (non-/v1) /tokenize + /detokenize routes.

    The OpenAI base_url ends in /v1; the tokenize routes live one level up.
    """
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    return root


def _post_json(url: str, payload: dict, api_key: str) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    if api_key and api_key != "EMPTY":
        req.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 (trusted localhost)
        return json.loads(resp.read().decode("utf-8"))


def _rendered_prompt(root: str, data: dict, api_key: str) -> str | None:
    """Best-effort recovery of the rendered prompt string from a /tokenize reply.

    Prefer the explicit `prompt` field; otherwise round-trip the returned token
    ids through /detokenize. Returns None if neither is available.
    """
    prompt = data.get("prompt")
    if isinstance(prompt, str) and prompt:
        return prompt
    tokens = data.get("tokens")
    if isinstance(tokens, list) and tokens:
        try:
            det = _post_json(f"{root}/detokenize", {"tokens": tokens}, api_key)
            text = det.get("prompt")
            if isinstance(text, str) and text:
                return text
        except (urllib.error.URLError, OSError, ValueError) as exc:
            print(
                f"[parity]     note  detokenize unavailable ({exc}); skipping text recovery"
            )
    return None


def check_prompt_parity(base_url: str, model: str, api_key: str) -> None:
    """Guard the rendered prompt matches the bare-<schema> training format.

    Soft by design: WARN on leftover tool-def markers, degrade gracefully if
    /tokenize is missing, and only assert the <schema> cue is present (that is
    what the plugin injects and what the LoRA was trained on).
    """
    root = _tokenize_base(base_url)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "tools": TOOLS,
        "tool_choice": "auto",
        "add_generation_prompt": True,
    }
    try:
        data = _post_json(f"{root}/tokenize", payload, api_key)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        print(
            f"[parity]     SKIP  /tokenize unavailable ({exc}); cannot inspect prompt parity"
        )
        return

    prompt = _rendered_prompt(root, data, api_key)
    if prompt is None:
        print(
            "[parity]     SKIP  server returned no rendered prompt text; cannot inspect parity"
        )
        return

    if '"parameters"' in prompt or '"function"' in prompt:
        print(
            "[parity]     WARN  rendered prompt still contains OpenAI tool-definition "
            "markers ('\"parameters\"'/'\"function\"') — the custom --chat-template is "
            "not stripping tool defs, so the LoRA will echo them instead of emitting RAIF."
        )
    else:
        print(
            "[parity]     PASS  rendered prompt is free of OpenAI tool-definition markers"
        )

    assert "<schema>" in prompt, (
        "rendered prompt is missing the '<schema>' cue — the plugin's adjust_request "
        f"did not inject it (or the template dropped it):\n{prompt}"
    )
    print(
        "[parity]     PASS  rendered prompt carries the '<schema>' cue (training parity)"
    )


def check_non_streaming(client: OpenAI, model: str) -> None:
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PROMPT}],
        tools=TOOLS,
        tool_choice="auto",
    )
    msg = r.choices[0].message
    assert msg.tool_calls, f"no tool_calls returned: {msg}"
    fn = msg.tool_calls[0].function
    assert fn.name == "get_weather", f"wrong tool: {fn.name}"
    args = json.loads(fn.arguments)  # raises if the plugin emitted non-JSON
    assert "city" in args, f"missing 'city': {args}"
    print(f"[non-stream] PASS  name={fn.name}  args={args}")
    print(
        f"             RAIF-G wire cost = {r.usage.completion_tokens} completion tokens"
    )


def check_streaming(client: OpenAI, model: str) -> None:
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PROMPT}],
        tools=TOOLS,
        tool_choice="auto",
        stream=True,
    )
    name, args = None, ""
    for chunk in stream:
        calls = chunk.choices[0].delta.tool_calls
        if not calls:
            continue
        if calls[0].function.name:
            name = calls[0].function.name
        if calls[0].function.arguments:
            args += calls[0].function.arguments
    if name == "get_weather" and args:
        json.loads(args)  # must assemble to valid JSON
        print(f"[stream]     PASS  name={name}  args={args}")
    else:
        print(
            f"[stream]     WARN  name={name!r} args={args!r} — the adapter likely "
            "does not emit the </raif> terminator; coarse streaming needs it."
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="raif")
    ap.add_argument("--api-key", default="EMPTY")
    args = ap.parse_args()
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    check_prompt_parity(args.base_url, args.model, args.api_key)
    check_non_streaming(client, args.model)
    check_streaming(client, args.model)
    print("e2e smoke: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
