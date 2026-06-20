"""End-to-end smoke test of the RAIF vLLM tool-call plugin.

Run against a `vllm serve ... --tool-call-parser raif` endpoint:

    python examples/e2e_smoke.py --base-url http://localhost:8000/v1 --model raif

The model emits RAIF-G; the plugin must hand the OpenAI client a *JSON* tool
call. Non-streaming correctness is the hard gate. Streaming is reported (it also
diagnoses whether the adapter emits the `</raif>` terminator the coarse streamer
needs). `completion_tokens` is printed as the RAIF-G wire cost on the live path.
"""

from __future__ import annotations

import argparse
import json
import sys

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
    print(f"             RAIF-G wire cost = {r.usage.completion_tokens} completion tokens")


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
    check_non_streaming(client, args.model)
    check_streaming(client, args.model)
    print("e2e smoke: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
