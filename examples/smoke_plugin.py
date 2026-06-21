"""End-to-end smoke for the single RAIF vLLM plugin (raif-vllm).

Run against a server started with the plugin installed and active:

    VLLM_PLUGINS=raif vllm serve <model> --enable-lora --lora-modules raif=<lora> \
      --reasoning-parser raif --enable-auto-tool-choice --tool-call-parser raif \
      --chat-template <raif_llama32.jinja>
    python examples/smoke_plugin.py --base-url http://localhost:8000/v1 --model raif

Asserts the three OpenAI paths behave, with NO client-side RAIF awareness — the
client sends and receives plain OpenAI shapes:

  1. plain chat        -> normal text content, untouched.
  2. tools             -> JSON `tool_calls` (RAIF-G decoded by the tool parser).
  3. response_format   -> `message.content` is valid JSON matching the schema
                          (RAIF-G decoded by the reasoning parser).

For the structured path it also reports the RAIF-G wire cost (the generated
`completion_tokens`) and, best-effort, compares it to the token count of the
equivalent JSON the client received — the RAIF token saving, measured live.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

from openai import OpenAI

WEATHER_TOOL = {
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

WEATHER_SCHEMA = {
    "type": "object",
    "properties": {
        "city": {"type": "string"},
        "unit": {"type": "string"},
        "temperature": {"type": "number"},
    },
    "required": ["city", "unit", "temperature"],
}

PROMPT = "What's the weather in Oslo in celsius? Make up a temperature."


def _root(base_url: str) -> str:
    root = base_url.rstrip("/")
    return root[:-3] if root.endswith("/v1") else root


def _tokenize_count(base_url: str, model: str, api_key: str, text: str) -> int | None:
    """Token count of a raw string via the server's /tokenize, or None."""
    body = json.dumps({"model": model, "prompt": text}).encode()
    req = urllib.request.Request(f"{_root(base_url)}/tokenize", data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    if api_key and api_key != "EMPTY":
        req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
            data = json.loads(resp.read().decode())
    except (urllib.error.URLError, OSError, ValueError):
        return None
    if isinstance(data.get("count"), int):
        return data["count"]
    toks = data.get("tokens")
    return len(toks) if isinstance(toks, list) else None


def check_plain_chat(client: OpenAI, model: str) -> None:
    r = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": "Say hello in one word."}]
    )
    msg = r.choices[0].message
    assert msg.content and not msg.tool_calls, f"plain chat not plain: {msg}"
    print(f"[plain]      PASS  content={msg.content!r}")


def check_tools(client: OpenAI, model: str) -> None:
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PROMPT}],
        tools=[WEATHER_TOOL],
        tool_choice="auto",
    )
    msg = r.choices[0].message
    assert msg.tool_calls, f"no tool_calls: {msg}"
    fn = msg.tool_calls[0].function
    assert fn.name == "get_weather", f"wrong tool: {fn.name}"
    args = json.loads(fn.arguments)  # raises if non-JSON
    assert "city" in args, f"missing city: {args}"
    print(f"[tools]      PASS  name={fn.name} args={args} "
          f"(RAIF-G wire = {r.usage.completion_tokens} tok)")


def check_response_format(client: OpenAI, model: str, base_url: str, api_key: str) -> None:
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PROMPT}],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "weather", "schema": WEATHER_SCHEMA},
        },
    )
    msg = r.choices[0].message
    assert not msg.tool_calls, f"response_format must use content, not tool_calls: {msg}"
    obj = json.loads(msg.content)  # raises if the reasoning parser emitted non-JSON
    assert "city" in obj, f"missing city: {obj}"
    raif_tok = r.usage.completion_tokens
    line = f"[resp_fmt]   PASS  content={obj} (RAIF-G wire = {raif_tok} tok)"

    json_str = json.dumps(obj, separators=(",", ":"))
    json_tok = _tokenize_count(base_url, model, api_key, json_str)
    if json_tok:
        saved = json_tok - raif_tok
        pct = 100 * saved / json_tok if json_tok else 0
        line += f"  | equivalent JSON = {json_tok} tok -> saved {saved} ({pct:.0f}%)"
    print(line)


def check_response_format_streaming(client: OpenAI, model: str) -> None:
    """Streamed `response_format`: deltas must concatenate to valid JSON.

    The reasoning parser has no `request` while streaming, so it decodes
    coarsely — buffering the RAIF-G until a framing terminator appears (plain
    chat never emits one), then emitting the decoded JSON. The client therefore
    still receives plain JSON across the stream, with no RAIF awareness.
    """
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PROMPT}],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "weather", "schema": WEATHER_SCHEMA},
        },
        stream=True,
    )
    chunks: list[str] = []
    for ev in stream:
        if not ev.choices:
            continue
        delta = ev.choices[0].delta
        if delta and delta.content:
            chunks.append(delta.content)
    content = "".join(chunks)
    assert content, "stream produced no content"
    obj = json.loads(content)  # streamed content must concatenate to valid JSON
    assert "city" in obj, f"missing city: {obj}"
    print(f"[resp_fmt/stream] PASS  content={obj} ({len(chunks)} delta(s))")


def check_json_object(client: OpenAI, model: str) -> None:
    """Schemaless `response_format={"type":"json_object"}` -> valid JSON content."""
    r = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": PROMPT + " Reply as JSON with city, unit and temperature.",
            }
        ],
        response_format={"type": "json_object"},
    )
    msg = r.choices[0].message
    assert not msg.tool_calls, f"json_object must use content, not tool_calls: {msg}"
    obj = json.loads(msg.content)  # raises if the reasoning parser emitted non-JSON
    print(f"[json_object]     PASS  content={obj} "
          f"(RAIF-G wire = {r.usage.completion_tokens} tok)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="raif")
    ap.add_argument("--api-key", default="EMPTY")
    args = ap.parse_args()
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    checks = [
        ("plain", lambda: check_plain_chat(client, args.model)),
        ("tools", lambda: check_tools(client, args.model)),
        ("resp_fmt", lambda: check_response_format(
            client, args.model, args.base_url, args.api_key)),
        ("resp_fmt/stream", lambda: check_response_format_streaming(client, args.model)),
        ("json_object", lambda: check_json_object(client, args.model)),
    ]
    failed: list[str] = []
    for name, fn in checks:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 — report every path, don't abort early
            failed.append(name)
            print(f"[{name}]  FAIL  {type(exc).__name__}: {exc}")

    if failed:
        print(f"smoke_plugin: FAIL ({', '.join(failed)})")
        return 1
    print("smoke_plugin: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
