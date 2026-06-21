"""Behavior tests for the RAIF inject step (no vLLM import).

`prepare_chat_request` is what the `render_chat` monkeypatch calls before chat
templating: it injects the compact `<schema>` cue into the prompt and, for the
`response_format` path, neutralizes vLLM's native JSON guided-decoding while
preserving the schema for the decoder. These tests exercise that behavior through
the public function on plain request-like objects.
"""

from __future__ import annotations

from types import SimpleNamespace

from raif_vllm.inject import prepare_chat_request

_WEATHER_RF = {
    "type": "json_schema",
    "json_schema": {
        "name": "weather",
        "schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
            "required": ["city"],
        },
    },
}


class _Model:
    """Stand-in for a vLLM pydantic request field (carries `model_dump`)."""

    def __init__(self, d):
        self._d = d

    def model_dump(self):
        return self._d


def _req(**kw):
    kw.setdefault("messages", [{"role": "user", "content": "weather in Oslo?"}])
    kw.setdefault("tools", None)
    kw.setdefault("tool_choice", None)
    kw.setdefault("response_format", None)
    kw.setdefault("vllm_xargs", None)
    return SimpleNamespace(**kw)


_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


def test_structured_request_injects_schema_cue():
    out = prepare_chat_request(_req(response_format=_WEATHER_RF))
    last = out.messages[-1]["content"]
    assert last.startswith("weather in Oslo?")
    assert last.endswith("</schema>")
    assert "city:s" in last


def test_pydantic_response_format_is_normalized():
    # vLLM passes response_format as a pydantic model, not a dict.
    out = prepare_chat_request(_req(response_format=_Model(_WEATHER_RF)))
    assert out.response_format is None
    assert out.vllm_xargs["raif_decl"] == "city:s\ndays:n?"
    assert out.messages[-1]["content"].endswith("</schema>")


def test_pydantic_tools_are_normalized():
    # vLLM passes tools as a list of pydantic models, not dicts.
    out = prepare_chat_request(_req(tools=[_Model(_WEATHER_TOOL)], tool_choice="auto"))
    assert out.messages[-1]["content"].endswith("</schema>")
    assert "city:s" in out.messages[-1]["content"]


def test_tools_request_injects_tool_cue():
    out = prepare_chat_request(_req(tools=[_WEATHER_TOOL], tool_choice="auto"))
    last = out.messages[-1]["content"]
    assert last.startswith("weather in Oslo?")
    assert last.endswith("</schema>")
    assert "city:s" in last


def test_plain_chat_is_untouched():
    original = [{"role": "user", "content": "hello there"}]
    out = prepare_chat_request(_req(messages=original))
    assert out.messages == original
    assert "<schema>" not in out.messages[-1]["content"]


def test_structured_neutralizes_native_json_and_stashes_decl():
    out = prepare_chat_request(_req(response_format=_WEATHER_RF))
    # Native JSON guided-decoding dropped so the model is free to emit RAIF-G.
    assert out.response_format is None
    # RAIF declaration preserved (as a string) for the decoder to recover.
    assert out.vllm_xargs["raif_decl"] == "city:s\ndays:n?"


def test_structured_preserves_existing_vllm_xargs():
    out = prepare_chat_request(
        _req(response_format=_WEATHER_RF, vllm_xargs={"keep": "me"})
    )
    assert out.vllm_xargs["keep"] == "me"
    assert "raif_decl" in out.vllm_xargs


def test_tools_path_does_not_touch_response_format_or_xargs():
    out = prepare_chat_request(_req(tools=[_WEATHER_TOOL], tool_choice="auto"))
    assert out.response_format is None  # was None; tools path leaves it alone
    assert out.vllm_xargs is None  # tools decode via the tool parser, no decl
