"""Behavior tests for the RAIF reasoning-parser pure helpers (no vLLM import).

The thin `RaifReasoningParser` shim that wires vLLM's types to these is exercised
separately behind `importorskip("vllm")`.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from raif_vllm.reasoning import has_terminator, route_and_decode

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


def _req(**kw):
    """A request exposing tools/tool_choice/response_format as attributes."""
    kw.setdefault("tools", None)
    kw.setdefault("tool_choice", None)
    kw.setdefault("response_format", None)
    return SimpleNamespace(**kw)


# ── routing / decode ─────────────────────────────────────────────────────────


def test_structured_request_decodes_to_json_content():
    reasoning, content = route_and_decode(
        "city=Oslo\ndays=5", _req(response_format=_WEATHER_RF)
    )
    assert reasoning is None
    assert json.loads(content) == {"city": "Oslo", "days": 5}


def test_structured_uses_schema_typing():
    _, content = route_and_decode(
        "city=Oslo\ndays=5", _req(response_format=_WEATHER_RF)
    )
    assert json.loads(content)["days"] == 5  # number, not "5"


def test_structured_fails_open_to_raw_text():
    raw = "\x00\x01 not raif at all"
    reasoning, content = route_and_decode(raw, _req(response_format=_WEATHER_RF))
    assert reasoning is None
    assert content == raw  # client still gets the generation, not an empty msg


def test_tools_request_passes_through_unchanged():
    # The tool parser decodes RAIF-G from content downstream — don't touch it.
    raw = "city=Oslo\nunit=celsius"
    reasoning, content = route_and_decode(
        raw, _req(tools=[{"type": "function"}], tool_choice="auto")
    )
    assert (reasoning, content) == (None, raw)


def test_tool_choice_none_with_response_format_decodes():
    _, content = route_and_decode(
        "city=Oslo",
        _req(
            tools=[{"type": "function"}],
            tool_choice="none",
            response_format=_WEATHER_RF,
        ),
    )
    assert json.loads(content) == {"city": "Oslo"}


def test_plain_chat_passes_through():
    raw = "The weather in Oslo is nice today."
    assert route_and_decode(raw, _req()) == (None, raw)
    assert route_and_decode(raw, _req(response_format={"type": "text"})) == (None, raw)


def test_decodes_from_stashed_raif_decl_after_inject():
    # Post-inject state: the inject step cleared response_format and stashed the
    # RAIF declaration in vllm_xargs. The parser decodes using that.
    req = _req(vllm_xargs={"raif_decl": "city:s\ndays:n?"})
    reasoning, content = route_and_decode("city=Oslo\ndays=5", req)
    assert reasoning is None
    assert json.loads(content) == {"city": "Oslo", "days": 5}


def test_stashed_empty_decl_still_decodes_schemaless():
    # json_object inject stashes raif_decl="" — presence of the key means RAIF.
    req = _req(vllm_xargs={"raif_decl": ""})
    _, content = route_and_decode("city=Oslo", req)
    assert json.loads(content) == {"city": "Oslo"}


def test_request_as_dict_is_supported():
    # vLLM hands a pydantic model; tests/other callers may pass a dict.
    _, content = route_and_decode("city=Oslo", {"response_format": _WEATHER_RF})
    assert json.loads(content) == {"city": "Oslo"}


def test_none_request_is_plain():
    assert route_and_decode("hello", None) == (None, "hello")


# ── streaming terminator detection ───────────────────────────────────────────


@pytest.mark.parametrize("marker", ["</raif>", "<|raif_end|>"])
def test_terminator_detected(marker):
    assert has_terminator(f"city=Oslo\n{marker}") is True


def test_no_terminator_in_plain_text():
    assert has_terminator("city=Oslo\nunit=celsius") is False
    assert has_terminator("The weather is nice.") is False
