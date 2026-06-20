"""Behavior tests for the RAIF vLLM tool-call parser's pure helpers.

These cover the RAIF-G <-> OpenAI-tool-call logic with NO vLLM import, so they
run on any machine. The thin `RaifToolParser` shim that wires vLLM's types to
these helpers is exercised separately behind `pytest.importorskip("vllm")`.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from raif_vllm import (
    CoarseToolCallStreamer,
    build_schema_block,
    decode_arguments,
    inject_schema,
    resolve_tool,
)

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


def test_decode_arguments_parses_raif_to_dict():
    assert decode_arguments("city=Oslo\ndays=3") == {"city": "Oslo", "days": 3}


def test_decode_arguments_fails_closed_on_truncated_leaf():
    # A cut-off final leaf (no separator) is unrepairable -> drop the call,
    # rather than execute a tool with partial arguments.
    assert decode_arguments("<raif>\ncity=Oslo\nlat") is None


def test_build_schema_block_wraps_declaration():
    assert build_schema_block(_WEATHER_TOOL) == "<schema>\ncity:s\n</schema>"


def test_build_schema_block_empty_when_no_fields():
    assert build_schema_block({"type": "function", "function": {"name": "ping"}}) == ""


_OTHER_TOOL = {"type": "function", "function": {"name": "send_email", "parameters": {}}}


def test_resolve_tool_from_forced_choice():
    choice = {"type": "function", "function": {"name": "send_email"}}
    tool = resolve_tool([_WEATHER_TOOL, _OTHER_TOOL], choice)
    assert tool["function"]["name"] == "send_email"


def test_resolve_tool_single_tool_auto():
    tool = resolve_tool([_WEATHER_TOOL], "auto")
    assert tool["function"]["name"] == "get_weather"


def test_resolve_tool_ambiguous_multi_tool_is_none():
    # Coarse tier can't tell which of several the model called from args alone.
    assert resolve_tool([_WEATHER_TOOL, _OTHER_TOOL], "auto") is None


def test_streamer_emits_name_with_id_on_first_update():
    events = CoarseToolCallStreamer("get_weather").update("<raif>\ncity=Os")
    assert len(events) == 1
    assert events[0]["name"] == "get_weather"
    assert events[0]["id"].startswith("chatcmpl-tool-")


def test_streamer_emits_args_once_block_terminates():
    s = CoarseToolCallStreamer("get_weather")
    s.update("<raif>\ncity=Os")  # name only
    assert s.update("<raif>\ncity=Oslo\n") == []  # no terminator yet -> wait
    assert s.update("<raif>\ncity=Oslo\n</raif>") == [{"arguments": '{"city":"Oslo"}'}]


def test_streamer_emits_args_at_most_once():
    s = CoarseToolCallStreamer("get_weather")
    s.update("<raif>\ncity=Oslo\n</raif>")  # name + args
    assert s.update("<raif>\ncity=Oslo\n</raif>") == []  # nothing more to send


def test_streamer_omits_args_when_terminated_but_malformed():
    # Fail closed: a terminated-but-unparseable block emits the name, no args.
    s = CoarseToolCallStreamer("get_weather")
    events = s.update("<raif>\nlat\n</raif>") + s.update("<raif>\nlat\n</raif>")
    assert all("arguments" not in e for e in events)


def test_inject_schema_appends_to_last_user_message():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Weather in Oslo?"},
    ]
    out = inject_schema(msgs, "<schema>\ncity:s\n</schema>")
    assert out[-1]["content"] == "Weather in Oslo?\n\n<schema>\ncity:s\n</schema>"
    assert out[0]["content"] == "sys"  # other messages untouched
    assert msgs[-1]["content"] == "Weather in Oslo?"  # input not mutated


def test_inject_schema_noop_on_empty_block():
    msgs = [{"role": "user", "content": "hi"}]
    assert inject_schema(msgs, "") == msgs


def test_inject_schema_leaves_non_string_content_alone():
    msgs = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    assert inject_schema(msgs, "<schema>\ncity:s\n</schema>") == msgs


# ── vLLM shim (skipped unless vLLM is installed; runs in the vLLM CI job) ─────


def _new_parser():
    """A RaifToolParser bypassing the vLLM base __init__ (which needs a real
    tokenizer); the parser methods under test use only module helpers + state."""
    import raif_vllm

    parser = raif_vllm.RaifToolParser.__new__(raif_vllm.RaifToolParser)
    parser._streamer = None
    return parser


def test_extract_tool_calls_returns_openai_tool_call():
    pytest.importorskip("vllm")
    out = _new_parser().extract_tool_calls(
        "city=Oslo", SimpleNamespace(tools=[_WEATHER_TOOL], tool_choice="auto")
    )
    assert out.tools_called is True
    assert out.tool_calls[0].function.name == "get_weather"
    assert out.tool_calls[0].function.arguments == '{"city":"Oslo"}'


def test_extract_tool_calls_drops_malformed_output():
    pytest.importorskip("vllm")
    out = _new_parser().extract_tool_calls(
        "<raif>\nlat", SimpleNamespace(tools=[_WEATHER_TOOL], tool_choice="auto")
    )
    assert out.tools_called is False
    assert out.tool_calls == []


def test_streaming_emits_name_then_args_delta():
    pytest.importorskip("vllm")
    p = _new_parser()
    req = SimpleNamespace(tools=[_WEATHER_TOOL], tool_choice="auto")
    d1 = p.extract_tool_calls_streaming("", "<raif>\ncity=Os", "", [], [], [], req)
    assert d1.tool_calls[0].function.name == "get_weather"
    d2 = p.extract_tool_calls_streaming(
        "", "<raif>\ncity=Oslo\n</raif>", "", [], [], [], req
    )
    assert d2.tool_calls[0].function.arguments == '{"city":"Oslo"}'
