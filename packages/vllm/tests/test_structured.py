"""Behavior tests for the RAIF `response_format` / structured-output helpers.

These cover the request<->RAIF logic for the JSON-mode path with NO vLLM import,
so they run on any machine (the thin reasoning-parser shim that wires vLLM's
types to these helpers is exercised separately behind `importorskip("vllm")`).
"""

from __future__ import annotations

import json

import pytest

from raif_vllm.structured import (
    build_response_schema_block,
    classify_request,
    decode_content,
    is_structured_response_format,
    response_format_schema,
    response_format_to_raif,
)

_WEATHER_RF = {
    "type": "json_schema",
    "json_schema": {
        "name": "weather",
        "schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "unit": {"type": "string"},
                "days": {"type": "integer"},
            },
            "required": ["city"],
        },
    },
}


# ── response_format_schema ───────────────────────────────────────────────────


def test_schema_extracted_from_json_schema():
    schema = response_format_schema(_WEATHER_RF)
    assert schema["properties"]["city"] == {"type": "string"}
    assert schema["required"] == ["city"]


def test_json_object_is_empty_schema():
    # A valid structured request that carries no schema.
    assert response_format_schema({"type": "json_object"}) == {}


def test_vllm_model_dump_uses_json_schema_inner_key():
    # vLLM's JsonSchemaResponseFormat stores the schema under the field name
    # `json_schema` (since `schema` is a reserved pydantic name); model_dump()
    # therefore emits the inner schema under `json_schema`, not `schema`.
    rf = {
        "type": "json_schema",
        "json_schema": {
            "name": "w",
            "description": None,
            "json_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
    assert response_format_schema(rf)["properties"]["city"] == {"type": "string"}


@pytest.mark.parametrize(
    "rf",
    [None, {"type": "text"}, "json_object", {}, {"type": "json_schema"}],
)
def test_non_structured_or_malformed(rf):
    # `{"type": "json_schema"}` with no inner schema degrades to `{}` (still
    # structured); everything else is None.
    result = response_format_schema(rf)
    if rf == {"type": "json_schema"}:
        assert result == {}
    else:
        assert result is None


def test_is_structured_predicate():
    assert is_structured_response_format(_WEATHER_RF) is True
    assert is_structured_response_format({"type": "json_object"}) is True
    assert is_structured_response_format({"type": "text"}) is False
    assert is_structured_response_format(None) is False


# ── cue construction ─────────────────────────────────────────────────────────


def test_to_raif_declaration_and_optionals():
    decl, degraded = response_format_to_raif(_WEATHER_RF)
    # Required field has no `?`; absent-from-required fields are optional.
    assert decl == "city:s\nunit:s?\ndays:n?"
    assert degraded == []


def test_schema_block_wraps_declaration():
    block = build_response_schema_block(_WEATHER_RF)
    assert block.startswith("<schema>\n")
    assert block.endswith("\n</schema>")
    assert "city:s" in block


def test_schema_block_empty_for_json_object():
    # No representable schema -> no cue (the model still free-generates RAIF-G).
    assert build_response_schema_block({"type": "json_object"}) == ""
    assert build_response_schema_block({"type": "text"}) == ""


# ── routing ──────────────────────────────────────────────────────────────────


def test_callable_tools_route_to_tools():
    assert classify_request([{"type": "function"}], "auto", None) == "tools"
    assert classify_request([{"type": "function"}], None, None) == "tools"


def test_tool_choice_none_falls_through_to_structured():
    # tools present but disabled -> the response_format wins.
    assert classify_request([{"type": "function"}], "none", _WEATHER_RF) == "structured"


def test_response_format_without_tools_is_structured():
    assert classify_request(None, None, _WEATHER_RF) == "structured"
    assert classify_request([], None, {"type": "json_object"}) == "structured"


def test_plain_chat_passes_through():
    assert classify_request(None, None, None) == "plain"
    assert classify_request(None, None, {"type": "text"}) == "plain"
    assert classify_request([], "none", None) == "plain"


# ── decode ───────────────────────────────────────────────────────────────────


def test_decode_content_schemaless():
    out = decode_content("city=Oslo\nunit=celsius\ndays=5")
    assert json.loads(out) == {"city": "Oslo", "unit": "celsius", "days": 5}


def test_decode_content_is_compact():
    # Compact separators (no spaces) — token-efficient, byte-stable.
    assert decode_content("city=Oslo") == '{"city":"Oslo"}'


def test_decode_content_with_declaration_types_numbers():
    decl, _ = response_format_to_raif(_WEATHER_RF)
    out = decode_content("city=Oslo\ndays=5", decl)
    assert json.loads(out)["days"] == 5  # typed as number, not "5"


def test_decode_content_fails_closed():
    # Unrepairable input -> None (caller keeps raw output rather than emit junk).
    assert decode_content("\x00\x01 not raif at all", "city:s") is None
