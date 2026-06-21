"""RAIF for the `response_format` / structured-output request path.

A fine-tuned model emits RAIF-G; for an OpenAI `response_format` request (the
JSON-mode / structured-output path, as opposed to tool calls) the model's RAIF-G
output is decoded to JSON and returned as the assistant message *content*. This
module holds the pure request<->RAIF logic with NO vLLM import, mirroring the
tool-call helpers in `raif_vllm` so it is fully unit-testable on any machine.

The conceptual pipeline for a structured request has three links:
  (a) inject a compact `<schema>` cue into the prompt (so the model knows the
      field names) -> `build_response_schema_block` + `raif_vllm.inject_schema`;
  (b) the model emits RAIF-G (free generation);
  (c) decode RAIF-G -> JSON for `message.content` -> `decode_content`.

Reuses `raif.schema_bridge` (JSON Schema -> compact RAIF declaration) and
`raif.decode` (RAIF-G -> JSON via the deterministic repair tier, fail-closed).
"""

from __future__ import annotations

import json
from typing import Any

from raif import decode
from raif.schema_bridge import json_schema_to_raif_schema


def to_plain(obj: Any) -> Any:
    """Normalize a vLLM request field to plain dict/list/scalar.

    vLLM hands `tools`/`response_format`/`tool_choice` as pydantic models (which
    have no `.get`/`isinstance(dict)`); the pure helpers expect JSON-shaped
    dicts. `model_dump()` recovers that shape; lists are normalized element-wise;
    anything already plain (dict, str, None) passes through.
    """
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, list):
        return [to_plain(x) for x in obj]
    return obj


def response_format_schema(response_format: Any) -> dict | None:
    """The JSON Schema carried by an OpenAI `response_format`, or None.

    `{"type": "json_schema", "json_schema": {"schema": {...}}}` -> the inner
    schema. `{"type": "json_object"}` -> `{}` (a valid structured request with no
    schema). Anything else (`{"type": "text"}`, absent, malformed) -> None.

    The inner schema lives under `schema` in the OpenAI wire format, but vLLM's
    `JsonSchemaResponseFormat` names that field `json_schema` (since `schema` is
    a reserved pydantic name), so its `model_dump()` emits `json_schema`. Accept
    either key.
    """
    if not isinstance(response_format, dict):
        return None
    kind = response_format.get("type")
    if kind == "json_schema":
        inner = response_format.get("json_schema") or {}
        schema = inner.get("schema")
        if not isinstance(schema, dict):
            schema = inner.get("json_schema")
        return schema if isinstance(schema, dict) else {}
    if kind == "json_object":
        return {}
    return None


def response_format_to_raif(response_format: Any) -> tuple[str, list[str]]:
    """`(raif_declaration, degraded_fields)` for an OpenAI `response_format`.

    Empty (`json_object`, schemaless, or wholly unrepresentable) -> `("", [])`.
    """
    schema = response_format_schema(response_format)
    if not schema:
        return "", []
    return json_schema_to_raif_schema(schema)


def build_response_schema_block(response_format: Any) -> str:
    """`<schema>...</schema>` prompt cue for a `response_format`, or "".

    The declaration is a cue only — the model is trained never to echo it. Empty
    when the request carries no representable schema (e.g. `json_object`).
    """
    decl, _degraded = response_format_to_raif(response_format)
    if not decl:
        return ""
    return f"<schema>\n{decl}\n</schema>"


def is_structured_response_format(response_format: Any) -> bool:
    """True if `response_format` asks for JSON (json_schema or json_object)."""
    return response_format_schema(response_format) is not None


def classify_request(tools: Any, tool_choice: Any, response_format: Any) -> str:
    """Route a chat request: `"tools"`, `"structured"`, or `"plain"`.

    Callable tools win (present and not disabled via `tool_choice="none"`) — they
    go through the tool-call parser. Otherwise a JSON `response_format` is
    `"structured"` (decode RAIF-G into message content). Everything else is
    `"plain"` and passes through untouched.
    """
    if tools and tool_choice != "none":
        return "tools"
    if is_structured_response_format(response_format):
        return "structured"
    return "plain"


def decode_content(model_output: str, raif_decl: str | None = None) -> str | None:
    """RAIF-G text -> compact JSON string for `message.content`, or None.

    Uses strict `decode` (with its deterministic repair tier); an output that
    still fails to parse yields None — fail closed rather than return malformed
    content. `raif_decl` is the RAIF declaration from `response_format_to_raif`
    (optional; schemaless decode still works, just with looser typing).
    """
    result = decode(model_output, raif_decl or None)
    if not result["ok"]:
        return None
    return json.dumps(result["value"], separators=(",", ":"))
