"""RAIF inject step: add the `<schema>` cue to the prompt before templating.

Because the tool-parser `adjust_request` hook is a no-op for prompt mutation
(chat templating renders the prompt first), the cue for BOTH the tools and the
`response_format` paths is injected here instead — from the `render_chat`
monkeypatch, which sees the full request before templating. Pure (no vLLM
import) and unit-tested; the monkeypatch is a thin wiring shim.
"""

from __future__ import annotations

from typing import Any

from .structured import (
    build_response_schema_block,
    classify_request,
    response_format_to_raif,
    to_plain,
)
from .tool_parser import build_schema_block, inject_schema, resolve_tool


def _get(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _set(obj: Any, name: str, value: Any) -> None:
    if isinstance(obj, dict):
        obj[name] = value
    else:
        setattr(obj, name, value)


def prepare_chat_request(request: Any) -> Any:
    """Inject the RAIF `<schema>` cue into `request.messages` before templating.

    Mutates and returns `request`. Routes to the tools cue or the
    `response_format` cue; plain chat is returned untouched.
    """
    # vLLM hands these as pydantic models; normalize to JSON-shaped dicts so the
    # pure helpers (which expect dicts) can read them. Only used for reading;
    # request.tools itself is left untouched for vLLM's own tool parser.
    tools = to_plain(_get(request, "tools"))
    tool_choice = to_plain(_get(request, "tool_choice"))
    response_format = to_plain(_get(request, "response_format"))

    kind = classify_request(tools, tool_choice, response_format)
    if kind == "tools":
        tool = resolve_tool(tools, tool_choice)
        block = build_schema_block(tool) if tool is not None else ""
    elif kind == "structured":
        decl, _degraded = response_format_to_raif(response_format)
        block = build_response_schema_block(response_format)
        # Stash the RAIF declaration (a plain string, so it fits `vllm_xargs`)
        # for the decoder, then clear `response_format` so vLLM does NOT apply
        # its native JSON guided-decoding — the model must be free to emit
        # RAIF-G. The reasoning parser recovers the schema from `raif_decl`.
        xargs = dict(_get(request, "vllm_xargs") or {})
        xargs["raif_decl"] = decl
        _set(request, "vllm_xargs", xargs)
        _set(request, "response_format", None)
    else:
        return request

    if block:
        _set(request, "messages", inject_schema(_get(request, "messages"), block))
    return request
