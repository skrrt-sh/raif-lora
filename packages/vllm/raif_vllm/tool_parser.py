"""vLLM tool-call parser plugin for RAIF-emitting models.

A fine-tuned model emits RAIF-G; this plugin converts it to OpenAI `tool_calls`
right before the response is assembled. The RAIF<->tool-call logic lives in pure
helpers (no vLLM import) so it is fully unit-testable; `RaifToolParser` is a thin
shim, defined only when vLLM is importable, that wires vLLM's types to them.

Depends on `raif-format` (>=0.6) for the decoder and the JSON-Schema bridge.
Range-tolerant across vLLM versions (import paths shimmed; see `_vllm_*` below).
Verified against vLLM v0.6.6, v0.9.2, and v0.23.0/main: the `ToolParser` method
signatures, the `register_module` decorator, and the CLI flags are stable; only
two import paths moved, which the shims below absorb.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from raif import decode
from raif.schema_bridge import tool_to_schema

# Closing mode markers the generation profile may emit to frame RAIF-G (ADR-0019).
# Their presence signals the args block is complete and safe to decode+emit.
_TERMINATORS = ("</raif>", "<|raif_end|>")


def _has_terminator(text: str) -> bool:
    return any(marker in text for marker in _TERMINATORS)


def _new_call_id() -> str:
    """A tool-call id in vLLM's `chatcmpl-tool-<hex>` shape (stdlib, no vLLM)."""
    return f"chatcmpl-tool-{uuid.uuid4().hex}"


class CoarseToolCallStreamer:
    """Coarse-tier streaming: emit the tool name immediately, then the whole
    arguments object once the RAIF-G block terminates. Stateful per request;
    each of name/arguments is emitted at most once. Pure (no vLLM types) — it
    returns plain event dicts the `RaifToolParser` shim maps to `DeltaMessage`.
    """

    def __init__(self, name: str, schema: Any | None = None) -> None:
        self._name = name
        self._schema = schema
        self._name_sent = False
        self._args_sent = False

    def update(self, current_text: str) -> list[dict]:
        events: list[dict] = []
        if not self._name_sent:
            self._name_sent = True
            events.append({"name": self._name, "id": _new_call_id()})
        if not self._args_sent and _has_terminator(current_text):
            args = decode_arguments(current_text, self._schema)
            if args is not None:
                self._args_sent = True
                events.append({"arguments": json.dumps(args, separators=(",", ":"))})
        return events


def build_schema_block(tool: dict) -> str:
    """`<schema>...</schema>` prompt cue for one tool, or "" if it has no fields.

    The declaration is a cue only — the model is trained never to echo it.
    """
    decl, _degraded = tool_to_schema(tool)
    if not decl:
        return ""
    return f"<schema>\n{decl}\n</schema>"


def resolve_tool(tools: list[dict], tool_choice: Any) -> dict | None:
    """The single tool the model output should be parsed against, or None.

    A named `tool_choice` selects that function; otherwise a lone tool is used.
    Several tools under `"auto"` are ambiguous from args alone (coarse tier),
    so this returns None and the caller emits no tool call.
    """
    if isinstance(tool_choice, dict):
        name = (tool_choice.get("function") or {}).get("name")
        for tool in tools or []:
            if (tool.get("function") or {}).get("name") == name:
                return tool
        return None
    if tools and len(tools) == 1:
        return tools[0]
    return None


def inject_schema(messages: list[dict], schema_block: str) -> list[dict]:
    """Append the `<schema>` cue to the last user message (fine_tune_plan §3.2).

    Returns a new list; the input is not mutated. No-op when the block is empty
    or the target message has non-string (e.g. multimodal) content.
    """
    if not schema_block:
        return messages
    out = [dict(m) for m in messages]
    for msg in reversed(out):
        if msg.get("role") == "user":
            if isinstance(msg.get("content"), str):
                msg["content"] = f"{msg['content']}\n\n{schema_block}"
            return out
    return out


def decode_arguments(model_output: str, schema: Any | None = None) -> dict | None:
    """RAIF-G text -> tool-call arguments dict, or None if unrepairable.

    Uses strict `decode` (which applies the deterministic repair tier); a model
    output that still fails to parse yields None — fail closed rather than emit a
    malformed tool call.
    """
    result = decode(model_output, schema)
    return result["value"] if result["ok"] else None


# ── vLLM shim ───────────────────────────────────────────────────────────────
# Everything below imports vLLM. It is guarded so this module stays importable
# (and the pure helpers above stay testable) on machines without vLLM installed.


def _vllm_toolparser_classes():
    """`(ToolParser, ToolParserManager)` across the import-path move (~v0.10)."""
    try:
        from vllm.tool_parsers import ToolParser, ToolParserManager  # newer
    except ImportError:
        from vllm.entrypoints.openai.tool_parsers import (  # <= v0.9
            ToolParser,
            ToolParserManager,
        )
    return ToolParser, ToolParserManager


def _vllm_protocol_types():
    """Tool-call payload types across the protocol-module move (~v0.10)."""
    try:
        from vllm.entrypoints.openai.engine.protocol import (  # newer
            DeltaFunctionCall,
            DeltaMessage,
            DeltaToolCall,
            ExtractedToolCallInformation,
            FunctionCall,
            ToolCall,
        )
    except ImportError:
        from vllm.entrypoints.openai.protocol import (  # <= v0.9
            DeltaFunctionCall,
            DeltaMessage,
            DeltaToolCall,
            ExtractedToolCallInformation,
            FunctionCall,
            ToolCall,
        )
    return {
        "DeltaFunctionCall": DeltaFunctionCall,
        "DeltaMessage": DeltaMessage,
        "DeltaToolCall": DeltaToolCall,
        "ExtractedToolCallInformation": ExtractedToolCallInformation,
        "FunctionCall": FunctionCall,
        "ToolCall": ToolCall,
    }


def _as_dict(obj: Any) -> Any:
    """vLLM request fields may be pydantic models or plain dicts/strings."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return obj


def _request_tools(request: Any) -> list[dict]:
    return [_as_dict(t) for t in (getattr(request, "tools", None) or [])]


def _request_tool_choice(request: Any) -> Any:
    return _as_dict(getattr(request, "tool_choice", None))


try:
    _ToolParser, _ToolParserManager = _vllm_toolparser_classes()
except ImportError:
    _ToolParser = None  # vLLM not installed — pure helpers above remain usable.


if _ToolParser is not None:
    _T = _vllm_protocol_types()

    def _events_to_delta(events: list[dict], types: dict) -> Any:
        """Map `CoarseToolCallStreamer` events to a single `DeltaMessage`."""
        if not events:
            return None
        calls = []
        for ev in events:
            if "name" in ev:
                calls.append(
                    types["DeltaToolCall"](
                        index=0,
                        id=ev["id"],
                        type="function",
                        function=types["DeltaFunctionCall"](name=ev["name"]),
                    )
                )
            else:
                calls.append(
                    types["DeltaToolCall"](
                        index=0,
                        function=types["DeltaFunctionCall"](arguments=ev["arguments"]),
                    )
                )
        return types["DeltaMessage"](tool_calls=calls)

    @_ToolParserManager.register_module("raif")
    class RaifToolParser(_ToolParser):
        """Decode RAIF-G tool calls into OpenAI `tool_calls` (coarse streaming)."""

        def __init__(self, tokenizer: Any, *args: Any, **kwargs: Any) -> None:
            # *args/**kwargs absorbs the `tools=` param added to __init__ in newer
            # vLLM while staying compatible with the older `(tokenizer)` form.
            super().__init__(tokenizer, *args, **kwargs)
            self._streamer: CoarseToolCallStreamer | None = None

        def adjust_request(self, request: Any) -> Any:
            # Keep RAIF-G framing terminators (`</raif>`, `<|raif_end|>`) in the
            # output so the parser can see them. Deliberately do NOT call
            # super().adjust_request (it sets `structured_outputs` to the tools'
            # JSON schema, forcing JSON output — the C16 trap) and do NOT inject
            # the `<schema>` cue here: `adjust_request` fires AFTER chat
            # templating, so prompt injection is a no-op. The cue is injected
            # pre-template by the `render_chat` hook (see `plugin.register`).
            request.skip_special_tokens = False
            return request

        def extract_tool_calls(self, model_output: str, request: Any) -> Any:
            tool = resolve_tool(_request_tools(request), _request_tool_choice(request))
            args = None if tool is None else decode_arguments(model_output)
            if args is None:
                return _T["ExtractedToolCallInformation"](
                    tools_called=False, tool_calls=[], content=model_output
                )
            return _T["ExtractedToolCallInformation"](
                tools_called=True,
                tool_calls=[
                    _T["ToolCall"](
                        function=_T["FunctionCall"](
                            name=tool["function"]["name"],
                            arguments=json.dumps(args, separators=(",", ":")),
                        )
                    )
                ],
                content=None,
            )

        def extract_tool_calls_streaming(
            self,
            previous_text: str,
            current_text: str,
            delta_text: str,
            previous_token_ids: Any,
            current_token_ids: Any,
            delta_token_ids: Any,
            request: Any,
        ) -> Any:
            tool = resolve_tool(_request_tools(request), _request_tool_choice(request))
            if tool is None:
                return None
            if self._streamer is None:
                self._streamer = CoarseToolCallStreamer(tool["function"]["name"])
            return _events_to_delta(self._streamer.update(current_text), _T)
