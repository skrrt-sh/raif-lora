"""RAIF reasoning-parser plugin: decode RAIF-G into `message.content`.

On vLLM (>=0.19), a reasoning parser runs for *every* chat completion when
`--reasoning-parser` is set — gated only on `if reasoning_parser:`, independent
of tools/`response_format`. Its `extract_reasoning(model_output, request)`
splits the generation into `(reasoning, content)`; `content` then becomes
`message.content` (and, for tool requests, is what the tool parser reads next).
That makes it the in-process seam for the `response_format` path: a structured
request's RAIF-G output is decoded to JSON and returned as `content`, with NO
proxy and NO change to the OpenAI client.

Routing (pure, `route_and_decode`):
  - `"structured"` (json `response_format`, no callable tools): decode RAIF-G ->
    compact JSON for `content`; fail OPEN to the raw text if decode fails (the
    client still gets the generation rather than an empty message).
  - `"tools"`: pass `content` through UNCHANGED so the tool parser can decode it
    into `tool_calls` downstream.
  - `"plain"`: pass through untouched.

The pure helpers take no vLLM import and are unit-tested; `RaifReasoningParser`
is a thin shim, defined only when vLLM is importable, that wires vLLM's types to
them (mirrors `raif_vllm.py`).
"""

from __future__ import annotations

from typing import Any

from .structured import (
    classify_request,
    decode_content,
    response_format_to_raif,
    to_plain,
)

# Closing markers the generation profile may emit to frame RAIF-G (ADR-0019);
# their presence signals a streamed block is complete and safe to decode.
_TERMINATORS = ("</raif>", "<|raif_end|>")


def _get(obj: Any, name: str) -> Any:
    """Read `name` off a request that may be a pydantic model or a plain dict."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def route_and_decode(model_output: str, request: Any) -> tuple[None, str]:
    """`(reasoning, content)` for a chat generation — reasoning is always None.

    Pure: `request` need only expose `tools`, `tool_choice`, `response_format`,
    `vllm_xargs` (dict or attribute access). Structured requests get RAIF-G
    decoded to JSON; tools and plain chat pass the output through unchanged.

    The inject step (`raif_inject`) clears `response_format` and stashes the RAIF
    declaration in `vllm_xargs["raif_decl"]`; when present, that key is the
    authoritative signal/schema (its presence means "this is a RAIF structured
    request"). Otherwise fall back to `response_format` (pre-inject / dict
    callers that did not pass through the monkeypatch).
    """
    xargs = to_plain(_get(request, "vllm_xargs"))
    if isinstance(xargs, dict) and "raif_decl" in xargs:
        decl = xargs["raif_decl"] or None
        decoded = decode_content(model_output, decl)
        return None, (decoded if decoded is not None else model_output)

    tools = to_plain(_get(request, "tools"))
    tool_choice = to_plain(_get(request, "tool_choice"))
    response_format = to_plain(_get(request, "response_format"))

    kind = classify_request(tools, tool_choice, response_format)
    if kind != "structured":
        return None, model_output

    decl, _degraded = response_format_to_raif(response_format)
    decoded = decode_content(model_output, decl)
    # Fail OPEN to raw text: the client asked for JSON, but an empty message is
    # worse than the model's literal output if RAIF-G decode could not repair it.
    return None, (decoded if decoded is not None else model_output)


def has_terminator(text: str) -> bool:
    """True once a RAIF-G framing terminator appears in streamed text."""
    return any(marker in text for marker in _TERMINATORS)


# ── vLLM shim ────────────────────────────────────────────────────────────────
# Everything below imports vLLM. Guarded so this module stays importable (and the
# pure helpers above stay testable) on machines without vLLM installed.


def _vllm_reasoning_classes():
    """`(ReasoningParser, ReasoningParserManager)` — import path stable >=0.19."""
    from vllm.reasoning import ReasoningParser, ReasoningParserManager

    return ReasoningParser, ReasoningParserManager


def _vllm_delta_message():
    try:
        from vllm.entrypoints.openai.protocol import DeltaMessage  # <=0.19-era
    except ImportError:
        from vllm.entrypoints.openai.chat_completion.protocol import DeltaMessage

    return DeltaMessage


try:
    _ReasoningParser, _ReasoningParserManager = _vllm_reasoning_classes()
except ImportError:
    _ReasoningParser = None  # vLLM not installed — pure helpers remain usable.


if _ReasoningParser is not None:
    _DeltaMessage = _vllm_delta_message()

    @_ReasoningParserManager.register_module("raif")
    class RaifReasoningParser(_ReasoningParser):
        """Decode RAIF-G into `message.content` for the `response_format` path.

        Non-streaming uses the full request (so it can read `response_format`);
        streaming has no request, so it decodes coarsely — buffering until a
        RAIF-G terminator appears, which plain chat never emits, then decoding
        schemaless. Pass-through otherwise.
        """

        def __init__(self, tokenizer: Any, *args: Any, **kwargs: Any) -> None:
            super().__init__(tokenizer, *args, **kwargs)
            self._content_sent_upto = 0

        def extract_reasoning(
            self, model_output: str, request: Any
        ) -> tuple[str | None, str | None]:
            return route_and_decode(model_output, request)

        def extract_reasoning_streaming(
            self,
            previous_text: str,
            current_text: str,
            delta_text: str,
            previous_token_ids: Any,
            current_token_ids: Any,
            delta_token_ids: Any,
        ) -> Any:
            # Coarse tier: a structured generation is decoded only once its RAIF-G
            # block terminates; until then (and for plain chat, which never emits
            # a terminator) stream the delta through unchanged.
            if not has_terminator(current_text):
                return _DeltaMessage(content=delta_text)
            decoded = decode_content(current_text)
            if decoded is None:
                return _DeltaMessage(content=delta_text)
            # Emit the decoded JSON suffix not yet sent (coarse: typically the
            # whole object at the terminating delta).
            suffix = decoded[self._content_sent_upto :]
            self._content_sent_upto = len(decoded)
            return _DeltaMessage(content=suffix)

        # Required abstract methods — RAIF-G has no separate reasoning channel,
        # so there is no reasoning/answer split: everything is content.
        def is_reasoning_end(self, input_ids: list[int]) -> bool:
            return True

        def extract_content_ids(self, input_ids: list[int]) -> list[int]:
            return input_ids
