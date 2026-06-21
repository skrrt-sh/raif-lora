"""The single `vllm.general_plugins` entrypoint: `register()`.

Loaded once at vLLM engine import (in every engine process). It wires RAIF into
a stock vLLM server with no proxy and no client changes:

  1. registers the `raif` reasoning parser (decode RAIF-G -> JSON content) and
     the `raif` tool parser (decode RAIF-G -> tool_calls) by importing their
     modules, whose `@...Manager.register_module("raif")` decorators self-register;
  2. monkeypatches `OpenAIServingRender.render_chat` so the compact `<schema>`
     cue is injected into the prompt BEFORE chat templating — the only in-process
     seam that sees the full request pre-template (the tool-parser `adjust_request`
     hook fires too late). For the `response_format` path it also neutralizes
     vLLM's native JSON guided-decoding so the model is free to emit RAIF-G.

Everything degrades to a no-op when vLLM (or the specific seam) is absent, so the
entrypoint is import-safe on any machine. Wire it up in `pyproject.toml`:

    [project.entry-points."vllm.general_plugins"]
    raif = "raif_vllm.plugin:register"

Verified end-to-end on vLLM 0.19.0 (CUDA-12, A40): plain chat / tools / json_schema
`response_format` all decode correctly through this one plugin.
"""

from __future__ import annotations

import contextlib
import functools
import importlib
import logging

from .inject import prepare_chat_request

logger = logging.getLogger("raif_vllm")

# Pre-template inject seams to wrap, in priority order: (module, class, method).
# The chat prompt is built here from `request.messages` BEFORE templating, so a
# wrapper can inject the `<schema>` cue and neutralize native JSON guided-decoding.
# `render_chat` (>=~0.19) receives the full request; paths drift across versions,
# so we try each and patch the first that exists. Tools still work without any of
# these (chat template + tool parser); only the response_format inject needs it.
_RENDER_SEAMS = (
    ("vllm.entrypoints.serve.render.serving", "OpenAIServingRender", "render_chat"),
    ("vllm.entrypoints.openai.chat_completion.serving", "OpenAIServingChat", "render_chat_request"),
)


def _wrap_with_prepare(cls: type, method: str) -> None:
    original = getattr(cls, method)

    @functools.wraps(original)
    async def wrapper(self, request, *args, **kwargs):
        # Never break serving on a RAIF prep error — degrade to native behavior.
        with contextlib.suppress(Exception):
            prepare_chat_request(request)
        return await original(self, request, *args, **kwargs)

    wrapper._raif_wrapped = True  # type: ignore[attr-defined]
    setattr(cls, method, wrapper)


def _install_render_patch() -> bool:
    """Wrap the first available pre-template inject seam. Returns True if patched.

    Idempotent (won't double-wrap). False means no seam was found (vLLM absent or
    an unrecognized version) — the response_format inject is then unavailable,
    but the tools path is unaffected.
    """
    for module_name, class_name, method in _RENDER_SEAMS:
        try:
            cls = getattr(importlib.import_module(module_name), class_name)
        except (ImportError, AttributeError):
            continue
        if getattr(getattr(cls, method, None), "_raif_wrapped", False):
            logger.info("RAIF render patch already installed on %s.%s", class_name, method)
            return True
        try:
            _wrap_with_prepare(cls, method)
        except AttributeError:
            continue
        logger.info("RAIF render patch installed on %s.%s", class_name, method)
        return True
    logger.warning(
        "RAIF: no pre-template inject seam found — response_format injection is "
        "DISABLED on this vLLM build (tools path unaffected). Tried: %s",
        ", ".join(f"{m}.{c}.{meth}" for m, c, meth in _RENDER_SEAMS),
    )
    return False


def register() -> None:
    """vLLM `general_plugins` entrypoint — register parsers + install the hook.

    Idempotent and import-safe without vLLM (registrations and the patch become
    no-ops). Importing the parser modules runs their `register_module` decorators.
    """
    from . import reasoning as _reasoning  # noqa: F401 - registers "raif" parser
    from . import tool_parser as _tool_parser  # noqa: F401 - registers "raif"

    logger.info("RAIF plugin: registered 'raif' reasoning + tool parsers")
    _install_render_patch()
