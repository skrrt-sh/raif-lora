"""Behavior tests for the single `general_plugins` entrypoint.

`register()` is the entrypoint vLLM calls at engine import. It must be safe to
call (and idempotent) even on a machine without vLLM — the parser registrations
and the `render_chat` monkeypatch degrade to no-ops there, so these run anywhere.
"""

from __future__ import annotations

from raif_vllm.plugin import register


def test_register_is_callable_without_vllm():
    # No vLLM installed in this env -> register must not raise (parser
    # registration + render_chat patch degrade to no-ops).
    register()


def test_register_is_idempotent():
    register()
    register()  # second call must not raise (e.g. double-wrapping the patch)
