"""RAIF for vLLM — one installable plugin for transparent RAIF token savings.

A fine-tuned model emits RAIF-G (a compact line format); this package makes a
stock vLLM server speak RAIF transparently to existing OpenAI clients, with no
proxy and no client changes. It is loaded as a single `vllm.general_plugins`
entrypoint (`raif = "raif_vllm.plugin:register"`); `register()` registers the
reasoning + tool parsers and installs the prompt-injection hook.

Pure helpers (`structured`, `inject`, `reasoning.route_and_decode`) carry no vLLM
import and are unit-tested on any machine; the parser shims and the `render_chat`
monkeypatch are the thin glue that wires vLLM's types to them.
"""

from __future__ import annotations
