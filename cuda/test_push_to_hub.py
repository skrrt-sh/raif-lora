"""Tests for the HF model-card generator in push_to_hub.

`build_model_card` is pure (it only reads optional run_meta.json/eval.json from the
adapter dir, which may be absent), so these run without the heavy HF deps. They pin
the one property that silently shipped wrong before: the license must follow the
*base model*, not be hardcoded to Llama.
"""

from __future__ import annotations

from pathlib import Path

import push_to_hub

_NO_ADAPTER = Path("/nonexistent-adapter")  # load_json returns None → card uses defaults


def test_llama_base_gets_llama_license():
    """A Llama base yields the Llama 3.2 license + 'Built with Llama' attribution."""
    card = push_to_hub.build_model_card(
        _NO_ADAPTER, "me/raif-llama-3.2-3b-lora", "unsloth/Llama-3.2-3B-Instruct"
    )
    assert "license: llama3.2" in card
    assert "Built with Llama" in card


def test_qwen_base_gets_apache_license_not_llama():
    """A Qwen base yields apache-2.0 and never leaks the Llama license/attribution."""
    card = push_to_hub.build_model_card(
        _NO_ADAPTER, "me/raif-qwen-0.5b-lora", "Qwen/Qwen2.5-0.5B-Instruct"
    )
    assert "license: apache-2.0" in card
    assert "license: llama3.2" not in card
    assert "Built with Llama" not in card
    assert "Qwen" in card
