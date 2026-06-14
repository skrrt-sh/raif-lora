"""Model registry for the RAIF examples — pick a base + adapter by short name.

Every published RAIF adapter was trained with PEFT/unsloth (torch format). The
local runtime here is MLX, which uses a different adapter layout, so each one is
converted to MLX once (a lossless rename + transpose; scale = alpha/rank = 2.0).
See `setup_adapter.py`.

Nothing needs to exist locally: bases come from `mlx-community/*` and adapters
from the published `skrrt-sh/*` LoRA repos on the Hub. If you happen to have the
local training artifacts (`models/`, `adapters-cuda/`), those are used instead to
skip the download.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import safetensors.numpy as sn

REPO = Path(__file__).resolve().parent.parent

# Every full-reg adapter targets all 7 projections on every layer.
LORA_KEYS = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]

MODELS: dict[str, dict] = {
    "llama-3b": {
        "label": "Llama-3.2-3B + RAIF  (the flagship, 100% valid fidelity)",
        "base_hf": "mlx-community/Llama-3.2-3B-Instruct-bf16",
        "base_local": REPO / "models" / "llama-3.2-3b-instruct-bf16",
        "peft_local": REPO / "adapters-cuda" / "full-reg",
        "peft_hf": "skrrt-sh/raif-llama-3.2-3b-lora",
        "mlx_dir": REPO / "adapters" / "llama-3b-mlx",
        "strip_think": False,
    },
    "qwen-0.5b": {
        "label": "Qwen2.5-0.5B + RAIF  (tiny & fast; ~98% valid fidelity)",
        "base_hf": "mlx-community/Qwen2.5-0.5B-Instruct-bf16",
        "base_local": None,
        "peft_local": REPO / "adapters-cuda" / "qwen05-tbl-full",
        "peft_hf": "skrrt-sh/raif-qwen2.5-0.5b-lora",
        "mlx_dir": REPO / "adapters" / "qwen-0.5b-mlx",
        "strip_think": False,
    },
    "qwen-4b": {
        "label": "Qwen3-4B + RAIF  (deployable agent model; ~14 GB)",
        "base_hf": "mlx-community/Qwen3-4B-Instruct-2507-bf16",
        "base_local": None,
        "peft_local": REPO / "adapters-cuda" / "qwen3-4b-full",
        "peft_hf": "skrrt-sh/raif-qwen3-4b-lora",
        "mlx_dir": REPO / "adapters" / "qwen-4b-mlx",
        "strip_think": True,  # Qwen3 emits a leading <think></think> block
    },
}

DEFAULT_MODEL = "llama-3b"

_THINK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)


def spec(model: str) -> dict:
    if model not in MODELS:
        raise SystemExit(f"unknown --model {model!r}; choose from {list(MODELS)}")
    return MODELS[model]


def base_path(s: dict) -> str:
    """Local base dir if present, else the Hub repo id (mlx_lm.load fetches it)."""
    local = s.get("base_local")
    if local and Path(local).exists():
        return str(local)
    return s["base_hf"]


def strip_think(s: dict, text: str) -> str:
    """Drop a leading <think>…</think> block (Qwen3) before decoding."""
    return _THINK_RE.sub("", text, count=1) if s.get("strip_think") else text


def resolve_peft_dir(s: dict) -> Path:
    """Local PEFT adapter if checked out, else download the published LoRA."""
    local = s.get("peft_local")
    if local and (Path(local) / "adapter_model.safetensors").exists():
        print(f"  using local adapter: {local}")
        return Path(local)
    print(f"  downloading adapter {s['peft_hf']} from the Hub...")
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=s["peft_hf"], allow_patterns=["adapter_*"]))


_KEY_RE = re.compile(
    r"^base_model\.model\.(model\.layers\.\d+\.(?:self_attn|mlp)\.\w+)\.lora_([AB])\.weight$"
)


def convert_to_mlx(peft_dir: Path, out_dir: Path, scale: float = 2.0) -> int:
    """PEFT LoRA -> MLX adapter. Rename keys, transpose A/B, write config.
    Returns the number of transformer layers covered."""
    src = sn.load_file(str(peft_dir / "adapter_model.safetensors"))
    out: dict = {}
    for k, v in src.items():
        m = _KEY_RE.match(k)
        if not m:
            raise ValueError(f"unexpected PEFT key, refusing to guess: {k!r}")
        out[f"{m.group(1)}.lora_{m.group(2).lower()}"] = v.T.copy()

    num_layers = max(int(re.search(r"layers\.(\d+)\.", k).group(1)) for k in out) + 1
    out_dir.mkdir(parents=True, exist_ok=True)
    sn.save_file(out, str(out_dir / "adapters.safetensors"))
    (out_dir / "adapter_config.json").write_text(json.dumps({
        "fine_tune_type": "lora",
        "num_layers": num_layers,
        "lora_parameters": {"rank": 32, "scale": scale, "dropout": 0.05,
                            "keys": LORA_KEYS},
    }, indent=2))
    return num_layers
