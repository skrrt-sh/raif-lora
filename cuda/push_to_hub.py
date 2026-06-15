"""Push a trained RAIF LoRA adapter to the Hugging Face Hub, with a model card
auto-generated from the adapter's run_meta.json + eval.json.

Auth: set HF_TOKEN, or run `huggingface-cli login` first (write-scope token from
https://huggingface.co/settings/tokens). The token is read from the environment
or the CLI's cache — it is never taken as a command-line argument.

    python cuda/push_to_hub.py --adapter ./adapters-cuda/full-reg \
        --repo <your-username>/raif-llama-3.2-3b-lora            # public
    python cuda/push_to_hub.py --adapter ./adapters-cuda/full-reg \
        --repo <your-username>/raif-llama-3.2-3b-lora --private

Uploads the whole adapter dir (adapter_config.json + adapter_model.safetensors +
tokenizer files + run_meta.json + eval.json). A LoRA adapter is tiny vs the base.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def load_json(p: Path):
    """Load JSON from the given path, or return None if it doesn't exist."""
    return json.loads(p.read_text()) if p.exists() else None


def _fmt_loss(x) -> str:
    """Render a loss value to 4 significant figures, or em-dash if missing/None."""
    return f"{x:.4g}" if isinstance(x, (int, float)) else "—"


def license_for(base_model: str) -> dict:
    """License id + attribution wording for the base model's family. Qwen2.5
    small bases (0.5/1.5/7B…) are Apache-2.0; Llama 3.2 carries Meta's community
    license. Returned dict feeds the card frontmatter + attribution section."""
    mid = base_model.lower()
    if "qwen" in mid:
        return {"id": "apache-2.0", "family": "Qwen2.5", "builtwith": "Built with Qwen",
                "attrib": "Derivative of Qwen2.5 — **Apache-2.0** (the Qwen2.5 small "
                          "bases are Apache-2.0 licensed)."}
    return {"id": "llama3.2", "family": "Llama 3.2", "builtwith": "Built with Llama",
            "attrib": "Derivative of Llama 3.2 — **Llama 3.2 Community License** "
                      'applies ("Built with Llama").'}


def build_model_card(adapter: Path, repo: str, base_model: str) -> str:
    """Build a Hugging Face model card (markdown) from adapter metadata.

    Reads run_meta.json and eval.json from the adapter dir if present, otherwise
    falls back to empty defaults. License/attribution and the token-cost note are
    derived from the base model's family via license_for().
    """
    meta = load_json(adapter / "run_meta.json") or {}
    ev = load_json(adapter / "eval.json") or {}
    hp = meta.get("hyperparams", {})
    res = meta.get("result", {})
    data = meta.get("data", {})

    rows = ""
    for name, stats in (ev.get("groups") or {}).items():
        n = stats.get("n", 0)
        if not n:
            continue
        rows += (f"| {name} | {100*stats['parse']/n:.0f}% | "
                 f"{100*stats['fidelity']/n:.0f}% | {n} |\n")
    gate = ev.get("gate") or {}
    gate_line = ("**Acceptance gate: PASS**" if gate.get("passed")
                 else "Acceptance gate: not fully met — see the per-group numbers below."
                 if gate.get("passed") is False else "")
    lic = license_for(base_model)
    # The ~10% real-world figure holds across tokenizers (cl100k −9, o200k −10,
    # Llama −9, Qwen −8, Mistral −8), so every card leads with it (no per-family
    # hedge). The −14% is the Llama-3.2/cl100k eval-corpus aggregate (table-heavy);
    # only the Llama card cites it, since Qwen lands nearer −12% there.
    token_note = ("- Token cost: ~10% fewer than minified JSON on real "
                  "function-call data; −14% on the table-heavy eval corpus."
                  if lic["family"].startswith("Llama")
                  else "- Token cost: ~10% fewer than minified JSON on real "
                       "function-call data (cross-tokenizer).")

    return f"""---
base_model: {base_model}
library_name: peft
license: {lic["id"]}
tags:
- lora
- peft
- raif
- function-calling
- structured-output
datasets:
- glaiveai/glaive-function-calling-v2
---

<p align="center">
  <img src="banner.jpg" alt="RAIF" width="640">
</p>

<h1 align="center">{repo.split('/')[-1]}</h1>

<p align="center">
  A LoRA adapter that makes <b>{base_model}</b> emit
  <a href="https://github.com/skrrt-sh/raif-standard">RAIF</a> instead of JSON for structured output.
</p>

RAIF — the Repairable AI Interchange Format — is a drop-in layer for the JSON a
model produces (structured outputs, JSON/strict mode, tool arguments alike). It
round-trips losslessly to JSON, repairs its own syntax errors, and costs ~10%
fewer tokens on real workloads (more on tables). This adapter brings those
properties to small, local, and self-hosted inference.

{gate_line}

## Results (parse = decodes; fidelity = byte-exact round-trip)

| group | parse | fidelity | n |
|---|---:|---:|---:|
{rows}
- **valid** = held-out split of in-training shapes; **holdout** = shapes withheld from training entirely.
{token_note}

## Training

| | |
|---|---|
| base | `{base_model}` |
| method | LoRA (PEFT) via unsloth |
| rank / alpha | {hp.get('rank')} / {hp.get('alpha')} |
| lora_dropout | {hp.get('lora_dropout')} |
| learning rate | {hp.get('learning_rate')} ({hp.get('lr_scheduler','constant')}) |
| seq length | {hp.get('max_seq')} |
| epochs / examples | {data.get('epochs')} / {data.get('examples_seen')} |
| final train / eval loss | {_fmt_loss(res.get('final_train_loss'))} / {_fmt_loss(res.get('final_eval_loss'))} |

Data: synthetic RAIF examples (with mechanism-carrier shapes) augmented with
real tool-call argument objects from `glaiveai/glaive-function-calling-v2`
(Apache-2.0), kept only where they round-trip losslessly. Full recipe:
[`RECIPE.md`](https://github.com/skrrt-sh/raif-lora/blob/main/RECIPE.md).

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("{base_model}")
tok = AutoTokenizer.from_pretrained("{repo}")
model = PeftModel.from_pretrained(base, "{repo}")
```

This model emits **RAIF, not JSON** — decode it at the output boundary with the
official codec (pure-stdlib, no `bun`, nothing to clone):

```sh
pip install raif-format        # or: uv add raif-format
```

```python
from raif import decode        # installs as `raif-format`, imports as `raif`

result = decode(model_output)  # {{"ok", "value", "repairs"}}
data = result["value"] if result["ok"] else None   # ordinary JSON, ready downstream
```

`decode_lenient()` recovers the intact leaves of a truncated stream. The codec is
the same one used to score this model, kept byte-identical across Python and
TypeScript by a shared conformance corpus.

## Links

- **Format spec & reference codec:** <https://github.com/skrrt-sh/raif-standard>
- **Decoder:** [`raif-format`](https://pypi.org/project/raif-format/) on PyPI · [`raif-format`](https://www.npmjs.com/package/raif-format) on npm
- **Training recipe, eval & the other RAIF models:** <https://github.com/skrrt-sh/raif-lora>

## License & attribution

{lic["attrib"]}
Trained in part on `glaiveai/glaive-function-calling-v2` (Apache-2.0) — attribute Glaive AI.
"""


def main() -> int:
    """Parse arguments, write the model card, and upload the adapter to the HF Hub."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", type=Path, required=True, help="adapter dir to upload")
    p.add_argument("--repo", required=True, help="target HF repo id, e.g. user/raif-llama-3.2-3b-lora")
    p.add_argument("--base-model", default="unsloth/Llama-3.2-3B-Instruct")
    p.add_argument("--private", action="store_true", help="create the repo private")
    p.add_argument("--no-card", action="store_true", help="don't write/overwrite README.md")
    args = p.parse_args()

    if not (args.adapter / "adapter_config.json").exists():
        raise SystemExit(f"✗ {args.adapter} doesn't look like a PEFT adapter dir "
                         f"(no adapter_config.json)")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    from huggingface_hub import HfApi
    api = HfApi(token=token)

    if not args.no_card:
        card = build_model_card(args.adapter, args.repo, args.base_model)
        (args.adapter / "README.md").write_text(card)
        print("wrote README.md model card")

    api.create_repo(args.repo, private=args.private, exist_ok=True, repo_type="model")
    print(f"uploading {args.adapter} → https://huggingface.co/{args.repo} ...")
    api.upload_folder(folder_path=str(args.adapter), repo_id=args.repo, repo_type="model")
    print(f"✓ done: https://huggingface.co/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
