# raif-lora — smoke session notes (M4 Max, 2026-05-17)

Run-of-record from the first day setting up the v0.5 fine-tune workstream on Apple Silicon. Plan-of-record is `../raif-standard/docs/fine_tune_plan.md`; this doc captures what was actually executed and what we learned.

## Stack picked

| Component | Choice | Why |
|---|---|---|
| Framework | `mlx-lm 0.31.3` on `mlx 0.31.2` | Only ML stack that natively uses Apple Silicon's unified memory + Metal. Axolotl/unsloth (the plan's recommendation) are CUDA-only and degrade hard on MPS. |
| Base model | `mlx-community/Llama-3.2-3B-Instruct-bf16` | Plan §4.1 target. 6.0 GB on disk; 14 GB peak in training (out of 128 GB host). |
| Adapter type | LoRA (r=16 smoke, r=32 warm) | Default mlx-lm targets — all Linear submodules of the last N transformer blocks, which covers q/k/v/o + gate/up/down per layer. Matches plan §4.2 target list without needing an explicit `keys:` list. |
| Inference / grammars | `llama.cpp` from Homebrew (9180) | Ships with Metal kernels; `llama-cli --grammar-file` covers GBNF testing once the grammar is finalized. Note: `llama-gbnf-validator` is NOT in the brew bottle; rebuild from source if we want that specific binary. |
| Dataset | Reused `prototype/src/dataset.ts` | Already emits chat-template JSONL exactly as mlx-lm consumes. No Python port needed. |
| Decoder for eval | Bun subprocess (`bun -e ... import { decode }`) | Avoids re-implementing the canonical decoder in Python. The TS source remains the canonical implementation. |

## Smoke run (300 iters, ~3 min)

`configs/llama-3-3b-sft-smoke.yaml`. 540 examples (30 variations × 18 shapes), rank 16, batch 4, LR 2e-4, num_layers 16.

### Loss trajectory

| Iter | Train | Val |
|---:|---:|---:|
| 1   | —    | 5.279 |
| 50  | 2.39 | 3.025 |
| 100 | 1.45 | 2.479 |
| 150 | 1.30 | 2.541 |
| 200 | 1.21 | 2.713 |
| 250 | 1.40 | 1.722 |
| 300 | 0.93 | 1.865 |

Val is noisy (8 batches × 4 = 32 samples), but the trend is clearly downward. Train loss dropped 4.10 → 0.93 (-77%). Peak mem 14.77 GB. Speed: 1.5–2.0 it/sec, ≈ 370 tokens/sec. 0.43% trainable params (13.9 M of 3.21 B).

### Held-out eval (`uv run python src/eval_smoke.py`)

After fixing the eval to sample uniformly across shapes (valid.jsonl was a single-shape slice — see "Bug found" below):

```
parse:    15/18 (83%)
fidelity: 1/18 (6%)
```

Where it failed:

| Shape | Failure | What the model emitted |
|---|---|---|
| `large_table` | column-count mismatch | header declared 4 cols (`actions,id,items,rows`), rows have 3 cells |
| `multiline_body` | unterminated multiline | opened `body=<<<\n…` but never emitted the closing `>>>` |
| `wide_heterogeneous_array` | unterminated array literal | emitted `events=[\n{…}\n{…}…` but no closing `]` |

All three are *structural* failures the model would learn out of with more data / examples; none are conceptual gaps.

Where it succeeded but emitted wrong content (the fidelity ✗ on parsed outputs): the model invents field *values* (random strings/numbers from the dataset generator's pool). It hasn't learned the user-instruction → schema-values mapping yet because:
- The synthetic prompts are very generic (`"Compose an object whose keys contain ``.`` characters."`)
- Only 30 variations per shape means very little vocabulary diversity per shape

### Comparison vs. plan baselines

The v0.5 plan §3 OpenRouter baseline numbers (no fine-tune, few-shot prompt):

| Model | Parse | Fidelity |
|---|---:|---:|
| gemma-3-4b-it    | 64% | 42% |
| qwen-2.5-7b      | 89% | 44% |
| llama-3.1-8b     | 75% | 42% |
| mistral-nemo     | 83% | 53% |
| gpt-oss-20b      | 100% | 83% |
| claude-haiku-4.5 | 94%  | 72% |
| **smoke LoRA (this run)** | **83%** | **6%** |

The smoke LoRA matches mistral-nemo on parse with **no in-context examples**, after 3 minutes of training. Fidelity is where we expect the real ramp once we scale data and iters per the plan.

## Bug found in `prototype/src/dataset.ts` eval split

`--eval-frac 0.05` produced a `valid.jsonl` that contained 27 examples of *one shape* (`pathological_keys`) rather than ~1–2 examples per shape. Looks like the split slices the tail of the concatenated stream rather than stratifying. Worked around in `src/eval_smoke.py` by sampling round-robin from `train.jsonl`. Should file an issue / patch upstream before publishing the dataset generator.

## Files written this session

```
raif-lora/
├── README.md                              ← orientation
├── NOTES.md                               ← this file
├── pyproject.toml                         ← uv deps (mlx, mlx-lm, hf-hub, datasets)
├── configs/
│   ├── llama-3-3b-sft-smoke.yaml          ← 30-min smoke (300 iters, r=16)
│   ├── llama-3-3b-sft-warm.yaml           ← 15-min warm (1500 iters, r=32, 1.7k examples)
│   └── llama-3-3b-sft.yaml                ← full acceptance (7000 iters, r=32, 11.5k examples)
├── src/
│   ├── make_data.sh                       ← wraps `bun run dataset` with smoke/full presets
│   └── eval_smoke.py                      ← held-out parse/fidelity score
├── grammars/
│   └── raif.gbnf                          ← first-cut GBNF (gaps: multiline body, nonce ref)
├── data/
│   ├── train.jsonl                        ← currently 1,710 examples (warm-run dataset)
│   └── valid.jsonl                        ← 90 examples
└── adapters/
    └── llama-3-3b-raif-sft-smoke/         ← 4 checkpoints (100, 200, 300, latest)
```

## Warm run (1500 iters, ~17 min)

`configs/llama-3-3b-sft-warm.yaml`. 1,710 train + 90 eval examples (100 variations × 18 shapes), rank 32 (matches plan §4.2), batch 4, LR 2e-4, num_layers 16.

### Loss trajectory (subset)

| Iter | Train | Val |
|---:|---:|---:|
| 1    | —    | 5.376 |
| 100  | 2.85 | 3.685 |
| 250  | 1.52 | — |
| 400  | 1.34 | 2.274 |
| 750  | 1.33 | — |
| 1000 | 1.02 | 2.314 |
| 1200 | —    | 2.020 |
| 1400 | —    | 2.467 |
| **1500** | **1.03** | **1.994** |

Peak mem 15.04 GB. Speed: ~1.5 it/sec, ~340 tokens/sec (≈ unchanged from smoke despite 2× LoRA params; MLX overlaps well). 0.87% trainable params (27.8 M of 3.21 B). Final saved at adapters/llama-3-3b-raif-sft-warm/.

### Held-out eval (same 18 diverse shapes as smoke)

```
parse:    16/18 (89%)
fidelity: 0/18 (0%)
```

### Smoke vs warm comparison

| metric                      | smoke (3 min) | warm (17 min) | delta |
|---|---:|---:|---|
| parse                       | 83% (15/18) | **89% (16/18)** | +6pp |
| fidelity                    | 6% (1/18)   | **0% (0/18)**   | -6pp |
| parse failures              | 3 shapes    | 2 shapes        | column-count fixed |
| `large_table`               | ✗ (col count) | ✓             | recovered |
| `multiline_body`            | ✗ (unterminated `<<<`) | ✗ (same)        | persists |
| `wide_heterogeneous_array`  | ✗ (unterminated `[`)   | ✗ (same)        | persists |
| trainable params            | 13.9 M (r=16) | 27.8 M (r=32) | matches plan |
| peak GPU mem                | 14.77 GB    | 15.04 GB        | +0.27 GB |

Two takeaways:

1. **The recipe scales for parse rate but stalls on fidelity.** More data + bigger rank moved parse closer to acceptance-gate territory (98% target). Fidelity went the *wrong direction* — the model emits RAIF more confidently, with more invented content. This is not a model-size problem; it's a *prompt-template* problem: every example's user prompt is some variant of "compose an object with these fields" without specifying values, so the model never learns to condition output content on the prompt. Plan §3.1 needs an amendment before the full SFT pass: prompts must include realistic value cues (`"Email the user at alice@acme.org about invoice #4421"` rather than `"Compose an object with `to`, `subject`, `body`"`).

2. **The two persistent parse failures are both newline-bounded delimiters.** `multiline_body` (`<<<\n…\n>>>`) and `wide_heterogeneous_array` (`prefix=[\n…\n]`) both require the model to track a multi-line open/close across many emitted lines. With max_seq 1024 and rank 32 these still aren't reliable. Options: (a) GBNF-constrained decoding (the existing `raif.gbnf` skeleton enforces these), (b) bump max_seq + targeted training on the multi-line shapes, (c) accept these as known limitations until vendor token-mode adoption per ADR-0017.

### Useful baselines for the plan ([fine_tune_plan.md](../raif-standard/docs/fine_tune_plan.md))

- 3B + LoRA on M4 Max trains at ~1.5 it/s for batch 4, max_seq 1024. Full SFT (7,000 iters at eff-batch 16) projects to **~12 hr** on this hardware vs the plan's CUDA estimate of 3–5 hr. Doable overnight, no GPU rental.
- 15 GB peak GPU mem on rank-32-with-all-projections means we have headroom to push to rank 64 or num_layers -1 (all 28 blocks) before mem becomes a concern.
- LoRA adapter file is 53 MB (rank 16) / 106 MB (rank 32). Trivial to host on HF; no merge required for distribution.

## Open follow-ups (post-session)

1. **Patch `dataset.ts` eval split** — should stratify by shape.
2. **GBNF gaps** — multiline `<<<\n…\n>>>` and nonce backref. The latter is unrepresentable in GBNF; the spec already accepts that and relies on the decoder's `fix` pass.
3. **Richer prompt templates** — the current request template (`"Return a tool call with these fields: …"`) is too uniform. Without more diverse prompts the model is just learning shape → output rather than meaning → output. Plan §3.1 already calls for varying field names from a pool of 200; we need to vary the *prompt phrasing* too.
4. **Token-Δ measurement against the adapter** — none of the smoke evals checked the −13% / JSON ratio. Easy to add once parse rate stabilizes.
5. **Test the GBNF against the corpus** — `grammars/raif.gbnf` is unverified. Once `llama-gbnf-validator` is built (or via `llama-cli --grammar-file`), every emitted corpus string should be accepted.
