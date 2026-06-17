"""Stack-agnostic scoring core for the RAIF LoRA eval.

Shared by the MLX path (`src/eval_smoke.py`) and the CUDA/unsloth path
(`cuda/eval_cuda.py`) so both report the *same* parse/fidelity meter and
their numbers stay comparable. This module imports neither mlx nor torch —
only stdlib + the bun/TS canonical decoder subprocess. The model, tokenizer,
and a `generate(model, tok, prompt, max_tokens, verbose)` callable are
injected into `eval_group`, which is what keeps it framework-free.

The eval meter itself is pinned by `src/test_eval_smoke.py`.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

from raif_bun import run_bridge

# Generation budget per example. 384 was sized for the original shapes, but the
# `tabular_report` carrier emits homogeneous tables up to ~18 rows × 7 cols (~450+
# tokens). At 384 the longest tables truncate mid-row → spurious parse failures
# (the model's output is correct, just cut off). 1024 covers the largest example
# while staying well under the 2048 eval seq length.
MAX_TOKENS = 1024

# Default eval data + sampling, shared by both stacks' CLIs.
VALID_FILE = Path("./data/valid.jsonl")
HOLDOUT_FILE = Path("./data/eval_holdout.jsonl")
N_SAMPLES = 10

# Qwen3 bases emit a leading `<think>…</think>` block before the answer (an empty
# one for the non-thinking Instruct-2507 base we train on). RAIF lives after it, so
# the decode boundary strips the block first. Shared by both stacks so MLX and CUDA
# score identically; a no-op for outputs without it (Llama, Qwen2.5).
_THINK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)


def strip_think_prefix(text: str) -> str:
    """Strip a leading `<think>…</think>` block from a model's raw output."""
    return _THINK_RE.sub("", text, count=1)


# Reads a JSON array of RAIF strings from the file named by RAIF_DECODE_INPUT
# and writes a JSON array of {ok, value?, repairs?, error?} — one bun process
# per batch.
BATCH_DECODE_SRC = """
import { decode } from "./src/raif.ts";
const path = process.env.RAIF_BRIDGE_INPUT;
const items = JSON.parse(await Bun.file(path).text());
const out = items.map((raif) => {
  try {
    const r = decode(raif);
    if (r && r.ok) return { ok: true, value: r.value, repairs: (r.repairs ?? []).length };
    const errs = (r && (r.errors ?? r.error)) || "(no error msg)";
    return { ok: false, error: "decode rejected: " + JSON.stringify(errs) };
  } catch (e) {
    return { ok: false, error: String(e) };
  }
});
process.stdout.write(JSON.stringify(out));
"""


def load_examples(path: Path) -> list[dict]:
    """Load a JSONL eval file into a list of example dicts ([] if absent)."""
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def sample_examples(examples: list[dict], n: int, seed: int) -> list[dict]:
    """Shape-stratified random sample: shuffle each shape bucket with the
    given seed, then take round-robin across shapes until n examples, so the
    sample covers the shape vocabulary instead of stacking on one shape."""
    rng = random.Random(seed)
    by_shape: dict[str, list[dict]] = {}
    for ex in examples:
        shape = ex.get("meta", {}).get("shape", "?")
        by_shape.setdefault(shape, []).append(ex)
    for bucket in by_shape.values():
        rng.shuffle(bucket)
    shapes = sorted(by_shape.keys())
    target = min(n, len(examples))
    out: list[dict] = []
    i = 0
    while len(out) < target:
        shape = shapes[i % len(shapes)]
        bucket = by_shape[shape]
        idx = i // len(shapes)
        if idx < len(bucket):
            out.append(bucket[idx])
        i += 1
    return out


def batch_decode(raifs: list[str]) -> list[dict]:
    """Decode many RAIF strings in ONE bun invocation.

    Returns a list (same order) of {ok: bool, value?, repairs?, error?}.
    """
    if not raifs:
        return []
    return run_bridge(BATCH_DECODE_SRC, raifs, timeout=max(60, 5 * len(raifs)))


def canon(obj: object) -> str:
    """Canonical-sort JSON for byte equality."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


# ── Stage gates (mirror ../ITERATION_PLAN.md "Stage gates") ──────────────────
# Numeric, eval-checkable thresholds expressed as fractions (0..1). `None` means
# "no constraint from this metric at this stage". Non-numeric gate clauses
# (token ratio, "multi-line shapes start parsing", "no held-out regression") are
# checked outside the eval and noted by the caller.
STAGE_GATES: dict[str, dict] = {
    "smoke": {"valid_fidelity": 0.50},
    "warm": {"valid_fidelity": 0.75, "holdout_fidelity": 0.2301},  # "> smoke's 23%"
    "mid": {"valid_parse": 0.95},
    "full": {
        "valid_parse": 0.98,
        "valid_fidelity": 0.95,
        "holdout_parse": 0.98,
        "holdout_fidelity": 0.95,
    },
}


def _frac(stats: dict | None, key: str) -> float | None:
    """A group's `key` count as a fraction of n, or None when nothing was scored."""
    if not stats or stats.get("n", 0) == 0:
        return None
    return stats[key] / stats["n"]


def evaluate_gate(stage: str, valid: dict | None, holdout: dict | None) -> dict:
    """Check this stage's numeric gate against scored groups. Returns
    {stage, passed, checks:[{metric, threshold, actual, ok}], note}."""
    gate = STAGE_GATES.get(stage)
    if gate is None:
        return {
            "stage": stage,
            "passed": None,
            "checks": [],
            "note": f"no gate defined for stage {stage!r}",
        }
    actual = {
        "valid_parse": _frac(valid, "parse"),
        "valid_fidelity": _frac(valid, "fidelity"),
        "holdout_parse": _frac(holdout, "parse"),
        "holdout_fidelity": _frac(holdout, "fidelity"),
    }
    checks = []
    passed = True
    for metric, threshold in gate.items():
        a = actual.get(metric)
        ok = a is not None and a >= threshold
        passed = passed and ok
        checks.append(
            {
                "metric": metric,
                "threshold": round(threshold, 4),
                "actual": None if a is None else round(a, 4),
                "ok": ok,
            }
        )
    note = (
        (
            "full-stage acceptance also requires ≤0.92× JSON tokens (bun bench) "
            "and no held-out regression — not checked here."
        )
        if stage == "full"
        else ""
    )
    return {"stage": stage, "passed": passed, "checks": checks, "note": note}


def print_gate(gate: dict) -> None:
    """Pretty-print an evaluate_gate result (per-metric ✓/✗ and PASS/FAIL)."""
    if gate.get("passed") is None:
        print(f"gate: {gate.get('note', '(none)')}")
        return
    print(f"── gate [{gate['stage']}] ──")
    for c in gate["checks"]:
        mark = "✓" if c["ok"] else "✗"
        act = "n/a" if c["actual"] is None else f"{100 * c['actual']:.0f}%"
        print(
            f"  {mark} {c['metric']:18s} ≥ {100 * c['threshold']:.0f}%   actual {act}"
        )
    print(
        f"  → {'PASS' if gate['passed'] else 'FAIL'}"
        + (f"   ({gate['note']})" if gate.get("note") else "")
    )


def write_results_json(path, payload: dict) -> None:
    """Write the eval payload as pretty JSON (creates parent dirs)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nWrote eval results to {p}")


def eval_group(
    name: str, examples: list[dict], model, tok, generate, max_tokens: int = MAX_TOKENS
) -> dict | None:
    """Score one group. Returns {parse, fidelity, repaired, n, skipped} or None
    if empty. `generate` is called as generate(model, tok, prompt=...,
    max_tokens=..., verbose=False) and must return the model's text output."""
    if not examples:
        print(f"[{name}] no examples — skipping group\n")
        return None

    print(f"── {name}: scoring {len(examples)} examples ──")

    # Recover expected JSON by routing the expected RAIF through bun (one batch).
    expected_results = batch_decode([ex["messages"][1]["content"] for ex in examples])
    kept: list[tuple[dict, object]] = []
    skipped = 0
    for ex, res in zip(examples, expected_results):
        shape = ex.get("meta", {}).get("shape", "?")
        if res.get("ok"):
            kept.append((ex, res.get("value")))
        else:
            skipped += 1
            print(
                f"    SKIP {shape}: expected RAIF failed to decode "
                f"({res.get('error', '?')[:120]})"
            )

    outputs: list[str] = []
    for ex, _ in kept:
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": ex["messages"][0]["content"]}],
            add_generation_prompt=True,
            tokenize=False,
        )
        out = generate(model, tok, prompt=prompt, max_tokens=max_tokens, verbose=False)
        outputs.append(strip_think_prefix(out).strip())

    decoded = batch_decode(outputs)

    parse_ok = 0
    fidelity_ok = 0
    repaired = 0
    rows: list[dict] = []  # per-example detail, for JSON export
    for i, ((ex, expected_json), out, res) in enumerate(zip(kept, outputs, decoded)):
        shape = ex.get("meta", {}).get("shape", "?")
        task = ex.get("meta", {}).get("task", "?")
        ok_p = bool(res.get("ok"))
        parse_mark = "✓" if ok_p else "✗"
        fid_mark = "—"
        fid_pass = False
        was_repaired = bool(ok_p and res.get("repairs", 0))
        if ok_p:
            parse_ok += 1
            if was_repaired:
                repaired += 1
            if canon(res.get("value")) == canon(expected_json):
                fidelity_ok += 1
                fid_pass = True
                fid_mark = "✓"
            else:
                fid_mark = "✗"
        err = res.get("error", "") or ""
        print(f"[{i}] {shape:30s} ({task:9s}) parse {parse_mark} fidelity {fid_mark}")
        if not ok_p:
            print(f"      error: {err.splitlines()[0] if err else '(empty)'}")
            print(f"      model output (first 200 chars): {out[:200]!r}")
        rows.append(
            {
                "shape": shape,
                "task": task,
                "parse": ok_p,
                "fidelity": fid_pass,
                "repaired": was_repaired,
                "error": (err.splitlines()[0] if err else None) if not ok_p else None,
                # Keep the output snippet only for failures, to keep the JSON small.
                "output": out[:400] if not (ok_p and fid_pass) else None,
            }
        )

    n = len(kept)  # skipped examples are excluded from the denominator
    if n == 0:
        print(f"[{name}] all {skipped} examples skipped — nothing to score\n")
        return {
            "parse": 0,
            "fidelity": 0,
            "repaired": 0,
            "n": 0,
            "skipped": skipped,
            "rows": rows,
        }
    print(
        f"\n[{name}] parse:    {parse_ok}/{n} ({100 * parse_ok / n:.0f}%)"
        f" — {repaired} via repair\n"
        f"[{name}] fidelity: {fidelity_ok}/{n} ({100 * fidelity_ok / n:.0f}%)\n"
        f"[{name}] skipped:  {skipped} (excluded from denominator)\n"
    )
    return {
        "parse": parse_ok,
        "fidelity": fidelity_ok,
        "repaired": repaired,
        "n": n,
        "skipped": skipped,
        "rows": rows,
    }


# ── Shared CLI driver (used by both src/eval_smoke.py and cuda/eval_cuda.py) ──


def add_common_eval_args(p) -> None:
    """Register the eval flags shared by both stacks (sampling, data files, output,
    gate) on an argparse parser. Stack-specific flags (adapter, checkpoint, max-seq)
    are added by the caller."""
    p.add_argument(
        "--n",
        type=int,
        default=N_SAMPLES,
        help=f"examples to sample per group (default {N_SAMPLES})",
    )
    p.add_argument(
        "--seed", type=int, default=0, help="RNG seed for example sampling (default 0)"
    )
    p.add_argument(
        "--valid",
        type=Path,
        default=VALID_FILE,
        help=f"in-training-shape eval file (default {VALID_FILE})",
    )
    p.add_argument(
        "--holdout",
        type=Path,
        default=HOLDOUT_FILE,
        help=f"held-out-shape eval file (default {HOLDOUT_FILE})",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="write full results JSON here (per-example rows + summary + gate)",
    )
    p.add_argument(
        "--gate",
        default=None,
        choices=["smoke", "warm", "mid", "full"],
        help="check this stage's ITERATION_PLAN gate and print PASS/FAIL; "
        "exit nonzero on FAIL",
    )


def run_eval(
    args, model, tok, generate, stack: str, extra_payload: dict | None = None
) -> int:
    """Score the valid + holdout groups with `generate`, print the summary and the
    optional stage gate, optionally write the results JSON, and return the process
    exit code (1 only when a requested gate fails). `stack` labels the payload;
    `extra_payload` carries stack-specific fields (adapter, checkpoint)."""
    groups = [
        ("valid (in-training shapes)", args.valid),
        ("holdout (withheld shapes)", args.holdout),
    ]
    results = []
    for name, path in groups:
        examples = sample_examples(load_examples(path), args.n, args.seed)
        results.append((name, eval_group(name, examples, model, tok, generate)))

    print("── summary ──")
    for name, stats in results:
        if stats is None or stats["n"] == 0:
            print(f"{name:30s} (no scored examples)")
            continue
        print(
            f"{name:30s} parse {stats['parse']}/{stats['n']} "
            f"({100 * stats['parse'] / stats['n']:.0f}%)  "
            f"fidelity {stats['fidelity']}/{stats['n']} "
            f"({100 * stats['fidelity'] / stats['n']:.0f}%)  "
            f"skipped {stats['skipped']}"
        )

    gate = None
    if args.gate:
        gate = evaluate_gate(args.gate, results[0][1], results[1][1])
        print()
        print_gate(gate)

    if args.out:
        write_results_json(
            args.out,
            {
                "stack": stack,
                **(extra_payload or {}),
                "n_per_group": args.n,
                "seed": args.seed,
                "groups": {name: stats for name, stats in results},
                "gate": gate,
            },
        )

    return 1 if (gate and gate.get("passed") is False) else 0
