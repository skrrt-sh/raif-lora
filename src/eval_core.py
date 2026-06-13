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
import os
import random
import subprocess
import tempfile
from pathlib import Path

# Resolve the prototype dir from THIS file's location so the bun decoder works
# regardless of the caller's cwd:  raif-lora/src/eval_core.py -> scratch/ ...
PROTOTYPE_DIR = (
    Path(__file__).resolve().parent.parent.parent / "raif-standard" / "prototype"
)

MAX_TOKENS = 384

# Reads a JSON array of RAIF strings from the file named by RAIF_DECODE_INPUT
# and writes a JSON array of {ok, value?, repairs?, error?} — one bun process
# per batch.
BATCH_DECODE_SRC = """
import { decode } from "./src/raif.ts";
const path = process.env.RAIF_DECODE_INPUT;
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
    fd, tmp_path = tempfile.mkstemp(suffix=".json", prefix="raif_decode_")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(raifs, f)
        res = subprocess.run(
            ["bun", "-e", BATCH_DECODE_SRC],
            capture_output=True,
            cwd=PROTOTYPE_DIR,
            timeout=max(60, 5 * len(raifs)),
            env={**os.environ, "RAIF_DECODE_INPUT": tmp_path},
        )
    finally:
        os.unlink(tmp_path)
    if res.returncode != 0:
        raise RuntimeError(
            f"bun batch decode failed: {res.stderr.decode('utf-8', 'replace')[:500]}"
        )
    return json.loads(res.stdout.decode("utf-8"))


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
    "warm":  {"valid_fidelity": 0.75, "holdout_fidelity": 0.2301},  # "> smoke's 23%"
    "mid":   {"valid_parse": 0.95},
    "full":  {"valid_parse": 0.98, "valid_fidelity": 0.95,
              "holdout_parse": 0.98, "holdout_fidelity": 0.95},
}


def _frac(stats: dict | None, key: str) -> float | None:
    if not stats or stats.get("n", 0) == 0:
        return None
    return stats[key] / stats["n"]


def evaluate_gate(stage: str, valid: dict | None, holdout: dict | None) -> dict:
    """Check this stage's numeric gate against scored groups. Returns
    {stage, passed, checks:[{metric, threshold, actual, ok}], note}."""
    gate = STAGE_GATES.get(stage)
    if gate is None:
        return {"stage": stage, "passed": None, "checks": [],
                "note": f"no gate defined for stage {stage!r}"}
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
        checks.append({"metric": metric, "threshold": round(threshold, 4),
                       "actual": None if a is None else round(a, 4), "ok": ok})
    note = ("full-stage acceptance also requires ≤0.92× JSON tokens (bun bench) "
            "and no held-out regression — not checked here.") if stage == "full" else ""
    return {"stage": stage, "passed": passed, "checks": checks, "note": note}


def print_gate(gate: dict) -> None:
    if gate.get("passed") is None:
        print(f"gate: {gate.get('note', '(none)')}")
        return
    print(f"── gate [{gate['stage']}] ──")
    for c in gate["checks"]:
        mark = "✓" if c["ok"] else "✗"
        act = "n/a" if c["actual"] is None else f"{100*c['actual']:.0f}%"
        print(f"  {mark} {c['metric']:18s} ≥ {100*c['threshold']:.0f}%   actual {act}")
    print(f"  → {'PASS' if gate['passed'] else 'FAIL'}"
          + (f"   ({gate['note']})" if gate.get("note") else ""))


def write_results_json(path, payload: dict) -> None:
    """Write the eval payload as pretty JSON (created parent dirs)."""
    import json as _json
    from pathlib import Path as _Path
    p = _Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(_json.dumps(payload, indent=2, ensure_ascii=False))
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
            print(f"    SKIP {shape}: expected RAIF failed to decode "
                  f"({res.get('error', '?')[:120]})")

    outputs: list[str] = []
    for ex, _ in kept:
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": ex["messages"][0]["content"]}],
            add_generation_prompt=True,
            tokenize=False,
        )
        out = generate(model, tok, prompt=prompt, max_tokens=max_tokens, verbose=False)
        outputs.append(out.strip())

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
        rows.append({
            "shape": shape, "task": task, "parse": ok_p, "fidelity": fid_pass,
            "repaired": was_repaired,
            "error": (err.splitlines()[0] if err else None) if not ok_p else None,
            # Keep the output snippet only for failures, to keep the JSON small.
            "output": out[:400] if not (ok_p and fid_pass) else None,
        })

    n = len(kept)  # skipped examples are excluded from the denominator
    if n == 0:
        print(f"[{name}] all {skipped} examples skipped — nothing to score\n")
        return {"parse": 0, "fidelity": 0, "repaired": 0, "n": 0,
                "skipped": skipped, "rows": rows}
    print(
        f"\n[{name}] parse:    {parse_ok}/{n} ({100*parse_ok/n:.0f}%)"
        f" — {repaired} via repair\n"
        f"[{name}] fidelity: {fidelity_ok}/{n} ({100*fidelity_ok/n:.0f}%)\n"
        f"[{name}] skipped:  {skipped} (excluded from denominator)\n"
    )
    return {"parse": parse_ok, "fidelity": fidelity_ok, "repaired": repaired,
            "n": n, "skipped": skipped, "rows": rows}
