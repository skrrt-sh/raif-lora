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
    for i, ((ex, expected_json), out, res) in enumerate(zip(kept, outputs, decoded)):
        shape = ex.get("meta", {}).get("shape", "?")
        task = ex.get("meta", {}).get("task", "?")
        ok_p = bool(res.get("ok"))
        parse_mark = "✓" if ok_p else "✗"
        fid_mark = "—"
        if ok_p:
            parse_ok += 1
            if res.get("repairs", 0):
                repaired += 1
            if canon(res.get("value")) == canon(expected_json):
                fidelity_ok += 1
                fid_mark = "✓"
            else:
                fid_mark = "✗"
        print(f"[{i}] {shape:30s} ({task:9s}) parse {parse_mark} fidelity {fid_mark}")
        if not ok_p:
            err = res.get("error", "")
            print(f"      error: {err.splitlines()[0] if err else '(empty)'}")
            print(f"      model output (first 200 chars): {out[:200]!r}")

    n = len(kept)  # skipped examples are excluded from the denominator
    if n == 0:
        print(f"[{name}] all {skipped} examples skipped — nothing to score\n")
        return {"parse": 0, "fidelity": 0, "repaired": 0, "n": 0, "skipped": skipped}
    print(
        f"\n[{name}] parse:    {parse_ok}/{n} ({100*parse_ok/n:.0f}%)"
        f" — {repaired} via repair\n"
        f"[{name}] fidelity: {fidelity_ok}/{n} ({100*fidelity_ok/n:.0f}%)\n"
        f"[{name}] skipped:  {skipped} (excluded from denominator)\n"
    )
    return {"parse": parse_ok, "fidelity": fidelity_ok, "repaired": repaired,
            "n": n, "skipped": skipped}
