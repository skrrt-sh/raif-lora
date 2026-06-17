#!/usr/bin/env python3
"""Live terminal demo: structured output from a 0.5B — schema in, data out.

A slide-based animation (rendered to assets/demo-0.5b.{mp4,gif} via
assets/demo-0.5b.tape) that mirrors the raif-standard "bun demo" look.

This is what RAIF is *for*. The agent is handed an **output schema** and some
**freeform details**, and its whole job is to fill the schema with those
details — the everyday structured-output / tool-call task. That's the prompt
both models get; it's exactly the `instruct` shape the LoRA was trained on
(here are the fields, here are the values, emit the record).

Nothing here is canned. Every output is generated **live** at record time by
loading Qwen2.5-0.5B (MLX) and streaming its real tokens, greedy (temp=0) so it
reproduces. Verdicts come from actually running `json.loads()` / `raif.decode()`
and comparing to the exact intended object. The comparison is FAIR: both models
get the **identical prompt** (the details + the `<schema>` block). The bare base
answers in prose/JSON and bloats, mistypes, or runs off the budget; the RAIF
LoRA answers in RAIF (its native format), which decode()s back to the exact
record:

  flat record    base is right — but 4× the tokens, wrapped in prose and a code
                 fence you'd have to parse back out; RAIF emits five clean lines
                 that decode directly.
  typed record   base silently reshapes the schema — a number becomes a string,
                 a top-level field gets misnested; RAIF decodes every field with
                 the right type and shape.
  multi-row      base bloats past the token budget and truncates mid-row (and it
                 never was JSON); RAIF decodes the whole table in far fewer tokens.

Each is shown twice — same prompt to the bare base (it struggles) then to the
RAIF LoRA (it holds). Run live with: uv run python examples/demo.py  (--fast).
"""

from __future__ import annotations

import json
import re
import sys
import time

import raif_models as M


# ── Catppuccin Mocha palette (truecolor) ────────────────────────────────────
def fg(r: int, g: int, b: int) -> str:
    return f"\033[38;2;{r};{g};{b}m"


def bg(r: int, g: int, b: int) -> str:
    return f"\033[48;2;{r};{g};{b}m"


RESET = "\033[0m"
BOLD = "\033[1m"

TEXT = fg(205, 214, 244)
GRAY = fg(108, 112, 134)
BLUE = fg(137, 180, 250)
GREEN = fg(166, 227, 161)
RED = fg(243, 139, 168)
YELLOW = fg(249, 226, 175)
TEAL = fg(148, 226, 213)
MAUVE = fg(203, 166, 247)
BADGE = BOLD + fg(30, 30, 46) + bg(137, 180, 250)

_GRAD = [
    (137, 180, 250),
    (140, 170, 252),
    (160, 168, 250),
    (180, 170, 249),
    (203, 166, 247),
    (203, 166, 247),
]
_LOGO = [
    "█████╗   █████╗  ██╗ ███████╗",
    "██╔═██╗ ██╔══██╗ ██║ ██╔════╝",
    "█████╔╝ ███████║ ██║ █████╗  ",
    "██╔═██╗ ██╔══██║ ██║ ██╔══╝  ",
    "██║ ██║ ██║  ██║ ██║ ██║     ",
    "╚═╝ ╚═╝ ╚═╝  ╚═╝ ╚═╝ ╚═╝     ",
]

MARGIN = "  "
INDENT = "    "
FAST = "--fast" in sys.argv

# ONE prompt, sent to both models — a fair comparison. The task is structured
# output: fill the given schema with the given freeform details. The bare base
# rambles/mistypes; the RAIF LoRA answers in RAIF (its native format), which
# decode()s back to the exact record. Built from each case's details + schema.
PROMPT_TMPL = "{details}\n\n<schema>\n{schema}\n</schema>"

# Real `instruct`-shape tasks (data/train.jsonl): details in freeform, the
# desired structure in <schema>. Sized so the FULL prompt and the FULL model
# output both fit on screen — nothing is summarized or capped.
CASES = [
    {
        "key": "flat",
        "details": (
            'Capture this as a record. Set cuisine to "Italian". The '
            'value of location is "New York". price_range should be '
            '"$$". Set open_now to true. The value of rating is 4.6.'
        ),
        "schema": [
            "cuisine:s",
            "location:s",
            "price_range:s",
            "open_now:b",
            "rating:n",
        ],
        "expected": {
            "cuisine": "Italian",
            "location": "New York",
            "price_range": "$$",
            "open_now": True,
            "rating": 4.6,
        },
        "budget": 256,
        "bug": "bloat",
        "fail_title": "Bare 0.5B — fill a record",
        "fail_sub": "Here's the schema and the details. Fill it in.",
        "ok_title": "+ RAIF LoRA — same prompt",
        "ok_sub": "Identical prompt; five lines that decode() straight back.",
    },
    {
        "key": "typed",
        "details": (
            'Fill these fields. For subject.last_name use "Tanaka". '
            'Assign "agent" to subject.role. subject.id should be 3129. '
            'The value of session is "quiet-lantern-330".'
        ),
        "schema": [
            "subject.last_name:s",
            "subject.role:s",
            "subject.id:n",
            "session:s",
        ],
        "expected": {
            "subject": {"last_name": "Tanaka", "role": "agent", "id": 3129},
            "session": "quiet-lantern-330",
        },
        "budget": 256,
        "bug": "wrong",
        "fail_title": "Bare 0.5B — a typed, nested schema",
        "fail_sub": "Same task — but the types and nesting have to be exact.",
        "ok_title": "+ RAIF LoRA — same prompt",
        "ok_sub": "Identical prompt; every type and path, exact.",
    },
    {
        "key": "table",
        "details": (
            "Build the rows from these line items.\n"
            '- properties[0].code 11, completed 1, extra "noble-marble"\n'
            '- properties[1].code 7, completed 2, extra "silver-falcon"\n'
            '- properties[2].code 23, completed 0, extra "amber-canyon"\n'
            '- properties[3].code 4, completed 5, extra "ivory-harbor"\n'
            '- properties[4].code 18, completed 3, extra "keen-meadow"\n'
            '- properties[5].code 9, completed 1, extra "plum-thicket"\n'
            '- properties[6].code 31, completed 4, extra "slate-lagoon"\n'
            '- properties[7].code 2, completed 2, extra "coral-bramble"'
        ),
        "schema": [
            "properties[]:o",
            "properties[].code:n",
            "properties[].completed:n",
            "properties[].extra:s",
        ],
        "expected": {
            "properties": [
                {"code": 11, "completed": 1, "extra": "noble-marble"},
                {"code": 7, "completed": 2, "extra": "silver-falcon"},
                {"code": 23, "completed": 0, "extra": "amber-canyon"},
                {"code": 4, "completed": 5, "extra": "ivory-harbor"},
                {"code": 18, "completed": 3, "extra": "keen-meadow"},
                {"code": 9, "completed": 1, "extra": "plum-thicket"},
                {"code": 31, "completed": 4, "extra": "slate-lagoon"},
                {"code": 2, "completed": 2, "extra": "coral-bramble"},
            ]
        },
        "budget": 200,
        "bug": "truncated",
        "fail_title": "Bare 0.5B — an 8-row table",
        "fail_sub": "Same task at scale — eight rows to lay down.",
        "ok_title": "+ RAIF LoRA — same prompt",
        "ok_sub": "Identical prompt; the whole table, decodes byte-exact.",
    },
]


def _sleep(s: float) -> None:
    time.sleep(s * (0.3 if FAST else 1.0))


def clear() -> None:
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def emit(s: str = "") -> None:
    sys.stdout.write(s + "\n")
    sys.stdout.flush()


def header(num: str, title: str, subtitle: str) -> None:
    clear()
    emit()
    emit()
    emit(f"{MARGIN}{BADGE} {num} {RESET}   {BOLD}{TEXT}{title}{RESET}")
    emit()
    emit(f"{MARGIN}{GRAY}{subtitle}{RESET}")
    emit()
    _sleep(0.4)


def logo_screen(tagline: str, footer: list[str] | None = None) -> None:
    clear()
    emit()
    emit()
    for row, art in zip(_GRAD, _LOGO):
        emit(f"{MARGIN}{BOLD}{fg(*row)}{art}{RESET}")
    emit()
    emit(f"{MARGIN}{GRAY}{tagline}{RESET}")
    if footer:
        emit()
        for ln in footer:
            emit(f"{MARGIN}{ln}")
    emit()


def _wrap(s: str, width: int = 76) -> list[str]:
    """Wrap a long string at comma/space boundaries so a literal payload shows on
    a couple of tidy lines instead of soft-wrapping to the terminal edge."""
    if len(s) <= width:
        return [s]
    parts = s.split(",")
    parts = [p + ("," if i < len(parts) - 1 else "") for i, p in enumerate(parts)]
    lines, cur = [], ""
    for p in parts:
        if cur and len(cur) + len(p) > width:
            lines.append(cur)
            cur = p
        else:
            cur += p
    if cur:
        lines.append(cur)
    return lines


def _wrap_words(s: str, width: int = 84) -> list[str]:
    """Word-wrap prose at whitespace so a long detail sentence shows on tidy
    lines instead of soft-wrapping mid-word at the terminal edge."""
    lines, cur = [], ""
    for word in s.split(" "):
        if cur and len(cur) + 1 + len(word) > width:
            lines.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}" if cur else word
    if cur:
        lines.append(cur)
    return lines or [""]


def show_task(details: str, schema: list[str]) -> None:
    """Render the shared prompt: freeform details, then the <schema> block."""
    emit(f"{INDENT}{TEAL}you ▸ {RESET}{TEXT}fill this schema with these details{RESET}")
    for raw in details.split("\n"):
        for line in _wrap_words(raw):
            emit(f"{INDENT}      {GRAY}{line}{RESET}")
    emit(f"{INDENT}      {MAUVE}<schema>{RESET}")
    for line in schema:
        emit(f"{INDENT}      {MAUVE}{line}{RESET}")
    emit(f"{INDENT}      {MAUVE}</schema>{RESET}")


_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def try_parse(text: str):
    """Best-effort JSON recovery from a base reply (strip fence / first span)."""
    cands = []
    m = _FENCE.search(text)
    if m:
        cands.append(m.group(1))
    for o, c in (("{", "}"), ("[", "]")):
        i, j = text.find(o), text.rfind(c)
        if 0 <= i < j:
            cands.append(text[i : j + 1])
    cands.append(text.strip())
    for c in cands:
        try:
            return json.loads(c), True
        except (json.JSONDecodeError, ValueError):
            continue
    return None, False


def canon(v) -> str:
    return json.dumps(v, sort_keys=True, separators=(",", ":"))


def _flatten(v, prefix: str = "") -> dict:
    """Map a value to {dotted.path: leaf} so two objects can be diffed by leaf."""
    out: dict = {}
    if isinstance(v, dict):
        for k, sub in v.items():
            out.update(_flatten(sub, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(v, list):
        for i, sub in enumerate(v):
            out.update(_flatten(sub, f"{prefix}[{i}]"))
    else:
        out[prefix] = v
    return out


def _typename(v) -> str:
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, (int, float)):
        return "number"
    return "string"


def describe_diff(expected, got) -> str:
    """One honest line naming how a parsed-but-wrong reply diverged: a misnested
    field, a coerced type, or a changed value."""
    exp, val = _flatten(expected), _flatten(got)
    for path, want in exp.items():
        if path not in val:
            return f"{path} got dropped or misnested"
    for path, want in exp.items():
        if _typename(val[path]) != _typename(want):
            return f"{path}: {_typename(want)} became a {_typename(val[path])}"
    for path, want in exp.items():
        if val[path] != want:
            return f"{path} changed value"
    return "the shape doesn't match"


class Model:
    """Qwen2.5-0.5B (MLX). Two instances: bare base, and base + RAIF LoRA."""

    def __init__(self, lora: bool = False) -> None:
        from mlx_lm import load
        from mlx_lm.sample_utils import make_sampler

        self.spec = M.spec("qwen-0.5b")
        if not (self.spec["mlx_dir"] / "adapters.safetensors").exists():
            sys.exit(
                "Adapter not built. Run: "
                "uv run python examples/setup_adapter.py --model qwen-0.5b"
            )
        self.model, self.tok = load(M.base_path(self.spec))
        if lora:
            from mlx_lm.utils import load_adapters

            self.model = load_adapters(self.model, str(self.spec["mlx_dir"]))
        self.model.eval()
        self.greedy = make_sampler(temp=0.0)

    def stream(
        self,
        prompt: str,
        max_tokens: int,
        color: str = TEXT,
        cps: float = 120.0,
        cap_lines: int | None = None,
    ):
        """Generate live, printing tokens as they arrive (indented, paced). If
        cap_lines is set, stop *printing* after that many lines but keep
        generating. Returns (full_text, n_tokens, was_capped)."""
        from mlx_lm import stream_generate

        prm = self.tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        out, lines, capped, suppress = "", 0, False, False
        sys.stdout.write(INDENT + color)
        sys.stdout.flush()
        for resp in stream_generate(
            self.model, self.tok, prompt=prm, max_tokens=max_tokens, sampler=self.greedy
        ):
            for ch in resp.text:
                if not suppress:
                    if ch == "\n":
                        lines += 1
                        if cap_lines and lines >= cap_lines:
                            suppress = capped = True
                            continue
                        sys.stdout.write("\n" + INDENT)
                    else:
                        sys.stdout.write(ch)
                    sys.stdout.flush()
                    _sleep(1.0 / cps)
            out += resp.text
        sys.stdout.write(RESET + "\n")
        sys.stdout.flush()
        out = M.strip_think(self.spec, out).strip()
        return out, len(self.tok.encode(out)), capped


def loading_screen() -> None:
    clear()
    emit()
    emit()
    for row, art in zip(_GRAD, _LOGO):
        emit(f"{MARGIN}{BOLD}{fg(*row)}{art}{RESET}")
    emit()
    emit(
        f"{MARGIN}{GRAY}loading Qwen2.5-0.5B (MLX) — every answer below is "
        f"generated live, greedy…{RESET}"
    )


# ── Slides ───────────────────────────────────────────────────────────────────
def slide_fail(m: Model, num: str, case: dict) -> int:
    """Bare base on the schema-fill task — show it struggle. Returns base tokens."""
    expected = case["expected"]
    prompt = PROMPT_TMPL.format(
        details=case["details"], schema="\n".join(case["schema"])
    )
    header(num, case["fail_title"], case["fail_sub"])
    show_task(case["details"], case["schema"])
    _sleep(0.4)
    emit()
    emit(f"{INDENT}{GRAY}── base 0.5B (live, full output) ──{RESET}")
    _sleep(0.3)
    out, ntok, _ = m.stream(prompt, case["budget"], cps=170.0)
    _sleep(0.4)
    emit()
    emit(f"{INDENT}{BLUE}json.loads(output){RESET}")
    _sleep(0.4)
    val, ok = try_parse(out)
    truncated = ntok >= case["budget"] - 4
    if not ok:
        if truncated:
            emit(
                f"{INDENT}{RED}✗ no usable record — it bloated past the "
                f"{ntok}-token budget and was cut off mid-row{RESET}"
            )
        else:
            emit(
                f"{INDENT}{RED}✗ no usable record — it answered in prose, "
                f"not structured data{RESET}"
            )
    else:
        faithful = canon(val) == canon(expected)
        if faithful:
            emit(
                f"{INDENT}{YELLOW}✓ correct — {RED}but {ntok} tok, wrapped in "
                f"prose and a code fence{RESET}"
            )
            emit(
                f"{INDENT}{GRAY}↑ you'd still have to find and parse the JSON "
                f"back out{RESET}"
            )
        else:
            emit(f"{INDENT}{YELLOW}✓ parses — {RED}but it's not your schema{RESET}")
            emit(f"{INDENT}{GRAY}↑ {describe_diff(expected, val)}{RESET}")
    _sleep(2.4)
    return ntok


def slide_success(m: Model, num: str, case: dict, base_tokens: int) -> None:
    from raif import decode

    expected = case["expected"]
    prompt = PROMPT_TMPL.format(
        details=case["details"], schema="\n".join(case["schema"])
    )
    header(num, case["ok_title"], case["ok_sub"])
    show_task(case["details"], case["schema"])  # the SAME prompt the base got
    _sleep(0.4)
    emit()
    emit(f"{INDENT}{GRAY}── 0.5B + RAIF LoRA (live, full output) ──{RESET}")
    _sleep(0.3)
    raif_text, rtok, _ = m.stream(prompt, case["budget"], cps=150.0)
    _sleep(0.3)
    res = decode(raif_text)
    if res.get("ok") and canon(res["value"]) == canon(expected):
        repairs = len(res.get("repairs") or [])
        decoded = json.dumps(res["value"], separators=(",", ":"), ensure_ascii=False)
        emit()
        lines = _wrap(decoded, 70)
        emit(f"{INDENT}{GRAY}decode() → {RESET}{TEXT}{lines[0]}{RESET}")
        for extra in lines[1:]:
            emit(f"{INDENT}           {TEXT}{extra}{RESET}")
        _sleep(0.3)
        win = (
            f"{rtok} tok vs base's {base_tokens}+ (cut off)"
            if base_tokens >= case["budget"] - 4
            else f"{rtok} tok vs base's {base_tokens}"
        )
        rep = "no repairs" if not repairs else f"{repairs} repair(s)"
        emit(f"{INDENT}{GREEN}✓ schema filled, byte-exact, {rep} · {win}{RESET}")
    elif res.get("ok"):
        emit(f"{INDENT}{YELLOW}✓ decodes (differs from intended){RESET}")
    else:
        emit(f"{INDENT}{RED}✗ decode failed: {res.get('error')}{RESET}")
    _sleep(2.8)


def main() -> int:
    sys.stdout.write("\033[?25l")
    try:
        logo_screen("Schema in, data out — structured output from a 0.5B.")
        _sleep(2.2)
        loading_screen()
        base = Model(lora=False)
        raif = Model(lora=True)
        for i, case in enumerate(CASES):
            bt = slide_fail(base, f"{2 * i + 1:02d}", case)
            slide_success(raif, f"{2 * i + 2:02d}", case, bt)
        logo_screen(
            "Same prompt: the bare 0.5B rambles and reshapes; RAIF fills the "
            "schema, byte-exact.",
            footer=[
                f"{BLUE}huggingface.co/skrrt-sh/raif-qwen2.5-0.5b-lora{RESET}",
                f"{BLUE}github.com/skrrt-sh/raif-lora{RESET}",
                "",
                f"{GRAY}Open source. Apache-2.0.{RESET}",
            ],
        )
        # Hold the call-to-action long enough that the screen recording (see
        # assets/demo-0.5b.tape) always ends on this frame, never on the shell
        # prompt that appears once the process exits.
        _sleep(8.0)
    finally:
        sys.stdout.write("\033[?25h" + RESET)
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
