"""Shared bridge to the canonical TypeScript RAIF implementation via `bun`.

Three callers shell out to `raif-standard/prototype/src/raif.ts`: the eval meter
(`eval_core`), the differential-test oracle (`_raif_oracle`), and the decoder
parity test (`test_raif_decode`). Each used to carry its own copy of the prototype
path, the `bun` availability probe, and the tempfile+subprocess plumbing — with
subtly different timeout formulas and error messages. This module is the single
place that knows how to run a bun script against the prototype, so the invocation,
the cwd, and the failure handling live in one spot.

The caller supplies the bun script (which reads its JSON input from the file named
by the `RAIF_BRIDGE_INPUT` env var) and the payload; `run_bridge` handles the
temp file, the subprocess, and parsing stdout back to JSON.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

# Resolve from THIS file so the bridge works regardless of the caller's cwd:
#   raif-lora/src/raif_bun.py -> scratch/ -> raif-standard/prototype
PROTOTYPE_DIR = Path(__file__).resolve().parent.parent.parent / "raif-standard" / "prototype"

# Env var the caller's bun script reads to find its JSON input file.
INPUT_ENV = "RAIF_BRIDGE_INPUT"


def available() -> bool:
    """True when both `bun` and the sibling `raif-standard` prototype are present."""
    return (
        subprocess.run(["which", "bun"], capture_output=True).returncode == 0
        and (PROTOTYPE_DIR / "src" / "raif.ts").exists()
    )


def run_bridge(script: str, payload, timeout: float) -> list:
    """Run `bun -e <script>` against `payload` (written to a temp file the script
    reads via `RAIF_BRIDGE_INPUT`) and return the JSON list it writes to stdout.

    Raises RuntimeError if bun exits non-zero (stderr is surfaced, truncated)."""
    fd, tmp = tempfile.mkstemp(suffix=".json", prefix="raif_bun_")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f)
        res = subprocess.run(
            ["bun", "-e", script],
            capture_output=True,
            cwd=PROTOTYPE_DIR,
            timeout=timeout,
            env={**os.environ, INPUT_ENV: tmp},
        )
    finally:
        os.unlink(tmp)
    if res.returncode != 0:
        raise RuntimeError(f"bun bridge failed: {res.stderr.decode('utf-8', 'replace')[:1000]}")
    return json.loads(res.stdout.decode("utf-8"))
