#!/usr/bin/env bash
# RAIF LoRA — RunPod / SSH-box bootstrap. Runs ON the GPU box (not your laptop).
#
# It is the cloud counterpart of cuda/README.md's prereqs: clone both repos as
# siblings, build the CUDA stack, install bun + the prototype decoder the eval
# meter shells out to, regenerate the (gitignored) data deterministically, run
# the meter oracle gate, then train+eval one stage to validate the port.
#
# One-liner on a fresh pod:
#   curl -fsSL https://raw.githubusercontent.com/skrrt-sh/raif-lora/main/cuda/cloud/bootstrap.sh | bash
#
# Knobs (env vars):
#   WORKROOT   parent dir holding the two sibling repos   (default /workspace/raif)
#   STAGE      stage to train at the end                  (default smoke)
#   RUN_STAGE  1 = train+eval STAGE; 0 = set up only      (default 1)
#   OPTIM      passthrough to the trainer --optim         (default unset → adamw_8bit)
#   LORA_REF   optional git ref to check out in raif-lora     (default: default branch)
#   STD_REF    optional git ref to check out in raif-standard (default: default branch)
#
# Why smoke by default: it is the FIRST time this CUDA code runs on a GPU. The
# smoke eval must roughly reproduce the MLX run (69% valid / 23% holdout fidelity)
# before any longer run is trustworthy — see ../README.md and ../../ITERATION_PLAN.md.

set -euo pipefail

WORKROOT="${WORKROOT:-/workspace/raif}"
LORA_REPO="${LORA_REPO:-https://github.com/skrrt-sh/raif-lora.git}"
STD_REPO="${STD_REPO:-https://github.com/skrrt-sh/raif-standard.git}"
STAGE="${STAGE:-smoke}"
RUN_STAGE="${RUN_STAGE:-1}"
LORA_REF="${LORA_REF:-}"   # optional git ref to check out in raif-lora (default: clone default branch)
STD_REF="${STD_REF:-}"     # optional git ref to check out in raif-standard

log() { printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }
die() { printf '\n\033[1;31mFATAL: %s\033[0m\n' "$*" >&2; exit 1; }

# ── 0. GPU present? ───────────────────────────────────────────────────────────
log "0. GPU check"
command -v nvidia-smi >/dev/null || die "no nvidia-smi — this is not a CUDA GPU box"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# ── 1. system deps ──────────────────────────────────────────────────────────
log "1. System deps (git, curl, unzip)"
if command -v apt-get >/dev/null; then
  apt-get update -qq && apt-get install -y -qq git curl unzip ca-certificates >/dev/null
fi

# Keep the big caches on the persistent volume (/workspace survives stop/restart),
# NOT the small ephemeral container disk — the torch + 3B-base downloads are GBs.
export HF_HOME="${HF_HOME:-$WORKROOT/.hf-cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$WORKROOT/.pip-cache}"
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR"

# ── 2. clone (or UPDATE) the repos as SIBLINGS so a re-run picks up fixes ──────
# eval_core resolves ../raif-standard/prototype, hence the sibling layout. An
# existing checkout is hard-reset to the latest ref — otherwise a re-run keeps
# running stale code (untracked data/ and adapters-cuda/ are left intact).
log "2. Clone/update repos as siblings under $WORKROOT"
mkdir -p "$WORKROOT"; cd "$WORKROOT"
clone_or_update() {  # <dir> <url> <ref-or-empty>
  local dir="$1" url="$2" ref="$3"
  if [ -d "$dir/.git" ]; then
    git -C "$dir" fetch --depth 1 origin "${ref:-HEAD}"
    git -C "$dir" reset --hard FETCH_HEAD
  else
    git clone --depth 1 ${ref:+--branch "$ref"} "$url" "$dir"
  fi
}
clone_or_update raif-lora     "$LORA_REPO" "$LORA_REF"
clone_or_update raif-standard "$STD_REPO"  "$STD_REF"

# ── 3. python env — install into the interpreter that OWNS torch ──────────────
log "3. Locate the image's torch interpreter"
cd "$WORKROOT/raif-lora"
deactivate 2>/dev/null || true   # leave any pre-activated venv from a failed prior run
# Install the userland stack straight into the interpreter that already has the
# image's torch — do NOT use a venv. A --system-site-packages venv can't "own" the
# system torch, so pip re-pulls torch/torchvision/xformers into it (multi-GB, and
# defeats the whole point). Installing into torch's own env lets pip treat torch/
# triton/nccl as already satisfied and skip them. (The pod is disposable, so
# mutating the image env is fine.)
BASE_PY=""
for cand in python python3 python3.12 python3.11 python3.10; do
  if command -v "$cand" >/dev/null 2>&1 && "$cand" -c 'import torch' >/dev/null 2>&1; then
    BASE_PY="$(command -v "$cand")"; break
  fi
done
[ -n "$BASE_PY" ] || die "no python on the box has torch — pick a RunPod PyTorch template (torch >= 2.5, CUDA >= 12.1)"
rm -rf .venv-cuda 2>/dev/null || true   # drop any stale venv from an older bootstrap version
PY="$BASE_PY"; PIP="$BASE_PY -m pip"
echo "torch interpreter: $BASE_PY"
$PIP install -q --upgrade pip wheel

log "3a. Install userland stack into the image torch (no torch reinstall)"
# Fail fast if the image torch is older than unsloth's floor, then install ONLY the
# userland stack (unsloth/trl/peft/…). torch/triton/nccl already live in this env,
# so pip recognizes them as satisfied and does not re-download them.
"$PY" - <<'PYEOF'
import sys, torch
print("image torch", torch.__version__, torch.version.cuda)
if tuple(int(x) for x in torch.__version__.split(".")[:2]) < (2, 5):
    sys.exit(f"FATAL: image torch {torch.__version__} < 2.5 — pick a newer RunPod "
             "PyTorch template (torch >= 2.5, CUDA >= 12.1) so torch isn't re-pulled.")
PYEOF
$PIP install -r cuda/cloud/requirements-cloud.txt

# ── 4. torch must see the GPU (we PRINT capability, never assert (12,0)) ───────
log "4. Verify torch sees the GPU"
"$PY" - <<'PYEOF'
import torch
print("torch:", torch.__version__, "| cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("torch cannot see a CUDA GPU — driver too old for these wheels?")
print("device:", torch.cuda.get_device_name(0),
      "| capability:", torch.cuda.get_device_capability())
PYEOF

# ── 5. bun + prototype decoder (the eval meter shells out to `bun -e`) ─────────
log "5. bun + prototype decoder"
if ! command -v bun >/dev/null; then
  curl -fsSL https://bun.sh/install | bash -s "bun-v1.3.13"   # pin: matches the lockfile author's bun
fi
export BUN_INSTALL="${BUN_INSTALL:-$HOME/.bun}"
export PATH="$BUN_INSTALL/bin:$PATH"
command -v bun >/dev/null || die "bun install failed / not on PATH"
( cd "$WORKROOT/raif-standard/prototype" && bun install --frozen-lockfile )

# ── 6. data — regenerate deterministically (gitignored; seed 0 == local) ──────
log "6. Regenerate data (make_data.sh full → 1235/65/500, seed 0)"
# Regenerate unless ALL THREE splits are present AND non-empty.
if ! { [ -s data/train.jsonl ] && [ -s data/valid.jsonl ] && [ -s data/eval_holdout.jsonl ]; }; then
  ./src/make_data.sh full
fi
wc -l data/*.jsonl

# ── 7. meter gate — MUST pass before trusting any training number ─────────────
log "7. Sanity gate: eval-meter oracle tests + data containment"
"$PY" src/test_eval_smoke.py
"$PY" src/check_data.py

# ── 7b. run manifest — what this run was built from (for reproducibility) ──────
mkdir -p logs
{
  echo "raif-lora   $(git -C "$WORKROOT/raif-lora" rev-parse HEAD)"
  echo "raif-standard $(git -C "$WORKROOT/raif-standard" rev-parse HEAD)"
  echo "bun         $(bun --version)"
  echo "image torch $("$PY" -c 'import torch;print(torch.__version__,torch.version.cuda)')"
} > "logs/run-manifest-$STAGE.txt"

# ── 8. train + eval one stage ─────────────────────────────────────────────────
if [ "$RUN_STAGE" = "1" ]; then
  OPTIM_FLAG=()
  [ -n "${OPTIM:-}" ] && OPTIM_FLAG=(--optim "$OPTIM")
  log "8. Train stage=$STAGE"
  "$PY" cuda/train_unsloth.py --stage "$STAGE" "${OPTIM_FLAG[@]}" 2>&1 | tee "logs/cuda-$STAGE.log"
  log "9. Eval stage=$STAGE  (target ≈ MLX smoke: 69% valid / 23% holdout fidelity)"
  "$PY" cuda/eval_cuda.py --adapter "./adapters-cuda/$STAGE" --n 13 2>&1 | tee "logs/cuda-$STAGE-eval.log"
  cat <<EOF

Done. If the smoke numbers roughly match (≈69% valid / ≈23% holdout fidelity),
the port is validated — climb the ladder (gates in ITERATION_PLAN.md):

  cd $WORKROOT/raif-lora
  export PATH="\$HOME/.bun/bin:\$PATH" HF_HOME="$HF_HOME"
  $BASE_PY cuda/train_unsloth.py --stage warm && $BASE_PY cuda/eval_cuda.py --adapter ./adapters-cuda/warm --n 13
  $BASE_PY cuda/train_unsloth.py --stage full && $BASE_PY cuda/eval_cuda.py --adapter ./adapters-cuda/full --n 13

Save the adapter OFF the pod BEFORE teardown — /workspace is wiped on terminate
and the adapter is gitignored. Works on any pod (no public IP needed):
  tar czf /workspace/$STAGE.tgz -C adapters-cuda $STAGE
  runpodctl send /workspace/$STAGE.tgz      # then 'runpodctl receive <code>' on your laptop
See cuda/cloud/README.md → "Pull the adapter back" for the scp alternative.
EOF
else
  log "Setup complete (RUN_STAGE=0). To train:"
  echo "  $BASE_PY cuda/train_unsloth.py --stage $STAGE && $BASE_PY cuda/eval_cuda.py --adapter ./adapters-cuda/$STAGE --n 13"
fi
