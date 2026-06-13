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

# ── 2. clone repos as SIBLINGS (eval_core resolves ../raif-standard/prototype) ─
log "2. Clone repos as siblings under $WORKROOT"
mkdir -p "$WORKROOT"; cd "$WORKROOT"
[ -d raif-lora ]     || git clone --depth 1 "$LORA_REPO"     raif-lora
[ -d raif-standard ] || git clone --depth 1 "$STD_REPO"      raif-standard
[ -n "$LORA_REF" ] && git -C raif-lora     checkout "$LORA_REF"
[ -n "$STD_REF" ]  && git -C raif-standard checkout "$STD_REF"

# ── 3. python env on the persistent volume ────────────────────────────────────
log "3. Python venv (.venv-cuda on $WORKROOT)"
cd "$WORKROOT/raif-lora"
deactivate 2>/dev/null || true   # leave any pre-activated venv (e.g. a failed prior run)
# Build the venv from the interpreter that ALREADY has the image's torch. RunPod
# images may keep torch in conda or a base venv — NOT necessarily python3.11 — so
# probe for it; assuming the wrong interpreter makes --system-site-packages inherit
# a torch-less site-packages (the "No module named 'torch'" failure).
BASE_PY=""
for cand in python python3 python3.11 python3.10; do
  if command -v "$cand" >/dev/null 2>&1 && "$cand" -c 'import torch' >/dev/null 2>&1; then
    BASE_PY="$(command -v "$cand")"; break
  fi
done
[ -n "$BASE_PY" ] || die "no python on the box has torch — pick a RunPod PyTorch template (torch >= 2.5, CUDA >= 12.1)"
echo "base python with torch: $BASE_PY"
# (Re)create the venv from THAT python so --system-site-packages inherits its torch.
# Rebuild if a prior run left a venv that can't see torch.
if [ ! -d .venv-cuda ] || ! .venv-cuda/bin/python -c 'import torch' >/dev/null 2>&1; then
  rm -rf .venv-cuda
  "$BASE_PY" -m venv --system-site-packages .venv-cuda
fi
# shellcheck disable=SC1091
source .venv-cuda/bin/activate
pip install -q --upgrade pip wheel

log "3a. Reuse the image's torch, install userland only (no torch reinstall)"
# The RunPod PyTorch template ships a matching torch; reinstalling it wastes paid
# GPU time. Print what the image has and fail fast if it's older than unsloth's
# floor (else pip would silently re-pull a multi-GB torch), then install ONLY the
# userland stack (unsloth/trl/peft/…) from the torch-free requirements-cloud.txt.
python - <<'PY'
import sys, torch
print("image torch", torch.__version__, torch.version.cuda)
if tuple(int(x) for x in torch.__version__.split(".")[:2]) < (2, 5):
    sys.exit(f"FATAL: image torch {torch.__version__} < 2.5 — pick a newer RunPod "
             "PyTorch template (torch >= 2.5, CUDA >= 12.1) so torch isn't re-pulled.")
PY
pip install -r cuda/cloud/requirements-cloud.txt

# ── 4. torch must see the GPU (we PRINT capability, never assert (12,0)) ───────
log "4. Verify torch sees the GPU"
python - <<'PY'
import torch
print("torch:", torch.__version__, "| cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("torch cannot see a CUDA GPU — driver too old for these wheels?")
print("device:", torch.cuda.get_device_name(0),
      "| capability:", torch.cuda.get_device_capability())
PY

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
python src/test_eval_smoke.py
python src/check_data.py

# ── 7b. run manifest — what this run was built from (for reproducibility) ──────
mkdir -p logs
{
  echo "raif-lora   $(git -C "$WORKROOT/raif-lora" rev-parse HEAD)"
  echo "raif-standard $(git -C "$WORKROOT/raif-standard" rev-parse HEAD)"
  echo "bun         $(bun --version)"
  echo "image torch $(python -c 'import torch;print(torch.__version__,torch.version.cuda)')"
} > "logs/run-manifest-$STAGE.txt"

# ── 8. train + eval one stage ─────────────────────────────────────────────────
if [ "$RUN_STAGE" = "1" ]; then
  OPTIM_FLAG=()
  [ -n "${OPTIM:-}" ] && OPTIM_FLAG=(--optim "$OPTIM")
  log "8. Train stage=$STAGE"
  python cuda/train_unsloth.py --stage "$STAGE" "${OPTIM_FLAG[@]}" 2>&1 | tee "logs/cuda-$STAGE.log"
  log "9. Eval stage=$STAGE  (target ≈ MLX smoke: 69% valid / 23% holdout fidelity)"
  python cuda/eval_cuda.py --adapter "./adapters-cuda/$STAGE" --n 13 2>&1 | tee "logs/cuda-$STAGE-eval.log"
  cat <<EOF

Done. If the smoke numbers roughly match (≈69% valid / ≈23% holdout fidelity),
the port is validated — climb the ladder (gates in ITERATION_PLAN.md):

  source $WORKROOT/raif-lora/.venv-cuda/bin/activate
  export PATH="\$HOME/.bun/bin:\$PATH" HF_HOME="$HF_HOME"
  python cuda/train_unsloth.py --stage warm && python cuda/eval_cuda.py --adapter ./adapters-cuda/warm --n 13
  python cuda/train_unsloth.py --stage full && python cuda/eval_cuda.py --adapter ./adapters-cuda/full --n 13

Save the adapter OFF the pod BEFORE teardown — /workspace is wiped on terminate
and the adapter is gitignored. Works on any pod (no public IP needed):
  tar czf /workspace/$STAGE.tgz -C adapters-cuda $STAGE
  runpodctl send /workspace/$STAGE.tgz      # then 'runpodctl receive <code>' on your laptop
See cuda/cloud/README.md → "Pull the adapter back" for the scp alternative.
EOF
else
  log "Setup complete (RUN_STAGE=0). To train:"
  echo "  python cuda/train_unsloth.py --stage $STAGE && python cuda/eval_cuda.py --adapter ./adapters-cuda/$STAGE --n 13"
fi
