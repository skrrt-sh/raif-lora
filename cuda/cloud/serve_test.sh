#!/usr/bin/env bash
# RAIF vLLM plugin — serve + end-to-end test on a RunPod / SSH GPU box.
#
# Runs ON the GPU box (not your laptop). Serves the published RAIF LoRA through
# vLLM with the `raif` tool-call parser, then asserts the OpenAI client gets a
# JSON tool call back (examples/e2e_smoke.py) and runs the importorskip shim
# tests (which now execute against real vLLM types).
#
# One-liner on a fresh pod (24 GB Ada/Ampere; vLLM image or PyTorch template):
#   curl -fsSL https://raw.githubusercontent.com/skrrt-sh/raif-lora/main/cuda/cloud/serve_test.sh | bash
#
# Knobs (env vars):
#   WORKROOT  parent dir holding the two sibling repos  (default /workspace/raif)
#   BASE      base model (ungated, no HF token needed)  (default unsloth/Llama-3.2-3B-Instruct)
#   ADAPTER   LoRA repo/path served as model id "raif"   (default skrrt-sh/raif-llama-3.2-3b-lora)
#   PORT      OpenAI server port                         (default 8000)
set -euo pipefail

WORKROOT="${WORKROOT:-/workspace/raif}"
BASE="${BASE:-unsloth/Llama-3.2-3B-Instruct}"
ADAPTER="${ADAPTER:-skrrt-sh/raif-llama-3.2-3b-lora}"
PORT="${PORT:-8000}"
LORA_REPO="${LORA_REPO:-https://github.com/skrrt-sh/raif-lora.git}"
STD_REPO="${STD_REPO:-https://github.com/skrrt-sh/raif-standard.git}"

log() { printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }
die() { printf '\n\033[1;31mFATAL: %s\033[0m\n' "$*" >&2; exit 1; }

log "0. GPU check"
command -v nvidia-smi >/dev/null || die "no nvidia-smi — this is not a CUDA GPU box"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

export HF_HOME="${HF_HOME:-$WORKROOT/.hf-cache}"
mkdir -p "$HF_HOME"

# Clone the two repos as siblings if a prior bootstrap hasn't already (eval +
# the plugin live in raif-lora; raif-format/stream/schema_bridge in raif-standard).
log "1. Repos (siblings under $WORKROOT)"
mkdir -p "$WORKROOT"; cd "$WORKROOT"
[ -d raif-lora/.git ]     || git clone --depth 1 "$LORA_REPO" raif-lora
[ -d raif-standard/.git ] || git clone --depth 1 "$STD_REPO"  raif-standard
PLUGIN="$WORKROOT/raif-lora/src/raif_vllm.py"
[ -f "$PLUGIN" ] || die "plugin not found at $PLUGIN"

# vLLM + the OpenAI client + raif-format installed EDITABLE from the sibling
# clone (the new stream/schema_bridge aren't on PyPI yet). `pip install vllm` is
# a no-op on a vllm/vllm-openai pod; on a PyTorch template it pulls vLLM's torch.
log "2. Install vllm + raif-format (editable)"
command -v vllm >/dev/null || pip install -q vllm
pip install -q openai pytest -e "$WORKROOT/raif-standard/packages/py"

log "3. Serve $BASE + LoRA '$ADAPTER' with the raif tool parser (port $PORT)"
vllm serve "$BASE" \
  --enable-lora --lora-modules "raif=$ADAPTER" \
  --max-lora-rank 32 \
  --enable-auto-tool-choice \
  --tool-parser-plugin "$PLUGIN" --tool-call-parser raif \
  --port "$PORT" >"$WORKROOT/vllm-serve.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT

log "4. Wait for the server to come up (up to ~5 min for model load)"
for _ in $(seq 1 100); do
  if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then break; fi
  kill -0 $SERVER 2>/dev/null || die "vllm exited early — see $WORKROOT/vllm-serve.log"
  sleep 3
done
curl -sf "http://localhost:$PORT/health" >/dev/null || die "server never became healthy"

log "5. End-to-end smoke (OpenAI client -> JSON tool_calls)"
python "$WORKROOT/raif-lora/examples/e2e_smoke.py" \
  --base-url "http://localhost:$PORT/v1" --model raif

log "6. Shim wiring tests (real vLLM types)"
pytest "$WORKROOT/raif-lora/src/test_raif_vllm.py" -v

log "DONE — RAIF vLLM plugin verified end-to-end. (server log: $WORKROOT/vllm-serve.log)"
