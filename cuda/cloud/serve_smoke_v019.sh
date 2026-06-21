#!/usr/bin/env bash
# RAIF single-plugin smoke on a CUDA-12 GPU box (vLLM 0.19, entry-point model).
#
# Runs ON the GPU box. Assumes the two working trees are already present (rsync'd)
# as siblings under $WORKROOT (NOT cloned — the plugin is uncommitted):
#   $WORKROOT/raif-standard/packages/py   (the `raif` format lib)
#   $WORKROOT/raif-lora/packages/vllm      (the `raif-vllm` plugin)
#
# Unlike the old serve_test.sh (vLLM 0.11 + --tool-parser-plugin FILE), this uses
# the installable plugin + the general_plugins entry point: `pip install raif-vllm`
# then `VLLM_PLUGINS=raif`. v0.19.0 is the LAST CUDA-12 vLLM and carries every hook
# the plugin needs (reasoning-parser content decode + the render_chat inject seam).
set -euo pipefail

WORKROOT="${WORKROOT:-/workspace/raif}"
BASE="${BASE:-unsloth/Llama-3.2-3B-Instruct}"
ADAPTER="${ADAPTER:-skrrt-sh/raif-llama-3.2-3b-lora}"
PORT="${PORT:-8000}"
PY="${PY:-python3.12}"
VLLM_PIN="${VLLM_PIN:-vllm==0.19.0}"

LORA_DIR="$WORKROOT/raif-lora"
STD_DIR="$WORKROOT/raif-standard"
PLUGIN_PKG="$LORA_DIR/packages/vllm"
CHAT_TEMPLATE="$LORA_DIR/cuda/cloud/raif_llama32.jinja"

log() { printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }
die() { printf '\n\033[1;31mFATAL: %s\033[0m\n' "$*" >&2; exit 1; }

log "0. GPU + interpreter"
command -v nvidia-smi >/dev/null || die "no nvidia-smi — not a CUDA GPU box"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
command -v "$PY" >/dev/null || die "interpreter '$PY' not found (set PY=)"
"$PY" --version
[ -d "$PLUGIN_PKG" ] || die "plugin package not found at $PLUGIN_PKG (rsync the tree first)"
[ -f "$CHAT_TEMPLATE" ] || die "chat template not found at $CHAT_TEMPLATE"

export HF_HOME="${HF_HOME:-$WORKROOT/.hf-cache}"
mkdir -p "$HF_HOME"

log "1. Install vLLM 0.19 (CUDA-12) + the raif packages (editable) + clients"
"$PY" -c 'import vllm' 2>/dev/null || "$PY" -m pip install -q "$VLLM_PIN"
# vLLM's prometheus instrumentator crashes on starlette 1.x ("'_IncludedRouter'
# object has no attribute 'path'") -> every /health returns 500 and the server
# never becomes healthy. Pin fastapi 0.115.6, which constrains starlette <0.42
# (routes still carry `.path`). Same pin the 0.11 runbook uses.
"$PY" -m pip install -q openai "fastapi==0.115.6" \
  -e "$STD_DIR/packages/py" \
  -e "$PLUGIN_PKG"
"$PY" -c 'import vllm, raif, raif_vllm; print("vllm", vllm.__version__, "| raif", raif.__version__, "| raif_vllm OK")'

log "2. Serve $BASE + LoRA '$ADAPTER' with the raif plugin (VLLM_PLUGINS=raif, port $PORT)"
# The chat template strips OpenAI tool-def JSON (tools path parity). The plugin's
# general_plugins entry point registers the parsers + installs the render_chat
# inject hook; --reasoning-parser raif lights up the response_format decode path.
VLLM_PLUGINS=raif "$PY" -m vllm.entrypoints.openai.api_server --model "$BASE" \
  --enable-lora --lora-modules "raif=$ADAPTER" \
  --max-lora-rank 32 --max-model-len 8192 --enforce-eager \
  --chat-template "$CHAT_TEMPLATE" \
  --reasoning-parser raif \
  --enable-auto-tool-choice --tool-call-parser raif \
  --port "$PORT" >"$WORKROOT/vllm-serve.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT

log "3. Wait for health (up to ~5 min for model load)"
for _ in $(seq 1 100); do
  curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 && break
  kill -0 $SERVER 2>/dev/null || die "vllm exited early — see $WORKROOT/vllm-serve.log"
  sleep 3
done
curl -sf "http://localhost:$PORT/health" >/dev/null || die "server never became healthy"

log "4. Confirm the plugin actually loaded (render patch + parser registration)"
grep -E "RAIF plugin:|RAIF render patch|no pre-template inject seam" "$WORKROOT/vllm-serve.log" || \
  printf '\033[1;33mWARN: no RAIF plugin log lines found — entry point may not have run.\033[0m\n'

log "5. Smoke: plain chat + tools + response_format (+ token saving)"
set +e
"$PY" "$LORA_DIR/examples/smoke_plugin.py" --base-url "http://localhost:$PORT/v1" --model raif
SMOKE_RC=$?
set -e

log "OVERALL"
if [ "$SMOKE_RC" -eq 0 ]; then
  printf '\033[1;32mOVERALL: PASS — RAIF single plugin verified (tools + response_format + plain).\033[0m\n'
  printf '(server log: %s)\n' "$WORKROOT/vllm-serve.log"
  exit 0
fi
printf '\033[1;31mOVERALL: FAIL (smoke rc=%d) — see %s\033[0m\n' "$SMOKE_RC" "$WORKROOT/vllm-serve.log"
exit 1
