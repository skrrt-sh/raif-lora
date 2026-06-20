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
# THE FIX (GPU e2e parity): when the client passes tools=[...], vLLM's default
# chat template also renders the OpenAI tool-definition JSON into the prompt, on
# top of the plugin's <schema> cue. The LoRA was trained ONLY on the bare
# <schema> cue, so it echoes the verbose tool defs instead of producing RAIF
# arguments. We serve with a custom --chat-template that renders ONLY the
# messages and IGNORES the tools variable, so the prompt the model receives is
# exactly the request + the plugin's injected <schema> block = training parity.
# We KEEP request.tools set on the wire (vLLM's --enable-auto-tool-choice gating
# and the plugin's extract_tool_calls both read it); only the template drops the
# tool rendering.
#
# ENVIRONMENT (RunPod A40, driver CUDA 12.9): pin vllm==0.11.0 (torch 2.8.0+cu128,
# OK on driver 12.9 — newer vLLM needs a CUDA-13 driver) and transformers>=4.56,<5
# (transformers 5.x removed all_special_tokens_extended, which vLLM 0.11's
# tokenizer init calls). The real interpreter with torch/vllm is python3.12.
#
# Knobs (env vars):
#   WORKROOT     parent dir holding the two sibling repos  (default /workspace/raif)
#   BASE         base model (ungated, no HF token needed)   (default unsloth/Llama-3.2-3B-Instruct)
#   ADAPTER      LoRA repo/path served as model id "raif"    (default skrrt-sh/raif-llama-3.2-3b-lora)
#   PORT         OpenAI server port                          (default 8000)
#   STD_BRANCH   raif-standard branch to clone               (default main)
#   LORA_BRANCH  raif-lora branch to clone                   (default main)
set -euo pipefail

WORKROOT="${WORKROOT:-/workspace/raif}"
BASE="${BASE:-unsloth/Llama-3.2-3B-Instruct}"
ADAPTER="${ADAPTER:-skrrt-sh/raif-llama-3.2-3b-lora}"
PORT="${PORT:-8000}"
LORA_REPO="${LORA_REPO:-https://github.com/skrrt-sh/raif-lora.git}"
STD_REPO="${STD_REPO:-https://github.com/skrrt-sh/raif-standard.git}"
STD_BRANCH="${STD_BRANCH:-main}"
LORA_BRANCH="${LORA_BRANCH:-main}"

# The image's real interpreter that actually carries torch/vllm is python3.12
# (plain `python3` on the stock RunPod image is an empty 3.10).
PY="${PY:-python3.12}"

log() { printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }
die() { printf '\n\033[1;31mFATAL: %s\033[0m\n' "$*" >&2; exit 1; }

log "0. GPU check"
command -v nvidia-smi >/dev/null || die "no nvidia-smi — this is not a CUDA GPU box"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
command -v "$PY" >/dev/null || die "interpreter '$PY' not found (set PY= to the python with torch/vllm)"
"$PY" --version

export HF_HOME="${HF_HOME:-$WORKROOT/.hf-cache}"
mkdir -p "$HF_HOME"

# Clone the two repos as siblings if a prior bootstrap hasn't already (eval +
# the plugin live in raif-lora; raif-format/stream/schema_bridge in raif-standard).
log "1. Repos (siblings under $WORKROOT)"
mkdir -p "$WORKROOT"; cd "$WORKROOT"
[ -d raif-lora/.git ]     || git clone --depth 1 --branch "$LORA_BRANCH" "$LORA_REPO" raif-lora
[ -d raif-standard/.git ] || git clone --depth 1 --branch "$STD_BRANCH" "$STD_REPO" raif-standard
PLUGIN="$WORKROOT/raif-lora/src/raif_vllm.py"
CHAT_TEMPLATE="$WORKROOT/raif-lora/cuda/cloud/raif_llama32.jinja"
[ -f "$PLUGIN" ]        || die "plugin not found at $PLUGIN"
[ -f "$CHAT_TEMPLATE" ] || die "chat template not found at $CHAT_TEMPLATE"

# Pinned env (see header). The heavy vLLM/torch install is skipped if vLLM is
# already present (e.g. a vllm/vllm-openai pod); on a PyTorch template the pin
# pulls the CUDA-12.8 torch that works on driver 12.9. The lightweight pins below
# ALWAYS run (idempotent) because the stock RunPod image ships bleeding-edge libs
# that outpace vLLM 0.11's loose deps:
#   transformers>=4.56,<5  — 5.x dropped all_special_tokens_extended (tokenizer init)
#   fastapi==0.115.6       — pins starlette <1.0; vLLM 0.11's prometheus
#                            instrumentator breaks on starlette 1.x
#                            ("'_IncludedRouter' object has no attribute 'path'")
# raif-format is installed EDITABLE from the sibling clone (the new
# stream/schema_bridge aren't on PyPI yet).
log "2. Install pinned vllm + transformers + fastapi + raif-format (editable)"
"$PY" -c 'import vllm' 2>/dev/null || "$PY" -m pip install -q "vllm==0.11.0"
"$PY" -m pip install -q "transformers>=4.56,<5" "fastapi==0.115.6" \
  openai pytest requests -e "$WORKROOT/raif-standard/packages/py"

log "3. Serve $BASE + LoRA '$ADAPTER' with the raif tool parser (port $PORT)"
# --chat-template renders ONLY the messages (ignores tools) -> training parity.
"$PY" -m vllm.entrypoints.openai.api_server --model "$BASE" \
  --enable-lora --lora-modules "raif=$ADAPTER" \
  --max-lora-rank 32 \
  --max-model-len 8192 \
  --enforce-eager \
  --chat-template "$CHAT_TEMPLATE" \
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

# Capture exit codes (don't let set -e abort before the OVERALL summary).
log "5. End-to-end smoke (OpenAI client -> JSON tool_calls)"
set +e
"$PY" "$WORKROOT/raif-lora/examples/e2e_smoke.py" \
  --base-url "http://localhost:$PORT/v1" --model raif
SMOKE_RC=$?

log "6. Shim wiring tests (real vLLM types)"
"$PY" -m pytest "$WORKROOT/raif-lora/src/test_raif_vllm.py" -v
PYTEST_RC=$?
set -e

log "OVERALL"
printf 'e2e smoke : %s (rc=%d)\n' "$([ "$SMOKE_RC" -eq 0 ] && echo PASS || echo FAIL)" "$SMOKE_RC"
printf 'shim tests: %s (rc=%d)\n' "$([ "$PYTEST_RC" -eq 0 ] && echo PASS || echo FAIL)" "$PYTEST_RC"
if [ "$SMOKE_RC" -eq 0 ] && [ "$PYTEST_RC" -eq 0 ]; then
  printf '\n\033[1;32mOVERALL: PASS — RAIF vLLM plugin verified end-to-end.\033[0m\n'
  printf '(server log: %s)\n' "$WORKROOT/vllm-serve.log"
  exit 0
fi
printf '\n\033[1;31mOVERALL: FAIL — see %s\033[0m\n' "$WORKROOT/vllm-serve.log"
exit 1
