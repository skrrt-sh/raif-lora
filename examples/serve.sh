#!/usr/bin/env bash
# Serve a RAIF LoRA over an OpenAI-compatible HTTP API (MLX), so any agent
# framework — Vercel AI SDK, the OpenAI SDK, LangChain, etc. — can talk to it.
#
#   examples/serve.sh                       # llama-3b -> http://127.0.0.1:8899/v1
#   MODEL=qwen-0.5b examples/serve.sh
#   MODEL=qwen-4b PORT=9000 examples/serve.sh
#
# Then point a client at http://127.0.0.1:$PORT/v1 with model "default_model",
# and pass the adapter path printed below in the request body as "adapters".
#
# NOTE: mlx-lm 0.31.x has a bug where the CLI --adapter-path is not actually
# applied per request, so the client must send "adapters": "<path>" in the body
# (see examples/ai-sdk/demo.ts). We still pass --adapter-path so this keeps
# working once the upstream bug is fixed.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-llama-3b}"
PORT="${PORT:-8899}"

# Resolve base + adapter paths from the example model registry.
read -r BASE ADAPTER < <(uv run python -c "
import examples.raif_models as M
s = M.spec('$MODEL'); print(M.base_path(s), s['mlx_dir'])")

if [ ! -f "$ADAPTER/adapters.safetensors" ]; then
  echo "Adapter for '$MODEL' not built." >&2
  echo "Run: uv run python examples/setup_adapter.py --model $MODEL" >&2
  exit 1
fi

echo "Serving $MODEL on http://127.0.0.1:$PORT/v1  (model id: default_model)"
echo "Client must send in the request body:  \"adapters\": \"$ADAPTER\""
exec uv run python -m mlx_lm server \
  --model "$BASE" \
  --adapter-path "$ADAPTER" \
  --port "$PORT"
