#!/usr/bin/env bash
# Autonomous train → eval → save-to-HF → (verified) self-destruct pipeline for the
# Qwen3-4B RAIF LoRA. Runs detached on the RunPod pod. The pod is TERMINATED only
# after the adapter + eval are confirmed present on the Hub — so a failed/partial
# save never destroys the only copy. The RunPod API key is read from pid 1's env
# at runtime (never hardcoded). Status is written to $LOG/STATUS at each gate.
set -uo pipefail
cd /workspace/raif/raif-lora
export PATH="$HOME/.bun/bin:$PATH" HF_HOME="/workspace/.cache/huggingface/"

OUT=adapters-cuda/qwen3-4b-full
LOG=logs/pod-qwen3-4b-full
REPO=skrrt-sh/raif-qwen3-4b-lora
BASE=unsloth/Qwen3-4B-Instruct-2507
mkdir -p "$LOG"
echo "STARTED $(date -u +%FT%TZ)" > "$LOG/STATUS"

# 1. Train (reduced: 6k iters / ~1.3 epochs, dropout 0 keeps fused kernels).
python cuda/train_unsloth.py --stage full --model "$BASE" --data ./data-tbl \
  --iters 6000 --lr 1e-4 --lora-dropout 0.0 --out "$OUT" --export-tar \
  > "$LOG/train.log" 2>&1
if [ $? -ne 0 ] || [ ! -f "$OUT/adapter_model.safetensors" ]; then
  echo "TRAIN_FAILED — pod kept up" >> "$LOG/STATUS"; exit 1
fi

# 2. Eval against the full gate (a gate FAIL is fine — we still save & report).
python cuda/eval_cuda.py --adapter "$OUT" --n 64 --gate full \
  --valid ./data-tbl/valid.jsonl --holdout ./data-tbl/eval_holdout.jsonl \
  --out "$OUT/eval.json" > "$LOG/eval.log" 2>&1
if [ ! -f "$OUT/eval.json" ]; then
  echo "EVAL_FAILED (no eval.json) — pod kept up for debugging" >> "$LOG/STATUS"; exit 1
fi

# 3. Stage clean artifacts into the adapter dir (drop intermediate checkpoints).
rm -rf "$OUT"/checkpoint-* "$OUT"/README.md
cp assets/banner.jpg "$OUT/banner.jpg" 2>/dev/null
cp "$LOG/train.log" "$OUT/train.log" 2>/dev/null
cp "$LOG/eval.log" "$OUT/eval.log" 2>/dev/null

# 4. Push to HF (private). Apache-2.0 card auto-generated from run_meta + eval.
python cuda/push_to_hub.py --adapter "$OUT" --repo "$REPO" \
  --base-model "$BASE" --private > "$LOG/push.log" 2>&1
PUSH_RC=$?

# 5. Independently VERIFY the upload before any teardown.
python - "$REPO" > "$LOG/verify.log" 2>&1 <<'PY'
import sys
from huggingface_hub import HfApi
repo = sys.argv[1]
files = set(HfApi().list_repo_files(repo))
need = {"adapter_model.safetensors", "adapter_config.json", "eval.json", "run_meta.json"}
missing = need - files
print("FILES:", sorted(files))
if missing:
    print("VERIFY_FAIL missing:", missing); sys.exit(1)
print("VERIFY_OK"); sys.exit(0)
PY
VERIFY_RC=$?

if [ $PUSH_RC -ne 0 ] || [ $VERIFY_RC -ne 0 ]; then
  echo "SAVE_FAILED push_rc=$PUSH_RC verify_rc=$VERIFY_RC — POD KEPT UP, artifacts in $OUT" >> "$LOG/STATUS"
  exit 1
fi
echo "ARTIFACTS_SAVED https://huggingface.co/$REPO (private) $(date -u +%FT%TZ)" >> "$LOG/STATUS"

# 6. Verified saved → self-destruct, IFF runpodctl is authed with a real account
#    key. The pod-injected RUNPOD_API_KEY is scoped down (can't manage pods), so we
#    do NOT use it; we rely on whatever the user configured via `runpodctl config
#    --apiKey <key>`. If that's missing/unauthorized, we SKIP teardown and leave the
#    pod up (artifacts are already safe on HF) rather than fail silently.
POD=$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^RUNPOD_POD_ID=//p')
if runpodctl get pod "$POD" >/dev/null 2>&1; then
  echo "SELF_DESTRUCT pod=$POD $(date -u +%FT%TZ)" >> "$LOG/STATUS"
  runpodctl remove pod "$POD" >> "$LOG/STATUS" 2>&1
else
  echo "SELF_DESTRUCT_SKIPPED runpodctl not authed (run: runpodctl config --apiKey <key>). Pod left UP; artifacts safe on HF." >> "$LOG/STATUS"
fi
