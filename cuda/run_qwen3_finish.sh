#!/usr/bin/env bash
# Finish the Qwen3-4B run whose TRAINING already completed (adapter saved) but
# whose eval crashed (missing raif_bun/bun on the fresh pod). Re-runs ONLY the
# tail: eval → save-to-HF → verify → self-destruct. No retraining. Mirrors steps
# 2–6 of run_qwen3_pipeline.sh.
set -uo pipefail
cd /workspace/raif/raif-lora || { echo "cd failed" >&2; exit 1; }
export PATH="$HOME/.bun/bin:$PATH" HF_HOME="/workspace/.cache/huggingface/"

OUT=adapters-cuda/qwen3-4b-full
LOG=logs/pod-qwen3-4b-full
REPO=skrrt-sh/raif-qwen3-4b-lora
BASE=unsloth/Qwen3-4B-Instruct-2507
mkdir -p "$LOG"
echo "FINISH_STARTED $(date -u +%FT%TZ)" >> "$LOG/STATUS"

[ -f "$OUT/adapter_model.safetensors" ] || { echo "NO ADAPTER — abort" >> "$LOG/STATUS"; exit 1; }

# 2. Eval against the full gate.
python cuda/eval_cuda.py --adapter "$OUT" --n 64 --gate full \
  --valid ./data-tbl/valid.jsonl --holdout ./data-tbl/eval_holdout.jsonl \
  --out "$OUT/eval.json" > "$LOG/eval.log" 2>&1
EVAL_RC=$?
[ $EVAL_RC -eq 0 ] || { echo "EVAL_FAILED rc=$EVAL_RC — pod kept up" >> "$LOG/STATUS"; exit 1; }
[ -f "$OUT/eval.json" ] || { echo "EVAL_FAILED (no eval.json) — pod kept up" >> "$LOG/STATUS"; exit 1; }

# 3. Stage clean artifacts.
rm -rf "$OUT"/checkpoint-* "$OUT"/README.md
cp assets/banner.jpg "$OUT/banner.jpg" 2>/dev/null
cp "$LOG/train.log" "$OUT/train.log" 2>/dev/null
cp "$LOG/eval.log" "$OUT/eval.log" 2>/dev/null

# 4. Push to HF (private), Apache-2.0 card from run_meta + eval.
python cuda/push_to_hub.py --adapter "$OUT" --repo "$REPO" \
  --base-model "$BASE" --private > "$LOG/push.log" 2>&1
PUSH_RC=$?

# 5. Verify upload before teardown.
python - "$REPO" > "$LOG/verify.log" 2>&1 <<'PY'
import sys
from huggingface_hub import HfApi
files = set(HfApi().list_repo_files(sys.argv[1]))
need = {"adapter_model.safetensors","adapter_config.json","eval.json","run_meta.json"}
missing = need - files
print("FILES:", sorted(files))
sys.exit(1 if missing else 0) if not missing else (print("MISSING", missing) or sys.exit(1))
PY
VERIFY_RC=$?

if [ $PUSH_RC -ne 0 ] || [ $VERIFY_RC -ne 0 ]; then
  echo "SAVE_FAILED push_rc=$PUSH_RC verify_rc=$VERIFY_RC — POD KEPT UP" >> "$LOG/STATUS"
  exit 1
fi
echo "ARTIFACTS_SAVED https://huggingface.co/$REPO (private) $(date -u +%FT%TZ)" >> "$LOG/STATUS"

# 6. Verified → self-destruct iff runpodctl is authed (skip otherwise; HF is safe).
POD=$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^RUNPOD_POD_ID=//p')
if runpodctl get pod "$POD" >/dev/null 2>&1; then
  echo "SELF_DESTRUCT pod=$POD $(date -u +%FT%TZ)" >> "$LOG/STATUS"
  runpodctl remove pod "$POD" >> "$LOG/STATUS" 2>&1
else
  echo "SELF_DESTRUCT_SKIPPED runpodctl not authed — pod UP, artifacts safe on HF." >> "$LOG/STATUS"
fi
