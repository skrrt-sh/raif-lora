#!/usr/bin/env bash
# Augmented dataset = synthetic + real.
#
#   synthetic (dataset.ts)        → mechanism coverage (multiline blocks,
#                                    pathological keys, nested arrays) + the
#                                    held-out shape-generalization probe.
#   real (Glaive FC v2, Apache-2.0) → distributional realism + field-name
#                                    diversity, from actual tool-call arg objects.
#
# The holdout file stays SYNTHETIC-ONLY — it tests shape generalization, which
# real data (no controlled shapes) can't measure. Real examples go to train +
# a small stratified valid slice only.
#
# One-time: download the dataset once from the LFS CDN (no rate limit):
#   curl -sL https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2/resolve/main/glaive-function-calling-v2.json \
#     -o /tmp/glaive_full.json
#
# Usage:  bash src/make_data_augmented.sh [smoke|warm|full] [glaive_json_path]
#         REAL_MAX=2285 controls the real:synthetic ratio (default ≈ 60/40 vs warm).
set -euo pipefail

cd "$(dirname "$0")/.."
data_dir="$(pwd)/data"
proto="../raif-standard/prototype"
mode="${1:-warm}"
glaive_file="${2:-/tmp/glaive_full.json}"
real_max="${REAL_MAX:-2285}"

if [[ ! -f "$glaive_file" ]]; then
  echo "✗ missing $glaive_file — download it first:" >&2
  echo "  curl -sL https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2/resolve/main/glaive-function-calling-v2.json -o $glaive_file" >&2
  exit 1
fi

echo "== synthetic ($mode) =="
bash src/make_data.sh "$mode"

# Match the real valid slice to the synthetic per-shape valid count, so
# check_data's ±1 stratification stays balanced across ALL shapes (the count
# differs by preset: warm→5, full→25).
syn_valid=$(wc -l < "$data_dir/valid.jsonl")
syn_shapes=$(python3 -c "import json;print(len({json.loads(l)['meta']['shape'] for l in open('$data_dir/valid.jsonl')}))")
per_shape=$(( syn_valid / syn_shapes ))
echo "  (synthetic valid: $syn_valid over $syn_shapes shapes → real valid-n=$per_shape)"

echo "== real (Glaive FC v2, max $real_max [0=all unique]) =="
( cd "$proto" && bun run src/ingest_glaive.ts --file "$glaive_file" --max "$real_max" \
    --valid-n "$per_shape" \
    --out-train "$data_dir/real_train.jsonl" --out-valid "$data_dir/real_valid.jsonl" )

echo "== merge (real → train + valid; holdout untouched) =="
cat "$data_dir/real_train.jsonl" >> "$data_dir/train.jsonl"
cat "$data_dir/real_valid.jsonl" >> "$data_dir/valid.jsonl"
rm -f "$data_dir/real_train.jsonl" "$data_dir/real_valid.jsonl"

echo "✓ augmented: train=$(wc -l < "$data_dir/train.jsonl"), valid=$(wc -l < "$data_dir/valid.jsonl"), holdout=$(wc -l < "$data_dir/eval_holdout.jsonl")"
echo "  verify:  uv run python src/check_data.py"
