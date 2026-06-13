#!/usr/bin/env bash
# Thin wrapper around the prototype's bun dataset generator.
# Presets: `smoke` (30/shape quick run), `warm` (100/shape, the 15-min warm
# run), `full` (500/shape, the v0.5 acceptance run).
#
# Usage: make_data.sh {smoke|warm|full} [holdout_shapes_csv|none]
#
# The second arg overrides the held-out shape list (plan §3.4). Held-out
# shapes are written ONLY to data/eval_holdout.jsonl, never to train.jsonl
# or valid.jsonl. Pass `none` to disable holdout entirely.

set -euo pipefail

cd "$(dirname "$0")/.."
data_dir="$(pwd)/data"
proto="../raif-standard/prototype"

mode="${1:-smoke}"
holdout="${2:-multiline_body,pathological_keys,large_table,deep_array_literal,flat_inline_object}"

case "$mode" in
  smoke)
    variations=30
    ;;
  warm)
    variations=100
    ;;
  full)
    variations=500
    ;;
  *)
    echo "usage: $0 {smoke|warm|full} [holdout_shapes_csv|none]" >&2
    exit 1
    ;;
esac
schema=0.7
adversarial=0.5

cd "$proto"
bun run src/dataset.ts \
  --variations "$variations" \
  --eval-frac 0.05 \
  --schema-frac "$schema" \
  --adversarial-frac "$adversarial" \
  --seed 0 \
  --holdout-shapes "$holdout" \
  --out-train "$data_dir/train.jsonl" \
  --out-eval  "$data_dir/valid.jsonl" \
  --out-holdout "$data_dir/eval_holdout.jsonl"
