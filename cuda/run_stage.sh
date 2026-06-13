#!/usr/bin/env bash
# One rung of the smoke→warm→full ladder, end to end, with the gates from
# ../ITERATION_PLAN.md enforced. This is the "ultimate" CUDA driver: it refuses
# to spend GPU time on a stage until the meter and the data are trustworthy, and
# refuses to advertise the next stage until this one's gate has actually passed.
#
#   bash cuda/run_stage.sh smoke
#   bash cuda/run_stage.sh warm
#   bash cuda/run_stage.sh full        # add EXTRA args after the stage, e.g.:
#   bash cuda/run_stage.sh full --micro-batch 2
#
# Steps (each fails the whole run on error — set -e):
#   1. meter   — src/test_eval_smoke.py   (4 oracle tests; the score must be honest)
#   2. data    — src/check_data.py        (containment / leakage / stratification)
#   3. train   — cuda/train_unsloth.py    (saves adapter + run_meta.json + .tgz)
#   4. eval    — cuda/eval_cuda.py --gate (parse/fidelity vs this stage's gate)
#
# Env knobs:  PY (python to use, default /usr/local/bin/python or `python`),
#             N  (eval samples per group, default 13).
set -euo pipefail

cd "$(dirname "$0")/.."          # repo root (raif-lora)

stage="${1:-}"
shift || true
case "$stage" in
  smoke|warm|full) ;;
  *) echo "usage: $0 {smoke|warm|full} [extra train args...]" >&2; exit 2 ;;
esac

PY="${PY:-$(command -v python3 || command -v python)}"
N="${N:-13}"
adapter="./adapters-cuda/${stage}"

echo "== [1/4] meter (oracle tests) =="
"$PY" src/test_eval_smoke.py

echo "== [2/4] data hygiene =="
"$PY" src/check_data.py

echo "== [3/4] train stage=${stage} =="
"$PY" cuda/train_unsloth.py --stage "$stage" --export-tar "$@"

echo "== [4/4] eval + gate =="
# eval_cuda exits nonzero when the gate FAILs (set -e turns that into a stop).
"$PY" cuda/eval_cuda.py --adapter "$adapter" --n "$N" \
  --gate "$stage" --out "${adapter}/eval.json"

echo
echo "✓ stage '${stage}' complete and gate PASSED."
echo "  adapter:   ${adapter}        (tarball: ${adapter}.tgz)"
echo "  results:   ${adapter}/eval.json   meta: ${adapter}/run_meta.json"
case "$stage" in
  smoke) echo "  next:      bash cuda/run_stage.sh warm" ;;
  warm)  echo "  next:      regen data @ ~300 var, then: bash cuda/run_stage.sh full" ;;
  full)  echo "  next:      acceptance gate met — pull ${adapter}.tgz off the pod and ship." ;;
esac
