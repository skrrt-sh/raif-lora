#!/usr/bin/env bash
# OPTIONAL — push this checkout's EXACT data/*.jsonl to the pod, instead of
# regenerating on the box. Run from your LAPTOP (where the local data lives).
#
# The bootstrap regenerates data deterministically (make_data.sh full, seed 0),
# so this is only needed if a bun/prototype version drift on the box produces a
# different split and you want a byte-identical comparison to the MLX smoke.
#
# REQUIRES FULL SSH — a RunPod pod with a public IP + exposed TCP port 22 (the
# `ssh root@<ip> -p <port>` form). Basic proxy SSH (POD_ID@ssh.runpod.io) does
# NOT support scp; on such a pod, send the files with runpodctl instead:
#   laptop:  tar czf data.tgz -C data train.jsonl valid.jsonl eval_holdout.jsonl && runpodctl send data.tgz
#   pod:     cd /workspace/raif/raif-lora && runpodctl receive <code> && tar xzf data.tgz -C data
#
# Usage:  cuda/cloud/upload_data.sh <ssh-host> <ssh-port> [remote-data-dir]
#   e.g.  cuda/cloud/upload_data.sh 1.2.3.4 40022
#
set -euo pipefail

HOST="${1:?ssh host required}"
PORT="${2:?ssh port required}"
REMOTE="${3:-/workspace/raif/raif-lora/data}"

cd "$(dirname "$0")/../.."          # → raif-lora root
[ -s data/train.jsonl ] || { echo "no local data/train.jsonl — generate it first"; exit 1; }

echo "Uploading data/{train,valid,eval_holdout}.jsonl → root@$HOST:$REMOTE"
ssh -p "$PORT" "root@$HOST" "mkdir -p '$REMOTE'"
scp -P "$PORT" data/train.jsonl data/valid.jsonl data/eval_holdout.jsonl "root@$HOST:$REMOTE/"
echo "Done. Re-run the eval on the pod against the same data."
