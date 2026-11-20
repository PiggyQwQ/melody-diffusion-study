#!/usr/bin/env bash
# CPU smoke run of the full pipeline: 2 training steps, checkpoint, provisional
# xt, reconstruction, generation and trajectory export. Output: $RUNS/smoke-<time>.
set -euo pipefail
source "$(dirname "$0")/common.sh"
cd "$REPO"
out="$RUNS/smoke-$(date +%Y%m%d-%H%M%S)"
export CUDA_VISIBLE_DEVICES=""
"$PY" training/train.py --dataset acfs_full --device cpu --steps 2 --batch-size 8 --train-windows 32 \
  --validation-windows 8 --validate-every 1 --warmup 1 --seed 0 --run-dir "$out/train"
"$PY" training/sample.py --device cpu make-xt --output "$out/xt-provisional-smoke.jsonl" --windows 4
"$PY" training/sample.py --device cpu reconstruct --checkpoint "$out/train/last.pt" \
  --xt "$out/xt-provisional-smoke.jsonl" --output "$out/recon.jsonl"
"$PY" training/sample.py --device cpu generate --checkpoint "$out/train/last.pt" --output "$out/gen.jsonl" --count 4
"$PY" training/sample.py --device cpu trajectory --checkpoint "$out/train/last.pt" --output-dir "$out/trajectory"
find "$out" -type f | sort
