#!/usr/bin/env bash
# CPU smoke run of the whole pipeline: 2 training steps, a small validation xt,
# reconstruction with one repository checkpoint, generation and trajectory export.
# Output: $RUNS/smoke-<time>.
set -euo pipefail
source "$(dirname "$0")/common.sh"
cd "$REPO"
out="$RUNS/smoke-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$out"
export CUDA_VISIBLE_DEVICES=""
"$PY" training/train.py --dataset acfs_full --device cpu --steps 2 --batch-size 8 --train-windows 32 \
  --validation-windows 8 --validate-every 1 --warmup 1 --seed 0 --run-dir "$out/train"
head -n 2 data/acfs-v261103/split-encode-v1/validation.jsonl | "$PY" -c \
  "import sys, json; [print(json.loads(line)['window_id']) for line in sys.stdin]" > "$out/ids.txt"
"$PY" training/make_xt.py --split validation --window-ids "$out/ids.txt" --times 10,90 --noise-seeds 0 \
  --out "$out/xt-smoke.jsonl"
"$PY" training/reconstruct.py --xt "$out/xt-smoke.jsonl" --out "$out/recon" --runs h2-acfs_full-s0 --device cpu
"$PY" training/sample.py --device cpu generate --checkpoint "$out/train/last.pt" --output "$out/gen.jsonl" --count 4
"$PY" training/sample.py --device cpu trajectory --checkpoint "$out/train/last.pt" --output-dir "$out/trajectory"
find "$out" -type f | sort | head -n 30
