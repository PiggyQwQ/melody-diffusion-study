#!/usr/bin/env bash
# Provisional outputs on the CPU (validation split only):
#   1. 100 validation windows chosen with random.Random(20261120), in file order;
#   2. xt with make_xt.py: t = 10, 25, 50, 75, 90, noise seeds 0 and 1 (1,000 cases);
#   3. reconstruction with all nine repository checkpoints;
#   4. 100 unconditional samples per H2 model and trajectories of the H2 seed-0 model;
#   5. nearest-neighbour reference of 300 validation windows (random.Random(20261124)).
# Usage: step4.sh <output-dir>   (the directory must not exist)
set -euo pipefail
source "$(dirname "$0")/common.sh"
out="$1"
if [ -e "$out" ]; then echo "exists: $out" >&2; exit 1; fi
mkdir -p "$out/provisional" "$out/generate"
cd "$REPO"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4
"$PY" training/scripts/select_windows.py --split validation --count 100 --seed 20261120 \
  --out "$out/provisional/window_ids_provisional.txt"
"$PY" training/make_xt.py --split validation --window-ids "$out/provisional/window_ids_provisional.txt" \
  --times 10,25,50,75,90 --noise-seeds 0,1 --out "$out/provisional/xt-provisional-validation.jsonl"
"$PY" training/reconstruct.py --xt "$out/provisional/xt-provisional-validation.jsonl" \
  --out "$out/provisional/recon" --runs all --device cpu --threads 4
for seed in 0 1 2; do
  "$PY" training/sample.py --device cpu generate --checkpoint "training/checkpoints/h2-acfs_full-s$seed.pt" \
    --output "$out/generate/generate__h2-acfs_full-s$seed.jsonl"
done
"$PY" training/sample.py --device cpu trajectory --checkpoint training/checkpoints/h2-acfs_full-s0.pt \
  --output-dir "$out/trajectory"
"$PY" training/sample.py --device cpu nn-reference --generated "$out"/generate/generate__*.jsonl \
  --output-dir "$out/nearest_neighbors" --count 300 --seed 20261124
