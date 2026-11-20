#!/usr/bin/env bash
# Step 3: the nine frozen training runs (2 layers x 128) plus a 2x128 overfitting
# record, on the CPU with four jobs of four threads each.
#   slots 1-3: H2 acfs_full seeds 0, 1, 2 (20,000 updates, best validation checkpoint)
#   slot 4:    overfitting record, then H3 acfs_matched and cantopop seeds 0, 1, 2
#              (cosine schedule of 10,000 updates stopped at 2,000; the step-2000
#              checkpoint last.pt is the official one)
set -euo pipefail
here="$(dirname "$(readlink -f "$0")")"
NET=(--layers 2 --width 128)
for seed in 0 1 2; do
  bash "$here/launch.sh" cpu:4 "h2-acfs_full-s$seed" --dataset acfs_full --steps 20000 \
    --validate-every 1000 --seed "$seed" "${NET[@]}"
done
queue="bash $here/train_job.sh cpu:4 s3-overfit-acfs_full-64win-L2-W128-20k --dataset acfs_full \
--train-windows 64 --batch-size 64 --steps 20000 --dropout 0 --validate-every 2000 --train-eval-windows 64 \
--memorizer-floor --seed 0 ${NET[*]}"
for dataset in acfs_matched cantopop; do
  for seed in 0 1 2; do
    queue="$queue; bash $here/train_job.sh cpu:4 h3-$dataset-s$seed --dataset $dataset --steps 2000 \
--schedule-steps 10000 --validate-every 250 --seed $seed ${NET[*]}"
  done
done
tmux new-session -d -s md-step3-queue "$queue"
echo "started md-step3-queue"
