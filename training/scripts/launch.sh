#!/usr/bin/env bash
# Start one training job on one GPU inside a detached tmux session.
# Usage: launch.sh <gpu-index> <run-name> <train.py arguments...>
# The run directory is $RUNS/<run-name>; stdout goes to $RUNS/logs/<run-name>.log.
set -euo pipefail
source "$(dirname "$0")/common.sh"
gpu="$1"; name="$2"; shift 2
mkdir -p "$RUNS/logs"
if [ -e "$RUNS/$name" ]; then echo "run exists: $RUNS/$name" >&2; exit 1; fi
if tmux has-session -t "md-$name" 2>/dev/null; then echo "session exists: md-$name" >&2; exit 1; fi
args=$(printf ' %q' "$@")
tmux new-session -d -s "md-$name" \
  "cd $REPO && CUDA_VISIBLE_DEVICES=$gpu $PY training/train.py --device cuda --run-dir $RUNS/$name$args > $RUNS/logs/$name.log 2>&1; echo EXIT \$? >> $RUNS/logs/$name.log"
echo "started md-$name on GPU $gpu"
