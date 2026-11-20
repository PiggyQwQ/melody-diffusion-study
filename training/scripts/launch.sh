#!/usr/bin/env bash
# Start one training job inside a detached tmux session.
# Usage: launch.sh <gpu-index | cpu:<threads>> <run-name> <train.py arguments...>
# The run directory is $RUNS/<run-name>; stdout goes to $RUNS/logs/<run-name>.log.
set -euo pipefail
source "$(dirname "$0")/common.sh"
target="$1"; name="$2"; shift 2
mkdir -p "$RUNS/logs"
if [ -e "$RUNS/$name" ]; then echo "run exists: $RUNS/$name" >&2; exit 1; fi
if tmux has-session -t "md-$name" 2>/dev/null; then echo "session exists: md-$name" >&2; exit 1; fi
case "$target" in
  cpu:*) threads="${target#cpu:}"
         env="CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=$threads MKL_NUM_THREADS=$threads"
         dev="--device cpu --threads $threads" ;;
  *)     env="CUDA_VISIBLE_DEVICES=$target"
         dev="--device cuda" ;;
esac
args=$(printf ' %q' "$@")
tmux new-session -d -s "md-$name" \
  "cd $REPO && $env $PY training/train.py $dev --run-dir $RUNS/$name$args > $RUNS/logs/$name.log 2>&1; echo EXIT \$? >> $RUNS/logs/$name.log"
echo "started md-$name on $target"
