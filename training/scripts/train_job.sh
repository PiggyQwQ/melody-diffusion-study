#!/usr/bin/env bash
# Run one training job in the foreground.
# Usage: train_job.sh <gpu-index | cpu:<threads>> <run-name> <train.py arguments...>
# The run directory is $RUNS/<run-name>; stdout goes to $RUNS/logs/<run-name>.log
# and ends with a line "EXIT <status>".
set -uo pipefail
source "$(dirname "$0")/common.sh"
target="$1"; name="$2"; shift 2
mkdir -p "$RUNS/logs"
log="$RUNS/logs/$name.log"
if [ -e "$RUNS/$name" ]; then echo "run exists: $RUNS/$name" | tee -a "$log" >&2; echo "EXIT 1" >> "$log"; exit 1; fi
case "$target" in
  cpu:*) threads="${target#cpu:}"
         export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads"
         dev=(--device cpu --threads "$threads") ;;
  *)     export CUDA_VISIBLE_DEVICES="$target"
         dev=(--device cuda) ;;
esac
cd "$REPO"
"$PY" training/train.py "${dev[@]}" --run-dir "$RUNS/$name" "$@" > "$log" 2>&1
status=$?
echo "EXIT $status" >> "$log"
exit $status
