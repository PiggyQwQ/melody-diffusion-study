#!/usr/bin/env bash
# Show this project's tmux sessions, the last log line of each run and our GPU processes.
set -uo pipefail
source "$(dirname "$0")/common.sh"
echo "sessions:"; tmux ls 2>/dev/null | grep '^md-' || echo "  none"
for log in "$RUNS"/logs/*.log; do
  [ -e "$log" ] || continue
  echo "$(basename "$log" .log): $(tail -n 1 "$log" | cut -c1-200)"
done
echo "gpu processes started from $WORK:"
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
  if tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q "$WORK"; then
    echo "  pid $pid $(nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader | grep "^$pid,")"
  fi
done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
