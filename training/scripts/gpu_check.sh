#!/usr/bin/env bash
# Sample GPU state three times, 20 s apart. A GPU counts as idle only if every
# sample shows < 500 MiB used, < 5 % utilisation and no compute process.
# Prints the samples and a final line "IDLE_GPUS=<comma list>".
set -euo pipefail

samples=3
interval=20
declare -A busy=()
indices=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' ')

for n in $(seq 1 "$samples"); do
  echo "sample $n $(date '+%F %T')"
  nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader,nounits |
    while IFS=', ' read -r idx uuid mem util; do echo "  gpu $idx mem ${mem}MiB util ${util}%"; done
  apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null || true)
  while IFS=', ' read -r idx uuid mem util; do
    procs=$(printf '%s\n' "$apps" | grep -c "$uuid" || true)
    if [ "$mem" -ge 500 ] || [ "$util" -ge 5 ] || [ "$procs" -gt 0 ]; then
      busy[$idx]=1
    fi
  done < <(nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)
  [ "$n" -lt "$samples" ] && sleep "$interval"
done

idle=()
for idx in $indices; do
  [ -z "${busy[$idx]:-}" ] && idle+=("$idx")
done
echo "IDLE_GPUS=$(IFS=,; echo "${idle[*]:-}")"
