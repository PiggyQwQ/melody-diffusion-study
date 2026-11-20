#!/usr/bin/env bash
# Reconstruct one xt file with all nine models (diffusion x3, one-shot argmax,
# one-shot sample x3) on the CPU, four models at a time with four threads each.
# Usage: reconstruct_all.sh <xt.jsonl> <output-dir>
# Writes <output-dir>/recon__<model>.jsonl and .meta.json; logs in <output-dir>/logs.
set -uo pipefail
here="$(dirname "$(readlink -f "$0")")"
source "$here/common.sh"; source "$here/models.sh"; source "$here/parallel.sh"
xt="$(readlink -f "$1")"; out="$(readlink -f "$2")"
mkdir -p "$out/logs"
cd "$REPO"
one() {
  "$PY" training/sample.py --device cpu reconstruct --checkpoint "$(official_checkpoint "$1")" \
    --xt "$xt" --output "$out/recon__$1.jsonl" > "$out/logs/recon__$1.log" 2>&1
  local s=$?; echo "recon $1 exit $s"; return $s
}
for model in "${MODELS[@]}"; do run_limited one "$model"; done
wait_all
echo "reconstruction finished, failures: $FAILED"
