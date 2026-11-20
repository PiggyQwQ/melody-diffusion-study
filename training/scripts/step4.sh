#!/usr/bin/env bash
# Step 4 on the CPU: provisional xt from the ACFS validation split, reconstruction
# and one-shot ablation with all nine models, unconditional generation with the
# three H2 models, and the trajectories of the H2 seed-0 model.
# Output: $RUNS/step4/{xt,recon-provisional,generate,trajectory}
set -uo pipefail
here="$(dirname "$(readlink -f "$0")")"
source "$here/common.sh"; source "$here/models.sh"; source "$here/parallel.sh"
out="$RUNS/step4"
mkdir -p "$out/xt" "$out/generate/logs"
cd "$REPO"
xt="$out/xt/xt-provisional-validation.jsonl"
[ -e "$xt" ] || "$PY" training/sample.py --device cpu make-xt --output "$xt"
bash "$here/reconstruct_all.sh" "$xt" "$out/recon-provisional"
gen() {
  "$PY" training/sample.py --device cpu generate --checkpoint "$(official_checkpoint "$1")" \
    --output "$out/generate/generate__$1.jsonl" > "$out/generate/logs/generate__$1.log" 2>&1
  local s=$?; echo "generate $1 exit $s"; return $s
}
traj() {
  "$PY" training/sample.py --device cpu trajectory --checkpoint "$(official_checkpoint h2-acfs_full-s0)" \
    --output-dir "$out/trajectory" > "$out/trajectory.log" 2>&1
  local s=$?; echo "trajectory exit $s"; return $s
}
for model in "${H2_MODELS[@]}"; do run_limited gen "$model"; done
run_limited traj
wait_all
echo "step 4 finished, failures: $FAILED"
