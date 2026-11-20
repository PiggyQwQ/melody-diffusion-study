# Run shell commands with at most $JOBS at a time; source this file.
# Each job uses OMP_NUM_THREADS threads on the CPU. Failures are counted in $FAILED.
JOBS="${JOBS:-4}"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${OMP_NUM_THREADS:-4}"
FAILED=0
run_limited() {
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n || FAILED=$((FAILED + 1)); done
  "$@" &
}
wait_all() {
  while [ "$(jobs -rp | wc -l)" -gt 0 ]; do wait -n || FAILED=$((FAILED + 1)); done
}
