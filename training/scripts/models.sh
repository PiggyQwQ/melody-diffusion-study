# The nine frozen models and their official checkpoints; source this file.
# H2: validation-best checkpoint; H3: the checkpoint at the fixed update 2000 (last.pt).
MODELS=(h2-acfs_full-s0 h2-acfs_full-s1 h2-acfs_full-s2
        h3-acfs_matched-s0 h3-acfs_matched-s1 h3-acfs_matched-s2
        h3-cantopop-s0 h3-cantopop-s1 h3-cantopop-s2)
H2_MODELS=(h2-acfs_full-s0 h2-acfs_full-s1 h2-acfs_full-s2)
official_checkpoint() {
  case "$1" in
    h2-*) echo "$RUNS/$1/best.pt" ;;
    h3-*) echo "$RUNS/$1/last.pt" ;;
  esac
}
