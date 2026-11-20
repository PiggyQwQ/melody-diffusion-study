#!/usr/bin/env bash
# Run the training test suite on the CPU and print the environment.
set -euo pipefail
source "$(dirname "$0")/common.sh"
cd "$REPO"
git log -1 --format='commit %H'
"$PY" -c "import sys, torch; from importlib.metadata import version as v; print('python', sys.version.split()[0], 'torch', torch.__version__, 'cuda', torch.version.cuda, *[f'{m} {v(m)}' for m in ('numpy', 'mido', 'pytest')])"
CUDA_VISIBLE_DEVICES="" "$PY" -m pytest -q -p no:cacheprovider training/tests "$@"
