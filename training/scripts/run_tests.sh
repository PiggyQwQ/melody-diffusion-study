#!/usr/bin/env bash
# Run the training test suite on the CPU and print the environment.
set -euo pipefail
source "$(dirname "$0")/common.sh"
cd "$REPO"
git log -1 --format='commit %H'
"$PY" -c "import torch, numpy, mido, pytest, sys; print('python', sys.version.split()[0], 'torch', torch.__version__, 'cuda', torch.version.cuda, 'numpy', numpy.__version__, 'mido', mido.__version__, 'pytest', pytest.__version__)"
CUDA_VISIBLE_DEVICES="" "$PY" -m pytest -q -p no:cacheprovider training/tests "$@"
