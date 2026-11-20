#!/usr/bin/env bash
# Start one training job inside a detached tmux session.
# Usage: launch.sh <gpu-index | cpu:<threads>> <run-name> <train.py arguments...>
set -euo pipefail
source "$(dirname "$0")/common.sh"
name="$2"
if tmux has-session -t "md-$name" 2>/dev/null; then echo "session exists: md-$name" >&2; exit 1; fi
cmd=$(printf ' %q' bash "$(dirname "$(readlink -f "$0")")/train_job.sh" "$@")
tmux new-session -d -s "md-$name" "$cmd"
echo "started md-$name on $1"
