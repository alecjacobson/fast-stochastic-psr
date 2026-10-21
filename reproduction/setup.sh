#!/usr/bin/env bash
set -euo pipefail
# Override to any filesystem with ~7 GB available. Current VM home is nearly full.
GEOSPSR_RUNTIME=${GEOSPSR_RUNTIME:-/opt/geospsr-runtime}
mkdir -p "$GEOSPSR_RUNTIME/tmp" "$GEOSPSR_RUNTIME/cache"
export TMPDIR="$GEOSPSR_RUNTIME/tmp"
export UV_CACHE_DIR="$GEOSPSR_RUNTIME/cache"
export UV_PYTHON_INSTALL_DIR="$GEOSPSR_RUNTIME/python"
if [ ! -x "$GEOSPSR_RUNTIME/venv/bin/python" ]; then
    uv venv --python 3.11 "$GEOSPSR_RUNTIME/venv"
fi
uv pip install --python "$GEOSPSR_RUNTIME/venv/bin/python" -r reproduction/requirements-lock.txt
if [ ! -e "$GEOSPSR_RUNTIME/upstream" ]; then
    git worktree add --detach "$GEOSPSR_RUNTIME/upstream" 2352f1ef98aab20e0303611c1fa2b0948acd2a05
fi
