#!/usr/bin/env bash
set -euo pipefail
GEOSPSR_RUNTIME=${GEOSPSR_RUNTIME:-/opt/geospsr-runtime}
export UV_CACHE_DIR="$GEOSPSR_RUNTIME/cache"
export TMPDIR="$GEOSPSR_RUNTIME/tmp"
if [ ! -x "$GEOSPSR_RUNTIME/render-venv/bin/python" ]; then
    uv venv --python "$GEOSPSR_RUNTIME/venv/bin/python" "$GEOSPSR_RUNTIME/render-venv"
fi
uv pip install --python "$GEOSPSR_RUNTIME/render-venv/bin/python" 'bpy==4.3.0' 'numpy==1.26.4'
