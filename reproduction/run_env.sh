#!/usr/bin/env bash
set -euo pipefail
GEOSPSR_RUNTIME=${GEOSPSR_RUNTIME:-/opt/geospsr-runtime}
export TMPDIR=${TMPDIR:-$GEOSPSR_RUNTIME/tmp}
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$GEOSPSR_RUNTIME/cuda-cache}"
mkdir -p "$CUDA_CACHE_PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda
export JAX_DEFAULT_MATMUL_PRECISION=highest
export MPLBACKEND=Agg
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)${PYTHONPATH:+:$PYTHONPATH}"
export GEOSPSR_RUNTIME
exec "$GEOSPSR_RUNTIME/venv/bin/python" "$@"
