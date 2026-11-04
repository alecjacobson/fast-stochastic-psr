#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
PYTHON=${PYTHON:-/opt/geospsr-runtime/venv/bin/python}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
"$PYTHON" -m pytest tests/test_nonperiodic.py -q
"$PYTHON" -m reproduction.nonperiodic_pilot --original
"$PYTHON" -m reproduction.nonperiodic_pilot --output reproduction/results/nonperiodic/scaling --n 2048 4096
"$PYTHON" -m reproduction.nonperiodic_pilot --output reproduction/results/nonperiodic/local --near-halfwidth .1
"$PYTHON" -m reproduction.nonperiodic_pilot --output reproduction/results/nonperiodic/stress --near-halfwidth .1 --lengthscale .15 --noise-variance .0001 --seed 416
"$PYTHON" -m reproduction.plot_nonperiodic_pilot
