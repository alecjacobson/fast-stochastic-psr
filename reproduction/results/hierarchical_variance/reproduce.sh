#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."

# First install the optional runtime dependencies and build the pinned backend:
# ./reproduction/run_env.sh reproduction/setup_higp.py
# See README.md in this directory. Each benchmark is an isolated process.
./reproduction/run_env.sh -m reproduction.run_hierarchical_scaling
./reproduction/run_env.sh -m reproduction.run_hierarchical_scaling \
  --n 131072 262144 --methods h2-gpu --variance-iterations 12 \
  --output reproduction/results/hierarchical_variance/large
./reproduction/run_env.sh -m reproduction.run_hierarchical_scaling \
  --n 8192 32768 73728 --methods cg-dense-float32 \
  --output reproduction/results/hierarchical_variance/controls
./reproduction/run_env.sh -m reproduction.run_hierarchical_scaling \
  --n 131072 --methods cg-dense-float32 --repeats 1 --variance-iterations 12 \
  --output reproduction/results/hierarchical_variance/controls
./reproduction/run_env.sh -m reproduction.audit_hierarchical_scaling
./reproduction/run_env.sh -m reproduction.compare_hierarchical_application \
  reproduction/results/hierarchical_variance/scaling/h2-gpu-73728-0.json \
  reproduction/results/hierarchical_variance/scaling/dense-float32-73728-0.json \
  --output reproduction/results/hierarchical_variance/application.json
for n in 8192 73728; do
  ./reproduction/run_env.sh -m reproduction.benchmark_hierarchical_sdd \
    --reference "reproduction/results/hierarchical_variance/references/$n.json" \
    --output "reproduction/results/hierarchical_variance/sdd/$n.json" \
    --eta .01 .03 --checkpoints 2048 8192 16384 32768 --budget 60
done
./reproduction/run_env.sh -m reproduction.compare_nonperiodic_geometry
./reproduction/run_env.sh -m pytest tests/test_nonperiodic.py tests/test_hierarchical.py -q
./reproduction/run_env.sh -m reproduction.summarize_hierarchical_variance
