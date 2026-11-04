#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../../.."

# Build optional pinned backend first; see ../README.md and ../METHOD.md.
# ./reproduction/run_env.sh reproduction/setup_higp.py
result_root=reproduction/results/hierarchical_variance/short
settings=(--lengthscale .03 --halfwidth .005 --h2-tolerance .001
          --leaf-size 40 --rank 8 --neighbors 24)
./reproduction/run_env.sh -m reproduction.run_hierarchical_scaling \
  "${settings[@]}" --n 8192 16384 32768 65536 73728 77824 \
  --output "$result_root/scaling"
./reproduction/run_env.sh -m reproduction.run_hierarchical_scaling \
  "${settings[@]}" --n 131072 --methods h2-gpu --variance-iterations 12 \
  --output "$result_root/large"
./reproduction/run_env.sh -m reproduction.run_hierarchical_scaling \
  "${settings[@]}" --n 32768 73728 --methods cg-dense-float32 \
  --variance-refinements 0 --output "$result_root/controls"
./reproduction/run_env.sh -m reproduction.run_hierarchical_scaling \
  "${settings[@]}" --n 131072 --methods cg-dense-float32 --repeats 1 \
  --variance-iterations 12 --variance-refinements 0 --output "$result_root/controls"
./reproduction/run_env.sh -m reproduction.audit_hierarchical_scaling --root "$result_root"
for n in 8192 32768 77824; do
  ./reproduction/run_env.sh -m reproduction.benchmark_hierarchical_sdd \
    --reference "$result_root/references/$n.json" --output "$result_root/sdd/$n.json" \
    --eta .1 .15 --checkpoints 2048 8192 16384 32768 49152 65536 98304 131072 --budget 60
done
./reproduction/run_env.sh -m reproduction.benchmark_hierarchical_sdd \
  --reference "$result_root/references/131072.json" --output "$result_root/sdd/131072.json" \
  --eta .15 --checkpoints 32768 65536 98304 131072 196608 262144 --budget 180
./reproduction/run_env.sh -m reproduction.compare_hierarchical_application \
  "$result_root/scaling/h2-gpu-77824-0.json" "$result_root/scaling/dense-float32-77824-0.json" \
  --output "$result_root/application.json"
./reproduction/run_env.sh -m reproduction.compare_hierarchical_application \
  "$result_root/sdd/77824-eta0.15-passing.json" "$result_root/scaling/dense-float32-77824-0.json" \
  --output "$result_root/application_sdd.json"
./reproduction/run_env.sh -m reproduction.compare_dense_scan_geometry \
  "$result_root/scaling/h2-gpu-77824-0.json" --output "$result_root/geometry"
./reproduction/run_env.sh -m reproduction.summarize_hierarchical_variance --root "$result_root"
./reproduction/run_env.sh -m pytest tests/test_nonperiodic.py tests/test_hierarchical.py -q
