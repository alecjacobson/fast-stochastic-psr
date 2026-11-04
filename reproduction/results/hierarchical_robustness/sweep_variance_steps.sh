#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
result_root=reproduction/results/hierarchical_robustness/extension
for steps in 2 4 8; do
  ./reproduction/run_env.sh -m reproduction.run_hierarchical_scaling \
    --n 196608 --repeats 1 --methods h2-gpu sparse-gpu --timeout 240 \
    --lengthscale .03 --halfwidth .005 --h2-tolerance .0001 --leaf-size 40 \
    --rank 8 --neighbors 24 --variance-iterations "$steps" --sparse-cutoff 11 \
    --order 4 --cross-tolerance 1e-6 --scan front --normal-noise .005 --seed 89 \
    --output "$result_root/pilot/iterations$steps/scaling"
  for method in h2-gpu sparse-gpu; do
    record="$result_root/pilot/iterations$steps/scaling/$method-196608-0.json"
    ./reproduction/run_env.sh -m reproduction.audit_hierarchical_variance "$record" \
      --reference "$result_root/references/196608.json" --random-queries 8
    ./reproduction/run_env.sh -m reproduction.compare_hierarchical_application \
      "$record" "$result_root/tight-field.json" \
      --output "$result_root/pilot/iterations$steps/application-$method.json"
  done
done
