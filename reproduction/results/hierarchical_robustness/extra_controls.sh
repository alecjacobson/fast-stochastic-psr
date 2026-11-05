#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
result_root=reproduction/results/hierarchical_robustness
for seed in 89 144 233; do
  ./reproduction/run_env.sh -m reproduction.run_hierarchical_scaling \
    --n 16384 32768 77824 --repeats 1 --methods sparse-gpu \
    --lengthscale .03 --halfwidth .005 --rank 8 --neighbors 24 \
    --variance-iterations 8 --sparse-cutoff 11 --order 4 --cross-tolerance 1e-6 \
    --scan front --normal-noise .005 --seed "$seed" \
    --output "$result_root/partial/seed$seed/scaling"
  ./reproduction/run_env.sh -m reproduction.audit_hierarchical_scaling --root "$result_root/partial/seed$seed"
  ./reproduction/run_env.sh -m reproduction.compare_hierarchical_application \
    "$result_root/partial/seed$seed/scaling/sparse-gpu-77824-0.json" \
    "$result_root/partial/seed$seed/scaling/dense-float32-77824-0.json" \
    --output "$result_root/partial/seed$seed/application-sparse-77824.json"
  candidate="$(./reproduction/run_env.sh - "$result_root/partial/seed$seed/sdd/77824.json" <<'PY'
import json, sys
a = json.load(open(sys.argv[1]))
passing = [(row['total_seconds'], c['passing_workload']) for c in a['configurations']
           for row in c['history'] if row.get('passes') and 'total_seconds' in row]
print(min(passing)[1] if passing else '')
PY
  )"
  if [[ -n "$candidate" ]]; then
    ./reproduction/run_env.sh -m reproduction.compare_hierarchical_application \
      "$candidate" "$result_root/partial/seed$seed/scaling/dense-float32-77824-0.json" \
      --output "$result_root/partial/seed$seed/application-sdd-77824.json"
  fi
done
./reproduction/run_env.sh -m reproduction.compare_hierarchical_application \
  reproduction/results/hierarchical_variance/sdd/73728-eta0.03-passing.json \
  reproduction/results/hierarchical_variance/scaling/dense-float32-73728-0.json \
  --output "$result_root/pilot/long-sdd-application.json"
./reproduction/run_env.sh -m reproduction.summarize_hierarchical_robustness
