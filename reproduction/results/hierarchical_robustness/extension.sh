#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
result_root=reproduction/results/hierarchical_robustness/extension
# Continue the first confirmation scan at higher density. No dense runtime is
# extrapolated beyond the device's factor-storage capacity.
./reproduction/run_env.sh -m reproduction.run_hierarchical_scaling \
  --n 16384 32768 77824 --repeats 1 --methods h2-gpu sparse-gpu --timeout 240 \
  --lengthscale .03 --halfwidth .005 --h2-tolerance .001 --leaf-size 40 \
  --rank 8 --neighbors 24 --variance-iterations 4 --sparse-cutoff 11 \
  --order 4 --cross-tolerance 1e-6 --scan front --normal-noise .005 --seed 89 \
  --output "$result_root/scaling"
./reproduction/run_env.sh -m reproduction.run_hierarchical_scaling \
  --n 196608 --repeats 3 --methods h2-gpu sparse-gpu --timeout 240 \
  --lengthscale .03 --halfwidth .005 --h2-tolerance .0001 --leaf-size 40 \
  --rank 8 --neighbors 24 --variance-iterations 4 --sparse-cutoff 11 \
  --order 4 --cross-tolerance 1e-6 --scan front --normal-noise .005 --seed 89 \
  --output "$result_root/scaling"
./reproduction/run_env.sh -m reproduction.audit_hierarchical_scaling --root "$result_root"
if [[ ! -f "$result_root/tight-field.json" ]]; then
  ./reproduction/run_env.sh -m reproduction.benchmark_hierarchical_variance \
    --n 196608 --method h2-gpu --lengthscale .03 --halfwidth .005 \
    --h2-tolerance 1e-7 --solve-tolerance 1e-10 --variance-iterations 0 \
    --order 4 --cross-tolerance 1e-6 --scan front --normal-noise .005 --seed 89 \
    --output "$result_root/tight-field.json"
fi
./reproduction/run_env.sh -m reproduction.audit_hierarchical_variance "$result_root/tight-field.json" \
  --reference "$result_root/references/196608.json" --random-queries 8
for method in h2-gpu sparse-gpu; do
  ./reproduction/run_env.sh -m reproduction.compare_hierarchical_application \
    "$result_root/scaling/$method-196608-0.json" "$result_root/tight-field.json" \
    --output "$result_root/application-$method.json"
done

# Check every actual query at the smaller four-step points against Cholesky.
for n in 16384 32768 77824; do
  for method in h2-gpu sparse-gpu; do
    ./reproduction/run_env.sh -m reproduction.compare_hierarchical_application \
      "$result_root/scaling/$method-$n-0.json" \
      "reproduction/results/hierarchical_robustness/partial/seed89/scaling/dense-float32-$n-0.json" \
      --output "$result_root/application-$method-$n.json"
  done
done
