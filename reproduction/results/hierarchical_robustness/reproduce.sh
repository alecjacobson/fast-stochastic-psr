#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
result_root=reproduction/results/hierarchical_robustness
settings=(--lengthscale .03 --halfwidth .005 --h2-tolerance .001
          --leaf-size 40 --rank 8 --neighbors 24 --variance-iterations 8)

# Tighten cross-basis compression, rather than merely increasing interpolation
# order. Seeds 21, 34 and 55 are retained pilots, not confirmation inputs.
for seed in 89 144 233; do
  ./reproduction/run_env.sh -m reproduction.run_hierarchical_scaling \
    "${settings[@]}" --n 16384 32768 77824 --repeats 1 \
    --methods h2-gpu dense-float32 --order 4 --cross-tolerance 1e-6 --scan front --normal-noise .005 \
    --seed "$seed" --output "$result_root/partial/seed$seed/scaling"
  ./reproduction/run_env.sh -m reproduction.audit_hierarchical_scaling \
    --root "$result_root/partial/seed$seed"
done

# Strong locality control with GPU graph construction, fixed cutoff selected
# separately from these scaling repetitions. All methods use order four here.
./reproduction/run_env.sh -m reproduction.run_hierarchical_scaling \
  "${settings[@]}" --n 32768 77824 --methods h2-gpu sparse-gpu \
  --sparse-cutoff 11 --output "$result_root/locality/scaling"
./reproduction/run_env.sh -m reproduction.run_hierarchical_scaling \
  "${settings[@]}" --n 131072 --methods h2-gpu sparse-gpu \
  --variance-iterations 12 --sparse-cutoff 11 --output "$result_root/locality/large"

# References are independent full-kernel computations from the preceding study.
for dir in scaling large; do
  for record in "$result_root/locality/$dir/"*.json; do
    [[ "$record" == *.status.json ]] && continue
    n="$(basename "$record" .json | cut -d- -f3)"
    probes=32; (( n > 73728 )) && probes=8
    ./reproduction/run_env.sh -m reproduction.audit_hierarchical_variance "$record" \
      --reference "reproduction/results/hierarchical_variance/short/references/$n.json" \
      --random-queries "$probes"
  done
done

for seed in 89 144 233; do
  if [[ ! -f "$result_root/partial/seed$seed/sdd/77824.json" ]]; then
    ./reproduction/run_env.sh -m reproduction.benchmark_hierarchical_sdd \
    --reference "$result_root/partial/seed$seed/references/77824.json" \
    --output "$result_root/partial/seed$seed/sdd/77824.json" --order 4 --cross-tolerance 1e-6 \
    --application-reference "$result_root/partial/seed$seed/scaling/dense-float32-77824-0.json" \
      --eta .1 .15 --checkpoints 2048 8192 16384 32768 49152 65536 98304 131072 --budget 90
  fi
  for n in 16384 32768 77824; do
    ./reproduction/run_env.sh -m reproduction.compare_hierarchical_application \
      "$result_root/partial/seed$seed/scaling/h2-gpu-$n-0.json" \
      "$result_root/partial/seed$seed/scaling/dense-float32-$n-0.json" \
      --output "$result_root/partial/seed$seed/application-$n.json"
  done
done
./reproduction/run_env.sh -m reproduction.summarize_hierarchical_robustness
./reproduction/run_env.sh -m pytest tests/test_nonperiodic.py tests/test_hierarchical.py tests/test_nonperiodic_sparse.py -q
bash reproduction/results/hierarchical_robustness/extra_controls.sh
bash reproduction/results/hierarchical_robustness/extension.sh
bash reproduction/results/hierarchical_robustness/extension_sdd.sh
bash reproduction/results/hierarchical_robustness/check_sparse_cutoff.sh
