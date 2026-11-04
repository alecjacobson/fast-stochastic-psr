#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
result_root=reproduction/results/hierarchical_robustness/extension
# One larger-size configuration, selected from the smaller seed-89 comparison;
# this is not an exhaustive large-N SDD tuning or timing repetition study.
if [[ ! -f "$result_root/sdd/196608.json" ]]; then
  ./reproduction/run_env.sh -m reproduction.benchmark_hierarchical_sdd \
    --reference "$result_root/references/196608.json" \
    --application-reference "$result_root/tight-field.json" \
    --output "$result_root/sdd/196608.json" --order 4 --cross-tolerance 1e-6 \
    --eta .15 --checkpoints 8192 16384 32768 65536 98304 131072 --budget 150
fi
if [[ -f "$result_root/sdd/196608-eta0.15-passing.json" ]]; then
  ./reproduction/run_env.sh -m reproduction.compare_hierarchical_application \
    "$result_root/sdd/196608-eta0.15-passing.json" "$result_root/tight-field.json" \
    --output "$result_root/application-sdd.json"
fi
./reproduction/run_env.sh -m reproduction.summarize_hierarchical_robustness
