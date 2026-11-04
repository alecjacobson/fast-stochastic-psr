#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
root=reproduction/results/hierarchical_robustness/extension
for cutoff in 9 10; do
  record="$root/pilot/sparse-cut${cutoff}.json"
  if [[ ! -f "$record" ]]; then
    [[ -f "$record.failure.json" ]] && continue
    if ./reproduction/run_env.sh -m reproduction.benchmark_hierarchical_variance \
      --n 196608 --method sparse-gpu --lengthscale .03 --halfwidth .005 \
      --rank 8 --neighbors 24 --variance-iterations 4 --sparse-cutoff "$cutoff" \
      --order 4 --cross-tolerance 1e-6 --scan front --normal-noise .005 --seed 89 \
      --output "$record" > "$record.log" 2>&1; then
      :
    else
      exit_status=$?
      python3 - "$record" "$exit_status" <<'FAILURE'
import json, re, sys
from pathlib import Path
record, status = sys.argv[1:]
log = Path(record+'.log').read_text()
errors = re.findall(r'^.*Error:.*$', log, flags=re.MULTILINE)
Path(record+'.failure.json').write_text(json.dumps(dict(status='failed', exit_code=int(status),
    reason=errors[-1] if errors else 'See log', log=record+'.log'), indent=2)+'\n')
FAILURE
      continue
    fi
  fi
  ./reproduction/run_env.sh -m reproduction.audit_hierarchical_variance \
    "$root/pilot/sparse-cut${cutoff}.json" --reference "$root/references/196608.json" --random-queries 8
  ./reproduction/run_env.sh -m reproduction.compare_hierarchical_application \
    "$root/pilot/sparse-cut${cutoff}.json" "$root/tight-field.json" \
    --output "$root/pilot/sparse-cut${cutoff}-application.json"
done
for n in 16384 32768 77824; do
  for method in h2-gpu sparse-gpu; do
    ./reproduction/run_env.sh -m reproduction.compare_hierarchical_application \
      "$root/scaling/$method-$n-0.json" \
      "reproduction/results/hierarchical_robustness/partial/seed89/scaling/dense-float32-$n-0.json" \
      --output "$root/application-$method-$n.json"
  done
done
./reproduction/run_env.sh -m reproduction.summarize_hierarchical_robustness
