# Exploratory records, including failures

These records are retained to make selection and debugging visible. They are
not inputs to the promoted timing curves.

- `solver-n*`: initial pilot sweeps. The parent-GPU monitor lacked permission.
  Frozen SDD confirmations fail at N=8192 and N=131072; the N=262144 setup
  exceeds its initially declared 4-GiB analytic-cache cap.
- `solver-verified-*`: adaptive SDD confirmations, still with the inaccessible
  parent-GPU monitor. Accuracy improvements do not repair their timing audits.
- `solver-audited-*`: per-MIG monitoring works. The N=262144 tuning run was
  interrupted because frequent full-residual checks dominated tuning; this
  interruption is recorded and the run remains incomplete.
- `solver-retuned-n262144`: batch-512 pilot selection with 1024-update checks,
  and independently confirmed solves. The promoted run reuses only the chosen
  hyperparameters/budget and checks every 256 updates after that budget.
- `solver-controlled-*`: intermediate analytic-CG controls and precision
  investigation. The N=32768 float64 factorization exceeds workspace memory.
  Other records precede the final fused float32 baseline. Do not mix these
  different implementations into a scaling curve.
- `variance-pilot-*`: small-cutoff exploration, including a coarse table that
  misses accuracy, a serialization failure, and the unequal-field packing
  failure that motivated #61. The first serialization failure was annotated
  manually; it is not a completed numerical benchmark.
- `variance-full-A-c50-g400-scaled`: full scan A passes quality after the packing
  fix, but predates the allocated-device timing audit and strong spatial/
  float32 baselines. Its query-only ratios are not final claims.

The raw JSON retains model, arguments, stages, failures, and any available
quality/audit information. Early files may lack source hashes or a usable
audit. They are historical diagnostics, not fully reproducible release
snapshots of each uncommitted intermediate implementation. Final benchmark
source, source/input hashes, and completed audited confirmations are supplied
in the promoted directories instead.
