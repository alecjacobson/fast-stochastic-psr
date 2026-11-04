# Practical scaling experiments

Tracking: [#59](https://github.com/alecjacobson/fast-stochastic-psr/issues/59).
These are constructed workloads on the released scorpion, not ContactFusion
sensor recordings. The aim is to select a useful operating regime, including
cases where NUFFT loses. All comparisons run sequentially on the same GPU.

## Observation scaling: accumulating a dense scan

Generate 524,288 distinct area-uniform visible-surface observations with the
existing `prepare_large_cloud` utility. Take nested prefixes from 512 through
262,144 points. Keep the object's domain, analytic periodic product-Matern32
kernel, length scale .01, per-axis variance .1, and observation-noise variance
.02 fixed. This is **infill scaling**, not growing the physical domain. More
measurements can change conditioning and iterative convergence.

Solve all three normal right-hand sides. Compare:

- GPU dense matrix construction, Cholesky factorization, and triangular solves.
- Compiled JAX stochastic dual descent, previously checked against pinned
  upstream updates. Pilot batches 128/512 and learning-rate multipliers 2/5;
  retain all attempted configurations and failures. At N=262,144 narrow the
  pilot to the stronger batch 512 from preceding sizes, try both multipliers,
  and check every 1,024 updates to reduce verification overhead. A final
  N=32,768 check also tests batches 1024/2048/4096 with both multipliers.
- NUFFT-CG at training cutoff 100 followed by analytic defect corrections.
  Each correction uses the same finite NUFFT solve and checks the full
  analytic system. Count the analytic geometry preparation and correction
  actions: this is not a pure finite-kernel NUFFT benchmark.
- Analytic CG using the same fast analytic action without NUFFT, to isolate
  the benefit of the NUFFT inner solves.

Quality requires a full analytic relative Frobenius residual <=1e-4 and
held-out mean relative L2 error <=0.2%. An analytic-CG reference is solved to
1e-10; 64 independent dense analytic rows check the action at <=1e-9 relative
error. Mean evaluation uses the same cutoff-50 NUFFT machinery for every
solver, including SDD. It therefore cannot account for solver differences.
The full-residual gate is stricter than a mean-only application necessarily
needs; results do not establish the crossover under a looser output-only gate.

Confirm SDD with independent seeds 500,501,502. Frozen pilot budgets that fail
remain failures. In the adaptive confirmation, continue in 256-update chunks
until the same full residual gate passes. Report update time, verification time,
and verification-geometry preparation separately. Update-only timings favor
SDD and are a lower bound on a deployable verified pipeline. Pilot tuning and
reference construction are recorded but excluded from candidate production
costs. Each repeated solve starts from zero; no previous solution is reused.

Report fresh-map preparation and solve separately. Shared software compilation
caches may already be warm, as can common evaluator kernels used by reference
validation. These are not cold-process startup measurements. Dense allocation
beyond the declared cap is not attempted; a missing dense timing is not an
extrapolated measured speedup. Report the required matrix bytes instead.

## Query/reuse scaling: marginal probability queries on a fixed scan

Use released scan A, full analytic observation covariance, and finite scalar/
cross cutoff 50. Prepare mean and exact marginal variance using the Cholesky
factor. Collapse all 3N whitened cross functions' squared Fourier series into
one variance series with doubled bandwidth. There is no Monte Carlo estimate,
rank truncation, or positivity clipping in this construction.

Compare direct table-backed Cholesky variance, ordinary type-2 NUFFT evaluation,
prepared interpolation-only NUFFT moment evaluation, and tuned direct SDD
variance solves. The reference table implements the original fifth-power
nonuniform knot scheme on GPU; increase density until accuracy gates pass.
Count its preparation separately: it depends on the model and can be shared
across maps. Use matrix batches for the direct Cholesky baseline, rather than
calling one triangular solve per query. SDD RHS batches and hyperparameters
are recorded. Pilot timing is not a production cost.

Also screen spatial grids at 32,64,128,256 nodes per axis. Screen interpolation
accuracy using the candidate only; **time the selected baseline's construction
using direct Cholesky evaluations**, then cubic spline-filter the grid. All
final validation uses the actual baseline grid. This is stronger than the
paper's cross-kernel cache: it caches posterior moments for one fixed map.
Its bounded domain must be stated. A second workload interleaves surface
queries with uniform periodic-workspace probes; its competing spatial grid
covers the entire periodic domain and uses periodic cubic interpolation.

Query fresh area-uniform mesh points perturbed by .01 coordinate noise. Each
repeat translates them, so answers are never reused. Rotation of method order
reduces systematic warmup bias. Check all outputs against the dense table
baseline, plus an independent exact finite-cross-kernel panel in each batch.
Require mean relative L2 <=0.2%, variance relative L2 <=0.02%, centered variance
contrast error <=1%, marginal inside-probability RMS <=0.05 percentage points,
maximum <=0.5 percentage points, and positive variances. These are numerical
benchmark thresholds, not calibrated physical application tolerances.

Marginal probabilities do not determine joint ray transmittance, collision
probability, or information gain. This experiment makes no joint uncertainty
claim. Keeping cutoff 50 fixed checks equivalence to the finite target; it
does not prove convergence to an infinite-bandwidth scalar field.

## Timing and provenance

Use synchronized GPU timing, three repeats, source/input SHA-256 hashes, pinned
runtime versions, and allocated-MIG process monitoring. Whole-run audit errors
or foreign processes disqualify timing claims. Periodic process sampling can
miss short-lived overlap and is not proof of exclusive access. Earlier runs
with an inaccessible parent-GPU audit remain archived and are excluded from
promoted curves. No unrelated process is stopped.

The primary log-log plots show only completed, accuracy-passing, audit-passing
measurements. Fit finite-range empirical exponents only where specified; they
are not asymptotic proofs. Include setup to distinguish repeated-query gains
from map-update latency. Any reuse break-even computed from measured components
is a cost-model prediction, not a measured long sequence of repeated batches.
