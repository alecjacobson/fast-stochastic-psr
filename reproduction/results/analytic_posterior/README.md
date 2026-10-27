# Coherent analytic posterior integration

This study connects the fixed analytic tail root to joint Fourier priors and
full analytic conditional solves. The target is a finite scalar/cross kernel
(cutoff 50) with the **full analytic vector training covariance**, including
observation noise. It is a direct-query numerical comparison, not a reproduction
of the released interpolant or a new claim of massive application speedup.

**Both A and B pass the paired numerical checks, but both are slower than the
cached dense baseline.** This establishes the coherent integration needed for
further acceleration; it does not transfer the million-point mean-fit speedup
to uncertainty-driven applications.

| Case | Observations | Production setup + 64 draw/query stages | Cached dense setup + 64 draw/query stages | Field error / fluctuation norm | Paired event disagreements |
|---|---:|---:|---:|---:|---:|
| A | 691 | 63.15 s | 2.180 s | 4.70e-12 | 0 |
| B | 17,419 | 265.62 s | 9.788 s | 9.13e-11 | 0 |

See the timing exclusions below; these are stage sums, not application wall
latencies. In B, the tail root takes about 2.11 s per draw and conditioning
about 1.95 s. The fitted model and reusable dense factors alone are not the
remaining uncertainty-preparation bottleneck.

## Pipeline

`AnalyticPosterior` streams complete coherent functions:

1. Draw independent real/imaginary Gaussian vector coefficients over the entire
   scalar band. Derive the scalar coefficients from those **same** vectors.
   The vector zero mode is retained; the scalar Poisson zero mode is zero.
2. Independently draw `sqrt(R) * white`, where
   `R = K_analytic - K_finite50 + noise*I`. The fixed Chebyshev polynomial uses
   the PSD/trace and analytic box bounds from the [root study](../analytic_tail/README.md).
3. Form observations minus the joint vector prior minus that tail/noise draw.
   Solve using cutoff-200 NUFFT initialization and full analytic defect
   correction, with relative full-system residual tolerance 1e-8.
4. Add the scalar prior to the cross-kernel correction. Subtract the fitted
   mean's deterministic centering offset, not a separate random offset per draw.

The implementation also supports analytic CG conditioning. The measurements
here use defect correction throughout. Batch-dependent transform plans are
replaced when necessary; coefficients are yielded one batch at a time. Retaining
yielded blocks increases caller-owned memory. The seed is reproducible for a
fixed batch schedule, not across arbitrary changes in batch size.

All parameters use analytic normalization, length scale .01, per-axis variance
.1 and observation-noise variance .02. Scalar/prior/tail transforms use float64
and NUFFT tolerance 1e-12. The finite fitting action uses float32, tolerance
1e-6 and iterative residual tolerance 1e-5 before analytic refinement. The
root's requested relative scalar error is 1e-7; numerical action, solve and
sampling errors are separate.

## Validation design

Three new integration tests and six root/refinement regressions pass. The
integration tests include partial batches and an **8,192-draw** moment check
against independently constructed trigonometric prior features and analytic
conditioning. This covers scalar/vector correlation, observation noise,
duplicate inputs, zero modes and full posterior covariance. The test record
retains the command, duration and source hashes in [tests.json](tests.json).

The scorpion experiments use the existing A/B inputs and 64 draws per case.
They evaluate 512 field queries, 24 rays with 256 steps, and 17 spheres with
256 surface points. Sphere placement uses only the deterministic dense mean
to find uncertain collisions. The same geometry is used for every method.

There are three sampling paths:

- **Production:** matrix-free tail root and analytically refined NUFFT solves.
- **Paired reference:** independent dense tail covariance and dense conditional
  solves, with the same finite prior and white noise. A uses an eigendecomposition
  for the principal root. B evaluates the same bounded polynomial with dense
  covariance actions; it does not compute a full exact principal root.
- **Dense Cholesky baseline:** cached dense analytic and residual Cholesky
  factors, with its own draws, solves, aggregation and queries. Its paths differ
  from the principal-root coupling, so it supplies a timing baseline and moment
  sanity check, not a paired-event reference.

Every production conditional RHS is checked against the independent dense
analytic matrix. Paired checks compare full sample fields, marginal/spatial
sample covariance, ray/collision events and the numeric trapezoidal next-view
score. A separate 32-query analytic posterior covariance is compared with the
sample moments for all three paths, using Gaussian mean/covariance standard
errors. These are sanity checks, not tight uncertainty estimates from 64 draws.

All production solve residuals are <=3.94e-12 on A and <=4.79e-11 on B. The
paired tail errors are <=2.20e-11 and <=3.51e-14 respectively; the B comparison
shares the polynomial approximation, as described above. Empirical covariance
differences between paired fields are <=7.45e-12 in relative Frobenius norm.
Both cases have exactly matching ray-transmittance and collision events and
next-view scores. Best views are uniquely 17 on A and 0 on B; uncertain
collision probabilities occur at all 17 A sites and 16 of 17 B sites.

Across the three sampling paths, the largest standardized mean/covariance
deviations from the 32-query analytic target are 3.20/3.30 on A and 2.97/3.01
on B. At 64 draws, the largest reported pointwise probability standard errors
are up to **6.25 percentage points**. Zero paired disagreement therefore
establishes numerical agreement on these draws, not high precision for the
underlying event probabilities. Raw [A](A/results.json) and [B](B/results.json)
records retain these distinctions. Both sampled GPU isolation audits and
retained-output replays pass.

## Timing scope and limits

Totals sum synchronized setup and sampling/query stages. They charge finite
prior generation to both production and dense methods, although the paired
experiment shares those random priors. Both use fast direct Fourier evaluation;
the dense baseline is therefore stronger than an untouched upstream query
implementation. Mean fitting, tail preparation, factors/plans, root actions,
conditional solves, aggregation and queries are included. Paired-reference
validation, geometry calibration, disk writes, Python bookkeeping and the
small RHS assembly operations outside stage timers are excluded. These are
measured stage totals, not full wall-clock application latency.

Each experiment is one instrumented 64-draw pass, not a repeated end-to-end
timing distribution. Preflight tests initialize native libraries. First-use
library costs can differ between stages. GPU process sampling can miss
short-lived overlap; its raw observations are retained.

This still does not establish absolute ray/collision probability accuracy,
geometry-discretization convergence, scalar bandwidth convergence, released
interpolant compatibility, physical scanning quality or massive-scale gains.
The existing million-point mean-fit speedup is not a posterior speedup.
[#46](https://github.com/alecjacobson/fast-stochastic-psr/issues/46) tracks the
remaining validation, and [#47](https://github.com/alecjacobson/fast-stochastic-psr/issues/47)
tracks reducing coherent-root cost and selecting its accuracy for applications.
[#48](https://github.com/alecjacobson/fast-stochastic-psr/issues/48) tracks a
different route for small fixed query panels: assemble and factor their
posterior covariance, charging all cross-kernel solves, to avoid a source-space
tail root. No speedup for that route is measured yet.

## Reproduce

```bash
./reproduction/run_env.sh -m pytest -q tests/test_analytic_posterior.py \
  tests/test_analytic_tail_gpu.py tests/test_analytic_refinement.py
./reproduction/run_env.sh -m reproduction.benchmark_analytic_posterior --case A --samples 64
./reproduction/run_env.sh -m reproduction.benchmark_analytic_posterior --case B --samples 64
./reproduction/run_env.sh -m reproduction.validate_analytic_posterior --case A
./reproduction/run_env.sh -m reproduction.validate_analytic_posterior --case B
```

The per-case JSON records contain all draw stages, conditioning histories,
source/input hashes and numerical checks. Committed small archives retain the
geometry, all field/event outputs, moment targets and the first complete
conditioning/tail control. Replay regenerates the first draw's full analytic
residual and paired tail root, and checks all retained outputs and moment
summaries. It does not regenerate the entire ensemble from its seed.
