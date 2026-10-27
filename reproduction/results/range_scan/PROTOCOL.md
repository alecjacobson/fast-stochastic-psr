# Range scan optimization protocol

Continue #43 from the exact serial control at 2a809c5. Before implementation,
measure far/direct CUDA kernel durations with GPU events on A/B, first execution
separately, then three calls. No Nsight Compute executable is available here;
this is stage timing, not hardware-counter evidence.

Implement bounded warp scans of the same decaying moment recurrence. Keep the
serial source and its archived benchmarks intact. Compare both backends with the
independent dense hyperbolic formula/image sum and periodic seam/tie tests.
Fixed full analytic covariance relative L2 gate 1e-7 per RHS, including a
noise-free comparison. Use the original three deterministic RHS (seed4501),
A691/B17419 points, leaf32, double precision. Preserve periodic closure, all
pairs and signed weights; no pruning or Fourier truncation. Report failures.

Measure setup, first call and three later calls separately, plus sampled process
isolation, retained geometry and additional scratch. Compare fresh serial and
scan calls; retain the earlier independent dense and NUFFT results as context
with their quality qualification. No extrapolation to CG/application gains.
A bounded warp scan is an intermediate step: if long serial tile chains remain,
track hierarchical chunk scans rather than introducing quadratic prefix work.

## Cached-geometry follow-up (declared after warp pilot)

The uncached warp pilot is retained unchanged: B scan107ms versus serial90ms,
with all numerical gates passing. Before further measurements, test cached
float64 source/target features and z decays for both serial and warp actions.
No arithmetic precision reduction or term pruning. Charge cache construction
and retained bytes, keep first and three subsequent calls, and use the same
RHS and independent archived full analytic controls. The additional storage
may limit massive scale; do not claim that the B memory tradeoff scales yet.

## Mean-fitting pilot (declared after cached-action measurements)

Cached warp actions reduce B to21.9ms with roundoff agreement, but retain349MB
and remain slower than the earlier cached dense action. Test complete fits on
A/B at ell=.01,noise=.02 and the documented B short/low-noise failure at
ell=.005,noise=.0004. Same observed normals, independent full analytic dense
matrix and double Cholesky reference. Cached range uses existing batched CG,
rtol5e-9,maxiter1000. Independent residual per RHS <=1e-8, weights relativeL2
<=1e-7 and centered direct-Fourier mean relativeL2<=1e-7 at all20k slice queries.
Scalar/cross cutoff50 is shared; this is not a full scalar-bandwidth test.

Record matrix construction/factorization, range plan/cache setup, first solve,
then3 subsequent solves for both. Record mean aggregation/centering/query stages
separately using common double NUFFT operators. All first timings and failures
retained. This is a single pilot, not an upstream SDD comparison, nor coherent
sampling or full application validation. Do not infer massive-scale performance.
