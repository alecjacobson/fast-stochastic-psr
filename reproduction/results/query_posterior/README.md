# Coherent sampling on fixed query panels

The query-space prototype passes covariance, field and event checks against a
cached dense query-space reference. It removes the source-space tail draw, but
**dense query covariance is still faster at A/B sizes**. Its promise is a
bounded-memory route to selected queries at large N; that scale has not yet
been measured for this implementation.

On B's 32-point covariance-control panel, preparation plus 8,192 correlated
samples takes **491.4 s through the cached dense source-space sampler, 9.01 s
through matrix-free query covariance, and 2.26 s through dense query covariance**.
The 54.5× pilot gain is relative to that source-space sampler, not the strongest
dense query reference or untouched GeoSPSR. This 32-point panel has a degenerate
joint-inside event and is not a collision or scanning benchmark.

## Method

For a fixed query set Q, assemble

`S_Q = K_ff(Q,Q) - sum_d C_d A^-1 C_d.T`, with `A=K_analytic+.02*I`.

The existing `FourierCorrectionCovariance` streams query basis blocks through
four NUFFT plans and a full analytic solver, avoiding storage of the N-by-M
cross matrix. The final query covariance and principal root require O(M²)
storage. Every query costs three conditioning RHS. Additional samples multiply
the fixed root by independent Gaussian white noise; they are jointly correlated
over the entire panel. New query locations require new covariance work.

The current A/B measurements use analytic range CG, which previously won over
NUFFT initialization plus correction at these N. The conditioner also supports
finite-NUFFT defect correction; that path is covered by independent small
controls, not a massive query-covariance benchmark here. The finite scalar/cross
band is 50, analytic length scale .01, per-axis variance .1 and noise variance .02.
Scalar NUFFTs use float64/eps 1e-12. The full analytic solve tolerance is 1e-10.

## Correctness and numerical failures retained

Seven targeted tests pass, including full covariance against independent finite
features, 16,384-draw moment checks, partial blocks, duplicate/nearly coincident
queries, and tiny/unequal RHS. [tests.json](tests.json) records their provenance.

The initial 1e-8 solve tolerance missed the fixed near-zero mean gate. Tightening
it exposed a tiny-scale float32 CG curvature failure in inner defect solves.
Normalizing each RHS in float64 before the finite solve and restoring its scale
resolves that failure; the original full analytic residual and output gates
remain unchanged. A new test includes 1e-20 and zero channels. Both failed test
stages are retained in [validation_history.json](validation_history.json).

The query root rejects material asymmetry and negative eigenvalues beyond
`100*eps*M*||S||_2`. It reports any roundoff-scale projection rather than adding
jitter. The 1,024-point application covariances have 214/215 tiny negative
eigenvalues, with maximum projections 5.07e-18/5.10e-18. Their budgets are
9.10e-15/8.52e-15, and root reconstruction errors are below 4.60e-14 relative.
Those budgets are floating-point heuristics, not allowances for solve/model
error or formal interval certificates. Small-panel roots require no projection.

All four measured cases pass full covariance relative error < 1e-7, marginal
variance relative error < 1e-7 and mean error/prior standard deviation < 1e-7.
Observed covariance errors are at most 5.10e-13. All retained-control replays and
sampled GPU isolation audits pass. Replay checks the saved covariances,
inputs, solve records, sample moments and events; it does not regenerate every
training solve or Fourier prior. Dense references use independent direct cross
rows and Cholesky solves but share the checked finite-scalar prior assembly.

## Workloads and measured costs

A has 691 observations; B has 17,419. Each has two fixed query panels:

- **32-point control:** the pre-existing off-/near-surface covariance queries.
  Its joint-inside probability is 1; only its field/covariance tests and sampling
  cost are informative.
- **1,024-point application panel:** existing rays 0, 8, 17 at 256 steps each and
  sphere 8 at 256 surface points. Coordinates and spatial resolution are retained
  from the previous coherent experiment. This is a subset of the 24-ray,
  17-sphere workload, not a replacement headline flagship comparison.

The following totals include preparation and the same sample count in every
method within each row. These are single-pass pilots, including first-use
effects; no repeated whole-pipeline timing distribution is claimed. They use
the NVIDIA L40 GPU; [environment.json](environment.json) records the runtime.

| Case / queries | Draws | Matrix-free query | Dense query | Dense source-space |
|---|---:|---:|---:|---:|
| A /32 |64 |1.409 s |.183 s |1.652 s |
| B /32 |8,192 |9.011 s |2.262 s |491.442 s |
| A /1,024 |64 |35.890 s |.893 s |1.661 s |
| B /1,024 |64 |266.251 s |5.408 s |7.743 s |

Once prepared, 32,768 query-space draws plus returned fields/event indicators on
the application panels take .0677 s on A and .0739 s on B. Dense-query sampling takes
.0683 s/.0737 s. Preparation costs 35.652 s/266.232 s for the matrix-free method and
.892 s/5.407 s for dense query covariance. We **did not measure 32,768 source-space
application draws**, and do not extrapolate an equal-count speedup there.

Each source-space baseline uses cached analytic and residual Cholesky factors,
double jointly drawn Fourier priors, batches of 8, reusable transform plans and
its own conditioning/query work. Query methods generate the same white noise
for paired accuracy; normal-generation cost is measured and charged to both.
All constructors and draw calls are GPU synchronized, including RHS assembly.
Totals charge common scalar/mean setup and prior covariance work to each
method. They exclude loading existing geometry, validation, file I/O and final
probability averaging/score analysis. They measure posterior preparation and
returned field/event arrays, not complete robot/scanning wall latency.

The first A application draw includes a .238 s first-use cost; its later, much
larger draw is faster. This effect is retained. The CUDA compilation cache was
preserved and moved to the runtime volume after home disk space became low;
`run_env.sh` now defaults its CUDA cache there and honors an explicit override.

## Application output and sampling confidence

For 32,768 samples, both A and B have **zero paired ray/collision disagreements**
between query methods. Paired field relative errors are 3.40e-8/4.39e-9; tiny
covariance eigenvalues make root differences larger than covariance differences.
Collision probabilities for sphere 8 are .911163/.532623. The source-space 64-draw
controls give .859375/.53125, consistent with their much larger sampling error;
these are independent root couplings, not paired event tests.

Hoeffding's inequality with a union bound over 768 ray-transmittance marginals
and one collision event gives a 95% simultaneous Monte Carlo radius of
**1.256 percentage points** at 32,768 draws:

`epsilon = sqrt(log(2*769/.05)/(2*32768))`.

This requires independent draws, not independent query locations. It applies
to the Gaussian distribution represented by the computed root; it excludes
numerical/model bias and spatial discretization. Propagating those probability
intervals through the numeric trapezoidal uncertainty score separates the best
tested ray: 17 on A, 0 on B. The 64-draw source controls cannot certify those
rankings with the same conservative bound. This is a result for these three
fixed rays and one fixed sphere, not complete sequential scanning or physical
collision accuracy.

## Remaining work

[#48](https://github.com/alecjacobson/fast-stochastic-psr/issues/48) remains open
for crossover optimization, adaptive query extension and massive measurements.
[#49](https://github.com/alecjacobson/fast-stochastic-psr/issues/49) tracks
propagating full analytic solve residuals into query-covariance error budgets,
which can support large-N checks without a dense reference. The larger panel
shows that even with cheap sampling, assembling all query covariance columns
can dominate. There is no new massive or full flagship application speedup.

## Reproduce

```bash
./reproduction/run_env.sh -m pytest -q tests/test_query_posterior.py tests/test_deflated_variance_gpu.py
./reproduction/run_env.sh -m reproduction.benchmark_query_posterior --case A --panel moments --samples 8192 --source-samples 64
./reproduction/run_env.sh -m reproduction.benchmark_query_posterior --case B --panel moments --samples 8192 --source-samples 8192
./reproduction/run_env.sh -m reproduction.benchmark_query_posterior --case A --panel applications --samples 32768 --source-samples 64
./reproduction/run_env.sh -m reproduction.benchmark_query_posterior --case B --panel applications --samples 32768 --source-samples 64
# Repeat this replay for each case/panel combination:
./reproduction/run_env.sh -m reproduction.validate_query_posterior --case B --panel applications
```

Raw records, complete query covariances and all retained field/event controls
are in [A 32](A_moments/results.json), [B 32](B_moments/results.json),
[A 1,024](A_applications/results.json), [B 1,024](B_applications/results.json).
