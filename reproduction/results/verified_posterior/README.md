# Coherent applications with independently verified conditioning

`HalfCubePosterior(..., refine_solves=True)` now checks both mean fitting and
coherent sample conditioning with serial float64 transforms of the identical
finite training cube. At most two loose native correction solves repair a failed
check. Acceptance depends on the independently evaluated residual, even when the
initial native solve reports success or failure. Defaults are unchanged.

The same option is available as `stream_applications --refine-solves`. Packed
solves, preconditioning, partial batches and early stream closure are covered by
tests. A narrower hybrid inverse verifies only its centered fluctuation solve
against the narrower cube, then adds the independently fitted wider-band mean.
It remains a hybrid approximation, rather than being silently changed into a
full-band posterior.

[Protocol](PROTOCOL.md) · [Raw A run](A/results.json) ·
[Raw three-million run](three_million/results.json) ·
[Issue #37](https://github.com/alecjacobson/fast-stochastic-psr/issues/37).

## What is compared

Each pair uses identical scalar-prior coefficients and identical conditioning
RHS, including observation noise. Production uses float32 conditioning and
aggregation, verified at actual residual <=2e-5. The reference uses a dense
finite-cube Cholesky factorization on A and float64 stored weights refined to
<=2e-7 on the large cloud; reference aggregation, queries and deterministic mean
centering use float64. The large reference shares the verification operator and
is a stronger numerical control, not an exact independent oracle.

All three million observations enter every action. The input matches the
[published cloud manifest](../large_inputs/scorpion_3000000/manifest.json).
Both methods use training cube 200 and scalar/cross cube 50, analytic
normalization, lengthscale .01, per-axis variance .1 and noise variance .02.
There are eight coherent draws, 512 field probes, 24 rays of 256 samples and
17 spheres with 256 surface probes. Sphere positions are calibrated around a
reference-mean crossing before drawing.

## Measured numerical agreement

| Input | Worst verified draw residual | Empirical variance difference | Empirical 32-query covariance difference | Ray / collision events | Selected view |
| --- | ---: | ---: | ---: | --- | --- |
| A, 691 | 1.1e-06 | 0.000264% | 0.000418% | identical | 18 / 18 |
| 3,000,000 | 2.37e-06 | 0.001464% | 0.002301% | identical | 4 / 4 |

Both declared numerical gates pass. A needs no corrections. At three million,
all eight initial native sample solves fail their residual gate; one correction
per draw repairs them. The mean initially reports native success, but the
double check finds a worst residual of 2.1395e-5 (threshold 2e-5); one correction
reduces it to 1.9477e-6. The stronger large-cloud reference requires one or two
further corrections per draw and reaches at most 1.9957e-7.

The fluctuation-scaled field differences are 0.000457% on A and 0.011664% at
three million. These include separate mean centering and scalar aggregation
precision. Both ensembles contain uncertain collision probes. View indices are
zero based; equality applies to this fixed discrete candidate set.

## Instrumented costs

| Input | Verified fit (s) | Eight verified conditional solves (s) | Prior + aggregation + queries (s) | Sum of production stages (s) | Sampled process peak (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| A, 691 | 0.911 | 7.626 | 0.637 | 9.174 | 21.57 |
| 3,000,000 | 28.377 | 202.759 | 0.892 | 232.027 | 23.39 |

Both runs overlap other GPU processes. These costs are diagnostic, with **no
speedup ratio**. The sum includes independent verification and correction but
excludes reference construction/queries, geometry calibration, validation,
hashing, disk export and process startup. It is a sum of synchronized measured
stages in an interleaved comparison, not a separate end-to-end timing. Sampled
memory includes reference work and allocator caches; it is not a production-only
peak or a continuous maximum. Raw records retain reference and wall times.

[Machine-readable summary](summary.json) · [22 targeted tests](tests.log) ·
[Default posterior-law regression](default-law-test.log).
The 23 unique targeted tests pass. Archive replay checks 100 saved array hashes
and reproduces all 32 field blocks exactly, with identical ray/collision events
and recomputed quality summaries:
[A verification](A/verification.json),
[three-million verification](three_million/verification.json).
These replay checks do not re-run the original residual measurements.

## Scope and remaining work

These comparisons isolate conditioning, aggregation and query precision under
shared random draws. They do not establish the native prior sampler's full
covariance law, convergence to the analytic infinite-band model, absolute
Monte Carlo accuracy, continuous collision geometry or an upstream speedup.
Eight paired samples are especially insufficient to certify small event
probability errors. Geometry refinement and larger, independent ensembles remain
required. Failed native residuals are retained inside each verified solve's
metadata; a failed native solve is never treated as a cheap matched-quality
baseline.

Double verification also retains a large FFT workspace. The option is explicit,
not an automatic replacement for the default sampler. The next performance
comparison must include verification, correction, mean fitting and preparation,
including preconditioner setup where used.

## Reproduce and replay

From the pinned GPU environment, after preparing the published large cloud:

```bash
./reproduction/run_env.sh -m pytest -q tests/test_verified_streaming.py tests/test_streaming.py tests/test_preconditioned_streaming.py tests/test_refinement.py tests/test_half_spectral.py::test_half_posterior_moments_partial_batch_and_nonhermitian_cache
./reproduction/run_env.sh -m reproduction.benchmark_verified_posterior --case A --output reproduction/results/verified_posterior/A --archive /opt/geospsr-runtime/verified-posterior/A
./reproduction/run_env.sh -m reproduction.benchmark_verified_posterior --case three_million --output reproduction/results/verified_posterior/three_million --archive /opt/geospsr-runtime/verified-posterior/three_million
./reproduction/run_env.sh -m reproduction.verify_verified_posterior --output reproduction/results/verified_posterior/A --archive /opt/geospsr-runtime/verified-posterior/A
./reproduction/run_env.sh -m reproduction.verify_verified_posterior --output reproduction/results/verified_posterior/three_million --archive /opt/geospsr-runtime/verified-posterior/three_million
./reproduction/run_env.sh -m reproduction.summarize_verified_posterior
```

Query geometry and paired fields/events are public. Large weights, RHS and
coefficient arrays stay in the local runtime archive; their payload hashes are
public. Replay checks those hashes and re-evaluates fields/events from saved
coefficients. It does not re-execute conditioning residual checks. Reproduction
requires regenerating the runtime archive; it is not a public binary download.
