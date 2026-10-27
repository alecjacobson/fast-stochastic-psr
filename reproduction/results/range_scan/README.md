# Cached periodic range sums: faster actions and a first mean-fitting pilot

Caching fixed geometric coefficients reduces the B17419 full analytic range
matrix action from **90.12 to 21.91 ms (4.11×)**. The cached warp scan retains
**349 MB**, versus 7.90 MB for the original range geometry. All three signed
RHS match the independent full analytic covariance to below **7e-15** relative
L2, including a separate noise-free check. These are action timings; the strong
cached dense action previously measured 16.41 ms and remains faster.

A first complete mean-fitting pilot gives **1.849 → 1.320 s (1.40×)** on B,
including CPU planning and cache construction, versus a new double-precision
dense Cholesky fit. It is **slower on A and the short-scale/low-noise B case**.
Reusing an existing dense factor is much faster. This is a single pilot against
our analytic double reference, not a comparison with the authors' complete
pipeline or compiled SDD. Coherent sampling and application gains remain open.

## What changed

[Stage timing](profile.json) places about 95% of the B serial action in far-block
sweeps: median 86.81 ms, versus 4.405 ms in direct leaf blocks. CUDA events measure
these stages; Nsight Compute was unavailable, so there is no hardware-counter
claim about the exact mix of exponentials, memory traffic and dependencies.

The first optimization assigns one warp to each of the 16 x/y moment features.
Each warp performs a 32-entry prefix scan in z using decaying moment transfers;
the block combines the 16 feature contributions. The infinite periodic initial
state and direct leaf blocks are unchanged. Long rectangles still traverse
32-entry tiles serially. This is **not yet a hierarchical scan across tiles**.

That uncached scan is accurate but **slower on B**, at 107.27 ms versus the fresh
90.12 ms serial control. This result is retained in [results.json](results.json).
The subsequent cache stores the source or target feature appropriate to each
entry/direction, plus adjacent, warp-stride, tile-entry and periodic z decays,
all in float64. No repeated exponential evaluation remains in the cached far
action. Direct leaf kernels still evaluate exponentials. No Fourier cutoff,
spatial pruning or precision reduction is introduced. The exact original
pair partition and its periodic decomposition are described in the
[serial report](../periodic_range/README.md).

The cache uses 368 bytes per far entry, independent of the RHS count, in addition
to original geometry. At B it adds 341,432,240 bytes, for 349,332,870 retained
bytes total. Its construction takes about 4 ms after kernel initialization;
the roughly one-second CPU geometry plan dominates setup. Temporary CPU planner
lists, GPU setup allocations and allocator pools are not counted in retained
bytes. Each scan block has 4,096 bytes of shared scratch and no additional global
action scratch beyond the input/output allocation. A default 1 GiB cache cap
rejects larger cache requests; **massive-scale feasibility is not established**.

## Fixed A/B action measurements

NVIDIA L40; three identical deterministic real RHS, seed4501; ell=.01,
per-axis variance=.1, noise variance=.02, leaf32. Four calls per method:
first retained separately, median of the next three shown below. Whole-call
wall times synchronize the device and include validation and allocation.

| Input / action | Setup (s) | First action (ms) | Later median (ms) | Retained geometry/cache (MB) |
| --- | ---: | ---: | ---: | ---: |
| A serial | .01485 | 138.28 | 2.297 | .115 |
| A uncached warp scan | .01436 | 2.446 | 1.984 | .115 |
| A cached serial | .05519 | 119.64 | .702 | 3.868 |
| A cached warp scan | .01411 | .853 | .465 | 3.868 |
| B serial | .98848 | 98.01 | 90.121 | 7.901 |
| B uncached warp scan | 1.02180 | 107.40 | 107.272 | 7.901 |
| B cached serial | .93759 | 23.84 | 23.776 | 349.333 |
| B cached warp scan | 1.08377 | 21.92 | 21.906 | 349.333 |

All numerical gates pass. The 4.11× ratio combines the fresh serial run and
subsequent cached follow-up; it is a single pilot, not a repeated independent
confirmation. The uncached and cached experiment records remain separate.
First calls include initialization where triggered under the existing disk
cache. Common code can already be initialized by a preceding method, so these
are not independently cold process comparisons. All raw timings are retained in
[results.json](results.json) and [cached/results.json](cached/results.json).

The 20 new tests cover warp and cache variants, multiple tiles, repeated z and
coincident points, seams, signed/unequal and zero RHS, length scales .001–3,
and independent analytic formulas. The original 13 serial controls, including
explicit pair coverage and an independent image sum, remain unchanged.

## Complete mean-fitting pilot

Inputs are the original observed normals. The analytic covariance is checked
against an independently constructed dense double matrix. Range CG reuses the
repository's existing batched recurrence at rtol5e-9, maxiter1000, with no
preconditioner. The reference builds and factors the full analytic matrix with
Cholesky. Quality gates are independent residual <=1e-8 per RHS, relative weight
error <=1e-7 per RHS, and relative centered mean error <=1e-7 across all 20,000
slice queries. Both methods use the same double direct-Fourier scalar/cross
cutoff50 for this mean check; scalar bandwidth convergence is outside scope.

| Input/model | Dense first fit (s) | Range first fit (s) | Range iterations | Largest independent residual | Centered mean relative error |
| --- | ---: | ---: | ---: | ---: | ---: |
| A, ell=.01, noise=.02 | .08486 | .09457 | 6 | 8.90e-10 | 2.69e-10 |
| B, ell=.01, noise=.02 | 1.84882 | 1.31959 | 17 | 2.15e-9 | 6.87e-10 |
| B, ell=.005, noise=.0004 | 1.84984 | 2.16998 | 54 | 4.83e-9 | 7.18e-10 |

First fit includes matrix construction/factorization/solve for dense and
geometry/cache construction/solve for range. It excludes input transfer, the
shared scalar operator, mean aggregation/centering/query, rendering and process
startup. The standard B mean stages separately take 73.06 ms dense and 75.74 ms
range; including them gives **1.922 versus 1.395 s (1.38×)**, still excluding
shared scalar-operator setup (14.20 ms). A's first mean timings are strongly
initialization-sensitive and do not justify a reconstruction ratio.

For reused fits, median dense triangular solve versus range CG is .296 versus
4.887 ms at A, 17.03 versus 398.32 ms at B, and 17.18 versus 1,216.98 ms in the
stress case. **The cached dense factor wins reuse decisively.** This backend's
possible value is avoiding dense factorization/storage, not replacing a factor
that already exists.

The short/low-noise case is the same B model where the earlier
[cube-bandwidth study](../training_bandwidth/README.md#failure-retained-shorter-length-scale-and-lower-noise)
failed. Range CG passes the analytic fit and finite-cross mean checks there,
but is slower than dense first fitting. This does not establish posterior
covariance or event accuracy, nor solve the coherent-prior problem.

[conditioning/results.json](conditioning/results.json) retains all first and
three subsequent solves, CG reports, independent residuals, setup/mean timings,
input/source hashes and process observations. Saved weights and full mean arrays
are in its three `*_fields.npz` files. Archive replay verifies eight action
comparisons and all three fit comparisons; a separate streamed dense action
rechecks every row of all three fitted systems and both reference/candidate
weights. All four timed runs pass sampled process-isolation checks; periodic
sampling cannot exclude short-lived interference.

## Remaining work and reproduction

[#43](https://github.com/alecjacobson/fast-stochastic-psr/issues/43) remains open:
profile cached execution, improve memory layout/cache size and CPU planning,
then test hierarchical tile scans if needed. The current cache is too large to
assume practical million-point operation. #18 still needs compiled SDD comparisons,
coherent priors and complete application tests. No massive-scale or application
speedup is claimed by this pilot. The main flagship figures remain unchanged.

[PROTOCOL.md](PROTOCOL.md) records the successive experiments before their
measurements, including the unsuccessful uncached scan.

```bash
./reproduction/run_env.sh -m pytest -q tests/test_periodic_scan.py tests/test_periodic_cached.py
./reproduction/run_env.sh -m reproduction.profile_periodic_range
./reproduction/run_env.sh -m reproduction.benchmark_periodic_scan
./reproduction/run_env.sh -m reproduction.benchmark_periodic_cached
./reproduction/run_env.sh -m reproduction.benchmark_range_conditioning
./reproduction/run_env.sh -m reproduction.validate_range_scan --gpu-residuals
```
