# Hybrid range scheduling and cached analytic leaves

A cached-leaf variant reduces the full analytic B17419 matrix action from
**24.70 to 9.96 ms** across the retained pilots. In a fresh complete mean-fit
pilot, setup plus fitting takes **.242 s versus 1.849 s (7.63×)** for our double
analytic dense Cholesky reference. The short-scale/low-noise B fit takes
**.618 versus 1.850 s (2.99×)**. Both pass independent full-row residuals and
all 20,000 centered mean queries. A is still slower, and reused dense factors
remain much faster.

At one million points, a fresh paired action comparison improves repeated
three-RHS calls **2.231 → 1.282 s (1.74×)** and setup **11.62 → 9.59 s**, but
increases retained device storage **18.51 → 27.31 GB**. Only 1,024 rows are
independently checked against all million sources. This is an analytic action
speedup within our implementation, **not a massive fitting, posterior, SDD or
application speedup**.

The native library is already compiled in these pilots. The previous isolated
cold build cost .98 s; that cost is excluded here and materially changes first
installation latency. GPU/library initialization is charged when triggered,
with all first timings retained. These are single pilots, not independent
confirmation runs or the authors' complete pipeline timings.

## Why change the scheduling and leaves?

[CUDA event profiling](profile.json) separates the million-point compact action
into **1.860 s far sweeps** and **.371 s direct leaves**. The far sweeps are also
profiled in disjoint rectangle-size buckets:

| Entries per far rectangle | Million groups | Entries | Isolated bucket median |
| --- | ---: | ---: | ---: |
| 1–32 | 332,405 | 9,511,118 | .225 s |
| 33–128 | 448,461 | 28,497,602 | .583 s |
| 129–512 | 88,187 | 24,173,477 | .356 s |
| 513–8192 | 18,643 | 32,323,709 | .417 s |
| >8192 | 787 | 22,148,638 | .361 s |

Separate launches change occupancy and scheduling, so bucket medians need not
sum to the unsplit kernel time. The summed bucket output agrees with the full
far action to 5.94e-16 at a million points (4.08e-16 at B). This is stage timing,
not hardware-counter proof of a particular bottleneck.

First, groups with at most 32 entries use a cached serial 16-feature sweep;
larger groups retain the 512-thread warp-scan kernel. The fixed threshold was
declared before measurements. The source/target partition and cached coefficients
are unchanged. It reduces B actions from 24.70 to 18.92 ms, while A is slightly
slower. Both outcomes are retained in [results.json](results.json).

Second, direct-leaf kernel matrices are computed once in float64. Subsequent
calls multiply those small matrices by source weights instead of reevaluating
exponentials. Cross leaves store zero entries for same-side pairs; full x leaves
include diagonal terms. We test leaf sizes 32 and 128, moving work between exact
far rectangles and exact direct blocks. There is no dropped interaction, Fourier
cutoff, reduced precision or altered periodic kernel.

Thirteen hybrid tests and ten cached-leaf tests pass. They cover short/long
length scales, seams, tied/coincident coordinates, signed/unequal/zero RHS,
multiple tiles, singleton cases, memory caps and independent dense formulas.
Explicit pair enumeration verifies that each pair is covered exactly once for
leaf sizes 8, 32 and 128. Hybrid versus compact, and leaf32 versus hybrid,
retain identical geometry arrays; leaf128 intentionally changes the partition.

## A/B cached-leaf pilot

Same seed4501 three signed RHS, ell=.01, per-axis variance=.1, noise variance=.02.
Every A/B row is checked against the archived independent analytic action.
Relative L2 gates are 1e-7 per RHS, both with and without diagonal noise. All
checks pass, with covariance errors below 7e-15. Setup, first call and all three
later calls are retained in [leaves/results.json](leaves/results.json).

| Input / backend | Setup (s) | First call (ms) | Median next3 (ms) | Retained MB |
| --- | ---: | ---: | ---: | ---: |
| A hybrid32 | .13949 | 32.369 | .556 | 1.666 |
| A cached leaf32 | .00342 | .484 | .454 | 2.358 |
| A cached leaf128 | .00235 | .404 | .378 | 2.596 |
| B hybrid32 | .07580 | 18.937 | 18.906 | 148.997 |
| B cached leaf32 | .07022 | 14.613 | 14.591 | 174.014 |
| B cached leaf128 | .05138 | 10.134 | 9.965 | 179.852 |

The first A hybrid setup includes .0954 s of scheduling initialization and is
retained, not discarded. Later methods can reuse initialized common code; these
first timings are not independently cold comparisons. Leaf128 is selected for
the subsequent fit/scale pilots from these two tested sizes; no additional
threshold tuning is hidden.

## Full analytic mean fits

The independent dense Cholesky reference uses the same observed normals and full
analytic covariance. Range CG uses the existing unpreconditioned batched solver,
rtol5e-9/maxiter1000. All cases pass independent residual<=1e-8, relative weights
<=1e-7 per RHS, and relative centered mean error<=1e-7 on all 20,000 slice queries.
The common scalar/cross representation is double direct Fourier cutoff50; this
is not a scalar bandwidth-convergence, covariance or event validation.

| Model | Dense first fit (s) | Leaf128 first fit (s) | CG iterations | Worst independent residual | Mean relative error |
| --- | ---: | ---: | ---: | ---: | ---: |
| A, ell=.01, noise=.02 | .07265 | .07915 | 6 | 8.90e-10 | 2.69e-10 |
| B, ell=.01, noise=.02 | 1.84857 | .24219 | 17 | 2.15e-9 | 6.87e-10 |
| B, ell=.005, noise=.0004 | 1.85017 | .61810 | 54 | 4.83e-9 | 7.18e-10 |

First fit includes dense matrix construction/factorization/solve or range
planning/coefficient-and-leaf caches/solve. Input transfer, shared scalar setup,
query work, rendering and process startup are excluded. B mean aggregation,
centering and queries add .06196 s dense and .06982 s range: total stages
**1.911 versus .312 s (6.12×)**, excluding common scalar setup of .02578 s.
Stress B totals 1.912 versus .678 s. A's mean timings are initialization-sensitive;
no A reconstruction speedup is claimed.

Median reused dense triangular solves are .272, 17.03, 17.17 ms for A/B/stress,
versus range CG at 4.277, 184.22, 562.17 ms. Dense-factor reuse remains the stronger
choice. All solves, residuals, setup costs and mean arrays are retained in
[conditioning/results.json](conditioning/results.json) and its `*_fields.npz` files.

## Million-point paired action

Both backends use the same fixed million-point prefix, seed4501 RHS and saved
seed4601 sample of 1,024 rows. Each reference row sums every source. Models are
constructed sequentially, releasing one before the next. Caps are 200M retained
plan entries and 32GiB combined coefficient/leaf caches, with 40GiB initially
free GPU memory required. Both complete within the caps.

| Backend | Setup | First action | Median next3 | Retained storage |
| --- | ---: | ---: | ---: | ---: |
| Native compact32 | 11.618 s | 2.345 s | 2.231 s | 18.513 GB |
| Cached leaf128 | 9.587 s | 1.283 s | 1.282 s | 27.314 GB |

Leaf128 reduces far entries from 116.65M to 88.57M. It retains 13.46 GB of far
coefficients and 13.29 GB of direct matrices, plus geometry/offsets. CPU planning
takes 8.94 s and leaf-matrix preparation .411 s. This is a larger-memory backend,
not a bounded-memory streaming solution. Retained storage excludes CUDA context,
allocator pools and setup temporaries. Source hashes, process RSS high-water,
first/repeated calls and process observations are retained in
[scale1m/results.json](scale1m/results.json).

Worst sampled covariance error is below 7.1e-15 for both methods. Sampled-row
agreement does not establish full-row, solver, posterior or application accuracy.
The native compact control is freshly measured here; the 1.74× ratio compares
full three-RHS actions, not the 1,024-row reference timing.

All five timed GPU process audits pass. Periodic process sampling can miss
short-lived interference. Archive replay verifies ten A/B action comparisons,
three fitted systems with independently streamed full-row residuals, and both
million-point outputs with regenerated sampled analytic reference sums.

## Remaining work

The original flagship figures remain unchanged. [#43](https://github.com/alecjacobson/fast-stochastic-psr/issues/43)
still tracks hierarchical inter-tile scans; long rectangles still traverse tiles
serially. [#44](https://github.com/alecjacobson/fast-stochastic-psr/issues/44) tracks
memory/work reduction. The analytic action is still far more expensive than
finite-cube NUFFT actions at different analytic accuracy, so
[#45](https://github.com/alecjacobson/fast-stochastic-psr/issues/45) now tracks using
it for occasional full-model verification/refinement of fast NUFFT fits. That
combination has no measured benefit yet. Correcting solves does not supply a
missing coherent analytic prior. Full fits, SDD comparisons, posterior covariance
and applications at scale remain open.

[PROTOCOL.md](PROTOCOL.md) records each experiment before its measurements.

```bash
./reproduction/run_env.sh -m pytest -q tests/test_periodic_hybrid.py tests/test_periodic_leaf_cache.py
./reproduction/run_env.sh -m reproduction.profile_range_groups
./reproduction/run_env.sh -m reproduction.benchmark_range_hybrid
./reproduction/run_env.sh -m reproduction.benchmark_range_leaves
./reproduction/run_env.sh -m reproduction.benchmark_leaf_conditioning
./reproduction/run_env.sh -m reproduction.benchmark_range_leaf_scale
./reproduction/run_env.sh -m reproduction.validate_range_hybrid --gpu-residuals
```
