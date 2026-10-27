# Compact periodic range cache and compiled geometry planning

The coefficient cache now stores **19 doubles per far entry instead of 46**.
At B17419, retained device storage falls **349.33 → 148.93 MB (57.4% smaller)**,
while action time increases **21.88 → 24.72 ms**. This is a memory/speed tradeoff,
not a faster action. Independent analytic covariance-action errors remain below
7e-15, including a separate check with the diagonal noise contribution removed.

Moving the unchanged stable-rank partition into C++ reduces B CPU planning from
**933.9 to 66.7 ms**, with **byte-identical plan arrays**. Complete compact setup
falls from .942 to .071 s in that comparison. The compiled library was cached;
a separate cold compilation/loading measurement takes **.980 s**.

A new analytic mean-fitting pilot, including geometry/cache setup, gives:

| Model | Dense double Cholesky first fit | Native compact first fit | Ratio | Independent residual, worst RHS |
| --- | ---: | ---: | ---: | ---: |
| A, ell=.01, noise=.02 | .0740 s | .0763 s | Dense slightly faster | 8.90e-10 |
| B, ell=.01, noise=.02 | 1.8487 s | .5229 s | **3.54×** | 2.15e-9 |
| B, ell=.005, noise=.0004 | 1.8509 s | 1.4468 s | **1.28×** | 4.83e-9 |

These are **single pilots with the native library already compiled**, compared
with our independent analytic double reference. The one-time .980 s cold library
build is excluded from these rows and can erase the stress-case advantage. GPU
and library initialization are included where triggered, but this is not an
independently cold process comparison. This is not an upstream pipeline or
compiled SDD speedup. Reusing an existing Cholesky factor remains much faster.

The exact analytic action also executes at one million observations, using
**18.51 GB retained device storage**, **11.13 s setup** and **2.231 s per repeated
three-RHS action**. Its 1,024 checked rows pass against all million sources.
That is sampled action evidence, **not full-row, fitting, posterior or application
validation**, and establishes no massive-scale speedup.

## Implementation and controls

The [preceding expanded-cache backend](../range_scan/README.md) remains unchanged.
[Cached stage timing](profile.json) measures B far sweeps at17.34ms and direct
leaves at4.40ms. CUDA events distinguish these stages; no hardware-counter profile
is available. The compact cache stores four one-dimensional features per x/y
axis and direction:16 coefficients instead of the expanded32 two-dimensional
products. The action forms each required x/y product when used. Coefficients
are stored by channel to make accesses across a warp contiguous.

Only adjacent z decay and the two periodic initial weights are retained.
At each warp-scan level, products of adjacent decays reconstruct the needed
longer transition. The final prefix product supplies the tile-entry transition.
No division by small exponentials, increasing exponential weights, reduced
precision, spatial pruning or Fourier truncation is used. The analytic periodic
closure is unchanged. The resulting19 float64 values require152 bytes per far
entry, down from368. No temporary expanded cache is allocated.

The C++ planner reproduces the Python partition: stable x rank splits, stable y
cross-block partitioning, local source bounds, stable z order and direct leaves.
It is compiled with `g++ -O3 -std=c++17 -shared -fPIC`, without fast-math, and
loaded through `ctypes`. The library is cached by source/flags hash under
`$GEOSPSR_RUNTIME/native-range` (or the system temporary directory if unset).
This optional backend requires a C++17 compiler. Library time and cached status
are recorded independently from CPU planning, device transfer and cache setup.
Retained-entry and cache-byte caps reject oversized plans; they are not a
streaming implementation or a bound on all allocator/process overhead.

Thirteen compact-cache tests pass at length scales .001,.01,.3,3, with seams,
coincident/repeated coordinates, signed/unequal/zero RHS, multiple tiles and a
separate expanded-cache control. Nine native-planner tests compare every plan
array and actions against Python, including singleton and tied inputs. The
first native test run had three incorrect exception expectations for invalid
zero caps: the implementation correctly raised ValueError rather than the
expected MemoryError. The test was corrected; all numerical/plan comparisons
had already passed. [Test history](test_history.json) retains that failure.

## A/B action and planning measurements

All actions use the same seed4501 three signed RHS, ell=.01, per-axis variance=.1,
noise variance=.02 and leaf32. Quality gates are1e-7 relative L2 per RHS, both
with and without noise. A/B reference actions use every input row.

| Backend / input | Setup (s) | First action (ms) | Median next3 (ms) | Retained MB |
| --- | ---: | ---: | ---: | ---: |
| Expanded, A | .05700 |119.71|.473|3.868|
| Compact, Python, A |.01460|.893|.528|1.665|
| Expanded, B |.94170|21.922|21.880|349.333|
| Compact, Python, B |.97413|24.738|24.719|148.927|
| Native compact, A (separate planner comparison) |.00319|.554|.524|1.665|
| Native compact, B (separate planner comparison) |.07083|24.737|24.701|148.927|

The native comparison also reruns Python compact setup/actions and checks hashes
of all nine plan arrays. [Native records](native/results.json) retain both
methods. [Initial cache records](results.json) retain the separate expanded/
compact comparison. First actions and all three later timings remain available;
common kernels can be initialized by earlier methods. No startup speedup is
inferred from different initialization states.

## Fitting quality and reuse

The fit protocol is unchanged from the preceding pilot: observed normals,
independent analytic dense matrix, double Cholesky, and batched unpreconditioned
CG with rtol5e-9/maxiter1000. All three cases pass independent residual<=1e-8,
weight-relative-error<=1e-7 per RHS and centered mean-relative-error<=1e-7 over
all20,000 slice queries. Mean errors are2.69e-10,6.87e-10 and7.18e-10. The shared
scalar/cross model uses direct double Fourier cutoff50; this is not a scalar
bandwidth-convergence test or a variance/event comparison.

Range CG takes6,17,54 iterations. Median reused solves are5.326,449.67,1372.55ms,
versus dense triangular solves of.273,17.03,17.17ms respectively. Standard B
fit plus mean aggregation/centering/query totals **1.911 vs .583 s (3.28×)**,
excluding shared scalar-operator setup of.0265s. Stress B totals1.912 vs1.507s.
A's mean times depend heavily on first-use initialization; no A reconstruction
speedup is claimed. Full first/subsequent solve, mean and setup costs are in
[conditioning/results.json](conditioning/results.json).

## Scaling the analytic action

Use fixed prefixes of the archived three-million-point scorpion cloud, the same
three seed4501 RHS and1,024 random query rows chosen with seed4601. The independent
analytic kernel sums **all sources** at each selected row. Saved controls include
the entire input point arrays, query row indices and sampled outputs; RHS are
reconstructed from the recorded seed and hash.

| N | Setup | CPU partition | First action | Median next3 | Retained device storage | Process RSS high-water after plan |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
|100,000|.706s|.623s|.278s|.162s|1.202GB|.836GB|
|1,000,000|11.127s|10.469s|2.349s|2.231s|18.513GB|1.990GB|

The100k CPU setup overlapped a separate CPU-only cold-library compilation; its
timing is retained with that qualification. The million run did not overlap
that compilation. Both use already compiled native libraries. Process RSS is
the cumulative CPU process high-water mark, including libraries and earlier
allocations; retained device bytes exclude CUDA context, allocator pools and
setup temporaries. The million plan stores116,654,544 far entries and15,509,704
direct entries. Caps were20M entries/4GiB cache at100k, and200M/32GiB at1M;
the latter also required40GiB initially free GPU memory. Both runs completed.

Worst sampled covariance-action error is6.51e-15 at100k and7.08e-15 at1M.
[100k records](scale100k/results.json) and [million records](scale1m/results.json)
retain all calls. The reference timings evaluate only1,024 rows and **must not**
be divided into full-action times as a speed comparison. Sampled agreement does
not prove accuracy of unsampled rows or a conditioned posterior.

All six timed GPU process audits pass; periodic sampling can miss short-lived
interference. Saved replay checks eight A/B action comparisons, three complete
fit comparisons with independently streamed full-row residuals, and both larger
sampled controls with regenerated independent reference sums.

## Remaining work and reproduction

[#44](https://github.com/alecjacobson/fast-stochastic-psr/issues/44) remains open
for further cache/work reduction and complete fits at scale. The million action
is memory-heavy and slow compared with prior finite-cube action timings, which
have different analytic error; there is no matched-quality speedup claim.
[#43](https://github.com/alecjacobson/fast-stochastic-psr/issues/43) still tracks
hierarchical inter-tile scans. Dense-factor reuse and compiled SDD must remain
strong controls. Coherent priors, covariance and full applications remain open
under#18/#6. This work does not establish intractability for upstream SDD.

[PROTOCOL.md](PROTOCOL.md) declares each follow-up before its measurements.

```bash
./reproduction/run_env.sh -m pytest -q tests/test_periodic_compact.py tests/test_native_range.py
./reproduction/run_env.sh -m reproduction.profile_cached_range
./reproduction/run_env.sh -m reproduction.benchmark_periodic_compact
./reproduction/run_env.sh -m reproduction.benchmark_native_range
./reproduction/run_env.sh -m reproduction.benchmark_range_cold_library
./reproduction/run_env.sh -m reproduction.benchmark_native_conditioning
./reproduction/run_env.sh -m reproduction.benchmark_range_scale
./reproduction/run_env.sh -m reproduction.benchmark_range_million
./reproduction/run_env.sh -m reproduction.validate_range_compact --gpu-residuals
```
