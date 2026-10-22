# All omitted block frequencies: covariance diagnostics and conditioning pilots

The positive-block backend now has an error diagnostic that includes **both
inside-cube corners and the infinite outer tail**. Dense controls validate the
two contributions independently. At one million observations, the fixed-panel
trace-bound estimate is **0.00253%** of the finite posterior
covariance Frobenius norm. This is an empirical covariance diagnostic, not a
full application accuracy certificate or a speedup.

## Covariance overlap and massive-input panel

All runs retain scalar/cross cutoff 50 and analytic normalization. Default
parameters are length scale .01, noise variance .02; stress uses .005 and .0004.
Per-axis kernel variance is .1. Inputs and the previously selected 16 queries
are fixed; hashes are in each result. Each of three seeds (71–73) draws 16,384
frequencies **per region**, for 49,152 corner and 49,152 outer frequencies after
pooling. These are validation frequencies, not posterior sample counts.

| Input | Core / extent | Actual covariance gap % | Estimated H norm % | Estimated trace bound % ± 1.96 SE | Corners' share of trace |
| --- | ---: | ---: | ---: | ---: | ---: |
| B, default | 100 / 200 | 0.001627 | 0.001627 | 0.004999 ± 0.000008 | 4.2% |
| B, stress | 150 / 800 | 0.015169 | 0.015173 | 0.046818 ± 0.000057 | 69.6% |
| 1,000,000, default | 100 / 200 | not computed | 0.000900 | 0.002533 ± 0.000005 | 4.5% |

The exact theorem is `0 <= V_full - V_block <= H_corner + H_outer`, a PSD
matrix inequality. Consequently `||gap||_F <= ||H||_F <= trace(H)`. The table
normalizes these quantities by `||V_block||_F`. Estimated H norms and traces
are not certified bounds. The reported interval applies only to the trace;
it is an empirical normal approximation, not rigorous or simultaneous coverage.
For the stress case, the trace is more conservative than the matrix norm:
the actual gap and exact H norm pass the earlier 0.02% covariance gate even
though the trace exceeds it. Neither that gate nor the table certifies every
joint direction or complete application probabilities.

At stress 150/800, corners contribute **69.6%**
of the trace bound. An outside-cube-only estimator would miss most of it.
[Issue #25](https://github.com/alecjacobson/fast-stochastic-psr/issues/25) tracks
whether adding selected two-high-axis blocks can improve the accuracy/cost
tradeoff. More transforms may erase any storage benefit.

[Default B records](B-default/results.json) · [Stress B records](B-stress/results.json) ·
[Million-point records](million-100-200/results.json) · [Method](../spectral_tail/METHOD.md).
Raw NPZ files retain covariance matrices, frequency draws and per-frequency
diagonal/trace samples. Small tests also verify the corner distribution by
explicit enumeration, stratum masses and independent feature quadratics.

## Numerical control and limitations

Dense controls use float64 direct separable covariance rows and Cholesky; no
extent-cubed reference spectrum is allocated. At large N the production run
uses float32, NUFFT eps 1e-6 and CG rtol 1e-5. A separate float64 run at eps/rtol
1e-9 checks probes [0, 8, 15]. Its finite covariance differs
by **0.000022%** in relative Frobenius
norm, with maximum variance difference
**0.000017%**.
This covers three probes, not all sixteen or a global numerical-error bound.
The feature diagnostic uses float64 phases/outer accumulation and float32
feature products in both runs. [Control records](million-double/results.json).

The million-point maximum pooled variance-bound estimate is
0.00338%; taking the maximum of
marginal normal interval endpoints gives
0.00340% and does
**not** provide simultaneous 95% coverage. Mean/centering, sample convergence,
full slices, ray/collision geometry, hitboxes and sequential decisions remain
separate requirements. This code does not add the omitted prior or solve the
full analytic posterior. The goal remains open.

## CPU screen for a positive corner refinement

Using the saved stress-panel frequencies and **unchanged old inverse weights**,
we screen disjoint two-high-axis boxes. Each added pair of high coordinates
extends from core150 to the selected edge; the third coordinate remains in
the core. Triple-high frequencies stay omitted. The partition is checked by
independent small-grid enumeration; it has no GPU backend yet.

| Edge | Added conjugate-paired stored modes | Captured corner trace contribution | Captured total trace contribution |
| --- | ---: | ---: | ---: |
| 200 | 4,515,000 | 29.9% | 20.8% |
| 250 | 18,060,000 | 58.8% | 40.9% |
| 300 | 40,635,000 | 74.9% | 52.1% |

These are contributions to the **existing diagnostic**, not predicted new
posterior errors. A new inverse changes its weights; the screening panel also
cannot serve as independent validation. Each nonempty refinement adds six
positive transform pairs, so repeated source spreading may outweigh the mode
count benefit. [Screening record](corner-screen.json); implementation and
matched-quality timing remain under issue #25.

## Centered block-inverse screening

A separate experiment keeps the cutoff200 cube prior, observed normals,
fitted mean, centering, eight stored random RHS and application geometry fixed.
Only the inverse applied to zero-mean fluctuations changes. This is a hybrid
law, different from the consistently finite block model audited above. Its
quality is assessed against the existing centered cube200 shared-draw control;
the tail diagnostic alone does not validate it.

| Inverse core / extent | Empirical covariance error | Analytic residual upper endpoint | Field gates | Eight-draw solve time |
| --- | ---: | ---: | --- | --- |
| 50 / 200 | 0.0402% | 0.0870% | **Fail** | 91.42 s (excluded: contention) |
| 100 / 200 | 0.0003% | 0.0463% | Pass | 99.50 s (single run) |

The existing covariance gate is 0.02%; it is unchanged. Analytic residuals use
4,096 independent held-out rows, excluding the 2,048 stopping rows; their
reported upper endpoints are empirical marginal intervals with a 0.1% gate.
An identical selected view or sampled event pattern cannot override failed
covariance checks. Eight draws do not establish event-probability convergence.
These are single-run screens, not repeated performance measurements. Timings
with foreign GPU activity are retained and excluded; no other job was stopped.
No new speedup is claimed. The earlier clean three-repetition cube150 result
is 38.14 s for the same eight RHS, with its own quality evidence in the
[centered-hybrid report](../centered_hybrid/README.md).
The block100 pilot passes the field and residual gates but takes 99.50 s in
its isolated single run, so it offers no speed advantage in this screen.
Reduced spectral storage did not translate into a faster conditional solve.
[Issue #26](https://github.com/alecjacobson/fast-stochastic-psr/issues/26)
proposes a different next trial: exploit conjugacy within the complete cube,
preserving the finite model while reducing the main FFT grid.

The [dense B block50 hybrid](centered-B-50/results.json) passes the fixed-panel
moment gates, but this alone does not predict million-point finite-ensemble
agreement. The CLI exposes `--block-core` only for centered fluctuations and
requires `50 <= core <= solve-cutoff <= 200`, preserving the held prior.
All pilot results, including failures, are retained.
The complete regression-suite output is archived in [tests.log](tests.log).

## Reproduce

Prerequisites are the matched B data, saved spectral-tail panels, million-point
cloud and common RHS scratch data described in the earlier reproduction reports.
Outputs may live under /opt scratch first to avoid filling the checkout disk.

```bash
./reproduction/run_env.sh -m reproduction.audit_block_tail --dataset B --output reproduction/results/block_tail/B-default
./reproduction/run_env.sh -m reproduction.audit_block_tail --dataset B --core 150 --extent 800 --lengthscale .005 --noise .0004 --output reproduction/results/block_tail/B-stress
./reproduction/run_env.sh -m reproduction.audit_block_tail --dataset 1000000 --output reproduction/results/block_tail/million-100-200
./reproduction/run_env.sh -m reproduction.audit_block_tail --dataset 1000000 --precision float64 --rtol 1e-9 --eps 1e-9 --probe-indices 0 8 15 --seeds 71 --output reproduction/results/block_tail/million-double
./reproduction/run_env.sh -m reproduction.audit_centered_hybrid --case B --cutoffs 200 --block-core 50 --output reproduction/results/block_tail/centered-B-50
./reproduction/run_env.sh -m reproduction.screen_corner_refinement --input reproduction/results/block_tail/B-stress --output reproduction/results/block_tail/corner-screen.json
./reproduction/run_env.sh -m reproduction.benchmark_nufft_batches solve --samples 8 --batch 1 --repeats 1 --centered-fluctuations --solve-cutoff 200 --block-core 50 --audit-analytic --stop-rows 2048 --holdout-rows 4096 --holdout-seed 2711 --scratch /opt/geospsr-runtime/common-rhs-128 --reference reproduction/results/centered_hybrid/million-200 --output reproduction/results/block_tail/centered-million-50
# Repeat the last command with --block-core 100 and output centered-million-100.
./reproduction/run_env.sh -m reproduction.summarize_block_tail
```
