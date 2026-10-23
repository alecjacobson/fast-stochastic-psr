# Reusable spatial preconditioning for posterior draws

**The tested spatial blocks do not reduce million-point iteration counts.** They do not establish a conditioning speedup; the default solver remains unchanged.

This experiment adds optional spatial block Jacobi to packed full-cube200 CG.
It compares the **same covariance operator, right-hand sides, prior coefficients
and query geometry**. The analytic block matrices are used only as an inverse
preconditioner; they do not replace the finite training covariance or change
the posterior prior. The mean is held fixed. All three repetitions and every
candidate in the [declared protocol](PROTOCOL.md) are retained.

## Setup and conditioning

NVIDIA L40, float32, NUFFT epsilon1e-6, CG tolerance1e-5, noise variance .02,
length scale .01, per-axis variance .1. Million-point runs condition eight common
draws in batches of two; B runs condition four draws at 17,419 observations.
The three solve repetitions reuse each preconditioner, while the setup-plus-solve
column charges construction once for each map's first ensemble. Source transfers,
NUFFT plan setup, mean fitting, prior generation and downstream queries are
outside these conditioning totals.

| Run | Preconditioner setup (s) | Median solves (s) | Setup + solves (s) | Iterations/batch | Retained preconditioner (MiB) | Whole-run sampled isolation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| baseline8 | 0.000 | 30.516 | 30.516 | 57–57 | 0.0 | False |
| block64 | 0.634 | 31.602 | 32.235 | 58–58 | 251.8 | True |
| block128 | 0.507 | 31.780 | 32.287 | 58–58 | 495.9 | True |
| block256 | 0.951 | 31.150 | 32.102 | 57–57 | 984.4 | True |
| B-baseline4 | 0.000 | 1.976 | 1.976 | 11–11 | 0.0 | True |
| B-block1284 | 0.102 | 1.729 | 1.831 | 9–9 | 8.7 | True |

**No timing ratio is claimed:** the declared whole-run isolation requirement fails. Timings remain diagnostics, and no repetition is dropped.

No candidate reduced million-point iterations, so this pilot does not justify a 128-draw timing expansion. Next experiments should test a shared coarse space or deflation rather than assume larger ensembles will remove per-solve overhead.

On B, block128 reduces iterations from 11 to 9; recorded setup-plus-solve time
is 1.831 s versus 1.976 s without preconditioning. This modest small-input
improvement does not carry over to the million-point case and does not establish
that NUFFT is preferable to the existing dense/SDD alternatives on B.

Periodic process samples cannot prove exclusive device access. Other jobs were
not stopped. Retained preconditioner storage includes inverse blocks and the
permutation; it excludes construction scratch, NUFFT plans, residuals and source
arrays. Raw process observations and stage timings are in each run's results.
Do not compare these full-cube200 conditioning numbers with earlier centered150
hybrid timings as if they used the same inverse model.

## Numerical evidence

| Candidate | Field error / fluctuations | Empirical covariance discrepancy | Independent finite-row residual | All numerical/event gates |
| --- | ---: | ---: | ---: | --- |
| block64 | 0.003441% | 0.002769% | 0.001015% | True |
| block128 | 0.004997% | 0.002169% | 0.001264% | True |
| block256 | 0.004626% | 0.002307% | 0.000950% | True |
| B-block1284 | 0.000906% | 0.000332% | 0.001140% | True |

The unchanged same-model gates require field/fluctuation error below .05%,
empirical variance and covariance discrepancies below .01%, and the existing
paired ray/collision/next-view gates. Independent double-precision cosine sums
check 32 finite-covariance rows at seed20261023; their residual threshold is
.002%. These sampled rows supplement the full NUFFT true residual, but do not
certify the full analytic kernel. Raw quality records retain all failures.

Small independent tests compare analytic preconditioner blocks with long
Fourier sums, and PCG solutions with dense finite-kernel solves. They cover
periodic seam coordinates, heteroscedastic noise, partial blocks, zero RHS,
initial-solution preservation and rejection of nonpositive preconditioners.
[Full test output](tests.log).
The production experiment uses homoscedastic noise; this does not validate
ContactFusion's observation model or sequential prior.

## Implementation and limits

`SpatialBlockJacobi` uses a deterministic Morton ordering of periodic point
coordinates and contiguous groups. It builds analytic principal blocks in
bounded batches, adds observation variances, checks Cholesky factorization and
stores symmetrized inverses. Construction uses double precision, application
uses the requested solve dtype. No silent jitter is added. Padding in the last
block is decoupled from actual observations. The grouping can split neighbors
at block or periodic seams; it is a heuristic, not a nearest-neighbor guarantee.

`solve(..., preconditioner=...)` opts into independent PCG recurrences for full,
folded or packed real systems. The existing unpreconditioned path is unchanged.
PCG checks positive operator/preconditioner curvature and reports the residual
against the original covariance action and original RHS norm. A converged
recurrence alone is insufficient. Explicit inverse storage costs O(Nb), and
local blocks need not resolve long-range modes. More elaborate overlapping or
coarse-space preconditioners must include their additional setup/application
cost and preserve the same posterior model.

These are solver experiments on eight/four shared draws, not independent
probability convergence studies or complete applications. The
[collision discretization failures](../query_refinement/README.md), absolute
Monte Carlo accuracy and full analytic-tail/scalar-band convergence remain
open. B's established dense/SDD alternatives are faster than its NUFFT fits;
this comparison does not establish NUFFT as the right small-input method.

## Reproduction

Use the [common-RHS preparation](../nufft_batches/README.md), with scratch outside
the git checkout. Run the baseline first, then each block size in a fresh process:

```bash
./reproduction/run_env.sh -m reproduction.benchmark_nufft_batches solve \
  --samples 8 --batch 2 --repeats 3 --pack-rhs \
  --scratch /opt/geospsr-runtime/common-rhs-128 \
  --output /opt/geospsr-runtime/preconditioning/baseline8
./reproduction/run_env.sh -m reproduction.benchmark_nufft_batches solve \
  --samples 8 --batch 2 --repeats 3 --pack-rhs --preconditioner-block 128 \
  --finite-rows 32 --scratch /opt/geospsr-runtime/common-rhs-128 \
  --reference /opt/geospsr-runtime/preconditioning/baseline8 \
  --output /opt/geospsr-runtime/preconditioning/block128
```

Repeat the candidate command for 64 and 256, keeping every output. For B use
`--dataset B --samples 4 --scratch /opt/geospsr-runtime/common-rhs-smoke`, with
its own baseline and block128 reference comparison. Compact fields and raw
results are published here. The summary generator reads those published files:
`./reproduction/run_env.sh -m reproduction.summarize_preconditioning`.
[Issue #30](https://github.com/alecjacobson/fast-stochastic-psr/issues/30) tracks
shared preconditioning; [#6](https://github.com/alecjacobson/fast-stochastic-psr/issues/6)
tracks broader conditioning/FMM approaches.
