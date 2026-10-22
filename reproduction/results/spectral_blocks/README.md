# Positive spectral blocks: GPU implementation and overlap validation

Work on [issue #22](https://github.com/alecjacobson/fast-stochastic-psr/issues/22).
Keep a complete core cube and six disjoint positive/negative slabs, each with
one high-frequency coordinate. Every scalar/cross mode is included if its
cutoff is no larger than the core. There are no negative inclusion-exclusion
weights and no overlap. Multiple-high-axis corners are omitted deliberately.

`geospsr.block_spectral.BlockKernel` now applies this finite covariance using
one core transform and three shifted, conjugate-paired slab transforms.
`reproduction.block_posterior.BlockPosterior` fits the mean and generates
coherent posterior draws with cached scalar coefficients. The scalar/cross
band must fit inside the core. This is an experimental finite posterior;
it does not restore the omitted analytic spectrum or change the default solver.

## Dense analytic overlap

The existing fixed 16-query panels and gates are reused without tuning their
locations. Scalar/cross cutoff is 50, per-axis variance is .1, and positions,
normals and input hashes are retained in the linked records. Posterior moments
use float64 Cholesky and independent direct separable covariance sums, avoiding
Monte Carlo error. The production float32 NUFFT operator and fitted mean are
checked separately against those finite matrices.

| Input | Core / extent | Mean error % | Covariance error % | Analytic residual % | All panel gates |
| --- | ---: | ---: | ---: | ---: | --- |
| A, default | 100 / 200 | 0.01541 | 0.000204 | 0.01573 | Pass |
| B, default | 100 / 200 | 0.00856 | 0.001627 | 0.01258 | Pass |
| B, default | 100 / 400 | 0.00154 | 0.000282 | 0.00190 | Pass |
| B, default | 150 / 400 | 0.00112 | 0.000202 | 0.00144 | Pass |
| B, stress | 100 / 200 | 0.84382 | 0.338054 | 1.64886 | **Fail** |
| B, stress | 100 / 400 | 0.30093 | 0.118841 | 0.55336 | **Fail** |
| B, stress | 150 / 400 | 0.11793 | 0.045947 | 0.21166 | **Fail** |
| B, stress | 150 / 800 | 0.03888 | 0.015167 | 0.06935 | **Pass** |

Default parameters are length scale .01 and noise variance .02; stress uses
.005 and .0004. Gates are mean <0.2%, variance/covariance <0.02%, inside-CDF
RMS <0.1 percentage points, maximum <2 points, and analytic residual <0.1%.
At stress 150/800, variance error is 0.01528%, CDF RMS is 0.01670 points and
maximum is 0.06681 points. The production fit's analytic residual is 0.06941%.
Across the ten recorded settings, NUFFT action error versus its direct finite
matrix is at most 0.000208%, and fitted mean discrepancy is at most 0.000453%.

The 150/800 setting is the first passing result here for this stress panel.
It stores 203,942,851 modes, including only the positive member of each slab
pair, versus 4,103,684,801 in the enclosing full cube. These are mode counts,
not measured peak-memory or runtime reductions. Full-slice, event, geometry
and massive-input validation remain open. This does **not** establish a full
application speedup or solve the general short-scale problem.

[A records](gpu_A/results.json) · [B records](gpu_B/results.json) ·
[Retained stress failures](gpu_B_stress/results.json) ·
[Wider stress pass](gpu_B_stress_wide/results.json).
Each directory also contains compact raw mean/covariance/probability arrays.
The single-run stage times are diagnostics, not a competitive benchmark.
B-default and stress-150/800 runs encountered other GPU processes; their
timings are excluded from performance claims. No other job was stopped.

## Implementation checks

Small independent integer-frequency sums cover odd/even slab widths, periodic
seams, clustered inputs, a zero-frequency core, a core-only cube, and float32
and float64 arithmetic. Tests check actions, positive covariance, CG residuals,
joint scalar/vector prior moments, posterior moments, partial draw batches and
retained RHS against dense solves. See [test output](gpu-tests.log).
The complete suite passes **53 tests**. A separate
[production-band smoke check](gpu_B_stress_wide/sampling/results.json) generates
four coherent draws at stress 150/800 and checks their conditional weights on
512 fixed independent direct covariance rows. The largest component residual
is 0.001073%, below its 0.002% numerical gate. Four draws do not establish
Monte Carlo or application convergence. Recorded device usage is sampled at
stage boundaries and is not a peak-memory measurement.

For a positive slab, a real matrix action contributes twice its real part.
Its prior coefficients instead use `sqrt(2*rho)` times proper complex Gaussian
noise before taking the real part. Using twice the real part for these random
draws would double the paired slab's variance. The scalar prior shares only the
core's low-frequency Gaussian coefficients; observation noise is independent.
Identical RNG seeds across different block/cube layouts do not imply paired
sample realizations.

```bash
./reproduction/run_env.sh -m pytest tests/test_block_spectral.py -q
./reproduction/run_env.sh -m reproduction.audit_block_posterior --case A --output reproduction/results/spectral_blocks/gpu_A
./reproduction/run_env.sh -m reproduction.audit_block_posterior --case B --output reproduction/results/spectral_blocks/gpu_B
./reproduction/run_env.sh -m reproduction.audit_block_posterior --case B --lengthscale .005 --noise .0004 --output reproduction/results/spectral_blocks/gpu_B_stress
./reproduction/run_env.sh -m reproduction.audit_block_posterior --case B --lengthscale .005 --noise .0004 --blocks 150:800 --output reproduction/results/spectral_blocks/gpu_B_stress_wide
./reproduction/run_env.sh -m reproduction.check_block_draws --output reproduction/results/spectral_blocks/gpu_B_stress_wide/sampling
```

## Original prior-mass design audit

An independent closed-form normalization of the infinite periodic Matérn
spectrum gives the following **prior diagonal mass**. These percentages are
not posterior covariance errors or acceptance gates. Noise regularization,
observation geometry and query directions still determine downstream error.

| Length scale | Core | Extent | Retained / cube modes | Omitted corners % | Outside cube % |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.01 | 50 | 200 | 10,211,201 / 64,481,201 | 0.70381 | 0.32320 |
| 0.01 | 100 | 200 | 32,361,201 / 64,481,201 | 0.01440 | 0.32320 |
| 0.01 | 150 | 200 | 54,451,201 / 64,481,201 | 0.00061 | 0.32320 |
| 0.01 | 100 | 400 | 80,842,401 / 513,922,401 | 0.01857 | 0.04128 |
| 0.01 | 150 | 400 | 163,172,401 / 513,922,401 | 0.00168 | 0.04128 |
| 0.005 | 50 | 200 | 10,211,201 / 64,481,201 | 10.56295 | 2.40554 |
| 0.005 | 100 | 200 | 32,361,201 / 64,481,201 | 0.53167 | 2.40554 |
| 0.005 | 150 | 200 | 54,451,201 / 64,481,201 | 0.02865 | 2.40554 |
| 0.005 | 100 | 400 | 80,842,401 / 513,922,401 | 0.72115 | 0.32440 |
| 0.005 | 150 | 400 | 163,172,401 / 513,922,401 | 0.08409 | 0.32440 |

Lower mode storage does not prove lower NUFFT workspace or faster solves:
the GPU algorithm repeats source spreading across blocks, with padded
FFT grids and plan overhead. It also needs coherent prior sampling and an
omitted-spectrum estimator that includes the excluded corners, not just the
outside-cube tail. Coherent finite-block sampling is implemented; the matched
performance study remains open.

Follow-up: the [stratified tail audit](../block_tail/README.md) now covers both
omitted regions, validates their separate covariance contributions against
dense references, and evaluates a million-point panel. It also retains centered
block-inverse pilot failures and screens possible corner refinements using
fixed old weights. Full application validation and competitive speedups remain
open; the initial mode-count table alone provides neither.

The normalization check also exposed an inherited mismatch at longer scales:
the released spectrum is smaller than the variance-normalized analytic
spectrum by `1/(1+exp(-sqrt(3)/ell))` per axis. At ell=1 this leaves a 38.66%
3D diagonal deficit even with an infinite Fourier band. At the .01/.005 scales
in the table, the factor rounds to one in float64. The current default corrects
this factor; `normalization='released'` explicitly preserves the old spectrum.
[Issue #24](https://github.com/alecjacobson/fast-stochastic-psr/issues/24) tracks
the correction, compatibility and affected tail diagnostics.
`normalization-diagnostic.json` records the coefficient-ratio check. The block
mass formula uses the normalized analytic law.

Tests enumerate small integer-frequency sets to verify coverage, disjointness,
symmetry and mass conservation, and compare the independent one-dimensional
normalization to production at the audited short scales. An explicit Fourier
sum also reconstructs the analytic covariance at long scales. Reproduce with:

```bash
./reproduction/run_env.sh -m reproduction.audit_spectral_blocks
```
