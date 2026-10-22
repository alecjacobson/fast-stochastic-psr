# Matched scorpion reproduction

The figures compare the released GeoSPSR calculation with a NUFFT calculation
on **identical inputs**, retaining the full analytic training covariance and
computing deterministic marginal variances. They replace the earlier
million-point image as the README's evidence of agreement.

## What was recovered from the release

[Upstream `experiments/example.py`](https://github.com/sholalkere/GeoSPSR/blob/2352f1ef98aab20e0303611c1fa2b0948acd2a05/experiments/example.py)
supplies the mesh, normalization, two camera positions/directions, field of view,
scan density 200, perturbation seed 0, noise, kernel parameters, truncation,
interpolation grid, and a horizontal slice at **y = −0.75**. It generates **691
observations** with the pinned geometry dependencies. Those exact settings form A.
The geometric scan functions are unchanged; their interactive plotting import
is deferred for headless execution.

The release does not supply a Figure-1-specific scene, a second slice height,
or the observations that generated the paper's raster. A height survey of the
released scan identifies **y = −0.35** as producing the upper panel's two separated
regions. This height is an explicit visual inference, not recovered metadata.
The figures follow [Figure 1 of the paper](https://arxiv.org/pdf/2503.19136), but
we do not claim pixel-identical reconstruction of its unpublished setup.

B increases only the camera raster density from 200 to 1,000, producing **17,419
observations**. It preserves the two cameras, noise parameters and seed, kernel,
planes and query resolution. Both methods receive the same actual observations,
normals and queries, with SHA-256 fingerprints recorded in their JSON files.
The ray samples and their perturbations are regenerated, not duplicated.

Both planes use 100 × 100 query positions over `[-π/2, π/2]²`. The cloud renderer
draws every observation; it does not insert a fitted mesh. Marker radii are
chosen for legibility and recorded in each `cloud_projection.json`.

## Three distinct calculations

1. **Untouched released APIs.** We import `compute_mean` and `compute_variance`
   from a separate upstream checkout, and run them on all 20,000 displayed
   queries in A and B. Both the original mean-centering call and grid means are
   retained. The API uses the analytic product-Matérn covariance and its released
   50³ trilinear cross-kernel interpolant. `sigma = 0.02` is added to the diagonal,
   exactly as in the example, despite the observation perturbation's standard
   deviation also being 0.02. We do not substitute the local corrected example's
   noise parameter.
2. **Cached reference.** We reuse a Cholesky factor and solve the same formula
   with the same released kernel. Its agreement with the untouched API is better
   than `5e-7` relative error for mean and `6e-8` for variance. This is the stronger
   baseline for warm query timings. Copies at the JAX/CuPy boundary own their
   storage and synchronize before JAX can recycle its output buffers; a lifetime
   error discovered during the large-case audit was fixed before the final runs.
3. **NUFFT, matched interpolant.** A direct NUFFT evaluates the underlying finite
   Fourier kernel, which differs from the release's interpolation. To match the
   actual release more closely, we Fourier-transform its periodic piecewise-linear
   nodal basis analytically, contract the tensor-product coefficients with its
   grid values, and truncate this interpolant representation to **201³ modes**.
   Grid values come from the original **101³-mode** Poisson kernel. NUFFT tolerance
   is `1e-6`. Both source aggregation and query evaluation are included in mean
   timings. Oversampling is 1.25 for point sets below 100,000 and 2 otherwise.

The deterministic variance formula is

```text
K = K_analytic(X, X) + 0.02 I = L Lᵀ
μ(q) = k_fv(q, X) K⁻¹ normals − mean_at_observations
variance(q) = prior_variance − ||L⁻¹ k_vf(X, q)||²
P(inside at q) = Φ(−μ(q) / sqrt(variance(q)))
```

The released example plots mean and variance only. The probability panels here
apply the same Gaussian CDF to the original and NUFFT outputs, using the
negative-inside convention of the released mesh normals and computed mean.

The squared norm sums over all observations and all three vector components.
We stream query batches of 1,024. There are **no Monte Carlo draws, diagonal
training approximations, point subsets, or spatial smoothing** in these figures.
Rendering uses bilinear pixel interpolation. Mean/variance color limits are
shared between compared methods for each slice, matching the release's linear
normalization; numeric limits are shown. Probability always uses `[0, 1]`.

## Agreement

| Quantity | A: 691 observations | B: 17,419 observations |
| --- | ---: | ---: |
| Mean relative L2 error | 0.1226% | 0.0305% |
| Variance relative L2 error | 0.00218% | 0.00740% |
| Probability RMS error, percentage points | 0.0479 | 0.0411 |
| Probability maximum error, percentage points | 0.621 | 1.665 |
| Maximum variance error / reference variance range | 0.650% | 0.224% |

Errors are measured before display interpolation. Probability maximum errors
occur near the boundary; a small mean difference matters more there. These are
measured approximation errors, not a claim of exact numerical identity.

[Absolute error maps A](A/errors.png) · [Absolute error maps B](B/errors.png) ·
[Comparison A](A/comparison.json) · [Comparison B](B/comparison.json)

The retained `nufft.npz` diagnostic uses the original finite Fourier kernel
directly. It has about 1.34%/1.27% mean error against the interpolated release in
A/B, despite accurately evaluating the underlying Fourier sum. Matching the
interpolant removes most of that discrepancy. The primary panels use
`compatible.npz`, not these diagnostic fields.

## Timings and scope

GPU: NVIDIA L40, 48 GB. Wall times synchronize the GPU. Other GPU jobs were run
sequentially. Clocks were not locked. Warm mean timings are medians of three
complete calls; variance timings are one repeated full pass after its first call.

| B: 17,419 observations, 20,000 queries | Released API | Cached reference | Matched NUFFT |
| --- | ---: | ---: | ---: |
| Mean, repeated call | 1.858 s | 0.3921 s | 0.02277 s |
| Variance, repeated call | 6.265 s | 1.172 s | 2.266 s |
| Sum of mean and variance repeated calls | 8.123 s | 1.564 s | 2.289 s |

The API's repeated calls include fitting and closure/compilation overhead;
the cached-reference and NUFFT query timings exclude fitting and kernel setup.
The **17.2× mean speedup** uses the stronger cached reference with the same
scope. The **81.6× ratio to the API mean call** includes those additional API
costs and should not be confused with a pure fast-summation speedup. Cached
reference variance is faster than our current NUFFT variance path. A is small
enough that its cached reference is faster for both fields; it is an agreement
check, not a large-N speed claim.

For a complete first reconstruction, summing the measured first-call stages
(kernel setup, fitting where separate, mean at observations, grid mean, variance)
gives **28.10 s → 2.96 s, or 9.5×**, for B. The analogous A sum is 23.37 s → 1.56 s.
These totals include compilation where applicable, but exclude Python/process
startup, geometry generation, file writes, rendering and probability-CDF
postprocessing. Kernel setup is 13.57 s for the released API versus 0.297 s for
the matched NUFFT representation in B. Full timing breakdowns are retained in
[original.json](B/original.json) and [compatible.json](B/compatible.json).

## Why the previous million-point teaser was insufficient

It changed point sampling, slice height, training covariance and color
normalization together, and estimated variance from 256 samples. The 8.8%
relative Monte Carlo standard error can dominate the small variance differences
visible at the released scan density. Its zero-based variance color scale also
hid that contrast. Those changes prevented an equivalence claim.

The million-point experiment remains useful as an explicitly finite-GP scaling
experiment, but its posterior is not the full analytic training model used here.
A million-point dense float32 covariance alone requires 4 TB. Panel B is sized
so both original and accelerated calculations can actually run. Matching the
full model at one million observations and accelerating deterministic variance
remain open work; the old 4,015× mean-only timing is not this comparison's speedup.

## Reproduce and validate

Follow the matched-figure commands in the [root README](../../../README.md).
`matched_teaser.py --backend original --literal-api` executes both the untouched
API and cached reference, asserting agreement. `--backend compatible` computes
the matched NUFFT fields independently. `compare_matched_teaser.py` computes
errors and renders the paired figures. `validate_matched_teaser.py` verifies
input hashes, field tolerances, API agreement, and NUFFT against direct Fourier
sums. The test suite additionally checks streamed conditioning against the
released implementation and the Fourier integrals of the interpolation basis.
