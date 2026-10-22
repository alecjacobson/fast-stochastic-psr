# Positive spectral block design audit

CPU-only design check for [issue #22](https://github.com/alecjacobson/fast-stochastic-psr/issues/22).
Keep a complete core cube and six disjoint positive/negative slabs, each with
one high-frequency coordinate. Every scalar/cross mode is included if its
cutoff is no larger than the core. There are no negative inclusion-exclusion
weights and no overlap. Multiple-high-axis corners are omitted deliberately.

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
the GPU algorithm would repeat source spreading across blocks, with padded
FFT grids and plan overhead. It also needs coherent prior sampling and an
omitted-spectrum estimator that includes the excluded corners, not just the
outside-cube tail. No GPU backend or speedup is implemented here.

The normalization check also exposes an inherited mismatch at longer scales:
the production spectrum is smaller than the variance-normalized analytic
spectrum by `1/(1+exp(-sqrt(3)/ell))` per axis. At ell=1 this leaves a 38.66%
3D diagonal deficit even with an infinite Fourier band. At the .01/.005 scales
in the table, the factor rounds to one in float64. The existing production
code is unchanged; [issue #24](https://github.com/alecjacobson/fast-stochastic-psr/issues/24)
tracks correction, released-code compatibility and affected tail diagnostics.
`normalization-diagnostic.json` records the coefficient-ratio check. The block
mass formula uses the normalized analytic law.

Tests enumerate small integer-frequency sets to verify coverage, disjointness,
symmetry and mass conservation, and compare the independent one-dimensional
normalization to production at the audited short scales. An explicit Fourier
sum also reconstructs the analytic covariance at long scales. Reproduce with:

```bash
./reproduction/run_env.sh -m reproduction.audit_spectral_blocks
```
