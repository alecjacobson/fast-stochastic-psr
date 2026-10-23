# Shared landmark Nyström preconditioning

**Rank 1,024 reduces million-point CG iterations from 57 to 38** in this pilot, with numerical and paired-event gates passing. It retains 3.815 GiB of basis data and costs 9.095 s to construct. Larger sample-count and coherent-producer validation remain open.

This pilot tests a global coarse correction after
[disjoint spatial blocks](../preconditioning/README.md) failed to reduce
million-point iteration counts. It preserves the packed full-cube200 covariance
action, original right-hand sides, prior coefficients, scalar cutoff50, solver
tolerance and query geometry. Only the inverse preconditioner changes.
The [protocol](PROTOCOL.md) fixes ranks 128, 512 and 1024 before running the sweep.

## Setup, conditioning and memory

NVIDIA L40; one million observations; eight common conditional draws in batches
of two; three solve repeats per configuration, with one shared setup. Noise
variance .02, length scale .01, per-axis variance .1, NUFFT epsilon1e-6 and
CG tolerance1e-5. Setup-plus-solve charges construction once for the first
ensemble on a map. Source transfer, transform plans, prior generation, mean
fitting, scalar aggregation and queries are outside these conditioning totals.

| Run | Setup (s) | Median solves (s) | Setup + solves (s) | Iterations/batch | Retained preconditioner (GiB) | Whole-run sampled isolation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| baseline8 | 0.000 | 48.603 | 48.603 | 57–57 | 0.000 | False |
| rank128 | 3.722 | 39.329 | 43.051 | 54–54 | 0.477 | False |
| rank512 | 2.349 | 25.366 | 27.715 | 45–45 | 1.907 | True |
| rank1024 | 9.095 | 22.289 | 31.385 | 38–38 | 3.815 | False |

**No timing ratio is claimed:** at least one declared whole-run isolation or quality requirement fails. All timings are retained as diagnostics; no favorable repetitions are selected.
Process sampling can miss brief overlaps and does not prove exclusive GPU use.
Other jobs were not stopped. Retained basis storage excludes construction
scratch, source arrays, CG vectors and NUFFT workspace. Raw results also retain
stage memory and sampled process observations. A larger rank can reduce
iterations while increasing time per iteration or setup cost; iteration count
alone is not an application speedup.

Rank 512 recorded setup-plus-eight-draw cost is 27.715 s versus 31.385 s for rank 1,024. Compare repeated solve cost, setup and storage separately; value for larger ensembles needs direct measurement. These are diagnostics, not timing-ratio claims.
[Issue #31](https://github.com/alecjacobson/fast-stochastic-psr/issues/31)
tracks compressing, streaming or avoiding the observation-by-rank basis before
scaling beyond one million inputs.

## Numerical controls

| Candidate | Retained rank | Basis orthogonality error | Field error / fluctuations | Covariance discrepancy | Independent finite-row residual | Numerical/event gates |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| rank128 | 128 | 1.43e-08 | 0.003303% | 0.003373% | 0.000873% | True |
| rank512 | 512 | 2.94e-08 | 0.003581% | 0.002137% | 0.001185% | True |
| rank1024 | 1024 | 4.98e-08 | 0.003096% | 0.001950% | 0.001205% | True |

Independent double-precision cosine sums check 32 covariance rows at seed20261023,
with a .002% residual gate, in addition to full NUFFT true residuals. Same-model
field error must be below .05% of the fluctuation norm; empirical variance and
covariance discrepancies below .01%. Paired ray/collision/next-view gates are
unchanged. These are numerical controls on eight common draws. They do not
establish absolute probability accuracy, continuous geometry, full analytic
training-tail or scalar-band convergence, or ContactFusion update semantics.
The [known collision sampling failure](../query_refinement/README.md) remains.

Tests compare covariance columns with independent long Fourier sums, the
preconditioner with a dense pseudoinverse/eigendecomposition, and conditioned
weights with an independent dense finite solve. Duplicate landmarks, periodic
seam coordinates, partial construction batches, both precisions, zero RHS and
invalid noise are covered. [Full test output](tests.log).

## Construction and application

Let C = K(X,L), W = K(L,L), and C W† Cᵀ = U diag(λ) Uᵀ. We apply
P⁻¹ = I − U diag(1 − (λ_min + σ)/(λ + σ)) Uᵀ, where σ is noise variance.
This spectral scaling follows Eq. 5.3 of
[Frangella, Tropp and Udell (2023)](https://tropp.caltech.edu/papers/FTU23-Randomized-Nystrom-SIMAX.pdf).
Our landmarks are uniform sampled columns, not a Gaussian sketch; the paper
distinguishes these constructions in §2.2.3. Its strongest Gaussian-sketch
guarantees do not automatically apply to this implementation.

The analytic product-Matern kernel constructs only P; PCG continues to solve
the original finite-cube system. Replacing the solve with the low-rank model
would change the posterior and is not performed here. Homoscedastic noise is
required by this implementation; heteroscedastic whitening remains separate.

Construction uses double precision in row blocks of 8192, NumPy landmark seed 41,
and relative eigenvalue cutoff 1e-10 in the two small Gram decompositions.
Discarded eigenmodes affect only the preconditioner. No silent jitter is added.
Setup also holds an N-by-rank double-precision feature array while constructing
the float32 basis; row batching bounds pairwise temporaries, not that array.
This pilot uses one landmark seed, and landmark sets at different ranks are not
required to be nested. It does not establish a seed-robust optimal rank.
The stored float32 basis is rechecked in double precision: the implementation
requires 1 − max(damping)·(1 + ‖UᵀU − I‖F) > 0. This is a numerical SPD guard,
not an interval-arithmetic proof. Setup includes the check. Eigenvalue ranges,
retained ranks, landmark hashes and approximate/analytic covariance traces are
recorded; trace capture is not posterior-error certification.

## Reproduction

Use the existing [common-RHS data preparation](../nufft_batches/README.md).
Run the following baseline, then each rank in a fresh process:

```bash
./reproduction/run_env.sh -m reproduction.benchmark_nufft_batches solve \
  --samples 8 --batch 2 --repeats 3 --pack-rhs --finite-rows 32 \
  --scratch /opt/geospsr-runtime/common-rhs-128 \
  --output /opt/geospsr-runtime/nystrom-preconditioning/baseline8
./reproduction/run_env.sh -m reproduction.benchmark_nufft_batches solve \
  --samples 8 --batch 2 --repeats 3 --pack-rhs --finite-rows 32 \
  --preconditioner-rank 512 --scratch /opt/geospsr-runtime/common-rhs-128 \
  --reference /opt/geospsr-runtime/nystrom-preconditioning/baseline8 \
  --output /opt/geospsr-runtime/nystrom-preconditioning/rank512
```

Repeat for ranks 128 and 1024, preserving all results. After copying compact
results/fields into this directory, regenerate this report with
`./reproduction/run_env.sh -m reproduction.summarize_nystrom`.
[Issue #30](https://github.com/alecjacobson/fast-stochastic-psr/issues/30) tracks
larger-sample validation, shared coarse spaces and setup/amortization costs.
