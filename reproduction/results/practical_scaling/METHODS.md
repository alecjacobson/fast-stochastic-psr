# Methodology and the original paper's notation

The [original paper, §§3.1–3.3](https://arxiv.org/html/2503.19136v1#S3)
uses N observations, M test locations, P posterior samples, L cross-covariance
Fourier modes, and L′ prior-sampling modes. Its direct cross-covariance work is
O(LMNP); its model-only interpolation table removes the L factor from subsequent
pairwise work. Dense factorization costs O(N³). The paper's main statistical
pipeline uses pathwise samples; the released implementation additionally offers
direct marginal variance through dense or SDD solves.

The following costs are derived for the implementations in this experiment,
not quoted complexities from the paper. Dimension, gridding tolerance, and
fixed RHS count are suppressed where appropriate.

| Operation | Pairwise / dense approach | Fourier approach |
|---|---|---|
| Mean or P correction fields after fitting | O(MNP) with the cross-kernel table | O(P(N+L log L)) aggregation, then O(P(M+L log L)) ordinary type-2 evaluation |
| Repeated queries of prepared fields | O(MNP) without a posterior-field cache | O(P L log L) grid preparation, then O(PM) interpolation-only queries |
| Marginal variance from a cached Cholesky factor | O(MN²), including all three component RHS | Exact moment-series preparation O(N³+N²+NL log L); then O(M+L log L), or O(M) with the spatial interpolation grid prepared |
| Mean solve | O(N³) Cholesky; approximately O(I_SDD bN) for batch b | O(I_CG(N+L_t log L_t)) finite-kernel CG, **plus analytic correction actions and geometry preparation** |

L_t is the training spectrum's mode count, distinct from L. I_CG and I_SDD
are measured iteration counts, not constants assumed independent of N. The
finite training kernel is not silently substituted for the full analytic
kernel: defect refinement repairs that discrepancy to the declared residual
tolerance. Its analytic action cost must be counted separately. A pure
analytic-CG ablation measures whether NUFFT actually helps the solve.

The variance series identity is straightforward. Let A=RRᵀ be the observation
covariance, and let c_d(q) be the cross-covariance vector for component d:

    v(q) = prior − Σ_d ||R⁻¹ c_d(q)||².

Each column of R⁻ᵀ defines a scalar Fourier function for each d. Squaring and
summing these 3N functions gives one Fourier series. A function supported on
[-K,K]³ has a square supported on [-2K,2K]³: at least (4K+1)³ grid samples
suffice to form that product without aliasing. The implementation pads to a
fast FFT size and crops back to the exact product support. No Monte Carlo
samples or low-rank approximation are involved. The same square accumulator
can represent a fixed sample moment estimator, but this experiment uses the
full Cholesky correction factor.

This is a **Cholesky–NUFFT hybrid**, not a faster conditioning algorithm. It
can be the right choice for a fixed map queried repeatedly, while the refined
NUFFT solve is a separate candidate for large maps that must be refitted.
The two techniques are not automatically one scalable variance pipeline:
forming the exact full correction factor still requires dense preparation.

A strong competing method also caches posterior moments on a spatial grid.
For H grid sites its preparation is O(HN²) after the factorization, and its
queries are O(M). Therefore there is **no asymptotic M-only superiority** over
an equally accurate prepared spatial cache. Compare preparation, accuracy,
memory, valid domain, and map-update frequency. Fourier aggregation removes
pairwise observation/query work during field construction; which side wins
in practice depends on N, bandwidth and domain. A model-only cross-kernel
table survives map changes, but either posterior-moment cache must be rebuilt
when observations or model parameters change.

Marginal moments also do not replace coherent posterior draws for joint
collision or ray events. Those events depend on cross-query correlations.
