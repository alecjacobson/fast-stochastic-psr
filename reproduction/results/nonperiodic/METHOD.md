# Non-periodic Poisson–Matérn reference

This experiment changes the probabilistic model. The original
[Sellán–Jacobson 2022 SPSR](https://www.dgp.toronto.edu/projects/stochastic-psr/)
is non-periodic; removing the periodic assumption of the current GeoSPSR
implementation is a step toward a more appropriate domain model. It does not
by itself establish better reconstruction or calibrated uncertainty. Numerical
agreement with the periodic implementation is not an acceptance criterion.

## Joint model and free-space solution

Let the three components of a zero-mean vector field be independent, with

\[
\operatorname{Cov}(v_d(x),v_e(y))=\delta_{de}k_v(r),\qquad
k_v(r)=a(1+\kappa r)e^{-\kappa r},\quad
\kappa=\sqrt 3/\ell,\quad r=\|x-y\|.
\]

Define the scalar field by \(\Delta f=\nabla\cdot v\) on \(\mathbb R^3\).
With the Fourier convention \(\widehat{\partial_d}=i\omega_d\),
\(\hat f=-i\omega\cdot\hat v/|\omega|^2\), so
\(S_f=S_v/|\omega|^2\). The low-frequency singularity is integrable in
three dimensions. Choose the stationary solution with covariance decaying
at infinite separation and no additional random constant mode. This is
not a claim that random field samples tend to zero at infinity.

The radial scalar covariance is the Newton potential of \(k_v\):

\[
g(r)=\frac1r\int_0^r s^2k_v(s)\,ds+
       \int_r^\infty s k_v(s)\,ds
=\frac{a}{\kappa^2}
  \frac{8-e^{-z}(8+5z+z^2)}{z},\quad z=\kappa r.
\]

Thus \(-\Delta g=k_v\), \(g(0)=3a/\kappa^2\), and
\(g(r)\sim8a/(\kappa^3r)\). The cross covariance has the sign

\[
c_d(q,x)=\operatorname{Cov}(f(q),v_d(x))=-\partial_{q_d}g(q-x).
\]

Writing \(g=a\kappa^{-2}h(z)\),
\(c=a[-h'(z)/z](q-x)\). At the origin,
\(h(z)=3-z^2/6+z^4/40-z^5/90+\cdots\) and
\(-h'(z)/z\to1/3\). Degree-18 Taylor evaluation below \(z=.5\)
avoids cancellation in the closed forms. Tests compare both kernels to
independent radial quadrature and verify the Poisson identity by finite
differences, including the sign of the cross kernel.

This model retains the independent vector field and its Poisson projection.
It does not identify observations of the full vector field with observations
of the gradient of a scalar GP; the former also has a divergence-free part.
It uses an isotropic kernel, unlike the current periodic product kernel.

## Dense conditioning oracle

For observed vectors \(y_{id}\), let
\(A_{ij}=k_v(\|x_i-x_j\|)+\eta_i\delta_{ij}\), where \(\eta_i\) is
**noise variance**. The same matrix is used for each independent component:

\[
\mu(q)=\sum_d c_d(q)^T A^{-1}y_d,\quad
V(q,q')=g(\|q-q'\|)-\sum_d c_d(q)^T A^{-1}c_d(q').
\]

One dense Cholesky factor costs O(N³) time and O(N²) storage. Direct marginal
variance costs O(N²M). The reference uses float64, explicit declared noise,
and no hidden jitter, variance clipping, minimum-variance subtraction or
level fitting. Joint covariance positivity is checked on independent panels.

The zero-mean scalar prior has a specified constant mode. A reconstruction
isovalue is a separate modeling choice. Subtracting a deterministic posterior
mean level and subtracting a random sample-dependent level have different
covariances. The pilot's zero-threshold Gaussian CDF checks numerical error
only; it is **not** an occupancy-calibration result. Before application
comparisons, sign, level, normal scaling and uncertainty conventions must be
fixed and reported.

## Regional variance experiment

Use tensor Chebyshev interpolation at r=p³ nodes in a fixed box:
\(c_d(q)\approx C_d\phi(q)\). If \(A=LL^T\), stack the three
\(L^{-1}C_d\) matrices into W and prepare

\[
H=W^TW,\qquad b=\sum_d C_d^TA^{-1}y_d.
\]

Online evaluation is \(\tilde\mu=\phi^Tb\) and
\(\tilde v=g(0)-\phi^TH\phi\): O(r²) work per query and O(r²) persistent
coefficient storage, independent of N at fixed r. The current producer still
costs O(N²r+Nr²), after dense O(N³) conditioning. It is a rank/accuracy
experiment, not the final asymptotic solution. The diagnostic whitened SVD
also uses dense solves and is not part of a proposed fast producer.

For the stacked exact and approximate whitened cross columns w and w̃,

\[
|v-\tilde v|\le2\|\tilde w\|\|w-\tilde w\|+\|w-\tilde w\|^2.
\]

We check this bound on held-out query points. Neither these sampled tests
nor a singular-value rank threshold prove uniform accuracy throughout a box.
Interpolation can overestimate the covariance correction and produce a
negative variance; such a result fails instead of being clipped.

The experiment is motivated by regional conditioning compression in
[Arora–Geoga](https://arxiv.org/html/2605.02574v3), but is not a reproduction
of their summary-conditioning method. Their favorable assumptions include
smoothness and separated prediction regions; near-surface boxes require
testing. [Chen–Stein](https://arxiv.org/abs/1711.05895) is a candidate route
to reusable hierarchical quadratic forms; its structured covariance is
itself a model approximation. FMM/IFMM can accelerate solves, but alone
does not eliminate one solve per query or per region.

## Original SPSR reference and comparison contract

The pilot calls the authors' linked gpytoolbox implementation, version 0.3.8,
with `output_variance=True` and `solve_subspace_dim=0`. The result manifest
hashes all Python source files in the package. It records explicit grid,
envelope, noise, global variance, queries and random seed. The upstream
implementation is unmodified.

Audited behavior: compact normal kernel of width `1.5*min(h)`; density-lumped
observation covariance; staggered vector grid; finite-difference gradient G;
Poisson operator GᵀG plus 10⁻⁶I; mean shifted by its average at observations;
full covariance obtained by two sparse-LU solves against a dense matrix;
reported variances shifted by minus their minimum plus noise² + 10⁻⁶.
G uses internal grid edges, with no periodic wrap. Boundary/gauge behavior
must be characterized before treating this as a free-space reference.

Changing grid spacing also changes the vector kernel and density estimate.
The 8³/12³/16³ runs establish reproducibility, **not** convergence of a fixed
continuous model. Original output is a reference algorithm, alongside
synthetic shape truth; it is not assumed to be the exact posterior of the
new model. Do not copy its variance shift into the dense oracle to force
agreement. A corrected or differently gauged baseline needs a separate label.

Two validation tracks follow:

1. Same-model numerical error: new accelerated implementation versus this
   dense oracle, including mean, variance, joint covariance and probabilities.
2. Modeling/application quality: original SPSR and known geometry versus the
   new model, using geometric error, thin structures, partial scans, uncertainty
   profiles, independent repeated-scan calibration and downstream decisions.
   A shape alone does not define a true latent implicit-field variance.

See issues [#69](https://github.com/alecjacobson/fast-stochastic-psr/issues/69)
and [#70](https://github.com/alecjacobson/fast-stochastic-psr/issues/70).
