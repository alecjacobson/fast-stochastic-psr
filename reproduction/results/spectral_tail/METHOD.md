# A posterior covariance bound for the omitted training spectrum

This diagnostic isolates **vector-training truncation**, keeping the scalar and
cross covariance at cutoff 50 and retaining observation noise. It uses the
product circle Matérn spectrum implemented by [Holalkere et al.](https://arxiv.org/abs/2503.19136)
and described by [Borovitskiy et al.](https://arxiv.org/abs/2006.10160).
The bound and diagnostic below are our derivation, not a claim from those papers.

## Exact statement

Let `B = K_finite + noise*I`, `D = K_analytic - K_finite`, and let `C_d` be
the training-to-query scalar/vector cross covariance for component `d`.
Because the omitted Fourier coefficients are nonnegative, `D` is positive
semidefinite. This requires the same kernel normalization and a training
bandwidth covering every scalar/cross mode.

Write `A_d = B^-1 C_d` and

```
V_finite = P - sum_d C_d.T B^-1 C_d
V_full   = P - sum_d C_d.T (B+D)^-1 C_d
H        = sum_d A_d.T D A_d
```

Then the following are **matrix inequalities**, not elementwise bounds:

```
0 <= V_full - V_finite <= H.
```

To see this, whiten by `B`: `X = B^-1/2 D B^-1/2 >= 0`. Each eigenvalue
`x >= 0` satisfies `0 <= x/(1+x) <= x`. Congruence by `B^-1/2 C_d` and
summation give the result. Thus the diagonal of `H` bounds each variance
increase, and `||V_full-V_finite||_F <= ||H||_F <= trace(H)`.
Off-diagonal entries need not increase; an entrywise upper bound would be wrong.

A second useful identity concerns coherent draws. If one samples the **full**
joint vector/scalar prior (including observation noise), but conditions with
the **finite** inverse, the resulting scalar covariance is `V_finite + H`.
Adding the omitted prior without correcting the solve therefore does not
produce `V_full`. It provides the upper side of this covariance bracket.
Its mean still uses the finite inverse. This identity does not authorize a
claim that Gaussian mixtures from random-feature sampling are exact GP draws.

Conversely, the conditional-solve benchmarks share a **finite** joint prior
between finite-inverse NUFFT and full-analytic-inverse SDD. With an exact full
inverse, the latter has covariance `V_shared = V_full - H_full`.
Write `Z_d=(B+D)^-1 C_d`; then `H_full=sum_d Z_d.T D Z_d`.
Whitening shows `0 <= x/(1+x)^2 <= x/(1+x)` for each omitted-tail eigenvalue,
and therefore

```
V_finite <= V_shared <= V_full <= V_finite + H.
```

This explains why the common-prior experiment still does not sample the full
analytic posterior. The bracket can support a separate error assessment; it
does not remove the need to check numerical solves, means and application
query coverage. Dense tests verify both hybrid covariance identities.

## Estimating H without an N-by-N matrix

Let `tau = sum_{n outside cube} rho(n)` and sample iid integer frequencies
with probability `rho(n)/tau` over the infinite complement of the training
cube. For each frequency compute

```
F_nd = sum_i exp(i n.x_i) A_id
H_sample(n) = tau * sum_d Re(conj(F_nd).T F_nd).
```

The expectation is `H`, and every sample is positive semidefinite. The feature
sum is streamed in source and frequency blocks. Work is `O(N R Q)`, where
`R` is the number of frequencies and `Q` the number of query probes. This is
a validation diagnostic, **not a new fast-summation backend**. No application
speedup is attributed to its runtime.

The one-dimensional spectrum is proportional to `(lambda^2+n^2)^-2`, with
`lambda=sqrt(3)/(2*pi*lengthscale)`. Inside the cube we sample the finite
categorical law. Outside, use a signed rounded Pareto proposal of shape 3,
starting at `cutoff+1/2`. For positive integer `n > cutoff`, its probability is

```
q(n) = (cutoff+1/2)^3 * ((n-1/2)^-3 - (n+1/2)^-3).
```

Convexity of `x^-4` gives `q(n) >= 3*(cutoff+1/2)^3*n^-4`. Accept with
probability `3*(cutoff+1/2)^3 / ((lambda^2+n^2)^2*q(n))`, which is at most
one. This gives the discrete infinite tail without a larger truncation.
Partition the 3D complement by its first outside axis: when the inside mass
per axis is `p`, these three disjoint cases have conditional weights
`[1,p,p^2]/(1+p+p^2)`. Later axes follow their unconditional laws.

## What the measurements establish

- A/B use independent float64 analytic and finite covariance matrices and
  dense solves. They check the matrix bracket, the actual covariance change,
  and the Monte Carlo estimator against `H` computed directly.
- Large N uses finite NUFFT-CG cross-covariance solves. A small residual in
  that system does not alone establish analytic-model fidelity. The tail
  estimator supplies different evidence about the covariance discrepancy.
- Query probes and their selection seed are fixed before tail sampling.
  Half come from 256 slice locations closest to the default mean zero set;
  half are sampled from the remainder. Stress runs retain this same panel.
- Three independent frequency seeds measure repeatability. Reported standard
  errors and marginal normal intervals are empirical, **not rigorous or
  simultaneous certificates**. A maximum over marginal interval endpoints
  does not have simultaneous 95% coverage.
- The theorem concerns exact solves and transforms. Floating-point,
  NUFFT and iterative errors remain separate; large-N estimates are not
  certified by the theorem alone.
- A later [normalization audit](../spectral_blocks/README.md) found an inherited
  Fourier/analytic Matérn scaling mismatch at long length scales. It is below
  float64 resolution for the .01/.005 settings reported here. At larger scales
  the in-band defect must also be included; the current outside-cube estimator
  alone is insufficient. Correction is tracked in
  [#24](https://github.com/alecjacobson/fast-stochastic-psr/issues/24).
- Sixteen query probes do not establish complete ray/collision probabilities,
  mesh convergence, posterior mean agreement, or matching quality everywhere.
  Covariance error relative to a matrix norm also need not control every
  nearly singular joint direction.

The source is `geospsr/spectral_tail.py`; the executable audit is
`reproduction/audit_spectral_tail.py`. Tests check infinite-tail expectations,
clustered observations, high-frequency phases, an independent explicit finite
Fourier sum, analytic normalization, the PSD bracket, and the hybrid-sampler
covariance identity.

## Separate prior and solve bandwidths

For issue #19, let `B` include the narrow training band and observation noise,
`D >= 0` contain the additional modes in a wider coherent prior, and `E >= 0`
contain all modes above that prior. Both bands must cover the scalar/cross
band. Define `A=B^-1 C`, `Z=(B+D)^-1 C`, with a sum over vector components
understood below. The hybrid covariance is

```
V_hybrid = V_B + A.T D A
V_prior  = P - C.T (B+D)^-1 C
R_D      = V_hybrid - V_prior >= 0.
```

Writing `X=B^-1/2 D B^-1/2`, the whitened expression for `R_D` is
`X^2 (I+X)^-1`, congruent by `B^-1/2 C`. Thus the covariance error relative to
the wider finite model is second order in the whitened added covariance.
This statement does **not** assume `B` and `D` commute. Relative to the full
analytic posterior, the remaining tail gives the two-sided matrix bound

```
R_D - Z.T E Z <= V_hybrid - V_full <= R_D.
```

For dense audits, a stable positive-semidefinite expression is
`R_D=(A-Z).T (B+D) (A-Z)`. It avoids subtracting two nearly equal posterior
covariances. With the same wider joint prior and held mean, `R_D` is also the
covariance of the **difference between paired hybrid and wider-posterior
draws**. That difference is independent of the wider posterior fluctuation:
its cross covariance vanishes because `C.T-Z.T(B+D)=0`.
Thus exact-law covariance error is second order even though paired field error
scales as the square root of that excess. Finite-ensemble paired covariance
differences also contain sampling cross terms; they need not equal `R_D`.

Mean error does not enjoy that second-order cancellation. To hold the mean
fixed while testing a cheaper inverse, generate centered fluctuations
`f_prior - C.T B^-1 (vector_prior + noise)` and then add the independently
computed wider/full mean. Simply solving the observed-data RHS with the
narrow inverse changes the mean. Large-scale full-mean error and application
query coverage still require validation. These identities motivate a trial;
they are not measured speedups or a claim of an exact full-posterior sampler.
An independent NumPy test checks noncommuting positive-definite matrices,
the joint-sampling covariance and the two-sided tail bound.
The [centered-hybrid report](../centered_hybrid/README.md) now tests this design
with exact dense A/B moments and paired million-point draws, retaining a
short-scale/low-noise failure and rejected cutoff-100 trial.

## A separate bound for posterior mean error

The same omitted covariance also bounds the mean discrepancy. Let
`alpha_d=B^-1 y_d`, `A_d=B^-1 C_d`, and

```
h_y = sum_d alpha_d.T D alpha_d
H   = sum_d A_d.T D A_d.
```

For exact solves and a fixed query panel, the vector difference
`delta_mu = mu_finite - mu_full` satisfies the matrix inequality

```
delta_mu delta_mu.T <= h_y H.
```

Whiten by `B` and stack the three vector components. The resolvent difference
has factor `X(I+X)^-1`. Cauchy–Schwarz in the `X` inner product bounds the
outer product by `H` times the data quadratic with factor `X(I+X)^-2`, which
is bounded by the quadratic with factor `X` defining `h_y`.
This does not require the kernel matrices to commute.

For a mean centered by a **fixed linear query functional**, replace each
cross-covariance column by its centered column before computing this mean
bound. This addresses the difference between each model's deterministic
centering offsets; it does not center random posterior fluctuations or change
their covariance. Query selection and the centering functional must be fixed.

An independent dense NumPy test checks the matrix inequality for noncommuting
covariances, three vector components and centered queries. No large-N mean
bound is measured here yet. It may be loose, and substituting empirical tail
estimates does not turn it into a rigorous certificate. Numerical solves and
full application coverage remain separate requirements of issue #17.

## Connection to a finite joint event, with mean held fixed

For a positive-definite finite query covariance `V`, let `G = V^-1/2 H V^-1/2`
have eigenvalues `g_i`. The actual covariance increment has whitened
eigenvalues `0 <= e_i <= g_i` (ordered consistently). For **identical means**,
the Gaussian divergence satisfies

```
KL(N(mu,V_full) || N(mu,V))
  = 1/2 sum_i (e_i - log(1+e_i))
 <= 1/2 sum_i (g_i - log(1+g_i)).
```

The inequality follows because `x-log(1+x)` increases for nonnegative x.
[Pinsker's inequality](https://stanford.edu/class/stats311/lecture-notes.pdf)
then bounds the difference in the probability of **any event on this finite
query vector** by `1/2 sqrt(sum_i (g_i-log(1+g_i)))`, capped at one. Here total
variation uses the supremum-over-events convention and KL uses natural logs.

This improves on a matrix-relative norm when a covariance panel is poorly
conditioned: whitening accounts for low-variance joint directions. Our report
checks the KL expression and bound with the dense overlap matrices. Large-N
values substitute an empirical estimate of H, so they remain estimates. They
do not incorporate posterior mean error, additional application query locations,
Monte Carlo probability error, or geometric discretization.
