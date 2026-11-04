# What changes algorithmically

With A=R R^T and component cross-covariance vectors c_d(q), marginal variance is

    v(q) = v0 - sum_d ||R^-1 c_d(q)||^2.

A column of R^-T supplies one real source-weight vector. For the Poisson
cross kernel its three component fields are the gradient of a single real
potential u_j. Consequently

    sum_j |grad u_j|^2 = (1/2) Laplacian(sum_j u_j^2)
                         - sum_j u_j Laplacian(u_j).

We form the two summed products on a grid large enough to retain all doubled
frequencies, then differentiate in Fourier space. Independently scaled u and
Laplacian(u) share one complex inverse FFT. Pairs of real source vectors also
share one type-1 NUFFT, using Hermitian reversal to recover each spectrum.
Batched triangular solves prepare R^-T in larger blocks before streaming its
columns into the transforms. The full factor is retained: no Monte Carlo or
low-rank error is introduced. Float32 transforms do introduce rounding error,
which is checked against independent double triangular-solve evaluations.

Cutoff K=50 means L=(2K+1)^3=1,030,301 cross modes. Variance support doubles to
[-100,100]^3; products use a 216^3 fast FFT grid, which is at least (4K+1)^3.
After constructing the variance coefficients, a cached interpolation evaluator
handles changing query coordinates without another FFT or any N-dependent
per-query solve. Both mean and variance, the normal CDF and ambiguity reduction
are included in probability query timings.

Using N, M and L in the [original paper](https://arxiv.org/abs/2503.19136)'s notation (with fixed spatial dimension,
accuracy and interpolation stencil):

| Method | One-time preparation | Subsequent marginal query work |
|---|---|---|
| Cholesky direct | O(N^3) | O(M N^2) |
| Direct SDD marginal solves | kernel/setup plus fitted mean | O(I b N M) for batch b and measured iteration count I |
| Cholesky-built spatial variance grid, H sites | O(N^3 + H N^2) | O(M) |
| Full-factor Fourier variance construction | O(N^3 + N^2 + N L log L) | O(M) after interpolation-grid preparation |

The N^3 term in the Fourier method includes both Cholesky and obtaining the
full inverse factor. This remains a dense-conditioning hybrid. It does not
claim an infinite-N asymptotic improvement over Cholesky conditioning, nor an
M-only asymptotic improvement over another prepared spatial cache. The useful
window is where the N L log L construction term grows more slowly than the
spatial grid's H N^2 work, many fresh queries amortize that construction, and
conditioning is still affordable. Local empirical slopes are not proofs of
asymptotic exponents.

The workload requires actual variance: occupancy probabilities depend on
mu/sqrt(v), and uncertainty scores average min(p,1-p). Constant-variance
ablations quantify this dependence. Marginal scores suffice for expected
pointwise uncertainty/occupancy counts; they do not describe correlated joint
collision or ray events. A changing observation set or covariance requires
rebuilding the variance cache. The synthetic workload is a candidate for
fixed-map inspection/planning, not a measured deployed sensor application.
