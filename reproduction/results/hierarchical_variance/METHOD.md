# Nonperiodic hierarchical conditioning and regional variance

This continues the [free-space joint model](../nonperiodic/METHOD.md). It changes
the periodic GeoSPSR model, and is not an exact implementation of Sellán–Jacobson
2022. The original SPSR method also has no periodic wrap. Numerical comparisons
use the same new model; comparisons with original SPSR use geometry and uncertainty
profiles, without requiring pointwise agreement.

## Accelerated path

The observation matrix is the isotropic vector Matérn-3/2 covariance plus declared
noise variance. A pinned [HiGP](https://github.com/huanghua1994/HiGP) implementation
builds a nested H² representation and adaptive factorized Nyström (AFN)
preconditioner. This is an algebraic, kernel-independent **FMM-style** hierarchy,
not an analytic Laplace FMM. Tree construction, preconditioner construction,
hierarchy export, GPU interaction blocks, and transfers are all timed.

The GPU implementation applies upward nested transfers, near/far interactions,
and downward transfers. It caches interaction blocks, not an N-by-N matrix.
AFN applies its small coarse factor and sparse approximate inverse on the GPU.
No full Cholesky factor occurs in this accelerated path. The small AFN coarse
factor (rank 50 in the long-correlation experiment, 8 in the short case) is
independent of N.

For a fixed contact box, tensor Chebyshev cross interpolation uses p³ nodes.
A thin SVD of the combined three-component cross matrix reduces 3p³ right-hand
sides to r columns U (r=21 in the long case, 19–20 in the short case). Compressing this matrix is
shared by all baselines. The variance requires only G=UᵀA⁻¹U; the mean requires
UᵀA⁻¹y. Neither requires a solve for each online query.

The mean is solved independently to relative residual 10⁻⁶. For approximate
variance solutions Z and residual R=U−AZ, use

    G = sym(UᵀZ + ZᵀR).
    UᵀA⁻¹U − G = RᵀA⁻¹R.

The variance error is second order in the solve error. This identity applies
to an SPD represented operator; it does not control H² approximation error by
itself. Eight variance iterations are used through N=77,824; twelve are used
for the larger extension. These are explicit experimental settings, not a
certified stopping rule. A four-iteration pilot failed the variance gate and
is retained. Issue [#79](https://github.com/alecjacobson/fast-stochastic-psr/issues/79)
tracks output-controlled stopping and positivity.

After projection, collapse the quadratic tensor polynomial into a scalar
Chebyshev variance field of degree 2p−2 per axis. The mean and variance are
evaluated with GPU Clenshaw recurrences. With p=4, the scalar field has 343
coefficients. This algebraic collapse introduces no further interpolation
approximation. It avoids evaluating a dense p³-by-p³ quadratic form per query.

## Work and storage

Let N be observations, M total queries, p fixed interpolation order, r retained
cross rank, t hierarchical rank, k coarse preconditioner rank, s sparse neighbor
count, and I iterations. The regional SVD costs O(N(3p³)²); it is linear in N
only when the box size, required accuracy, and p stay fixed.

| Producer | Leading conditioning / regional preparation work | Matrix storage |
|---|---|---|
| Dense Cholesky | O(N³ + N²(r+3) + N(3p³)²) | O(N²) |
| Dense-action PCG | O(I N²(r+3)) plus preconditioner and SVD | O(N²) cached, or bounded row tiles |
| H² + AFN PCG | hierarchy/AFN setup + O(I T_H(N,t)(r+3)) + SVD | O(N t²) for the implementation's cached blocks |
| Mini-batch SDD | O(T b N(r+3)) for T iterations and batch size b, plus SVD | linear for fixed b and r |

For bounded ranks and spatial interaction lists, T_H is linear in N, with
tree/search setup typically O(N log N). AFN has fixed-size local factorizations
and a small coarse matrix. Rank, neighbor count and iteration growth must be
measured; a uniform asymptotic bound is not established by a fixed-size pilot.
All methods receive the same O(M p³) scalar query cache. Repeated queries do not
give H² a free advantage over a Cholesky baseline that could use the same cache.

## Application and comparison contract

The synthetic scan samples an ellipsoid with axes (1,.8,.6), using true outward
unit normals. Nested observation sets have seed 12. The Matérn length scale is
.3, vector component variance .0025, and observation noise variance .000025.
The long-case region is the box of halfwidth .05 centered at the known surface contact
(1,0,0). The isovalue is the posterior mean at that contact, a deterministic
level choice; variance is not shifted or clipped. This does not condition on
an additional random scalar observation.

The winning short-correlation case uses length scale .03, box halfwidth .005,
H² tolerance .001, leaf size 40, AFN coarse rank 8 and 24 neighbors. Domain,
kernel parameters, region and output gates stay fixed while N increases.
See [its report](short/README.md) for all results and the SDD counterexample.

Sixteen batches each contain 1,024 candidate trajectories with 64 points,
for 1,048,576 queries. Every batch has different transverse ray positions.
Compute marginal inside probabilities Φ((level−mean)/sqrt(variance)) and choose
the trajectory with greatest mean Bernoulli ambiguity 4p(1−p). These are
marginal probabilities, not joint collision probabilities. Query positions
and synthetic observations are generated outside solver timing for every
method. Their generation is common to all methods. Common CUDA/query-kernel
initialization is separately reported; all method-specific setup is timed.

The strong dense baseline uses a single in-place Fortran-order float32 GPU
Cholesky factor. Mean and Gram construction share **one** triangular solve
for [U,y]. Avoidable factor copies and two-sided full solves are not charged.
The dense CG control receives the same AFN preconditioner. Its float32 inner
steps use float64 residual accumulation and iterative refinement because
unrefined float32 CG stalled or failed output accuracy. All refinement work
is included in its timing. SDD receives the same regional cross compression
and both the standard moment calculation and an optional full-kernel residual
correction. The latter can be conservative for stochastic iterates with errors
orthogonal to the cross basis; it is not assumed to be faster or more accurate.
The cheaper passing option is selected. Its reported preparation includes any
final correction action. Earlier checkpoint quality evaluations are excluded
and recorded separately, giving SDD the benefit of offline stopping selection.
Passing checkpoints execute all online query batches. Failed budgets are
reported without extrapolating a completion time.

## Independent numerical audit

Held-out queries include random interior points, all eight box corners and its
center. Acceptance requires maximum variance error / prior variance ≤10⁻⁴,
maximum marginal probability error ≤10⁻³, and positive predicted variances.
The independent reference uses direct cross kernels, not regional interpolation.
It refines its solve against **full float64 kernel actions**. A float32 Cholesky
factor is only the initial reference solver, not the accuracy authority. For
larger sets that do not fit a dense factor, a tighter H² solve initializes
refinement; full-kernel residuals provide the independent check.

For the exact model A ≥ noise_variance I. The residual-corrected direct variance
has absolute error bounded by sum_d ||R_d||² / noise_variance. A conservative
Cauchy–Schwarz bound controls mean error; endpoint evaluations propagate these
intervals to probabilities. Reported audit errors include these reference
error allowances. Bounds assume ordinary floating-point arithmetic error is
negligible relative to the allowances; they are not interval-arithmetic proofs.
Reference construction is outside timing and saved separately with residuals,
source hashes and settings. Sampled audit success is not a uniform box theorem.

This is a specialized local-query workload. Global reconstruction, many widely
separated boxes, kernel fitting, acquisition updates, partial scans, and
calibrated uncertainty on repeated real scans remain separate research tasks.
