# Coherent analytic tail-root prototype

The matrix-free tail-plus-noise root passes small complete-covariance controls
and paired draws on the released 691-point scorpion. **It is slower than dense
residual Cholesky at this size.** Current million-point bounds require many
expensive analytic actions; no massive root or application speedup is claimed.

This advances [#46](https://github.com/alecjacobson/fast-stochastic-psr/issues/46).
[#47](https://github.com/alecjacobson/fast-stochastic-psr/issues/47) tracks tighter
bounds, cheaper roots and application-based accuracy selection. It does not
change the flagship figures or the qualified mean-fit result.

## Model and bounds

The existing dense `Posterior.draw(exact=True)` samples the independent residual

`R = K_analytic - K_finite + noise_variance * I`.

With the analytically normalized finite spectrum, the omitted Fourier
coefficients are positive. Thus `Delta = K_analytic - K_finite` is PSD, its
diagonal equals the omitted spectral mass, and `Delta <= K_analytic` in PSD
order. We enclose R using:

- Lower bound: observation-noise variance.
- Upper bound: the smaller of noise plus `N * omitted_mass`, and the existing
  analytic spatial-box row-sum bound (including noise).

The box grid is 128³. Float64 evaluations have explicit numerical padding;
these are exact-arithmetic enclosures evaluated numerically, **not directed-
rounding certificates**. The bounds exclude numerical errors in the actions.

`FixedChebyshevRoot` selects a Chebyshev-Lobatto polynomial and degree using a
Bernstein-ellipse interpolation bound before seeing any Gaussian RHS. The
requested relative scalar-root error here is 1e-7. In exact arithmetic this
implies relative covariance error at most `2*error + error²` and preserves
Gaussianity because the map is fixed and linear. Clenshaw evaluation stores a
constant number of RHS arrays. Numerical action and recurrence errors are
checked separately.

`AnalyticTailAction` subtracts a double-precision, unnoised, analytically
normalized NUFFT covariance action from the noisy analytic range action. It
rejects finite actions that include noise, use released normalization or use
float32. Both actions must use identical coordinates and kernel parameters;
the caller is responsible for that match. No diagonal-tail substitution occurs.

## Released scorpion overlap

All cases use the same 691 float32 coordinates, per-axis variance .1, cutoff 50,
double-precision transforms at eps 1e-12, upsampfac 2, and analytic cached leaves
of size 128. Three independent vector-component RHS correspond to the residual
part of **one scalar posterior draw**, not three complete posterior samples.
The polynomial uses only the bound, never the measured dense eigenvalues.

| Case | Length scale / noise | Degree | Root, median subsequent call | Dense Cholesky draw, median subsequent call | Paired relative error |
|---|---|---:|---:|---:|---:|
| A | .01 / .02 | 8 | .1582 s | .0000876 s | 2.20e-11 |
| A short / low noise | .005 / .0004 | 44 | .8709 s | .0001524 s | 3.14e-12 |
| A long | 1 / .00001 | 2 | .03980 s | .0001044 s | 1.01e-9 |

Each method has one first call and three subsequent calls. Setup plus first
draw is .3341 / .9487 / .1074 s for the matrix-free method, versus .07489 /
.006967 / .003606 s for dense residual Cholesky. The dense first A factorization
includes a substantial first-use cost; the other cases reuse initialized
libraries. Preflight tests initialized the native range library. These are
component pilots, not cold-install or complete-posterior benchmarks.

Matrix-free totals include bound computation, polynomial construction, spectrum,
NUFFT plans, analytic plan and root action. Dense totals include the spectrum,
residual covariance construction, Cholesky and draw. Reference eigendecomposition
and validation are excluded from both timings. Dense Cholesky and the principal
root produce the same intended covariance but different paths for identical
white noise, so paired accuracy uses an independent principal-root reference.

All three cases pass action error <=1e-8 and paired root error <=1e-7. The full
covariance of the **dense evaluation of the same polynomial**, checked on every
basis RHS, has whitened operator error <=9.36e-10. This is not a claim that all
691 matrix-free basis RHS were evaluated. The tiny GPU unit tests do evaluate
every matrix-free basis RHS, including seam, duplicate, short-scale/low-noise
and long-scale cases. They also check conditional scalar covariance against
independent finite features, including correlations with the vector prior.

The 23 targeted CPU/GPU tests pass. Independent CPU replay reconstructs the
analytic and finite matrices using direct trigonometric sums, checks the
saved GPU draws and eigenvalues, and verifies polynomial covariance error.
The benchmark process audit passes; periodic samples can miss short-lived jobs.

## Scale feasibility: bounds only

These use the existing matched B and million-point inputs. They measure bound
construction, **not root execution**. Every degree requires that many analytic
range actions plus finite NUFFT actions per RHS batch. No elapsed root time is
extrapolated from earlier action measurements.

| Input | Length scale / noise | Degree at cutoff 50 | Degree at cutoff 200 |
|---|---|---:|---:|
| B, 17,419 points | .01 / .02 | 38 | 17 |
| B, 17,419 points | .005 / .0004 | 230 | 230 |
| Million | .01 / .02 | 281 | 147 |
| Million | .005 / .0004 | 1,633 | 1,633 |

The equal stress-case degrees come from the analytic box bound being smaller
than the tail trace at both cutoffs. The bounds can be loose: for A at standard
settings the actual largest residual eigenvalue is .020256, versus an upper
bound of .030665. Tightening the interval may substantially reduce work.
The 1e-7 root tolerance is an initial correctness target; downstream tasks may
permit a looser tolerance, which must be validated alongside sampling error.

The root component has not yet been integrated with the verified full analytic
conditional solve and streamed scalar/vector prior generation. Complete
posterior preparation, ray/collision probabilities, view rankings, and massive
application quality remain open. The 3.61× million-point mean-fit gain cannot
yet be applied to those workloads.

## Reproduce

```bash
./reproduction/run_env.sh -m pytest -q tests/test_square_root.py \
  tests/test_analytic_tail_gpu.py tests/test_inverse_root.py tests/test_inverse_root_gpu.py
./reproduction/run_env.sh -m reproduction.benchmark_analytic_tail
./reproduction/run_env.sh -m reproduction.validate_analytic_tail
```

The [raw record](results.json) retains setup/action times, bounds, coefficients'
parameters, source/input hashes and the GPU process audit. The small control
archives retain white noise, paired outputs, eigenvalues and polynomial
coefficients. The [CPU replay record](validation.json) is separate.
