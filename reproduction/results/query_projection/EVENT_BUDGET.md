# Finite-grid event probability diagnostics

A PSD solve/truncation budget can bound changes in ray and collision
probabilities, even when using a different coupling to draw the two fields.
This is a conditional mathematical bound, evaluated in ordinary floating
point. It is not a total application-error certificate. The original
numerical quality gates remain unchanged; rank 384 remains rejected.

Let X denote the exact finite-grid Gaussian and Y the approximation. If
`Cov(Y)-Cov(X)=E >= 0`, there exists a coupling
`Y=X+delta+eta`, with eta independent of X, covariance E, and deterministic
mean difference delta. Suppose `|delta_i|<=d_i`, `E_ii<=u_i`, and the known
approximate marginal is `Y_i~N(mu_i,s_i)`. For any t>0, opposite signs imply
`|Y_i|<=t+d_i` or `|eta_i|>t`. Thus

```
P(sign(X_i) != sign(Y_i))
 <= min(1, 2(t+d_i)/sqrt(2*pi*s_i)
          * exp(-max(|mu_i|-t-d_i,0)^2/(2*s_i))
          + erfc(t/sqrt(2*u_i))).
```

The first term bounds the Gaussian density over the interval around zero;
the second bounds the centered Gaussian difference tail. Independence of eta
and Y is unnecessary. Consequently, the same formula also works with any
Gaussian factor coupling if u bounds the squared row norms of the factor
difference, even without PSD covariance ordering. This observation supplies
a possible route to handling other numerical covariance errors; those bounds
have not yet been established for the massive pipeline.

This derivation improves the original issue-52 density bound, which used
X's unknown mean and lower variance `s_i-u_i`. The implementation retains
that older option for comparison. The new bound uses the known approximate
variance, handles `u_i>=s_i`, and is never larger on the common valid domain.
Zero perturbation, deterministic variance, and unknown mean intervals are
handled explicitly. A minimum over the fixed 128-value geometric threshold
grid `t/sqrt(u) in [0.5,30]` preserves the mathematical upper bound. It is
not numerical interval arithmetic or a claim of optimal thresholds.

Union bounds across prefixes give ray-transmittance probability bounds;
a union across all sphere nodes bounds the collision-event probability.
These compare event laws through an existence coupling. They do not bound
disagreements under an arbitrary shared-root coupling.

## Retained B panel

The saved controls use 17,419 observations, three 256-node rays (0,8,17),
one 256-node sphere (8), and 32,768 independent Gaussian draws. We evaluate
three explicit scenarios: equal means, the observed difference from the
independent dense mean reference, and a solve-only mean bound derived from
all-observation residuals. The observed difference remains an oracle
diagnostic. Results below use the residual-derived bound, without using the
dense fitted answer.

| Rank | Full ray 0 | Full ray 8 | Full ray 17 | Sphere 8 | Original numerical gate |
| --- | ---: | ---: | ---: | ---: | --- |
| 478 | 0.00084846 | 0.00100494 | 0.00029854 | 0.00640655 | Pass |
| 384 | 0.01323861 | 0.01414131 | 0.00658595 | 0.11109872 | **Fail** |

Values are probability fractions. Rank 478's sphere-law budget is about
0.641 percentage points; rank 384's is 11.11 points. The simultaneous
95% Monte Carlo radius is 1.256 percentage points over 769 events.
Adding these quantities gives sphere-8 intervals of approximately
[0.513660, 0.551587] and [0.408967, 0.656279], respectively, under the stated
assumptions. They exclude additional numerical, model and geometry errors.

Prefix-specific law bounds plus the simultaneous Monte Carlo radius yield
scanning-score intervals `[0.333333,0.352941]`, `[0.019608,0.019608]`, and
`[0,0]`. Ray 0 remains separated for both ranks. This is a decision on the
retained three-ray grid, not a guarantee for the full application scene or
for continuous rays, and it does not rescue rank 384's failed numerical gate.

## Mean fitting and empirical centering

Let `r_d=y_d-A alpha_d` be the mean-fit residual for vector component d.
Joint-prior positivity and `A>=noise_min I` imply

```
|mean_fit_error(q)|
 <= sqrt(kff(q,q)) * sqrt(sum_d ||r_d||^2 / noise_min).
```

The implementation subtracts the empirical mean field over all observations
to choose its isovalue. Triangle inequality includes that error by adding the
average scalar-prior standard deviation at those observations to the first
factor. For a stationary prior this doubles the factor. This bound requires
no dense fitted reference and no new query-conditioning solves.

For B, the recorded all-row residual norms are approximately
`[2.9492e-9, 4.2910e-9, 9.5597e-10]`, the noise floor is 0.02, and the scalar
prior standard deviation is 0.00169531. The centered mean error bound is
**1.2693e-10** after rounding upward for display. Including it changes the
rank-478 sphere-law bound to **0.00640655**, with the same separated best
view. Residual norms use the analytic action in ordinary floating point;
this is a conditional solve-only bound. Action error and mean/centering
field-evaluation error are still excluded, so this is not a numerical
interval certificate.

## Validation and reproduction

Nineteen CPU tests pass. They include the analytic centered bivariate-normal
sign formula, independent scalar quadrature with unequal means, correlated
Gaussian perturbations and duplicate nodes, an indefinite covariance
*difference* with a valid factor coupling, zero errors/variances, tiny scales,
invalid inputs, and scanning-score interval enclosure on a nonuniform grid.
Four additional tests compare the centered/uncentered mean bound against
independent nonstationary feature models at two noise levels. Two initial test expectations were corrected: the finite threshold grid
retains a positive 30-sigma tail, and floating-point 0.1+0.2 is not exactly
0.3. No algorithm or application gate was changed for those corrections.

```
./reproduction/run_env.sh -m pytest -q tests/test_event_probability_budget.py
./reproduction/run_env.sh -m reproduction.evaluate_event_probability_budget reproduction/results/query_projection/B_rauto_c50_s2_b4 reproduction/results/query_projection/B_r384_c50_s2_b4
```

Fetch the release controls as described in the parent report first. Each
`event_budget.json` retains node/prefix bounds, all three mean scenarios, source
and control hashes, and all exclusions. Issue #52 remains open for justified
mean/numerical covariance budgets and prospective application-driven rank/
solve selection. No extra GPU benchmark was run concurrently with the live
million-point pilot.
