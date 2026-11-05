# Whole-box bounds for regional moment caches

The represented fields are Chebyshev polynomials in the same physical box:

\[
h_i(q)=\mathrm{level}_i-\mu_i(q),\quad
v_i(q)=v_{0,i}-c_i(q),\quad
p_i(q)=\Phi(h_i(q)/\sqrt{v_i(q)}).
\]

The inputs to this comparison are the stored binary64 coefficient values,
interpreted as exact real numbers. Different polynomial orders are zero-padded;
levels may differ, including the separately anchored levels in the benchmark.
Centers and halfwidths must match exactly. The probability bound includes level
differences. Variance error is normalized by the reference prior variance.

On the normalized cube, every Chebyshev basis product has magnitude at most one.
Thus the sum of absolute coefficients bounds polynomial magnitude; the constant
coefficient plus/minus the sum of the remaining magnitudes bounds its range.
We compute these bounds for both numerators, both variances, and their differences.
If the variance lower bound is nonpositive, the result is inconclusive; no
variance clipping is allowed.

For positive variance, global derivative bounds give

\[
|p_1-p_2|\leq
\frac{\epsilon_h}{\sqrt{2\pi v_{\min}}}
+\frac{\epsilon_v}{2\sqrt{2\pi e}\,v_{\min}}.
\]

The implementation uses slightly larger constants 0.4 and 0.121. Within each
sub-box it also bounds the numerator's magnitude above and below over the
convex hull of both fields' ranges. With those bounds denoted by
\(h_{\max},h_{\min}\), and the variance upper bound by \(v_{\max}\), set
\(E\geq\exp(-h_{\min}^2/(2v_{\max}))\). Valid local derivative bounds are

\[
L_h=0.4E/\sqrt{v_{\min}},\qquad
L_v=\min(0.121/v_{\min},\;0.2h_{\max}E/v_{\min}^{3/2}).
\]

The comparison uses \(L_h\epsilon_h+L_v\epsilon_v\), capped at one.
The exponential upper bound is the reciprocal of a lower bound on the first
21 positive Taylor terms of \(\exp(t)\). Downward clipping of \(t\) at 64
prevents overflow without invalidating the upper bound. This avoids relying on
the rounding accuracy of a library exponential or CDF implementation.

If a bound does not meet the tolerances, the algorithm bisects the box, choosing
an axis by coefficient variation. The Chebyshev restriction matrices for
\(T_k((t\pm1)/2)\) are generated using exact rational arithmetic. Interval
matrix application encloses each arithmetic operation. Leaves form a complete
partition of the original box. The result is `bounded` only if every leaf has
positive variance lower bounds and meets both tolerances. Exceeding the leaf or
depth budget yields `inconclusive`, which is **not** evidence that the actual
error exceeds tolerance. Unresolved leaves remain in the aggregate bounds.
The output retains the subdivision path to the worst leaf. The comparison CLI
also evaluates both fields on a 3x3x3 grid in that leaf, providing
targets for follow-up checks. These point observations are reported separately; they
do not change an inconclusive interval result into a formal failure bound.

Every bound-producing sum, difference, product, division and square root uses
outward binary64 rounding via `nextafter`; reductions round at every pairwise
addition. The assumptions are IEEE-754 round-to-nearest basic arithmetic and
square root, with gradual underflow. Overflow/invalid arithmetic raises an
exception rather than accepting a comparison. Axis selection and queue priority
are heuristics and do not affect enclosure validity.

## What this does and does not establish

This is a bound on **real polynomials represented by the saved coefficients**.
It does not include floating-point error in the CPU/GPU query evaluator, the
conversion from a quadratic cache to polynomial coefficients, interpolation,
cross-basis truncation, hierarchy approximation, solves, or model calibration.
Agreement between two approximate caches does not establish agreement with the
exact GP. The existing direct full-kernel probe audits remain separate evidence.

The bound is independent of the number of subsequent query points and the
training-point count once the fields have been built. Its cost instead depends
on polynomial degree, the number of subdivisions, and separation from zero
variance. A conservative or near-threshold case can consume the leaf budget.
This is a validation primitive for issues #79/#83/#93, not yet an automatic
production stopping rule or a proof of the solver's asymptotic complexity.

For a padded degree array with side length \(d\) and \(L\) partition leaves,
the current tensor restriction implementation takes \(O(Ld^4)\) arithmetic and
at most \(O(Ld^3)\) working storage. A single-box coefficient bound costs
\(O(d^3)\). These costs exclude cache construction and use fixed-dimensional
three-dimensional boxes. They do not imply that the required degree or number
of leaves stays constant for arbitrary growing datasets or query regions.

## Reproduction

From the repository root:

```sh
./reproduction/run_env.sh -m pytest tests/test_regional_bounds.py -q
./reproduction/run_env.sh -m reproduction.run_regional_bounds --output /tmp/regional-bounds-replay
```

The replay is CPU-only. It uses the six selected full-query comparisons in
`../poissonrecon_large/eagle` and the order-six/order-seven counterexample.
It checks recorded artifact hashes where the older comparison provides them.
Every new bound result records both input hashes and implementation hashes.
The counterexample changes interpolation order **and** variance iterations;
it does not isolate either source of error.
