# Whole-box probability bounds and larger Torso validation

The six selected Eagle comparisons from the
[large-scan study](../poissonrecon_large/README.md) now have **whole-box bounds**
on the differences between their represented moment/probability fields. This
strengthens the accuracy comparison underlying the full-Eagle **8.08×**
mean-and-variance speedup: its stored H² and locality polynomials agree within
the 0.001 probability tolerance everywhere in the contact box.

| Eagle points / control | Observed million-query CDF difference | Whole-box upper bound | Sub-boxes |
|---|---:|---:|---:|
| 32,768 / locality | 0.00029027 | 0.00044117 | 1 |
| 32,768 / float32 Cholesky | 0.00000677 | 0.00000907 | 1 |
| 77,824 / locality | 0.00055331 | 0.00098172 | 4 |
| 77,824 / float32 Cholesky | 0.00014996 | 0.00023442 | 1 |
| 196,608 / locality | 0.00066066 | 0.00093671 | 4 |
| 796,825 / streamed locality | 0.00091743 | **0.00099935** | 25 |

Displayed upper bounds are rounded upward; raw values are in
[bounds/summary.json](bounds/summary.json). The full-Eagle bound takes **about 0.3 s
on the CPU**, independent of the number of subsequent queries. All six variance
difference bounds are also below the 1e-4 prior-normalized tolerance, and both
fields have positive variance lower bounds over the complete box.

The order-six/order-seven counterexample is retained: both fields pass their
17-point audits, but their observed full-query probability difference is
0.00149244. The whole-box comparison **does not accept it**: its upper bound is
0.00178102 after the 256-leaf budget. An upper bound above tolerance means
`inconclusive`; it is the independently observed query difference that proves
this pair fails the tolerance. This pair changes both interpolation order and
variance iterations, so it is not an isolated iteration experiment.

The bounds also locate a useful follow-up region. Evaluating a 3x3x3 grid in
the worst sub-box finds probability difference **0.00103032** at physical
coordinate **(1.000625, -0.005, -0.0025)**. This is a new 27-probe numerical
observation that catches the pair's failure without replaying a million queries;
it remains separate from the interval bound and from an exact-GP reference.

These bounds apply to the **real polynomials represented by the saved binary64
coefficients**. They do not include query-evaluation roundoff, construction or
interpolation error, or error to the exact GP. The direct full-kernel point
audits remain separate; neither form of numerical agreement establishes model
calibration or equivalence to original Sellán–Jacobson SPSR.

See [METHOD.md](METHOD.md) for derivation, outward-rounding assumptions,
subdivision, complexity, and CPU-only replay commands. Both the NUFFT and
nonperiodic H² routes remain available.

## Validation

The CPU suite passes **33 tests**, covering the existing free-space posterior
and the new interval bounds: exact rational restriction checks, dense sampled
cross-checks, unequal orders/levels, positive-variance recovery by subdivision,
an interior tolerance violation, invalid inputs, overflow and very small
variances. A fresh replay into a separate directory reproduces the six bounded
comparisons and the unaccepted counterexample. Input and implementation SHA256
hashes accompany every bound result.
