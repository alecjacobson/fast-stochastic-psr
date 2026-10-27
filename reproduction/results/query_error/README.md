# Residual-corrected query covariance

This experiment uses approximate finite-NUFFT query solves, one full analytic
residual per block, and a quadratic correction to the assembled covariance.
It avoids strict analytic convergence for every query RHS. The target remains
the full analytic training covariance and finite scalar/cross spectrum 50.
The separate mean fit still uses analytic residual tolerance 1e-10.

## Error identity and scope

Let `X_d=C_d.T`, `R_d=X_d-A W_d`, and `A >= noise_min I`. Summing the three
independent vector components, define

```
S_raw = Kff - sym(sum_d X_d.T W_d)
S_corrected = S_raw - sym(sum_d W_d.T R_d)
S_corrected - S_exact = sum_d R_d.T A^-1 R_d
0 <= S_corrected - S_exact <= U = sum_d R_d.T R_d / noise_min.
```

These are exact-arithmetic identities. The covariance error after correction
is quadratic in the residual and positive semidefinite. Given a positive
definite computed covariance, let `eta` be the largest eigenvalue of
`S_corrected^-1/2 U S_corrected^-1/2`. If `eta<1`, the relative operator error
with respect to the exact covariance is at most `eta/(1-eta)`.
Nearly singular panels get an explicit noninformative result; no artificial
variance floor is inserted. Absolute and marginal budgets remain available.

The measured residuals cover **all observations and all query components**.
The budgets exclude cross/prior NUFFT error, numerical error in the analytic
action, mean/centering error, roundoff and PSD projection. They are not interval
certificates or automatic ray/collision probability guarantees. Small dense
controls and sampled direct rows separately check numerical consistency.

The preset solve-only relative budget is **1e-4 (0.01%)**. This is deliberately
different from the earlier strict query-posterior controls' 1e-7 covariance-L2
gate; no timing ratio between those quality targets is claimed. Both the
budget and independently observed covariance error are reported below.

## Dense overlap: 17,419 observations, 32 queries

Same B inputs and query panel as the previous
[fixed-query study](../query_posterior/README.md). This is a covariance-control
panel with a degenerate joint-inside event, not an application benchmark.

| Finite training cutoff | Setup + mean + corrected covariance/root | Raw covariance L2 error | Corrected covariance L2 error | Relative solve-only bound | Budget passes |
| --- | ---: | ---: | ---: | ---: | --- |
| 50 | 4.581 s | 1.0762e-3 | 2.5502e-5 | 1.3728e-4 | No |
| 100 | 5.716 s | 1.2136e-4 | 2.9280e-7 | 1.0980e-6 | Yes |
| 200 | 22.249 s | 1.6064e-5 | 3.3818e-9 | 1.3552e-8 | Yes |

All errors and bounds in the table are fractions, not percentages. Cutoff 100
does **not** pass the earlier 1e-7 covariance-L2 gate; cutoff 200 does. Cached
dense query covariance remains faster: the earlier dense run took 2.26 s
including 8,192 draws. These are single-pass pilots, not repeated medians.

Independent dense Cholesky checks the residual-correction identity to about
4.1e-14 relative to the target covariance. The PSD upper bound and entrywise
envelopes also pass. Raw and corrected matrices are both saved, so the
correction's benefit can be distinguished from accuracy already supplied by
the approximate solves. Mean error divided by prior standard deviation is
below 6.9e-10. Validation costs about 2.03 s separately from preparation.

Preparation is the sum of synchronized GPU/host stages: analytic/finite/scalar
setup, strict mean fit and centering, query covariance including all analytic
residuals, correction/budget assembly, and the query root. It excludes process
startup, input loading, independent validation and file output. Small RHS
snapshots for validation are captured inside the measured query solve. These
are not complete application wall-latency measurements.

## Million-point covariance control

Both pilots use the same million observations and 32 queries and pass the
preset 1e-4 solve-only budget:

| Finite training cutoff | Setup + mean + covariance/root | Query preparation alone | Relative solve-only bound | Sampled process memory peak |
| --- | ---: | ---: | ---: | ---: |
| 100 | **166.28 s** | 119.33 s | **7.0293e-5 (0.00703%)** | 34.56 GB |
| 200 | 246.95 s | 199.69 s | 3.4723e-7 (0.0000347%) | 37.30 GB |

The lower bandwidth reduces full preparation by **1.49× within our prototype**
at this preset solve-error budget. This is a single-pass comparison, not a
speedup over GeoSPSR or the best possible configuration. Both runs fit a fresh
mean at analytic tolerance 1e-10; their weights differ by 3.06e-9 relative L2
from the separately published full analytic-CG confirmation. The two computed
covariances differ by 6.24e-6 relative L2; neither is an independent exact
million-point covariance reference.

The 1,024 independent direct rows use every one of the million sources and all
96 query RHS. Maximum cross-kernel discrepancy divided by RHS norm is 1.87e-12;
the corresponding analytic-action discrepancy is below 8.65e-15. Full residual norms
used in the covariance budget are computed over all one million rows. The
direct-row check costs 1.36 s separately and does not certify a global numerical
error bound. Periodic process sampling can miss transient memory peaks and
short-lived foreign work. No foreign GPU process was observed in either run.

There is no independent dense covariance reference at this N, no same-input
SDD covariance timing, and no million-point application speedup established.

## Cost and remaining work

The prototype retains weights and residuals, requiring **O(3NM)** storage,
in addition to the analytic kernel plan and O(M²) query matrices. It does not
inherit the uncorrected assembler's O(Nb) streaming memory bound. At one
million observations and 32 queries those two retained arrays alone occupy
1.536 GB; concatenation and component products need transient storage too.
After validation the arrays are released.

The next steps are output-budget-driven correction/stopping, larger and
near-surface application panels, and a same-input comparison against the
strongest compiled SDD pipeline. Large dense memory requirements alone do not
establish that the authors' iterative method is intractable. Follow
[#49](https://github.com/alecjacobson/fast-stochastic-psr/issues/49) and
[#48](https://github.com/alecjacobson/fast-stochastic-psr/issues/48).
[#50](https://github.com/alecjacobson/fast-stochastic-psr/issues/50) tracks bounded
memory for the residual correction at application-size query panels.

## Reproduce

```
./reproduction/run_env.sh -m pytest -q tests/test_query_error.py tests/test_query_posterior.py tests/test_analytic_refinement.py
./reproduction/run_env.sh -m reproduction.benchmark_query_error --case B --cutoff 50
./reproduction/run_env.sh -m reproduction.benchmark_query_error --case B --cutoff 100
./reproduction/run_env.sh -m reproduction.benchmark_query_error --case B --cutoff 200
./reproduction/run_env.sh -m reproduction.benchmark_query_error --case million --cutoff 200
./reproduction/run_env.sh -m reproduction.benchmark_query_error --case million --cutoff 100
./reproduction/run_env.sh -m reproduction.validate_query_error reproduction/results/query_error/B_c50 reproduction/results/query_error/B_c100 reproduction/results/query_error/B_c200 reproduction/results/query_error/million_c100 reproduction/results/query_error/million_c200
```

The benchmark refuses to overwrite existing results: preserve the checked-in
directories elsewhere before regenerating them. Run GPU commands serially.
The million-point mean check needs the hash-verified `plain_weights.npy` from
the [analytic-refinement release](https://github.com/alecjacobson/fast-stochastic-psr/releases/tag/analytic-refinement-v1)
at `/opt/geospsr-runtime/analytic-refinement/confirmation/plain_weights.npy`.
The CPU replay uses only retained controls. The incomplete first JSON export
and cutoff-50 quality failure remain in [validation history](validation_history.json).
All twelve targeted tests and all five completed-run artifact replays pass;
the cutoff-50 B run remains a reported budget failure.
