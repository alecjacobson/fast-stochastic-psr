# Prior-basis query posteriors

The B ray/sphere panel now prepares in **98.89 s**, with the same 1e-7 numerical
output gates and zero ray/collision disagreements across 32,768 paired draws.
This is **1.79× faster than our adaptive full-query method (177.37 s)** and
**2.69× faster than our earlier strict matrix-free method (266.23 s)**. The
strong cached dense query reference remains faster at B (**5.407 s**).
These are single-pass pilots, not complete upstream application speedups.

## Method and bounds

Choose an orthonormal query basis Q from the scalar-prior covariance Kff.
For each independent vector component, X_d=C_d.T and W_d approximates
`A^-1 X_d Q`, with full analytic `A=Kvv+.02 I`. NUFFT actions assemble
`L=sum X_d.T W_d`; full all-observation analytic residuals assemble
`H=sum W_d.T R_d` and `U=sum R_d.T R_d/.02`, where `R_d=X_d Q-AW_d`.
No full training/query cross matrix is formed.

```
G = sym(Q.T L-H)
S_projected = Kff-L Q.T-Q L.T+Q G Q.T
S_projected-S_exact = sum (X_d-A W_d Q.T).T A^-1 (X_d-A W_d Q.T).
```

The error is positive semidefinite in exact arithmetic. With
`T=(I-QQ.T)Kff(I-QQ.T)`, the joint-prior PSD property gives, for any eta>0,

```
error <= (1+eta) T + (1+1/eta) Q U Q.T = upper.
```

We choose `eta=(||U||F/||T||F)^(2/3)`, which minimizes the Frobenius bound
when the two terms have orthogonal support. Zero-tail/residual limits are
handled separately. `||upper||F/(||S_projected||F-||upper||F)` bounds relative
Frobenius error when its denominator is positive. Each relative marginal
error is bounded by `upper_ii/(S_projected_ii-upper_ii)` when positive.
These are solve/truncation bounds, not interval certificates: prior/cross/
analytic-action numerical errors, mean error and root projection remain
separate. Negative prior eigenvalues within the fixed roundoff allowance
are clipped only when constructing the tail bound, and the projection is
recorded; the covariance formula uses the original symmetric prior.

All queries share one Gaussian covariance/root. Samples on separate batches
are not treated as independent, and new query panels require new work.
Reduced W/R vectors use the host-backed storage from the preceding study:
O(3Nr) host storage plus bounded GPU product chunks and O(M²+Mr) query arrays.

## B: same 17,419 observations and 1,024 queries

The panel contains rays 0,8,17 and sphere 8, with 256 samples each. The scalar/
cross cutoff is 50, query conditioning uses finite cutoff 50 plus two analytic
defect steps, and the separate mean fit retains analytic tolerance 1e-10.

The initial automatic rank rule uses only the prior: leave Frobenius tail
at most `rtol/8` of the prior norm. It selects **478 directions**, reducing
conditioning from 3,072 to 1,434 RHS. This heuristic alone is not a posterior
guarantee: the assembled covariance must pass both error budgets. A failure
is reported explicitly; automatic rank escalation is not implemented yet.

| Rank | Full preparation | Covariance L2 error | Maximum marginal error | Frobenius bound | Marginal bound | Quality |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 384, manual | 80.71 s | 1.06e-8 | **1.56e-7** | 6.63e-6 | 4.73e-5 | **Fail, retained** |
| 478, prior-only selection | **98.89 s** | **6.04e-11** | **1.19e-9** | **1.33e-8** | **9.89e-8** | **Pass** |

Errors/bounds are fractions, not percentages. Both runs have zero event
disagreements in 32,768 paired draws; this does not rescue the rank-384
marginal/budget failure. Rank 478 has full-field relative difference 5.81e-9.
Sphere-8 collision probability is 0.53262329, and the finite-grid 95%
simultaneous Monte Carlo radius is 1.256 percentage points over 769 events.
Propagated intervals separate ray 0 from the other two tested views. Numerical/
model bias and spatial discretization are not included in this sampling bound.

Independent dense controls check the PSD inequality; the minimum eigenvalue
of `upper-(S_projected-S_reference)` is -2.37e-20, within the recorded
roundoff allowance. All 23 targeted tests pass, covering zero/incomplete/full
bases, duplicate/near-duplicate queries, weak/zero components, tiny variance,
zero tail/residual limits and independent dense Fourier features. Both CPU
artifact replays pass, with the rank-384 quality failure preserved.

Preparation includes prior/basis work, all reduced solves and residuals, host
transfers/products, covariance assembly and root, plus setup and mean fitting.
It excludes process startup, input loading, independent validation, artifact
writes and final probability/score summaries. Draw RNG/field/event stages are
reported separately. Sampled peak GPU memory for rank 478 is 8.32 GB and host
RSS 1.37 GB; no peak-GPU-memory improvement is claimed. Partial final batches
can retain additional NUFFT plans, so the massive pilot uses a batch multiple.

## Massive panel and remaining work

A separate one-million-observation run of the **same 1,024-query panel** is
underway, using rank 512 (prior-spectrum headroom and no partial final batch),
cutoff 100, two defect steps and batches of four. Its outcome is not yet
established. It will retain local vector checkpoints and independently check
sampled direct rows against all sources. No massive application speedup,
strong same-input SDD comparison, or upstream intractability follows from the
completed B controls.

[#51](https://github.com/alecjacobson/fast-stochastic-psr/issues/51) tracks rank
selection, solve effort and massive validation. [#52](https://github.com/alecjacobson/fast-stochastic-psr/issues/52)
tracks finite-grid event-law bounds from PSD covariance budgets; the preliminary
equal-mean diagnostics there are not total numerical/model error certificates.
Geometry/Monte Carlo convergence and the full application suite remain open.

## Reproduce

```
./reproduction/run_env.sh -m pytest -q tests/test_query_projection.py tests/test_query_error.py tests/test_query_posterior.py
./reproduction/run_env.sh -m reproduction.benchmark_query_projection --case B --steps 2
./reproduction/run_env.sh -m reproduction.benchmark_query_projection --case B --rank 384 --steps 2
./reproduction/run_env.sh -m reproduction.benchmark_query_projection --case million --rank 512 --cutoff 100 --steps 2 --batch 4 --save-vectors
```

Benchmarks refuse to overwrite existing results. Preserve those directories
before rerunning and serialize GPU work. B source hashes correspond to
`dd1de63`; the later script adds optional local vector archival. The first CPU
replay exposed a query-hash representation mismatch: inputs are float32, while
the model saves their exact float64 promotion. Replay now verifies both the
original input hash and exact promotion; neither values nor hashes were altered.

B controls and all paired event arrays are public release assets listed in
[the hash manifest](release_manifest.json). Fetch and replay them with:

```
./reproduction/run_env.sh -m reproduction.fetch_query_host --manifest reproduction/results/query_projection/release_manifest.json
./reproduction/run_env.sh -m reproduction.validate_query_projection reproduction/results/query_projection/B_rauto_c50_s2_b4 reproduction/results/query_projection/B_r384_c50_s2_b4
```

The shared fetch command accepts either study's manifest. The massive run is
not included in this release while it is incomplete.
