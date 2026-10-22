# Full-cube symmetry: a measured conditioning gain

At one million observations, a positive frequency half plus a 2D zero-axis
plane reduces the eligible median for eight conditional draws from
**38.16 s to 31.38 s (1.22×)**.
Every frequency of the finite training cube is retained. This is an incremental
conditioning improvement over our existing cube150 solver, not a new complete
application or upstream SDD ratio.

## Matched workload and repeat selection

The comparison keeps the cutoff200 cube prior and mean, scalar/cross cutoff50,
centered cutoff150 inverse, eight stored random RHS, source ordering, offset,
and ray/collision query geometry fixed. Only the inverse's numerical
representation changes. The underlying wider-prior/narrower-inverse hybrid
is the same as in the [earlier report](../centered_hybrid/README.md); its
full-analytic and application-convergence limitations remain.

| Implementation | Eligible / total repetitions | Eligible median | Sampled process-memory peak |
| --- | ---: | ---: | ---: |
| Full cube150 | 4 / 6 | 38.16 s | 5.25 GiB |
| Folded cube150 | 5 / 6 | 31.38 s | 2.88 GiB |

The [protocol](PROTOCOL.md) declares six repetitions per implementation,
at least three eligible repetitions, and the median of **all** eligible
measurements. No selection by speed occurs. Each solve interval must have
preceding/following GPU-process observations, no foreign PID/error in that
bracket, and no sampling gap longer than two intervals. Whole-run audits are
also retained; a job outside a solve need not invalidate an independently
bracketed solve. Periodic sampling can miss short jobs and is not proof of
exclusive GPU use. No other project was stopped.

Memory is the largest observed own-process usage in eligible windows, sampled
at approximately one-second intervals and including bracketing observations.
It may miss short-lived allocations and is **not an exact allocator peak**.
Source transfer, plan creation, prior generation and later queries are outside
the solve timer and are recorded separately. The three-repetition preliminary
runs had whole-run contention and do not supply the reported ratio.

[Full records](full150-six/results.json) · [Folded records](half150-six/results.json) ·
[All repetitions and summary](summary.json).

## Numerical and larger-sample checks

This is a **same-model** comparison: the numerical gates stay at field error /
fluctuation norm <0.05% and empirical variance/covariance errors <0.01%.
The folded eight-draw covariance error is 0.001011%.
Ray/collision outputs and selected view agree in the recorded paired check.
Analytic residual holdouts are separate from these same-model gates and retain
the existing 0.1% target. They still concern the underlying finite-prior hybrid.

The 128-draw comparison against the prior full cube150 run also passes: field
error / fluctuation norm is 0.000486%, empirical
covariance error is 0.000596%, and the
maximum held-out analytic residual upper endpoint is
0.07897%.
The raw 128-draw solve takes 527.81 s; its timing is **excluded** because its process-audit interval did not pass.
This is one confirmation, not repeated 128-draw performance evidence. Both
the original and folded implementations use finite common priors; this check
does not establish absolute Monte Carlo convergence or equivalence to the
full analytic posterior. [128-draw records](half150-128/results.json).

## Coherent priors and a usable posterior class

`HalfCubePosterior` implements fitting, coherent finite-cube draws, optional
centered narrower solves, and the existing prepared-query cache. Small tests
compare joint vector/scalar and posterior moments with independent explicit
Fourier sums, including all three possible folding axes, partial batches and
cached evaluation of non-Hermitian coefficient arrays. Action/CG tests also
cover odd/even widths, periodic seams, zero modes and float32/64 arithmetic.
See [complete test output](tests.log).

A production cutoff200 check generates four coherent draws at one million
observations. Rebuilding its first canonical prior on a complete 3D grid gives
RHS discrepancy 0.000102% and scalar
coefficient discrepancy 0.000000%.
Independent direct finite-covariance rows give maximum residual
0.001694% against the 0.002% gate.
Its mean differs from the saved full cube200 mean by
0.000613% on the same 16 probes.
Four draws and 512 validation rows are an execution/numerical check, not
posterior probability convergence. [Production records](production-draws/results.json).

For real weights, negative folded frequencies are the conjugates of the
positive half. Matrix actions add twice the positive half's real part plus
the zero-axis plane. Prior coefficients instead use amplitude `sqrt(2*rho)`
on the positive half and `sqrt(rho)` on the plane. The scalar low modes share
those same Gaussian draws. Negative-half scalar coefficients may be zero;
taking the real Fourier sum supplies the conjugate contribution. Seeded
realizations need not match a separately generated full-cube sampler.

```python
from reproduction.half_posterior import HalfCubePosterior

model = HalfCubePosterior(points, normals, training_cutoff=200)
model.draw(128, batch=1, solve_cutoff=150)  # existing centered hybrid law
model.prepare_query_cache()
mean = model.mean(queries)
samples = model.samples(queries)
```

Omit `solve_cutoff` for the consistently coupled finite-cube posterior.
Changing the inverse band is a separate modeling approximation; folding alone
does not change it. Dense/compiled-SDD references still win some smaller
workloads. There is no claim that this representation restores omitted analytic
frequencies or that one million points are intractable for the authors' SDD.
[Issue #26](https://github.com/alecjacobson/fast-stochastic-psr/issues/26) tracks
remaining full-workload coverage; [#27](https://github.com/alecjacobson/fast-stochastic-psr/issues/27)
proposes packing real RHS in full complex transforms as another speed trial.

## Reproduce

Use the existing common-RHS scratch dataset and prior reference artifacts.
Raw measurements here use NVIDIA L40, float32, CG rtol1e-5 and NUFFT eps1e-6.

```bash
./reproduction/run_env.sh -m reproduction.benchmark_nufft_batches solve --samples 8 --batch 1 --repeats 6 --centered-fluctuations --solve-cutoff 150 --audit-analytic --stop-rows 2048 --holdout-rows 4096 --holdout-seed 2711 --scratch /opt/geospsr-runtime/common-rhs-128 --reference reproduction/results/centered_hybrid/million-150 --output reproduction/results/half_cube/full150-six
./reproduction/run_env.sh -m reproduction.benchmark_nufft_batches solve --samples 8 --batch 1 --repeats 6 --centered-fluctuations --solve-cutoff 150 --half-cube --audit-analytic --stop-rows 2048 --holdout-rows 4096 --holdout-seed 2711 --scratch /opt/geospsr-runtime/common-rhs-128 --reference reproduction/results/half_cube/full150-six --output reproduction/results/half_cube/half150-six
./reproduction/run_env.sh -m reproduction.check_half_draws --output reproduction/results/half_cube/production-draws
./reproduction/run_env.sh -m reproduction.benchmark_nufft_batches solve --samples 128 --batch 1 --repeats 1 --centered-fluctuations --solve-cutoff 150 --half-cube --audit-analytic --stop-rows 2048 --holdout-rows 4096 --holdout-seed 2711 --scratch /opt/geospsr-runtime/common-rhs-128 --reference reproduction/results/centered_hybrid/million-150-128 --output reproduction/results/half_cube/half150-128
./reproduction/run_env.sh -m pytest tests -q
./reproduction/run_env.sh -m reproduction.summarize_half_cube
```
