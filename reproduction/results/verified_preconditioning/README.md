# Verified preconditioning at three million observations

Fused compact Nyström conditioning is compared to plain packed CG and the
stronger float32 monolithic Nyström baseline, with independently checked
same-cube residuals and shared coherent posterior draws. The initial experiment
finds a downstream covariance failure even though residual checks pass. No
initial timing ratio is eligible. A separately declared tighter-verification
follow-up is tracked in [issue #38](https://github.com/alecjacobson/fast-stochastic-psr/issues/38).

[Initial protocol](PROTOCOL.md) · [Initial summary](summary.json) ·
[Tighter follow-up protocol](STRICT_PROTOCOL.md) ·
[Preconditioning issue #33](https://github.com/alecjacobson/fast-stochastic-psr/issues/33).

## Initial residual threshold: 2e-5

| Method | Eight verified solves (s) | Coarse setup (s) | Setup + solves (s) | Empirical covariance difference | Numerical gates | Sampled process peak (GiB) |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| Plain packed CG | 126.363 | 0 | 126.363 | 0.010967% | **FAIL** | 27.00 |
| Float32 monolithic Nyström | 113.409 | 35.282 | 148.691 | 0.003596% | pass | 37.71 |
| Fused float16 Nyström | 101.772 | 35.277 | 137.050 | 0.002664% | pass | 31.98 |

All three runs pass sampled GPU isolation. All tested ray/collision events and
view choices match the archived tighter reference. The plain solver exceeds the
unchanged 0.01% covariance-difference limit, so the predeclared all-method
quality gate excludes **all timing ratios from this experiment**. Its pairwise
field checks against both preconditioned paths also fail. The methods are not
made comparable by ignoring this small failure.

Every packed conditional solve passes its independently recomputed residual
without a correction. In contrast, the earlier folded-transform controls on
these identical RHS needed one correction each. Their different costs are not
combined into a new cross-experiment timing ratio. Independently verified
residual tolerance still needs a downstream accuracy budget; it is not itself
a covariance guarantee.

## Scope

The published three-million source, eight seed701 RHS, scalar priors, geometry
and tighter reference come from the [verified posterior study](../verified_posterior/README.md).
All source and prior/RHS hashes match. Training cube200, scalar/cross cube50,
noise variance .02 and the underlying finite posterior are unchanged. Both
coarse models use rank1024, seed41 and bounded two-pass construction. The latter
avoids a 24.6GB feature temporary for float32 while double verification is
resident; it is not claimed to be the fastest unlimited-memory construction.

These are conditional solves from **already prepared coherent RHS and priors**.
Verification, any corrections and first packed-plan/kernel use are included.
Mean fitting is separately reported (about 25.2 s per run), as are aggregation
and queries (about .5 s for all eight). Prior generation, input transfers,
geometry, validation, hashing/export and startup are excluded from table sums.
The CuPy disk compilation cache may be warm. Setup + solves is a measured stage
sum, not a separately measured end-to-end application time. Peak memory is
sampled process memory including model, verification and allocator caches.

One eight-draw experiment per method does not establish an absolute Monte Carlo
error, convergence to the analytic infinite-band model, continuous collision
geometry, robust timing statistics or an upstream speedup. The reference is a
stronger same-finite-cube numerical control, not an exact independent oracle.

## Reproduce

Regenerate the local archive from the linked verified-posterior study first.
Run methods sequentially on the GPU:

```bash
./reproduction/run_env.sh -m reproduction.benchmark_verified_preconditioning --method plain --output reproduction/results/verified_preconditioning/plain --archive /opt/geospsr-runtime/verified-preconditioning/plain
./reproduction/run_env.sh -m reproduction.benchmark_verified_preconditioning --method float32 --output reproduction/results/verified_preconditioning/float32 --archive /opt/geospsr-runtime/verified-preconditioning/float32
./reproduction/run_env.sh -m reproduction.benchmark_verified_preconditioning --method fused16 --output reproduction/results/verified_preconditioning/fused16 --archive /opt/geospsr-runtime/verified-preconditioning/fused16
./reproduction/run_env.sh -m reproduction.summarize_verified_preconditioning
./reproduction/run_env.sh -m reproduction.verify_verified_preconditioning --output reproduction/results/verified_preconditioning/plain --archive /opt/geospsr-runtime/verified-preconditioning/plain
```

Repeat the last command for float32 and fused16. Public field/event artifacts
are small; full weights and coefficients remain in a regenerable local runtime
archive, with public payload hashes. Verification replays coefficient fields and
events and checks archive hashes; it does not rerun original residual checks.
