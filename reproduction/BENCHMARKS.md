# Benchmark contract and coverage

Goal and work queue: [#1](https://github.com/alecjacobson/fast-stochastic-psr/issues/1).
This contract separates measured speed from evidence that the same application
is being solved. New results must state which checks they meet.

## References and quality

1. **Released implementation:** pinned upstream commit
   `2352f1ef98aab20e0303611c1fa2b0948acd2a05`, untouched APIs where available.
2. **Strong cached reference:** identical algebra with reused factors, kernel
   rows shared across samples and reasonable GPU batch sizes. Check against the
   released APIs on feasible cases.
3. **Corrected statistical reference:** explicitly identify noise, covariance or
   sampling corrections. Such a reference is not the untouched stochastic API.
4. **Intended-model reference:** independent direct Fourier calculations and
   dense analytic conditioning where feasible, with cutoff convergence. Matching
   a coarse interpolation table alone does not establish model accuracy.

Record observations, normals, queries and their hashes; seeds, covariance and
noise conventions; Fourier/prior/interpolation cutoffs; solver and NUFFT
tolerances; coherent sample count; geometric discretization; exact baseline
provenance. Separate numerical error, model/truncation error, sampling error and
geometric discretization error. A small finite-system residual does not certify
the full analytic model. Report failures and difficult cases too.

### Initial gates

These thresholds are declared before the corresponding new benchmark runs.
They are working acceptance criteria, not universal application guarantees.
Changing them requires a reason and reclassification of earlier results.

| Comparison | Initial gate | Additional reporting |
| --- | --- | --- |
| Matched Figure 1 fields (existing) | Mean relative L2 <0.2%, variance <0.02%, probability RMS <0.1 percentage points and maximum <2 points | Error maps, shared color scales, deterministic reference |
| Paired ray evaluation | RMS transmittance difference ≤1 percentage point; maximum ≤2 points | Paired standard error, per-sample event disagreement, full curves |
| Paired collision evaluation | Maximum probability difference ≤2 percentage points | Paired standard error, uncertain objects, surface point convergence |
| Paired next-view scoring | Reference score regret ≤two ray steps | Full scores, selected view, ties, multi-seed decision stability |
| Hitboxes / sequential scan / massive-scale posterior | Gates pending task-specific convergence studies | No matching-quality claim until those gates are specified and passed |

Paired evaluation gates only check numerical evaluation of the same prepared
draws. They do not certify the draw distribution, Monte Carlo convergence,
continuous collision geometry or downstream scan quality. At 128 samples the
worst-case marginal probability standard error is 4.42 percentage points;
sharing draws can resolve much smaller *backend differences*. Independent
sample-count/seed studies remain necessary for absolute application accuracy.

**Gate revision 2:** the initial A/B runs used only the RMS ray threshold. B's
direct-Fourier diagnostic passed that threshold but differed by 17.19 percentage
points at one ray location. The maximum-error gate above was added after this
finding; earlier RMS-only passes are not sufficient. B's initial collision
trajectory also produced only 0/1 probabilities. Require uncertain collision
cases before claiming uncertainty validation. Tracked in
[#10](https://github.com/alecjacobson/fast-stochastic-psr/issues/10).

For an application benchmark, use equal statistical accuracy, not automatically
equal sample counts. Samples must be spatially coherent for joint events. Test
small and large input/query counts, partial and clustered scans, length scales,
noise levels, and uncertainty regions where sign decisions are sensitive.

## Timing and memory

- Synchronize GPU work at stage boundaries. Preserve first-call and at least
  three repeated query measurements; report medians and individual values.
- Separate data generation, kernel setup, fitting, prior/residual sampling,
  conditional solves, source aggregation, field queries, event reductions,
  mesh extraction, and acquisition/rebuild costs.
- Report preparation plus first application result and repeated-query time;
  identify amortization assumptions. Never present a supplied-weight kernel
  action as an end-to-end application speedup.
- Cache and batch fairly: reference factors, cross blocks across samples and
  fixed-query results may be reused whenever the accelerated path also benefits
  from that fixed workload. Repeating identical queries solely to measure warm
  kernels must be distinguished from an application that can cache its answer.
- Include plan creation/transfers unless explicitly timed as preparation. Record
  hardware/software, memory measurement method, peak live/workspace memory,
  repetitions, concurrent GPU use, precision and timing exclusions.
- A recorded run is not promoted if quality gates fail or remain unverified.
  Cold-stage sums from a shared diagnostic harness are not independently
  optimized end-to-end application measurements.

## Coverage and next evidence

| Paper example / application | Present evidence | Missing work / issue |
| --- | --- | --- |
| Scorpion mean, variance, statistical slices (Fig. 1) | Matched A/B; original API verification | Faster variance [#4](https://github.com/alecjacobson/fast-stochastic-psr/issues/4) |
| Length-scale/detail and Fourier/grid convergence (Figs. 2–4, 14, 16) | Kernel checks, compatible interpolant | Controlled downstream accuracy sweeps [#5](https://github.com/alecjacobson/fast-stochastic-psr/issues/5) |
| Posterior sampling and local evaluation (Figs. 6, 10) | Corrected CPU/GPU moment checks; optimized preparation and fairly batched solve comparisons | Full-model massive prior/covariance and sample convergence [#3](https://github.com/alecjacobson/fast-stochastic-psr/issues/3), [#13](https://github.com/alecjacobson/fast-stochastic-psr/issues/13) |
| Uncertainty-aware hitboxes (Fig. 5) | Faster complete small demo preserves paired mesh connectivity; clipping noted | Geometry/uncertainty convergence and controlled upstream comparison [#4](https://github.com/alecjacobson/fast-stochastic-psr/issues/4) |
| Collision probability (Fig. 7) | Paired uncertain probes, two B seeds; cached query acceleration | Geometry/sample convergence and complete preparation costs [#3](https://github.com/alecjacobson/fast-stochastic-psr/issues/3) |
| Ray transmittance and next-view scores (Fig. 9) | Paired moving queries; cached query acceleration; reference-grid convergence | Broader decision/ray/sample convergence [#3](https://github.com/alecjacobson/fast-stochastic-psr/issues/3) |
| Sequential scanning (Fig. 8) | One seed, approximate sampler | Multi-seed acquisition/refit benchmark [#7](https://github.com/alecjacobson/fast-stochastic-psr/issues/7) |
| Input/output scaling and solver convergence (Figs. 11–13, 15) | Training-bandwidth audit and converged compiled SDD mean fits through 1m observations | Batched conditional-solve/parameter coverage [#6](https://github.com/alecjacobson/fast-stochastic-psr/issues/6), [#15](https://github.com/alecjacobson/fast-stochastic-psr/issues/15) |
| Massive-scale applications | Million-point conditioning/sampling pilot; independent SDD/NUFFT residual holdout | Complete application quality bridge and larger impractical-upstream cases [#8](https://github.com/alecjacobson/fast-stochastic-psr/issues/8) |

Scenes not present in the release remain labeled reconstructed task scenarios.
Audit additional supplementary examples as they are incorporated; file new
issues for missing coverage or discovered numerical/performance failures.

The [cached-query report](results/cached_queries/README.md) adds function-level
checks on 257 independent queries, including 32-point empirical covariance.
Before those runs, numerical cache gates were set to error / fluctuation norm
<5e-4 and empirical variance/covariance relative errors <1e-4. These isolate the
cache approximation; they do not replace model or sample-count convergence.

The [training-bandwidth report](results/training_bandwidth/README.md) declares
additional overlap gates: full query covariance relative L2 ≤0.02% and full
analytic training residual ≤0.1%, alongside the field/probability gates above.
Its massive-input pilot requires the empirical bootstrap 95% upper endpoint for
512 sampled analytic residual rows to be <0.1%, for both observed-normal and
conditional-sampling right-hand sides. This sampling gate is not a global bound
and does not certify full-model application quality. Training cutoff 150 passes
the million-point mean check but fails the conditional-draw check; cutoff 200
passes both in the eight-draw pilot. Short-scale/low-noise B fails all tested
cutoffs, tracked in [#14](https://github.com/alecjacobson/fast-stochastic-psr/issues/14).

The [spectral-tail diagnostic](results/spectral_tail/README.md) provides separate
posterior-covariance evidence: a PSD bound validated against dense analytic A/B
conditioning and estimated on a fixed 16-query million-point panel. Its
frequency-sampling standard errors are empirical; they are not rigorous or
simultaneous certificates. Production/tighter numerical settings are compared
separately. Passing this limited covariance panel does not close the mean,
centering, joint-event, sample-count or geometry requirements above.

## Beyond feasible upstream runs

Validate overlapping sizes first, then increase distinct real scan observations
and query counts separately. Measure resource limits under declared budgets.
Include the released iterative solver: a 4 TB dense matrix does not establish
that every upstream algorithm is infeasible. Extrapolations must state their
assumptions and cannot be labeled measured speedups. Sampled analytic residuals
and overlap checks support, but do not prove, full-model fidelity at larger N.

The [iterative conditioning report](results/iterative_baseline/README.md) now
confirms a tuned, compiled SDD baseline at one million observations. It is
feasible within a 240-second update budget. The confirmation uses 1,024 analytic
rows independent of the 512 tuning/stopping rows. Both methods pass the sampled
residual gate, but this does not certify full-model posterior covariance or
joint-event accuracy. The 27.2× first-fit ratio is one confirmation measurement,
not a complete-application result. Three warm fit repetitions cover A/B/100k;
the million-point fit is not repeated.

The [conditional-solve report](results/conditional_solves/README.md) shares analytic
kernel rows across all SDD sample RHS. At B the cached dense solve wins; at one
million points the eight-draw solve ratio is 4.39×, not the mean-fit ratio. The
massive pilot's vector prior is truncated, so it does not certify full-model
posterior accuracy. Paired field gates were declared as error/fluctuation norm
<0.2%, variance/covariance errors <0.02%, alongside existing local event gates.
Both solvers pass independent per-draw analytic residual checks. Bootstrap
intervals are marginal, not simultaneous guarantees.

Optimized exact sampler preparation is a separate same-method result. CPU/GPU
randomness shares a posterior law but not realizations; independent moment tests
and paired same-RNG controls validate it. The full application wall measurement
includes fitting through mesh export and excludes process/data startup and
plotting. It is a single run per variant, not a runtime confidence interval or
proof of geometric/sample convergence.
