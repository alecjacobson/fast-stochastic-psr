# Fixed contrast-sensitive correction study

Run released A (691) and B (17,419) on every one of their 20,000 fixed slice
queries. Preserve full analytic vector covariance, scalar/cross cutoff 50,
noise variance .02, and direct Fourier evaluation. This studies the direct
model; it does not replace the matched-interpolant teaser comparisons.

Use a dense factor A=L L^T and independent Gaussian vectors xi. The correction
coefficients aggregate alpha=L^-T xi, so the evaluated correction has covariance
C A^-1 C^T. This removes the full Fourier prior from marginal-variance sampling.
It produces correction functions, not posterior draws suitable for ray or
collision events. Dense N-by-N storage remains required. Validate against an
independent explicit Fourier cross sum and dense covariance before benchmarking.

For each dataset, use independent seeds 401, 503, 607, batch eight, and fixed
looks at 128, 512, 2,048 and 8,192 independent draws. Use correction-only moments:
no pilot weights, clipping, adaptive stopping or selection of favorable seeds.
Accumulate second moments with memory proportional to query count, retaining
only each checkpoint's variance estimate. All seeds reach the maximum count.

Require relative L2 error over centered spatial variance contrast <=1%, inside
probability RMS <=.05 percentage points and maximum <=.5 percentage points,
and no nonpositive estimates. These are prospective exploratory marginal gates;
passing does not by itself establish action-ranking or full application quality.
Use the same fixed mean for both probabilities, isolating variance estimation.

Compute a streamed float64 dense reference with cross-sum epsilon 1e-12. Compare
against the stronger deterministic float32 streamed cross-factor variance path,
including its query setup and all three fresh evaluations. Require that baseline
also pass the same quality gates. Record common fitting/mean-query cost separately.
Record stochastic Gaussian generation, triangular solve and source aggregation
as generation; query/transfer and moment reduction separately; cumulative wall
time includes checkpoint/statistics overhead. Source loading, reference matrix,
factorization, reference queries and full instrumented work wall are recorded.
Factor setup is shared across stochastic and deterministic methods, not free.

Report whole-run process isolation and retain every failure. A speedup ratio
requires all seeds to pass at the SAME checkpoint, baseline quality to pass,
and whole-run isolation to pass. Otherwise report costs and failures without a
ratio. No historical-machine timing comparison, posterior-event or upstream
speedup follows. Intervals use Gaussian correction chi-square bounds with a
Bonferroni allocation across queries and four looks, excluding numerical bias.

The runtime had to be restored from the pinned lock on this VM; large runtime
caches are absent. These experiments use committed A/B inputs. Do not reuse
historical million-point timings as a substitute for new larger-scale work.

Source transform plans are reused within each draw stream, including a separate
plan for a partial final batch. One query plan is timed explicitly and reused
across fixed-size batches. Include that setup in any eligible stochastic total.
