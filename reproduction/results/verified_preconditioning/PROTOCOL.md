# Verified three-million-point conditional solves

Predeclared before these runs. Compare plain packed CG, float32 monolithic
Nyström and fused float16 Nyström in three fresh, sequential processes, in
that order. Use the published three-million input and all eight seed701 RHS
and scalar priors from `verified_posterior/three_million`. Validate source,
geometry and archived array hashes. These are fixed existing numerical controls,
not new independent data or a held-out Monte Carlo study.

All use the full training cube200, scalar cube50, analytic normalization,
ell .01, per-axis variance .1, noise variance .02, float32 eps1e-6, rtol1e-5.
Use packed solves, batch one, zero starts. Verify each mean and conditional
solve with serial float64 same-cube actions at eps1e-10; acceptance actual
residual <=2e-5, up to two corrections at native rtol .1. Include verification
and correction in measured solve time. No relaxed tolerance or hybrid inverse.

Both coarse models use rank1024, landmark seed41, row blocks8192, and two-pass
construction to bound memory. Float32 stores the basis without compression and
uses monolithic GEMM. Fused float16 decodes in registers and accumulates float32.
Both retain the existing post-storage SPD guard. Two-pass construction is used
also for the stronger float32 baseline to avoid its full 24.6GB temporary
feature array coexisting with double verification. Report storage and setup;
this is not the fastest-possible unlimited-memory float32 construction.

One first-use eight-draw experiment per method; no reruns chosen to recover
favorable timing/isolation. Record fit, coarse setup, each conditioning solve,
aggregation, queries, transfers/export and wall time. First conditioning includes
packed-plan creation and first use of preconditioner kernels. Existing CuPy disk
compilation caches may be warm. No untimed solve warmup. Compare the eight-solve
sums and setup-inclusive first eight, separately from fit-inclusive components.
Prior generation is excluded because coherent RHS/prior arrays are loaded from
the validated archive. Do not call these complete application timings.

Compare all methods to the archived stronger reference (float64-stored weights
verified <=2e-7, double aggregation/mean centering). Reuse its frozen geometry and
fields; prior/RHS hashes must match exactly. Existing field gates: fluctuation
L2 <5e-4; empirical variance/covariance L2 <1e-4. Existing v3 event gates and
uncertain-collision coverage must pass. Also compare methods pairwise on these
same draws. Record all native failures and independent correction histories.

A timing ratio requires all three complete runs, all numerical gates and all
three whole-run sampled-isolation checks. Otherwise publish diagnostic times
without ratios. Even eligible ratios are a single eight-draw pilot, not a
robust timing confidence interval, absolute probability accuracy, convergence
to the analytic model, a geometry guarantee or an upstream speedup. Report
sampled process memory as including fitted model, verification and caches.
