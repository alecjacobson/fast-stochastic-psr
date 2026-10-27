# Packed real source aggregation and exact planar query contraction

Declared before these full-slice runs. The instrumented original B forward
factor action spends ~3.74 s in source aggregation, ~.15 s in the triangular
solve and ~.13 s in query evaluation. These synchronized diagnostic stage times
are not a speedup baseline. Test the dominant source work by packing independently
scaled pairs of real channels into one complex type-1 transform and reconstructing
individual spectra using conjugate reversed modes. Restore the complete odd
centered cube before cross-kernel multiplication. Zero channels remain zero.

Run three variants in separate processes on A and B: fresh generic batch16,
packed observation-side source transforms (factor forward and correction draw
aggregation), and packing plus exact planar query transforms (both directions
in the factor and field evaluation). Source-only packing leaves query plans
unchanged. Planar contraction retains all Fourier modes, original coordinates
and heights, grouping exact equal coordinates after the standard precision cast.
No snapping, reduced bandwidth, altered data density or interpolation grid.
Double control factors always use unpacked generic 3D transforms.

Use exactly the original factor-space pilot's selected settings: A rank256 /
128 evaluation draws, B rank512 /32 draws; factor batch16, field batch8. Retain
confirmation sketch/evaluation pairs2709/709,2801/801,2907/907 across variants.
No new calibration or rank/count tuning. Keep every 20,000 original query, full
analytic vector covariance, scalar/cross cube50 and noise variance .02. Verify
input/reference hashes and retain the prior independent double variance reference.

Require all seeds to pass contrast relative L2≤1%, probability RMS≤.05pp,
max≤.5pp, strictly positive variance and double deterministic-control discrepancy
≤0.1% of reference spatial contrast. Also require each candidate's paired variance
outputs to differ from the fresh generic execution by ≤0.1% of reference contrast.
Do not claim a ratio if any required quality gate or whole-run sampled isolation
fails. No repeats to rescue quality, isolation, memory or a poor timing result.
Report all three variants and all per-seed costs, including failures.

Measure three fresh streamed deterministic variance evaluations in each process.
Include sketch generation, both factor actions, QR, control/state construction,
plan setup, all evaluation generation/query, projection/reduction and one output.
Keep shared fit/mean separate; exclude validation references, export and process
startup from first-estimate stage sums. Compare candidate medians to their fresh
deterministic baseline and to the fresh generic first-estimate median only if
both executions qualify. This is one three-seed confirmation per method/case,
not application wall time or broad timing-distribution evidence.

Small explicit Fourier/factor tests check source packing, scaling across unequal
channels, odd/zero channels, adjoints, complex coefficients, seams, axis choices,
partial batches and paired retained normals. The standalone planar query test
uses all original B slice points and three repeats per direction, with setup
reported, without translating its query-only timings into application speedups.

Dense factorization and the 3N-row basis remain. Neither packing nor planar
queries establishes a massive-scale inverse, moving-query reuse, released-
interpolant compatibility, or correlated event/application quality. Keep those
requirements open and do not replace the main original-code teaser images.
