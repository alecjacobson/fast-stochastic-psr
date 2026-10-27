# Full analytic verification/refinement of finite-cube NUFFT fits

Declared before implementation measurements under #45. Use same A/B normals and
queries, full analytic training kernel and leaf128 exact range action. Cases:
A/B ell=.01 noise=.02 and B ell=.005 noise=.0004. Shared scalar/cross cutoff50.
Independent double dense Cholesky reference. Existing full analytic gates:
all-row residual<=1e-8 per RHS, weights/centered mean<=1e-7. Mean gate covers all
20k slice queries; not posterior covariance or scalar bandwidth convergence.

Finite initial solver: packed float32 cube200, eps1e-6, upsamp1.25, CG rtol1e-5,
maxiter500. Compare (1) full analytic CG from zero, rtol5e-9/maxiter1000;
(2) analytic CG initialized by that NUFFT fit; (3) outer defect correction,
checking the full analytic residual after each step and approximately solving
its correction with the same finite-cube inner solver. Accumulate updates in
double. At most8 corrections; inner rtol1e-5 relative to each correction RHS.
This is separately verified defect correction, not ordinary fixed-SPD PCG.
Retain any failed finite/outer controls. No retuning for convergence/timing.

Measure first plus3 subsequent runs, charging initial NUFFT solve to both
combined methods and both model setups to first-use totals. Dense factor/reuse
and exact-from-zero are strong references. Account for analytic verification
inside each method; independent dense audit is validation-only, recorded apart
from method timing. Include native cache status and source/input hashes.
Only declare larger-scale follow-up if this pilot is numerically credible.
No complete application, upstream SDD or massive speedup inferred from A/B.

## Million-point pilot

A/B and stress cases pass after analytic CG or defect correction, but both
combined methods are slower than direct analytic CG there; retain these losses.
Stress defect correction needs7 corrections. Before scale measurements, test
one million identical points/normals: fixed first1M of archived3M cloud, verifying
point hash against the published range control. Same standard ell=.01/noise=.02,
finite cube200 and solver tolerances. Only the standard model is tested here;
no extrapolation from it to the short/low-noise case.

Measure first and one subsequent run of plain analytic CG, finite fit, warm
analytic CG and defect correction. Include both setups and initial finite fit
for combined totals; no preconditioner or post-result retuning. Native leaf128,
200M entry/32GiB cache cap, initially40GiB GPU free. Save complete weight arrays
in runtime storage with hashes; publish normals, query means and sampled
independent controls. All-row residuals use the analytic range action. Audit
1,024 seed4601 rows separately with direct analytic sums over all1M sources.
Compare weights and20k centered slice means to independently converged plain
analytic CG; this reference is not dense Cholesky at scale. Gates unchanged:
all-row and sampled residual1e-8, weights/means1e-7. Preserve finite failures.
No covariance/coherent-prior/application/SDD speedup claimed from this pilot.

## Runtime-loss confirmation

The first million pilot completed both repetitions and passed every gate, but a
VM/runtime reset then removed the ephemeral Python environment and full-weight
archives. Public-worktree inputs, means, sampled weights/residuals, source hashes
and results survived. Do not claim full-weight replay of that pilot or overwrite
its timings. Restore the locked environment and run a separate single-pass
confirmation of all four methods using the surviving identical input arrays.
Keep the same tolerances/settings and record new compilation/setup costs. Publish
confirmation full weights as durable release assets with array/file hashes.
Replay the original pilot's retained means/samples, and the confirmation's full
weights and independent audits. No attempt to recreate the lost exact hashes.
