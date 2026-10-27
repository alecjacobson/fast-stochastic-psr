# Three-million-point oracle output stopping

This follows issue #36. Keep the existing source, noise variance .02, length
scale .01, per-axis variance .1 and scalar/cross cutoff 50. Verify the source and
the saved cutoff-250 reference weights against their published hashes. The
reference is a bandwidth-checked finite-kernel mean, not an exact analytic solve.
Historical residual gates and comparisons remain unchanged.

Before observing these outputs, fix calibration planes y=-.4 and -.7 and
held-out planes y=-.5 and -.8. These differ from the earlier display planes.
They share the same underlying scan; they are not independent scenes. Evaluate
the mean on 1025 and 2049 square grids over [-pi/2, pi/2] in x,z. Extract every
zero contour without filtering. Refine and compare extracted polylines using
the existing Hausdorff brackets with sample spacing 1e-4.

Record two target levels, coarse and strict: boundary upper distance no larger
than respectively .001 or .0001 times domain width. Both also require relative
L2 mean error <=.002 on each fine grid, matching contour/open-component counts
to the reference, stable counts between the two resolutions, and consecutive
grid upper distance <=the boundary target for both candidate and reference.
These are numerical benchmark tolerances, not physical or universal application
acceptance criteria, continuous-curve error bounds, or topology guarantees.

Select the first recorded checkpoint passing BOTH calibration planes for each
target. Report the held-out result at that same checkpoint; do not select a
later iteration using held-out results. Keep all checkpoint results, including
failure and any later regression. This is an **oracle reference-based accuracy
study**, not a deployable stopping rule. Reference preparation and verification
costs must be reported separately; they are required if using the oracle.

NUFFT uses the production unpreconditioned conjugacy-folded cutoff-200 CG solve,
rtol 1e-5/epsilon 1e-6, observing iterations 4,8,16,32,64 and its final iterate
(at most 1000). The optional observer must not alter the iterates, validated
against the same solve without an observer. Run a fresh process once.

SDD uses the compiled released algebra, batch512, eta5, lr=5*512/N,
momentum .9, Polyak .01, seeds 0 and 1. Record iterations
128,256,512,1024,2048,4096,8192,16384, in a fresh process per seed. Always run
all checkpoints; the selected endpoint is retrospective, not an actual early
termination. Do not extend this schedule after inspecting accuracy.

For each checkpoint separately time aggregation, all-source centering,
calibration fields/contours/metrics, and held-out fields/contours/metrics.
Record setup and solver/update time between checkpoints. Calibration-path cost
includes setup, all solver work and all prior calibration checks, excludes
held-out confirmation, file I/O and reference preparation. Report those excluded
costs and the full instrumented wall separately. Oracle-inclusive cost adds
reference preparation; confirmation-inclusive cost also adds the selected
held-out check. No deployment speedup follows from an oracle-only selection.

GPU process samples cover the whole run. Publish timing ratios only if both
SDD seeds confirm the selected calibration endpoint on held-out planes, the
NUFFT endpoint also confirms, and all whole-run isolation audits pass. Timing
precision is limited by one process per configuration and the monitoring
overhead; do not compare to historical hardware times.

Retain compact checkpoint contour archives and raw summary metrics. Store the
larger fields/coefficients outside Git with payload hashes. Recompute numerical
quality from these arrays and contour vertices before publication. Do not
reinterpret this as posterior variance, correlated ray/collision, full surface,
next-view, ContactFusion, or upstream-intractability validation.
