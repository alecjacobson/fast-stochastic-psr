# Spatial block preconditioning pilot

Compare the existing packed full-cube200 solver against Morton-ordered analytic
principal-block inverse preconditioners with block sizes 64,128,256. Noise
variance .02, length scale .01, per-axis variance .1, scalar cutoff50, tolerance
1e-5, two-draw batches. Keep the covariance action, eight common conditional RHS
and prior coefficients identical. No centered narrower inverse is used.

Run the baseline and each candidate in separate processes, three repeats each.
Retain every result and failure. Report setup separately and setup plus solves;
setup is paid once per map. Require all compared whole-run process audits to
pass before reporting any timing ratio. No attempts to select faster reruns.
Iteration counts and numerical quality remain useful under contention.

Use the existing same-model field/covariance/event gates against baseline,
including numeric next-view integration. Compare independent finite-covariance
residual rows on any candidate selected for larger validation. Small dense
spectral controls test PCG and preconditioner symmetry/positive definiteness,
partial blocks, periodic coordinates and heteroscedastic noise.

This tests repeated conditional solves on common finite-model RHS, not full
application speed, absolute probability quality, geometric convergence, full
analytic-tail convergence or ContactFusion. Consider 128 draws only if the
pilot suggests a useful setup/solve tradeoff, with an explicit new validation
plan. Retain failed or slower candidates, including small-N costs.

## Small-input confirmation

Before inspecting candidate sweep outcomes, also declare a four-draw B overlap
using the existing common-rhs-smoke data (17,419 observations), same full cube200,
packed batch2 and three repeats. Compare the baseline and block128; retain
setup, numerical gates and any slowdown. This checks small-input overhead and
is not a claim that NUFFT beats the established faster dense/SDD choices there.
