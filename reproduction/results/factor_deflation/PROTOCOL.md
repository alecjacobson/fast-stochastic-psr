# Whitened-factor deflation: fixed prospective experiment

Declared before any full-slice factor-deflation run. Work under #40/#39/#4.
Retain every 20,000 A/B slice query, source/normal arrays, analytic vector
covariance, noise variance .02 and direct scalar/cross cube50. Use the committed
double variance reference after source/query/payload verification. The released
interpolant and correlated posterior events are separate models/workloads.

With H=C L3^-T, choose an independent Gaussian query sketch Omega, then
V=orth(H^T Omega), U=H V. Estimate diag(H H^T) as row_sum(U^2) plus second
moments of H xi-U(V^T xi), retaining the exact same xi used in field generation.
Subtract from known scalar prior variance without clipping. V uses double QR;
forward/adjoint actions and Gaussian field generation use the same float32
factor/cross kernel. Compute U again using double analytic Cholesky and double
NUFFT; require control-diagonal discrepancy ≤0.1% of reference spatial contrast.
A small explicit-Fourier/dense test independently checks both factor directions,
component layout and retained Gaussian pairing. Numerical error is separate from
the exact-arithmetic projection identity.

Sketch ranks64/128/256/512, evaluation counts32/128/512/2048, field batch8,
factor-action batch4. Pilot/evaluation seed pairs2401/401,2503/503,2607/607.
Each rank has its own independently timed sketch, adjoint action, QR, forward
action and control construction. Share evaluation fields across rank candidates
and a rank-zero raw-moment control, with individual residual-reduction timings.
Every seed reaches every look. Count sketch generation, both factor actions,
QR, persistent V/U allocation, transform plans, source generation, query,
projection/reduction and one final variance/probability output. Keep dense fit
and mean cost separate. Reference checks, pilot search and export are excluded
from candidate first-estimate stage sums. Keep all unsuccessful candidates.

Require contrast relative L2≤1%, probability RMS≤.05 percentage points,
max≤.5 points, positive variance everywhere and the control precision gate.
Measure three fresh streamed deterministic variances per process as the strong
baseline. Select lowest median total first-estimate cost among rank/count pairs
passing all three calibration seeds. No new ranks/counts to rescue a failure.

Confirm the selected setting in a new process with fixed independent pairs
2709/709,2801/801,2907/907, a fresh three-repeat deterministic baseline, and
whole-run sampled GPU-isolation audit. Publish timing ratios only if every
confirmation seed, deterministic quality and isolation gate passes. Do not
retry to rescue isolation or failed quality. Report all component costs and
per-seed ranges. Compare earlier query-space results descriptively, without
claiming a fresh paired timing ratio against that earlier execution.

Dense factor memory is quadratic in N; V has 3N rows and U has query_count
rows. No extrapolated massive-scale win, arbitrary moving-query reuse, or
complete-application speedup follows from this experiment. Extension requires
validated inverse-root actions and new query-geometry tests.
