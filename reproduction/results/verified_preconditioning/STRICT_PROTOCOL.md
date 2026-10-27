# Follow-up after the packed baseline covariance failure

Declared after the initial plain and float32 runs completed and while fused16
was still running. The initial plain packed run passes its 2e-5 independently
checked residual threshold but fails the unchanged empirical covariance gate:
0.010967% versus a 0.01% limit. Keep that entire experiment, without timing ratios.
The float32 coarse control passes the downstream gates.

Expose an optional `verification_rtol`, defaulting to the existing 2*rtol,
independent of native recurrence tolerance. Require refinement to be enabled
when specifying it. This is an explicit accuracy budget, not a relaxed gate.
Test its validation and correction behavior before starting the follow-up.

Repeat the same three fresh processes, order plain/float32/fused16, with a
uniform actual verification threshold **2e-6** for both mean and conditional
solves. Keep native CG rtol1e-5, correction CG rtol.1, cap two, identical archived
RHS/prior, geometry, preconditioner settings and every downstream quality gate.
Use the same tighter archived reference. No per-draw tolerance selection.
The follow-up is justified by numerical failure, not by seeking favorable timing
or isolation. Do not combine selected timings across the two experiments.

Preserve original records under plain/float32/fused16 and put this separate
experiment under strict/plain, strict/float32 and strict/fused16. Retain both
protocols and implementation manifests. Apply the same all-three-runs numerical
and whole-run sampled-isolation gates before reporting any follow-up ratio.
Setup and prior-generation exclusions are unchanged. Ratios, if eligible, are
single eight-draw numerical pilots. Independent seeds, geometries and parameter
regimes remain necessary before changing any default tolerance (issue #38).
