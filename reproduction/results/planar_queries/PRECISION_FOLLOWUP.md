# Independent spectrum reference before full variance comparison

The first all-B-query microbenchmark failed its planned 1e-5 relative spectrum
comparison to native 3D type1 (observed1.76e-4). Preserve this failure in
initial-micro-failure; source is commit a7ff655. It establishes no timing ratio.

A 69-mode direct-sum diagnosis on the same real strengths and original float32
coordinates finds aggregate errors: generic32/up1.25 1.28e-3; planar32/up1.25
3.33e-5; planar32/up2 9.28e-6; planar64/up2 1.53e-12. These are selected-mode,
unweighted Fourier errors, not posterior variance errors. They do not invalidate
the prior kernel-weighted variance checks, nor do those checks establish high
accuracy of every unweighted transform mode.

Before any full variance execution, set planar 2D oversampling to2 (generic 3D
remains1.25) and compare BOTH query directions to a full double 3D reference
(eps1e-12,up2) on the same input arrays. Preserve the 1e-5 relative spectrum/field
threshold and report each method's pass/fail. Generic32 is a performance control,
not an exact numerical oracle. No query-only ratio if either method fails this
microbenchmark gate. This corrects the reference, not the finite kernel or data.
The initial failure remains visible.

All full-slice variance, probability, positivity, double-control and paired
output thresholds in ../packed_factor_sources/PROTOCOL.md remain unchanged.
No rank/count/seeds are reselected. Include the larger planar grids and all
contraction/plan costs. If the revised planar microbenchmark fails, keep it out
of the full-variance candidate run and retain the failed record; packed-source
work can still proceed independently.
