# Fixed-rank larger-sample and streaming validation

Hold rank1024 and landmark seed41 fixed after the eight-draw pilot. Condition
all128 common full-cube200/scalar50 draws, packed batch2, rtol1e-5, noise variance
.02. Report full128 and draws8..127 (120 pilot-held-out draws) separately using
unchanged same-model field/variance/covariance and paired-event gates. Geometry
is reused, not held out. Independent32-row finite residual threshold .002%.

Reference is the archived nufft128 at commit8fa5204: that implementation hardcodes
FourierCovariance(200), uses the same source/RHS/geometry hashes and noise.02,
and starts from the fitted mean. Starting guesses do not alter the converged
finite model. Its timings are not a contemporary packed-solver timing baseline.
Recompute all paired-event scores using the current numeric5%-95% convention;
do not reuse its historical gate-version2 score metadata.

One confirmation process, no speedup ratio from historical timings. Retain
setup, solve, fields, finite-row checks, audit and all failures. Then validate
an explicit reusable model preconditioner in the coherent streamed producer:
identical prior/RHS with the same seed and batch schedule, partial batches,
early-exit cleanup and reuse across streams. Keep default draws unpreconditioned.
A million-point coherent8-draw control must compare unpreconditioned retained
samples with preconditioned streaming on the same fitted mean and seed, with
independent finite rows and unchanged field/covariance/event gates. This checks
integration, not absolute MC/geometry/analytic-tail convergence or speed.

No timing reruns to salvage a ratio; a future qualified paired timing experiment
needs its own protocol. Preserve existing application-quality failures.

## Discoveries after the protocol was fixed

The source audit found legacy RHS hashes were computed before a float64-to-
float32 storage conversion. A read-only regeneration of all128 pairs and the
four-draw B cache checks the exact stored RHS/prior bytes and recovers the old
hashes. New preparation hashes stored dtypes and includes prior hashes. This
provenance correction is separate from numerical and timing gates.

New streaming tests exposed a fixed low-noise float32 covariance failure at
the default tolerance. Preserve the initial failures and all12 tolerance/
transform-accuracy sweep outcomes. Selecting rtol3e-6 at eps1e-6 for that
stress test is exploratory; it does not change the predeclared million-point
rtol1e-5 or any acceptance gate. No sweep case becomes a performance claim.
