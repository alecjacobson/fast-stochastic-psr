# Fixed-query correction-covariance deflation

Declared before full-slice runs. Target the same direct full-analytic-vector /
scalar-cross-cube50 model as correction_contrast, noise variance .02, every
20,000 committed A/B slice query. This is not the released interpolant, a new
coherent sampler, or a replacement for the matched teaser images.

For correction covariance B=C A^-1 C^T, build a double-precision orthonormal
basis Q from independent Gaussian correction pilot fields. Compute BQ by
NUFFT cross/transpose actions and the unchanged dense inverse. The deterministic
diagonal contribution is 2 diag(QQ^T B)-diag(QQ^T B QQ^T). Accumulate second
moments of independent centered residual fields (I-QQ^T)c, then subtract the
combined correction diagonal from known scalar prior variance. No clipping;
negative controls/variance estimates are retained. This identity assumes exact
actions and an independent fixed pilot; numerical errors are separately checked.

Pilot ranks16/64/256, evaluation looks32/128/512/2048, batch8. Three independent
pilot/evaluation seed pairs1401/401,1503/503,1607/607. Each rank's own pilot,
QR and covariance-action costs are measured; evaluation draws are shared across
ranks and a rank-zero raw-moment control. Every seed reaches every look. Covariance
actions use four-column blocks with explicit shared plan setup charged once to
every candidate. Source generation plans and query plan setup are included.
The total number of sampled fields is pilot rank plus evaluation count; BQ also
requires rank covariance actions (two triangular solves each), whose cost is
not hidden by the sample count.

Use the archived float64 dense variance reference only after validating source
and query hashes, and measure three fresh float32 streamed deterministic
variances as the strong baseline. Use the same fitted mean for both probability
estimates. Require unchanged contrast L2<=1%, probability RMS<=.05pp,
maximum<=.5pp, and positive variance at every query. Independently recompute BQ
with float64 operators/analytic factorization and eps1e-12 for every pilot;
require the deterministic contribution discrepancy <=0.1% of reference spatial
contrast. This precision control is not an independent algorithmic oracle;
small explicit-Fourier/dense tests independently check adjoint signs and layout.

Select each case's fastest complete candidate (including pilot/QR/BQ/plan setup,
sample generation/query/reduction and one final output) that passes all three
pilot/evaluation pairs at the same checkpoint. This selection is exploratory.
If no candidate passes, retain the failure without additional unplanned settings.

Before claiming a timing ratio, confirm the selected setting in a new process
on three prospectively fixed pilot/evaluation pairs1709/709,1801/801,1907/907,
with three new deterministic baseline evaluations. A confirmation ratio requires
all three new pairs to pass unchanged quality/precision gates, the deterministic
baseline to pass, and the entire confirmation's sampled-isolation audit to pass.
Use only confirmation timings for ratios; do not select repetitions for good
isolation. No additional repeats to rescue a failure. Report per-seed ranges
and common fit/mean cost. These are fixed-query marginal first-estimate costs,
not posterior events, moving-query amortization, model/geometry convergence or
an upstream/complete-application speedup. Large-N conditioning remains a separate
step after this dense control.
