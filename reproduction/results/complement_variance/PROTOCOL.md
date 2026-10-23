# Marginal variance pilot

Before running any pilot, fix the released 691-point A input, analytic vector
covariance, scalar/cross cutoff 50, length scale .01, variance .1 per axis,
observation-noise variance .02, and the corrected coherent pathwise sampler.
This is the direct model, not the original interpolated Figure 1 pipeline.

Use 128 evenly spaced indices from the released 20,000 slice queries and 128
evenly spaced observed points. Use seeds 107, 211, 307, each with 32 independent
pilot draws followed by 128 scoring draws. Batch size is four. Retain all seeds.
Freeze weights from the pilot's component second moments. Do not use the exact
reference or the scoring draws to tune weights; count pilot costs explicitly.

Compare known-mean posterior second moments, prior variance minus correction
second moments, and their weighted combination. Never clip negative estimates.
Compute the analytic training matrix and Cholesky in float64 and cross sums at
NUFFT epsilon 1e-12, separately from the float32 sampler. This reference shares
spectral code; an independent real-feature dense GP test checks the decomposition.
Compare sampled components against dense conditioning of the actual saved RHS
and a higher-accuracy evaluation of the same scalar prior coefficients.

Per seed, require numerical component error below .05% relative L2 and combined
variance error below 1% relative L2, inside-probability RMS below .5 percentage
points and maximum error below 2 percentage points. Report raw posterior and
correction-only errors as well. These are exploratory marginal-quality gates,
not replacements for established same-draw numerical gates or application
quality requirements. Failure is retained; do not adjust thresholds afterward.

Report the posterior-to-prior variance range, empirical component cross moments,
negative counts and simultaneous component-interval coverage/width. Exact-law
intervals assume independent Gaussian draws, correct centering/coupling and no
numerical bias. Intersecting intervals are allowed to be empty. These are
marginal-variance diagnostics, not correlated event-probability certificates.

All stages and a whole-run GPU process audit are recorded for diagnostics.
This pilot establishes no runtime ratio: it has no fixed-quality stopping rule
or equally optimized baseline, and includes dense reference work. A theoretical
oracle estimator-variance ratio is not a measured sample-count or speedup ratio.
A/B action ranking, larger scales, streaming memory, extra query costs and
independent confirmation beyond this pilot remain future work.
