# Paired real RHS: comparison declared before repeated timings

Compare packed full-cube transforms with the best folded-cube implementation
on the same million-point input, eight stored draws, cutoff200 prior/mean,
centered cutoff150 inverse, offset and application geometry. Test batch sizes
one and two for the packed implementation; compare both against folded batch
one. A single packed batch-one pilot precedes this protocol and is excluded
from repeated performance summaries.

Run six repetitions for each configuration. Use the median of **all** intervals
passing the existing bracketing GPU-process audit, requiring at least three
per configuration. If fewer than three qualify, collect one additional batch
of six and combine all eligible observations. Retain excluded timings; never
select by runtime or stop another project's GPU jobs. Periodic process sampling
does not prove exclusive GPU use. Report each tested configuration separately. Use the faster eligible packed
configuration for one 128-draw numerical confirmation, and report its timing
only with its own process-audit qualification; do not call that single run
repeated 128-draw evidence.

Keep the strict same-model gates: field error / fluctuation norm <0.05%,
empirical variance and covariance errors <0.01%, and existing event/uncertainty
coverage and analytic residual gates. Packing must not change the finite
covariance or noise model. Dynamic-range channel tests separately check
roundoff leakage. The centered hybrid's model limitations remain unchanged.

These timings cover conditioning, including CG control and true residuals.
Setup, prior generation and application queries are outside that timer.
Sampled process-memory maxima are not exact allocator peaks. A larger-draw
confirmation and coherent-sampler checks are required before integration is
advertised; neither alone proves complete-application or analytic-posterior
accuracy. No new upstream or ContactFusion ratio follows from this experiment.
