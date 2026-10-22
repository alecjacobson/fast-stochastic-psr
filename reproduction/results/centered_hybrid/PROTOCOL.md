# Wider prior, cheaper centered inverse: declared experiment

Use the cutoff-200 coherent vector prior and fitted mean, scalar/cross cutoff 50,
and a cutoff-100/150/200 inverse for centered prior-plus-noise fluctuations.
The scalar prior, observed data, mean weights, offset, random RHS and query
geometry stay fixed. This is an experimental hybrid law, not an exact posterior
at the narrower bandwidth. The coupled sampler remains the default.

Before the trials, retain the existing cross-model gates: field error divided
by fluctuation norm <0.2%, empirical variance/covariance errors <0.02%, local
ray/collision probability differences within the benchmark contract, view regret
within two ray steps, and uncertain collision coverage. A centered cutoff-200
control uses the stricter same-model numerical gates. All analytic residual
holdouts remain recorded with the existing <0.1% gate; passing output checks
alone does not silently turn a failed residual into a full protocol pass.

Exact float64 dense moments at released A/B and the existing short-scale,
low-noise B stress case use fixed queries from the prior spectral-tail audit.
Panel gates versus the analytic model are mean error <0.2%, variance/covariance
error <0.02%, probability RMS <0.1 pp and maximum <2 pp. This panel does not cover
full application geometry. The held mean uses the wider finite model; its error
against the analytic mean is assessed separately.

At one million points, replay the saved common RHS and priors. Start with eight
draws and three timing repetitions, then expand a promising setting. Solver
tolerances apply to the centered RHS in these variants, so the cutoff-200
centered control is necessary. Retain failed settings. Publish speed ratios only
when both configurations pass the required quality checks and sampled GPU
isolation; shared-host contention invalidates performance claims.
