# Fixed-polynomial correction sampling pilot

Implement a fixed Chebyshev inverse-root map and a bounded streaming correction
sampler. Coefficients/degree depend on the source and declared spectral interval,
never on the Gaussian RHS. This is a control for future rational/Krylov sampling,
not a claim that polynomial sampling is fast or solves Monte Carlo variance noise.

Use the complete finite training cube 200, scalar/cross cube 50, length scale
.01, per-axis variance .1 and noise variance .02. Compare the fixed polynomial
with the existing full finite joint-prior-plus-noise/CG correction extraction.
Both target C A^-1 C^T. Neither output is a posterior function sample for event
queries. The pathwise baseline creates an unused scalar prior through the
existing joint-prior routine; retain that qualification instead of presenting
it as a minimal vector-only implementation.

Run the released A source, then the first one million points of the published
three-million-point source, then all three million, in fresh processes serially.
The million-point prefix is a new workload, not the earlier million-point scan.
Verify source hashes; every selected observation enters the covariance action.

Set a degree-independent spectral lower bound to the noise variance. Use a
128-cubed periodic box-count convolution upper bound, capped by the covariance
trace bound, computed before any draw. The box envelope uses decreasing positive
analytic kernel entries; a finite analytic-normalized Fourier kernel inherits
the analytic PSD upper bound. Float64 evaluation and a 1e-8 relative pad are
numerical safeguards, not directed-rounding certification. NUFFT error remains
separate. Record the upper bound and degree for the trace-only control too.

Select Chebyshev-Lobatto degree for an exact-arithmetic scalar relative error
bound <=1e-4 using Bernstein ellipses, capped at degree 20,000. If the cap fails,
retain the failure. Use float32 GPU actions (epsilon1e-6), two independent
Gaussian field draws, seed401, batch1. A draw has three independent normal RHS.
The polynomial length is fixed; no adaptive stopping or per-RHS fitted basis.

For every draw separately time preparation, aggregation and 128 selected slice
queries. Shared source/query plans and operator setup are counted separately.
Validate each polynomial draw using the round trip A p(A)^2 xi versus xi:
relative L2 <=.001 for EACH of its three RHS. This is a random-direction
numerical diagnostic, not an operator norm or covariance guarantee. Validation
cost is excluded from draw cost and reported explicitly. On released A also
compare the same Gaussian RHS to a dense symmetric inverse root of the SAME
finite kernel: require relative weights error <=.001, and correction-field
relative L2 <=.001. Dense construction/eigendecomposition costs are separate.

For the pathwise baseline use two draws, seed509, the same finite cube and
rtol1e-5; preserve true finite residuals and require the existing solver's
convergence flag. Keep every outcome. Do not compare individual realizations
across different valid Gaussian couplings as if they were the same sample.

GPU ownership/memory is sampled throughout. There are only two draws per method
and one process per size; no complete-application or Monte Carlo convergence
claim. Publish costs without a speedup ratio unless every quality check and
whole-run isolation audit passes; even then scope any ratio to draw preparation.
Archive compact query fields and hashes; full RHS, weights and coefficients stay
outside Git. Existing small dense covariance, seam/duplicate/low-noise tests and
streamed partial-batch tests must pass before the scaling run.
