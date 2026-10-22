# Same finite-cube representation: declared comparison

Compare the full cube and positive-half-plus-zero-plane implementation on the
same million-point input, eight saved conditional RHS, cutoff200 prior/mean,
centered cutoff150 fluctuation inverse, offset and application geometry.
No model frequency is removed. Retain the stricter same-model numerical gates:
field error / fluctuation norm <0.05%, empirical variance/covariance <0.01%,
plus the existing event/uncertainty coverage and analytic residual checks.
The held hybrid model still has the separately documented full-analytic gap.

Run six repetitions per implementation. Report the median of **all** eligible
repetitions only if at least three per implementation have sampled GPU isolation.
Eligibility uses a preceding and following process observation, rejects foreign
PIDs/errors in that bracket and rejects observation gaps exceeding two sampling
intervals. Whole-run and per-stage audits are both retained. No selection by
runtime is permitted. If fewer than three qualify, an additional six-repetition
batch may be collected; combine all eligible repetitions from both batches.
Contended measurements remain archived and excluded. Other projects are never
stopped. Periodic sampling cannot prove exclusive GPU use.

The measured interval is conditional solving, including CG control and true
residual checks. Plans, source transfer, prior generation and later application
queries are separate. A passing result is a conditioning speedup, not a new
complete-application or SDD ratio. Sampled process memory is a lower-resolution
observation of peak usage, not an exact allocator peak. A larger-sample run and
coherent-prior/posterior checks must precede claims about sampling preparation.
