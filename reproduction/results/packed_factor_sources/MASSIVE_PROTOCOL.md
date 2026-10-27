# Three-million-point aggregation carryover control

Declared before execution. After the A/B runs finish, test source packing on
all three million observations using the eight archived, verified fused16 strict
posterior weights from verified_preconditioning/strict/fused16. Verify point,
weight, prior and geometry hashes. Reuse the archived coherent priors, offset,
training cube200, scalar/cross cube50 (its original query oversampling2), and all existing field,
ray and collision queries. Do not refit, resample, alter bandwidth or truncate N.

Compare cached generic type1 with cached packed-real type1, at the existing
streaming batch1 (three real channels become two complex channels). Share the
same resident input arrays and generic query plan. Measure each source-plan
setup and three repetitions of all eight aggregations/prior additions and
queries; reuse plans in both methods. Report raw per-pass costs, warm medians
and setup plus the first pass. Transfers are separate, and query setup is
reported separately. No new batch-width or repetition selection.

Every pass must preserve existing coupled-field gates: error over fluctuation
<5e-4, empirical variance and covariance relative differences<1e-4. Require exact
tested ray/collision events and view choice against archived outputs. Save the
final fields, coefficients and their hashes; replay for both methods. A timing
ratio requires all repetitions and whole-run sampled isolation to pass. Retain
failure and do not retry to rescue quality/isolation.

This is prepared-weight aggregation/query performance at massive N, not a new
conditioning, prior-generation, full-application or upstream speedup. The eight
fields are insufficient for absolute Monte Carlo/application convergence. State
this distinction even if the numerical controls and aggregation timings improve.
