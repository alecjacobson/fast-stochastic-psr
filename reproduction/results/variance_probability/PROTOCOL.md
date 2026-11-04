# Variance-dependent probability workload

This follow-up was designed after the small-A spatial-cache failure and the
8,192-observation construction pilots. It tests a fixed reconstructed object
queried repeatedly for marginal occupancy and uncertainty. The query geometry
is synthetic; this is not a ContactFusion replay or a joint collision test.

Use released case B with nested seed-92 observation subsets, the original
analytic product-Matérn observation covariance (lengthscale .01, per-axis
variance .1, noise variance .02), and the same finite scalar/cross cutoff 50.
No change to the model or accuracy limits is allowed to obtain a crossover.
Use one allocated 1g.24gb Blackwell MIG device, sequential GPU jobs, and save
source/data hashes, package versions and the MIG process audit.

The main replay has N=16,384, 32 fresh batches of M=65,536 locations (2,097,152
queries total). Half the locations are independent near-surface mesh samples;
half probe the full periodic workspace. Each batch applies a different small
translation. Compute mean, marginal variance, inside probability Phi(mu/sqrt(v)),
and mean ambiguity min(p,1-p). Rank the candidate batches by mean ambiguity.
This measures marginal uncertainty scoring; it does not estimate the probability
of any collision, coherent posterior samples, or physical sensor calibration.

The NUFFT candidate retains full analytic Cholesky conditioning. Collapse the
full correction factor into a doubled-bandwidth variance field. Use the gradient
product identity, paired real source NUFFTs, float32 source/product FFT work
with double accumulation, and batched double inverse-factor solves. Charge
all preparation and actual query/score work. This is a Cholesky–NUFFT hybrid,
not a claim of matrix-free variance conditioning.

Controls: actual float32 Cholesky marginal queries; direct SDD marginal queries;
and a Cholesky-built quintic spatial variance cache given the same prepared
NUFFT mean. Screen spatial sizes against an independent exact-double panel,
then time an actual Cholesky-built grid and validate fresh queries. The prior
cubic full-moment cache and all unsuccessful screens remain in pilot records.
The model-only cross-kernel table may be reused free by every baseline; report
its cold cost separately. Give SDD the prepared Cholesky mean to isolate variance.
Tune SDD batch/step size among the existing four configurations; retain every
pilot and iteration count. Censor its replay after 180 seconds of query work,
checking between native chunks, retaining actual completed counts and their
accuracy. Do not extrapolate the censored total or call it a completed solve.

Existing gates: mean relative L2 <= .002; variance relative L2 <= .0002;
centered variance contrast relative L2 <= .01; probability RMS <= .0005 and
maximum <= .005; every variance positive, no clipping to pass. Compare full
batches against independently validated float32 Cholesky and exact-double
held-out panels. Also report candidate-ranking regret and prior/constant-variance
ablations. The strict variance gates are part of the claim: a probability-only
or looser-accuracy application may choose a cheaper spatial approximation.

Retain cold construction scaling at smaller nested N, plus the large-N actual
replay. Plot measured preparation and query costs on log axes. Do not infer
an asymptotic M-only win over a prepared spatial cache: both have O(M) queries.
Both exact approaches retain cubic conditioning; the favorable construction
terms are O(N^2 + N L log L) versus O(H N^2) for H spatial sites, at fixed
bandwidth/accuracy. Report local slopes as empirical, not infinite-N proofs.

Revision after full-batch reference failure: the first N=16,384 replay was
stopped when the table-interpolated direct mean produced probability maximum
errors above .005 on 65,536 queries, although the smaller exact panels passed.
The repeated experiment gives direct Cholesky, SDD and spatial-variance controls
the same accurate prepared mean. Variance construction, model and gates are
unchanged. Retain the interrupted record as failed-reference-B16384.json.

Accounting: the Fourier candidate is charged spectrum, analytic observation
matrix, double Cholesky, mean fit/aggregation/centering, full variance-series
construction and evaluator preparation, plus measured probability/score work.
Baseline totals grant the analytic matrix and reusable model-only cross table
for free. They include the measured float32 factor, mean fit/aggregation/centering
and prepared mean; the spatial baseline additionally includes actual variance
grid construction. This is deliberately favorable to the baselines. Tuning,
independent validation and benchmark logging are outside pipeline timings and
remain in the raw records. The SDD censor reports query work alone.
