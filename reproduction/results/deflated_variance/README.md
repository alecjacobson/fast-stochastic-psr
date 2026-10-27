# Fixed-query covariance deflation for marginal variance

This prototype resolves a covariance subspace from independent pilot draws,
computes its diagonal contribution with NUFFT covariance actions, and estimates
only the projected residual from evaluation draws. All 20,000 A/B slice queries
and the existing 1% spatial-contrast / probability gates are retained.
It targets the direct analytic-vector/scalar50 model, not the released interpolant.
It does not produce coherent posterior samples for event queries.

[Protocol](PROTOCOL.md) · [Filtering follow-up](FILTERED_PROTOCOL.md) ·
[Issue #39](https://github.com/alecjacobson/fast-stochastic-psr/issues/39).

The unfiltered pilots select rank256 with128 evaluation draws on A and512 on B;
all three independent pilot/evaluation pairs pass at those settings. Pilot
preparation is included. Independent confirmation remains to be completed before
any timing ratio is claimed. Seven targeted tests pass, including explicit
Fourier/dense covariance actions, partial RHS batches, projection identities,
Monte Carlo moments and retained negative estimates.

The first attempt failed JSON serialization after its first seed in each case.
Both exited with code1; partial artifacts/logs are preserved under
pilot-serialization-failed and are not used for results. Casting the normalization
to a Python float fixes the reporting type without changing arithmetic, ranks,
seeds, sample counts or gates. Completed pilot records are under pilot/A and B.
