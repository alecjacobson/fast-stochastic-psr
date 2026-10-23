# Declared nested-query refinement and checked replay

Generate 256 consistently coupled cube200/scalar50 posterior draws for the
existing million-point input, seed51, batch2, with the existing packed inverse
and numerical gates. Save complete, hashed coefficient blocks plus mean, offset,
geometry and model/source provenance. Generation checkpoints are 8,32,128,256;
their previous absolute-probability target remains separate and unchanged.

On those same draws, evaluate nested ray grids with 256/511/1021/2041/4081
positions, and nested sphere-surface sets with 256/512/1024/2048/4096 points.
Preserve all original coordinates exactly. Compare each adjacent pair at their
common coordinates and retain all observed failures. Evaluate ordinary type-2
and cached Fourier fields/events on every level, with the existing 0.05% field
error/fluctuation and 2pp paired-event gates. These are numerical controls,
not a full analytic-kernel validation.

For each adjacent pair, count the independent draws with ANY event disagreement
across its rays/positions or objects. This is one Bernoulli event per draw.
Its probability bounds the absolute discrepancy in any individual event
probability. Report one-sided exact binomial upper bounds with alpha=.05/8
for the four ray and four sphere comparisons (Bonferroni 95% simultaneous
coverage). Target: upper bound <=2pp. These bounds concern differences between
two finite discretizations, not continuous geometry, absolute probability
accuracy or errors shared by both grids. There is one predeclared final look
for this geometry bound. Keep the eight original comparisons fixed if exploring
additional grids later; label additional tests separately.

Report next-view scores at every resolution and fine-grid regret of coarse-grid
choices. Validate archive integrity, complete coverage and replay of source-run
coarse/fine event counts before interpreting refinement. Coefficients and fitted
mean are fixed during replay; do not refit or regenerate to improve agreement.

Run three fresh-process query replays; retain every run. Report setup, archive
I/O/hash validation, cached and direct queries, reductions and full work wall
separately. Timing ratios require all three runs' process audits to pass; otherwise
retain timings as diagnostics without a ratio. Conditioning preparation is paid
once and is not included in a prepared replay-only timer. Other GPU jobs are
never stopped. No complete application or upstream speedup follows unless the
separate model, sampling, geometry and timing requirements are met.
