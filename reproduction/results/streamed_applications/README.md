# Streaming coherent application statistics

At one million observations, **128 coherent draws now flow through cached
field, ray, collision and next-view queries without retaining the sample
coefficient ensemble**. The largest yielded coefficient block is
15.72 MiB (two draws), independent
of the total requested sample count. This removes the coefficient-list plus
concatenation limit from this streaming path; the original `draw()` still
retains its ensemble. This is an execution and convergence diagnostic, not a
new matching-quality application speedup.

## Numerical checks

The [eight-draw control](million-control8-corrected/results.json) compares
retained and streamed samples on the same fitted model and seed/batch schedule.
Its covariance discrepancy is 0.000484%;
sampled ray/collision events and the selected view agree. Fields and covariance
pass the unchanged strict same-model gates. The initial pre-score-correction
pilot is retained in [legacy-control8](legacy-control8/results.json).

All 128 streamed draws pass independently summed finite-covariance rows:
maximum relative residual **0.001059%**
against 0.002%. Cached fields differ from ordinary type-2 evaluation by at most
**0.001153%** of the fluctuation norm,
against 0.05%. These 32 validation rows are a numerical sample, not a full-system
or analytic-model certificate. The fit and prior use a consistently coupled
cube200 vector model and cutoff50 scalar/cross model; no narrower hybrid inverse
is used. Full analytic-tail and scalar-band convergence remain separate work.

## Probability and geometry convergence remain open

| Draws | Simultaneous MC radius | Ray refinement max change | Sphere refinement max change | Selected view |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 96.50 pp | 0.000 pp | 0.000 pp | 4 |
| 32 | 48.25 pp | 3.125 pp | 0.000 pp | 14 |
| 128 | 24.12 pp | 3.125 pp | 0.000 pp | 14 |

The [declared protocol](PROTOCOL.md) applies a 95% Hoeffding union bound across
all reported event locations and all three checkpoints. It requires independent
draws but does not assume independent spatial queries. The target is a radius
of at most **two percentage points**; the 128-draw run **does not pass**.
This conservative bound is not an estimate of actual error. The next-view
score bounds likewise do not certify a stable optimal action. Uncertainty
coverage includes 15 nontrivial sphere-probe probabilities.

Refinement inserts ray midpoints (256 → 511 positions) and adds sphere-surface
points (256 → 512), preserving all old coordinates exactly. The table reports
paired empirical changes, not a bound on continuous geometry error. These are
sphere-surface collision proxies, not full solid collision tests. The
[prefix diagnostics](summary.json) also compare 8 and 32 draws with the final
128-draw estimate; the latter includes those prefixes and is not independent
ground truth. No complete matching-quality claim follows from this run.

Next-view scores now integrate a floating-point uncertainty-band indicator.
The inherited Boolean trapezoid undercounted consecutive true intervals. The
[historical audit](../view_score_audit/README.md) preserves source records and
checks the consequences; new diagnostics use gate version 3. Sequential scan
trajectories without saved transmittance still require rerunning.

## Execution and memory accounting

The single instrumented run takes 560.24 s,
including 6.17 s fitting and 553.84 s in the
streaming loop. The loop includes independent validation, refined geometry,
transfers, online statistics and checkpoint writes. Sample-generation stage sums
are prior 4.52 s, conditional solve 543.79 s,
and scalar aggregation 3.57 s. Other stages and all
batch records are in [results.json](million128/results.json).
The whole-run process audit **did not pass** sampled isolation. This is one run,
not repeated optimized application timing. Do not compare this wall time directly
with the earlier fixed-RHS centered150 conditioning-only measurements.

CuPy live allocations observed at yields range from 0.90 to
0.90 GiB; the largest sampled own-process usage is
21.26 GiB. These are different measurements: CuPy live bytes
omit native-library/context allocations, while process observations include
reserved pools but can miss transient peaks. Neither is an exact device peak.
The tests check release of previous coefficient arrays and bounded live storage
over 128 small-problem draws; large-input records show every yielded batch.

## API and reproduction

```python
from contextlib import closing
from reproduction.half_posterior import HalfCubePosterior
from geospsr.cached_spectral import CachedFourierEvaluator

model = HalfCubePosterior(points, normals, training_cutoff=200)
with closing(model.draw_batches(128, seed=51, batch=2, pack_rhs=True)) as batches:
    for block in batches:
        cache = CachedFourierEvaluator(block['coefficients'])
        fields = cache.evaluate(queries) - model.offset
        # Reduce these coherent draws into event counts or moments here.
        del fields, cache, block
```

Keep batching and seed fixed when comparing realizations. Use `closing` for
early exits or consumer exceptions. Only one stream may be active per model;
streaming clears the previously retained sample ensemble/cache, and mean
queries remain available. Tests cover partial batches, packed and folded solves,
optional centered inverses, cleanup and online moments/counts against retained
draws and independent dense controls. [Test output](tests.log).

The benchmark can optionally export coefficient blocks to CPU/disk using
`--coefficient-dir`; this enables later loading for moving queries, but is not
a checkpoint/resume protocol. Do not treat an incomplete run as a full ensemble.

```bash
./reproduction/run_env.sh -m reproduction.stream_applications --samples 8 --checkpoints 8 --retained-control --output reproduction/results/streamed_applications/million-control8-corrected
./reproduction/run_env.sh -m reproduction.stream_applications --output reproduction/results/streamed_applications/million128
./reproduction/run_env.sh -m pytest tests -q
./reproduction/run_env.sh -m reproduction.summarize_streamed_applications
```

The source/geometry scratch dataset is the existing common-RHS million-point
dataset; fresh coherent samples are generated rather than replaying stored RHS.
[Issue #20](https://github.com/alecjacobson/fast-stochastic-psr/issues/20) tracks
memory/scaling, and [#3](https://github.com/alecjacobson/fast-stochastic-psr/issues/3)
tracks complete application accuracy and timing.
