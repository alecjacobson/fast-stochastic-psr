# Marginal variance from the conditioning correction

On the released 691-point scorpion, **128 evaluation draws reduce variance
error from 12–13% to 0.17–0.18%** using the independent conditioning correction.
All three fixed seeds pass the predeclared marginal-quality gates. Each also
uses **32 separate pilot draws** to freeze combination weights. This is an
estimator-quality pilot, **not a measured speedup or reproduction of the
Figure 1 variance image**. [Protocol](PROTOCOL.md) · [Issue #34](https://github.com/alecjacobson/fast-stochastic-psr/issues/34).

## Why this helps

For the consistently conditioned Gaussian model, write a scalar prior draw as
`g0 = g + c`, where `g` is the centered posterior and `c` the conditioning
correction. They are independent, with variances `V` and `W = V0 − V`.
Thus both `mean(g²)` and `V0 − mean(c²)` estimate posterior variance `V`.
Their estimator variances are `2 V² / S` and `2 W² / S` for `S` independent
draws, with known zero means. A fixed weighted combination remains unbiased.

The oracle weight on the posterior estimate is `W² / (V² + W²)`. We estimate
that weight from the independent pilot; exact reference values never select
weights. At weakly constrained points the small correction is easier to
estimate in absolute terms. At strongly constrained points the posterior
estimate is preferable. The tests cover both regimes. No draw partners or
spatial query points are counted as independent extra samples.

## Released scorpion pilot

The model uses full analytic vector covariance, cutoff-50 scalar/cross spectra,
corrected observation-noise variance .02 and direct Fourier queries. It is not
the released interpolated teaser pipeline. Queries are 128 fixed slice locations
and 128 observed positions. Reference variances use dense float64 Cholesky and
epsilon-1e-12 cross sums; these share spectral code with the sampler. The separate
real-feature dense test independently verifies the decomposition and shows that
an inconsistent inverse breaks it.

| Seed | Posterior-only variance error | Correction-only variance error | Combined variance error | Combined inside-probability RMS / max (pp) |
| --- | ---: | ---: | ---: | ---: |
| 107 | 13.165% | 0.182% | 0.183% | 0.0148 / 0.0683 |
| 211 | 12.216% | 0.167% | 0.165% | 0.0139 / 0.0705 |
| 307 | 12.700% | 0.176% | 0.176% | 0.0151 / 0.0592 |

Variance errors are relative L2 against the deterministic reference. No estimate
is clipped. No negative estimates or empty intervals occur in these runs.
Maximum numerical component error against dense conditioning of the saved RHS
is about 0.000214% relative L2, far below the Monte Carlo errors here.
Empirical posterior/correction cross moments are reported with their sampling
standard errors; their small observed values are not a proof of independence.

## Small spatial variation still needs validation

Posterior variance is **97.06–99.996% of prior variance** on this query panel.
That favors the correction estimator, but total-variance error can conceal
errors in the features shown by a rescaled variance colormap. The following
diagnostics were added after inspecting the pilot, without changing its gates.
For both, the numerator is the combined estimator's absolute L2 error. The
denominator is either the variance reduction `V0 − V` or the centered spatial
variation `V − mean(V)` across these fixed queries.

| Seed | Error / variance reduction | Error / spatial contrast | Reference covered by component intervals | Largest interval width / true variance |
| --- | ---: | ---: | ---: | ---: |
| 107 | 12.47% | 19.40% | 256/256 | 3.49% |
| 211 | 11.30% | 17.58% | 256/256 | 3.69% |
| 307 | 12.05% | 18.74% | 256/256 | 3.37% |

The intervals use two chi-square component intervals per query with Bonferroni
allocation across all 256 queries, targeting 95% simultaneous coverage under
the exact independent Gaussian law. Numerical/model error is not included.
Observed coverage in three seeds does not establish empirical calibration.
As another post-hoc context baseline, simply returning the prior variance has
**1.464%** variance error and **0.1237pp**
inside-probability RMS error. This baseline needs no posterior samples.

The spatial-contrast errors remain substantial. These results therefore do
not establish matched teaser colors, uncertainty-based action rankings, ray
transmittance, collision probabilities or full application quality. A covariance
matrix assembled from such estimates is not automatically positive semidefinite.
The exact-law identity also does not automatically apply to the narrower hybrid
inverse used by some earlier large-scale experiments.

For context, a read-only audit of the existing deterministic matched teaser
arrays gives **0.296% / 0.081%** error relative
to spatial contrast for A/B, respectively. Those use their full 20,000-query
panels and the matched interpolant; they are separate experiments. Their
previously reported range-normalized errors remain available in the
[matched reproduction report](../matched/README.md). The new stochastic pilot
does not replace those images or their stronger deterministic quality evidence.

## Full-slice follow-up

The [subsequent A/B study](../correction_contrast/README.md) uses all 20,000 slice
queries, a cheaper inverse-factor correction sampler and fixed counts through
8,192 draws. It still misses a prospective 1% contrast target on both datasets;
the deterministic baseline passes. This confirms that the small total-variance
error above does not yet establish matching-quality application gains.

## Cost and next steps

Recorded preparation, component-query and reference timings are retained in
[results.json](A/results.json). GPU process isolation fails, and the pilot has
no equally optimized fixed-quality timing baseline. No runtime ratio follows.
The protocol requested all stage timings, but reference-matrix assembly and
per-seed dense component checks were not timed separately. This instrumentation
gap is retained explicitly; the recorded times do not cover the entire work.
Even without contention, additional pilot draws, component queries and setup
must be included. The very large theoretical oracle estimator-variance ratios
in the raw record are **not measured sample-count or application speedups**.
At A, dense deterministic variance is also a strong baseline.

Next: independent fixed-quality sample-count studies at A/B, preservation of
spatial contrast and action rankings, streaming component statistics and
million-point consistently coupled sampling. Keep the strongest deterministic
and stochastic baselines and all preparation-through-query costs.

## Reproduction

```bash
./reproduction/run_env.sh -m reproduction.benchmark_complement_variance \
  --output /opt/geospsr-runtime/complement-variance/A
./reproduction/run_env.sh -m pytest -q tests/test_complement_variance.py
./reproduction/run_env.sh -m reproduction.summarize_complement_variance
```

Copy benchmark artifacts into this report's `A/` directory before summarizing.
The summary independently recomputes frozen weights, estimates and intervals
from the saved component draws. The complete suite passes **131 tests** ([log](full-tests.log)).
