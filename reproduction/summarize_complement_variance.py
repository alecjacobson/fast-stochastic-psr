"""Recheck marginal estimates and expose the small spatial-contrast scale."""
import json
import re
from pathlib import Path
import numpy as np
from scipy.special import ndtr
from reproduction.complement_variance import estimate, pilot_weights


OUT = Path(__file__).resolve().parent/'results/complement_variance'


def main():
    raw = json.loads((OUT/'A/results.json').read_text())
    assert raw['complete'] and raw['seeds'] == [107,211,307]
    ref = np.load(OUT/'A/reference.npz')
    variance, mean, v0 = ref['variance'], ref['mean'], float(ref['prior_variance'])
    reduction = ref['correction_variance']
    np.testing.assert_allclose(variance+reduction, v0, rtol=1e-14)
    rows, diagnostics, results = [], [], []
    for run in raw['runs']:
        seed = run['seed']; data = np.load(OUT/f'A/seed_{seed}.npz')
        g,c = data['posterior_draws'],data['correction_draws']
        assert g.shape == c.shape == (160,256)
        beta = pilot_weights(g[:32],c[:32])
        estimates = estimate(g[32:],c[32:],v0,beta)
        for key,value in estimates.items():
            np.testing.assert_array_equal(value,data[key])
        metrics = {}
        for method in ['posterior','correction','combined']:
            value = estimates[method]
            error = value-variance
            metric = {'relative_variance_l2':float(np.linalg.norm(error)/np.linalg.norm(variance)),
                      'variance_reduction_relative_l2':float(np.linalg.norm(error)/np.linalg.norm(reduction)),
                      'spatial_contrast_relative_l2':float(np.linalg.norm(error)/np.linalg.norm(variance-variance.mean()))}
            np.testing.assert_allclose(metric['relative_variance_l2'],run['metrics'][method]['relative_variance_l2'],rtol=1e-13)
            metrics[method] = metric
        r=run['metrics']; m=metrics['combined']
        rows.append(f'| {seed} | {100*r["posterior"]["relative_variance_l2"]:.3f}% | '
                    f'{100*r["correction"]["relative_variance_l2"]:.3f}% | '
                    f'{100*r["combined"]["relative_variance_l2"]:.3f}% | '
                    f'{100*r["combined"]["probability_rms"]:.4f} / {100*r["combined"]["probability_max"]:.4f} |')
        diagnostics.append(f'| {seed} | {100*m["variance_reduction_relative_l2"]:.2f}% | '
                           f'{100*m["spatial_contrast_relative_l2"]:.2f}% | '
                           f'{run["intervals"]["queries_covered"]}/256 | '
                           f'{100*run["intervals"]["maximum_width_over_true_variance"]:.2f}% |')
        results.append({'seed':seed,'metrics':metrics,'declared_gates_pass':run['pass']})
    prior_probability = ndtr(-mean/np.sqrt(v0))
    exact_probability = ndtr(-mean/np.sqrt(variance))
    prior_error = float(np.linalg.norm(reduction)/np.linalg.norm(variance))
    prior_probability_rms = float(np.sqrt(np.mean((prior_probability-exact_probability)**2)))
    matched = {}
    for case in ['A','B']:
        folder = OUT.parent/'matched'/case
        original = np.load(folder/'original.npz')['variance'].astype('d')
        candidate = np.load(folder/'compatible.npz')['variance'].astype('d')
        matched[case] = float(np.linalg.norm(candidate-original)/np.linalg.norm(original-original.mean()))
    full = OUT/'full-tests.log'
    tests = re.findall(r'(\d+) passed',full.read_text()) if full.exists() else []
    test_text = (f'The complete suite passes **{tests[-1]} tests** ([log](full-tests.log)).' if tests else
                 'The three new independent estimator tests pass ([log](tests.log)); the preceding GPU suite passed 128 tests.')
    result={'scope':raw['scope'],'declared_gates_pass':raw['all_quality_gates_pass'],
            'runs':results,'prior_only_relative_variance_l2':prior_error,
            'prior_only_probability_rms':prior_probability_rms,
            'published_matched_teaser_contrast_errors':matched,
            'posthoc_diagnostics':['variance reduction norm','centered spatial contrast norm','prior-only baseline'],
            'protocol_deviation':'reference matrix assembly and dense per-seed component checks were not timed separately',
            'speedup_claim':False}
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    (OUT/'README.md').write_text(f'''# Marginal variance from the conditioning correction

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
{chr(10).join(rows)}

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
{chr(10).join(diagnostics)}

The intervals use two chi-square component intervals per query with Bonferroni
allocation across all 256 queries, targeting 95% simultaneous coverage under
the exact independent Gaussian law. Numerical/model error is not included.
Observed coverage in three seeds does not establish empirical calibration.
As another post-hoc context baseline, simply returning the prior variance has
**{100*prior_error:.3f}%** variance error and **{100*prior_probability_rms:.4f}pp**
inside-probability RMS error. This baseline needs no posterior samples.

The spatial-contrast errors remain substantial. These results therefore do
not establish matched teaser colors, uncertainty-based action rankings, ray
transmittance, collision probabilities or full application quality. A covariance
matrix assembled from such estimates is not automatically positive semidefinite.
The exact-law identity also does not automatically apply to the narrower hybrid
inverse used by some earlier large-scale experiments.

For context, a read-only audit of the existing deterministic matched teaser
arrays gives **{100*matched['A']:.3f}% / {100*matched['B']:.3f}%** error relative
to spatial contrast for A/B, respectively. Those use their full 20,000-query
panels and the matched interpolant; they are separate experiments. Their
previously reported range-normalized errors remain available in the
[matched reproduction report](../matched/README.md). The new stochastic pilot
does not replace those images or their stronger deterministic quality evidence.

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
./reproduction/run_env.sh -m reproduction.benchmark_complement_variance \\
  --output /opt/geospsr-runtime/complement-variance/A
./reproduction/run_env.sh -m pytest -q tests/test_complement_variance.py
./reproduction/run_env.sh -m reproduction.summarize_complement_variance
```

Copy benchmark artifacts into this report's `A/` directory before summarizing.
The summary independently recomputes frozen weights, estimates and intervals
from the saved component draws. {test_text}
''')
    print(json.dumps(result))


if __name__=='__main__':main()
