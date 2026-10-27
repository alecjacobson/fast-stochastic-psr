"""Audit and plot all frozen-fit contour refinements without solver speedups."""
import argparse
import json
from pathlib import Path
import numpy as np
from reproduction.contour_distance import curve_distance, curve_statistics


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', type=Path, default=Path(__file__).resolve().parent / 'results/large_contours')
    a = p.parse_args(); out = a.results
    raw = json.loads((out / 'results.json').read_text())
    if not raw['complete']:
        raise ValueError('incomplete grid evaluation')
    files = np.load(out / 'contours.npz')
    def get(name, size, h):
        key = f'{name}_{size}_{h}'
        count = raw['methods'][name]['grids'][f'{size}_{h}']['components']
        curves = [files[f'{key}_{i}'] for i in range(count)]
        stats = curve_statistics(curves)
        for k, v in stats.items():
            np.testing.assert_allclose(v, raw['methods'][name]['grids'][f'{size}_{h}'][k], rtol=1e-12)
        return curves
    width = raw['domain'][1] - raw['domain'][0]
    result = {'scope': 'distances between extracted two-dimensional mean-zero polylines',
              'domain_width': width, 'sample_spacing': 1e-4,
              'refinement_upper_gate': 1e-4 * width, 'refinement': {}, 'comparison': {}}
    refine_rows = []; compare_rows = []
    def bracket(d):
        return f'{d["hausdorff_lower"]:.6g}–{d["hausdorff_upper"]:.6g}'
    for name in raw['methods']:
        result['refinement'][name] = {}
        for h, height in enumerate(raw['heights']):
            history = []
            for low, high in zip(raw['grids'], raw['grids'][1:]):
                previous = get(name, low, h); current = get(name, high, h)
                d = curve_distance(previous, current)
                d['grids'] = [low, high]
                d['stable_component_counts'] = all(
                    curve_statistics(previous)[k] == curve_statistics(current)[k]
                    for k in ['components', 'open_components'])
                d['refinement_pass'] = d['stable_component_counts'] and d['hausdorff_upper'] <= result['refinement_upper_gate']
                history.append(d)
            result['refinement'][name][str(h)] = history
            final = history[-1]
            refine_rows.append(f'| {name} | {height:.2f} | {bracket(final)} | '
                               f'{len(current)} | {"pass" if final["refinement_pass"] else "FAIL"} |')
    for name in ['half200', 'sdd']:
        result['comparison'][name] = {}
        for h, height in enumerate(raw['heights']):
            history = []
            for size in raw['grids']:
                d = curve_distance(get(name, size, h), get('half250', size, h))
                d['grid'] = size
                history.append(d)
            result['comparison'][name][str(h)] = history
            final = history[-1]
            compare_rows.append(f'| {name} | {height:.2f} | {bracket(final)} | '
                                f'{100*final["hausdorff_lower"]/width:.5f}–{100*final["hausdorff_upper"]/width:.5f}% |')
    result['all_final_refinements_pass'] = all(v[-1]['refinement_pass']
        for method in result['refinement'].values() for v in method.values())
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), layout='constrained')
    styles = [('half250', 'black', '-', 'NUFFT cutoff 250'),
              ('half200', '#008daf', '--', 'NUFFT cutoff 200'),
              ('sdd', '#d14772', '-.', 'SDD, 600 s update budget')]
    for h, height in enumerate(raw['heights']):
        location = np.array(result['comparison']['sdd'][str(h)][-1]['sampled_max_location'])
        radius = .015 * width
        all_points = np.concatenate(get('half250', raw['grids'][-1], h))
        for column, ax in enumerate(axes[h]):
            for name, color, ls, label in styles:
                for i, curve in enumerate(get(name, raw['grids'][-1], h)):
                    ax.plot(*curve.T, color=color, ls=ls, lw=1.1, label=label if i == 0 else None)
            ax.set_aspect('equal'); ax.set(xlabel='x', ylabel='z')
            ax.grid(alpha=.15)
            if column == 0:
                ax.set(xlim=(all_points[:,0].min()-.04, all_points[:,0].max()+.04),
                       ylim=(all_points[:,1].min()-.04, all_points[:,1].max()+.04),
                       title=f'Mean-zero contours, y={height:.2f}')
                ax.add_patch(Rectangle(location-radius, 2*radius, 2*radius, fill=False, ec='.5', lw=.8))
            else:
                ax.set(xlim=(location[0]-radius, location[0]+radius),
                       ylim=(location[1]-radius, location[1]+radius),
                       title='Zoom at largest sampled SDD discrepancy')
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=3, fontsize=9)
    fig.savefig(out / 'contours.png', dpi=180); fig.savefig(out / 'contours.pdf')
    plt.close(fig)
    (out / 'README.md').write_text(f'''# Three-million-point mean-zero slice contours

The frozen [mean fits](../large_fit/README.md) are now checked on progressively
finer slice grids. This separates small global field error from displacement of
the displayed zero boundary. It does not introduce a new application pass/fail
threshold or a solver timing ratio.

[Protocol](PROTOCOL.md) · [Raw measurements](results.json) ·
[All refinement and comparison distances](summary.json) ·
[All extracted contour vertices](contours.npz).

## Final-grid comparison

All three million observations, saved weights and original centering constants
are reused; the weights and source hashes are verified. The old 20,000-query
panel is replayed before evaluating the new grids. Scalar/cross cutoff stays 50.
Training cutoff 250 is an internal reference, not a full analytic solution.

| Method versus cutoff 250 | Slice y | Hausdorff bracket (model units) | Bracket / domain width |
| --- | ---: | ---: | ---: |
{chr(10).join(compare_rows)}

![Mean-zero contours and magnified differences](contours.png)

Every connected curve is retained. These are actual extracted 2049-square
contours; the right panels magnify the neighborhood of the largest sampled SDD
discrepancy on each plane. The full-plane panels crop to the contour extents;
the evaluation domain remains [-pi/2, pi/2] squared. No displacement is amplified.
Colors/line patterns distinguish the models and do not encode uncertainty.

## Numerical refinement

Each method is evaluated directly on nested 257, 513, 1025 and 2049 square grids
at y=-.35 and -.75. The final 1025-to-2049 comparison is:

| Method | Slice y | Hausdorff bracket (model units) | Final components | Refinement check |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(refine_rows)}

The predeclared refinement check requires unchanged component/open-component
counts and upper distance <={result['refinement_upper_gate']:.7f}, or 0.01% of
the domain width. All final checks pass: **{result['all_final_refinements_pass']}**.
All earlier grids are retained in the summary, including any failed checks.
This is an empirical refinement check, not a proof of continuous zero-set or
topology convergence. Marching squares can miss components between samples.

Distances refer to the **extracted polylines**. Each segment is sampled at gaps
<=1e-4; exact nearest-neighbor distances between sample sets are converted to
lower and upper Hausdorff bounds using both sample-cover radii. The brackets
exclude floating-point roundoff, field-evaluation error and error between a
polyline and the continuous zero set. They are not statistical confidence
intervals. Unit tests cover known distances, alternative tessellations, missing
components and open endpoints.

## Interpretation and scope

A passing global mean norm alone cannot supply a task-specific tolerance on
boundary displacement. Conversely, the training residual does not tell us how
much displacement a downstream application can tolerate. These measurements
provide model-coordinate distances for selecting a prospective, meaningful
geometry criterion in [issue #36](https://github.com/alecjacobson/fast-stochastic-psr/issues/36).
The historical residual and mean gates are unchanged.

These are the same two display planes, with new grid discretizations. They are
not held-out slice geometry, a full 3D surface, posterior variance, a calibrated
probability or a correlated event. SDD's single saved budget endpoint does not
establish its earliest time to a given contour tolerance. No baseline is shown
intractable. Query timings and sampled GPU occupancy are retained for audit;
this saved-fit check does not include solver fitting and makes no speedup claim.

## Reproduce

First reproduce the three frozen fits in `../large_fit`. Then run:

```bash
./reproduction/run_env.sh -m reproduction.benchmark_large_contours \\
  --source /opt/geospsr-runtime/large-cloud-3000000 \\
  --fits /opt/geospsr-runtime/large-fit \\
  --output /opt/geospsr-runtime/large-contours
./reproduction/run_env.sh -m reproduction.summarize_large_contours \\
  --results /opt/geospsr-runtime/large-contours
./reproduction/run_env.sh -m pytest -q tests/test_contour_distance.py
```

The mean grids stay outside Git; their hashes are in the raw record. Compact
contours and the report are published. The summarizer can reproduce distances
and figures using only the committed JSON and contour archive.
''')
    print(json.dumps({'all_final_refinements_pass': result['all_final_refinements_pass'],
                      'final_comparisons': {n: {h: v[-1] for h, v in d.items()}
                                            for n, d in result['comparison'].items()}}, indent=2))


if __name__ == '__main__':
    main()
