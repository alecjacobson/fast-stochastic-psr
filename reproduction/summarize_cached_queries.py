"""CPU-only report of cached Fourier query measurements and quality checks."""
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent / 'results/cached_queries'


def read(name):
    return json.loads((ROOT / name / 'results.json').read_text())


def main():
    runs = {name: read(name) for name in ['A-moving', 'B-moving', 'B-seed43']}
    lines = ['# Cached posterior queries', '',
        'This experiment accelerates **repeated queries of prepared posterior samples**.',
        'It retains every Fourier mode and uses the public cuFINUFFT interpolation-only',
        'interface after caching the oversampled spatial grids. It does not yet accelerate',
        'full-model fitting/sampling to a million observations. See the',
        '[cuFINUFFT interface documentation](https://finufft.readthedocs.io/en/v2.5.1/c_gpu.html).', '',
        '## Matched released-kernel application queries', '',
        'A has 691 observations; B has 17,419. Both use 128 coherent posterior draws,',
        '24 rays × 256 locations and 17 spheres × 256 surface locations. The spheres',
        'are placed around a mean-surface crossing using deterministic calibration,',
        'without inspecting realized sample events. Every backend gets identical probes.',
        'Query positions translate between the three timed repeats; no query answers',
        'are cached. All timings synchronize the GPU. Probability errors below compare',
        'the final common translated geometry; independent field/covariance checks use',
        '257 additional locations. Seeds 42 and 43 are both retained.', '',
        '| Run | Strong reference rays | Cached compatible rays | Ratio | Strong reference collisions | Cached compatible collisions | Ratio |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, r in runs.items():
        ref, fast = r['backends']['reference_cached'], r['backends']['compatible_cached']
        values = []
        for kind in ['rays', 'collision']:
            a, b = ref[kind]['median_seconds'], fast[kind]['median_seconds']
            values += [f'{a*1000:.2f} ms', f'{b*1000:.2f} ms', f'{a/b:.2f}×']
        lines.append('| ' + name + ' | ' + ' | '.join(values) + ' |')
    lines += ['', 'The reference shares each released interpolated cross-kernel block across all',
        'samples, reuses the analytic factorization and **also caches its prior grids**.',
        'The accelerated compatibility path uses the existing 201³-mode representation',
        'of the released kernel. Both receive the same corrected pathwise draws and',
        'mean offset. This is stronger cached algebra, not untouched upstream stochastic',
        'API timing; the sampler includes observation noise and the omitted vector',
        'spectrum. Query timings include sample evaluation, joint events and view scores.', '',
        '| Run | Ray RMS / max error (percentage points) | Collision max error (points) | Uncertain reference spheres | View-score regret | Paired gate |',
        '| --- | ---: | ---: | ---: | ---: | --- |']
    for name, r in runs.items():
        q = r['quality']['compatible_cached']
        lines.append(f"| {name} | {100*q['rays']['rms_probability_difference']:.3f} / {100*q['rays']['max_probability_difference']:.3f} | {100*q['collision']['max_probability_difference']:.3f} | {r['backends']['reference_cached']['collision']['uncertain_query_count']} / 17 | {q['next_view']['reference_score_regret']:.5f} | {'pass' if q['compatibility_pass'] else 'FAIL'} |")
    lines += ['', 'Gates: ray RMS ≤1 percentage point and maximum ≤2, collision maximum ≤2,',
        'next-view regret ≤two ray steps. These compare paired numerical evaluations.',
        'With 128 draws the worst-case marginal Monte Carlo standard error is 4.42',
        'percentage points. Absolute probability convergence, sphere/ray discretization,',
        'multi-step scanning and original unpublished application scenes remain open.', '',
        '## Does caching change the random functions?', '',
        'Independent tests cover random spectra, periodic translations, odd/even FFT',
        'grids, production-bandwidth corner modes, changed query positions and partial',
        'sample batches against direct Fourier sums and ordinary NUFFT. Two real-valued',
        'functions are packed into one complex grid using their Hermitian spectra;',
        'this halves storage without discarding frequencies. The 1.6× padded grid',
        'limits float32 cancellation observed with the smaller 1.25× grid.', '',
        '| Run | Cache field error / fluctuation norm | Empirical variance relative error | Empirical covariance relative error | Cache vs ordinary event changes |',
        '| --- | ---: | ---: | ---: | ---: |']
    for name, r in runs.items():
        f = r['field_checks']['compatible']
        q = r['quality_vs_full_compatible']['compatible_cached']
        fraction = max(q[k]['event_disagreement_fraction'] for k in ['rays', 'collision'])
        lines.append(f"| {name} | {f['error_over_fluctuation_l2']:.3g} | {f['empirical_variance_relative_l2']:.3g} | {f['empirical_covariance_relative_l2']:.3g} | {fraction:.3g} |")
    lines += ['', 'The covariance check uses the first 32 of 257 independent random queries.',
        'These coupled finite-sample diagnostics isolate caching error, not population',
        'covariance or model/truncation error. The numerical gates were fixed before',
        'these runs: field error / fluctuation norm <5e-4, variance and covariance',
        'relative errors <1e-4. Invalid checks stop the benchmark.', '',
        '## Preparation, memory and amortization', '',
        '| Run | Shared fit/mean + corrected draw preparation | Additional compatible spectrum/aggregation/grid setup | Reference prior-grid setup | Approx. ray batches to amortize extra setup |',
        '| --- | ---: | ---: | ---: | ---: |']
    for name, r in runs.items():
        s = r['seconds']
        common = s['common_fit_and_direct_mean_setup'] + s['common_corrected_draws_and_direct_aggregation']
        extra = sum(s[k] for k in ['compatible_kernel_setup', 'compatible_sample_aggregation', 'compatible_grid_setup'])
        ref_extra = s['reference_prior_grid_setup']
        saving = r['backends']['reference_cached']['rays']['median_seconds'] - r['backends']['compatible_cached']['rays']['median_seconds']
        count = str(math.ceil(max(0, extra - ref_extra) / saving)) if saving > 0 else 'does not amortize'
        lines.append(f'| {name} | {common:.2f} s | {extra:.2f} s | {ref_extra:.3f} s | {count} |')
    b = runs['B-moving']
    lines += ['', 'This break-even estimate assumes the reference kernel table already exists',
        'and the measured shared conditioning/draws are available. It does not count',
        'the original slow table setup as an advantage, and is not an independently',
        'optimized cold-start application benchmark. Shared preparation still includes',
        'direct sample aggregation for all paths. Calibration, independent validation',
        'and ordinary baseline runs are diagnostic overhead outside query timings.', '',
        f"For 128 draws, the compatible cache payload is **{b['cache_bytes']['compatible']/2**30:.2f} GiB**;",
        f"the direct and reference-prior cache payloads are **{b['cache_bytes']['direct']/2**30:.2f} GiB** each.",
        'These are payload sizes, not total GPU peaks; coefficients, factorization,',
        'runtime allocators and workspaces require additional memory. Caches are tested',
        'sequentially. An initial unpacked compatible-cache allocation failed; the raw',
        '`A-unpacked-memory-failure` record is retained. Packing resolved that failure.', '']
    memory = ROOT / 'memory-samples.csv'
    if memory.exists():
        values = []
        for line in memory.read_text().splitlines():
            try: values.append(float(line.split(',')[-1]))
            except ValueError: pass
        if values:
            lines += [f'The external 100 ms `nvidia-smi` monitor observed a maximum of **{max(values)/1024:.2f} GiB**',
                      'across the monitored validation queue. This includes allocator/context memory;',
                      'shorter transients can be missed and monitoring began during the first run.', '']
    fixed = {n: read(f'B-grid{n}-fixed') for n in [100, 200, 400]}
    assert len({json.dumps(r['input_sha256'], sort_keys=True) for r in fixed.values()}) == 1, 'refinement inputs differ'
    lines += ['## Removing interpolation: a convergence bridge', '',
        'The released 50³ kernel interpolation changes local probabilities appreciably.',
        'To distinguish that bias from direct-NUFFT error, refine the same nonuniform',
        'trilinear table while keeping analytic conditioning, draws, mean offset and',
        'query hashes fixed. Table nodes use the independently validated finite-Fourier',
        'kernel. This is a refined reference algorithm, not the untouched released setting.', '',
        '| Reference grid | Direct ray RMS / max difference (points) | Direct collision max difference (points) | Direct paired gate | Strong reference / cached direct ray time |',
        '| --- | ---: | ---: | --- | ---: |']
    for n, r in fixed.items():
        q = r['quality']['direct_cached']
        a = r['backends']['reference_cached']['rays']['median_seconds']
        btime = r['backends']['direct_cached']['rays']['median_seconds']
        lines.append(f"| {n}³ | {100*q['rays']['rms_probability_difference']:.3f} / {100*q['rays']['max_probability_difference']:.3f} | {100*q['collision']['max_probability_difference']:.3f} | {'pass' if q['compatibility_pass'] else 'FAIL'} | {a*1000:.2f} / {btime*1000:.2f} ms |")
    lines += ['', 'This supports using direct Fourier evaluation for the intended finite-scalar,',
        'analytic-vector posterior rather than permanently reproducing a coarse table.',
        'It does not validate a truncated *training* covariance at massive N.', '',
        '![Uncertain collisions and interpolation convergence](comparison.png)', '',
        '## Reusable API and hitbox check', '',
        '`Posterior.prepare_query_cache()` enables cached mean/sample evaluation for',
        'arbitrary later query coordinates. Redrawing invalidates the sample cache.',
        'The application demo accepts `--cache-queries` and a separate `--output` path.',
        'The [complete demo output check](application_cache_check.json) found identical',
        'ray probabilities, next-view scores and collision probabilities; volume mean',
        'and standard deviation differ by less than 1e-6 relative L2. All three hitbox',
        'meshes retain identical faces, with maximum vertex displacement below 0.00011',
        'in normalized scene coordinates. Existing domain clipping remains unchanged.',
        'Single-call sample-volume evaluation fell from 128.0 to 11.9 ms; this is a',
        'same-method implementation comparison, not a validated upstream hitbox speedup.',
        'Preparation varied by about a second between those runs and dominates the',
        'overall demo, so the isolated query improvement is not a cold-start claim.', '',
        '## Reproduce', '', '```bash',
        './reproduction/run_env.sh -m pytest -q tests',
        './reproduction/run_env.sh reproduction/paired_applications.py --case A --cached-grids --uncertain-collisions --query-jitter .002 --output reproduction/results/cached_queries/A-moving',
        './reproduction/run_env.sh reproduction/paired_applications.py --case B --cached-grids --uncertain-collisions --query-jitter .002 --output reproduction/results/cached_queries/B-moving',
        './reproduction/run_env.sh reproduction/paired_applications.py --case B --cached-grids --uncertain-collisions --query-jitter .002 --seed 43 --output reproduction/results/cached_queries/B-seed43',
        '# Repeat with reference grids 100, 200, 400 and corresponding output paths:',
        './reproduction/run_env.sh reproduction/paired_applications.py --case B --cached-grids --uncertain-collisions --backends direct --reference-grid 200 --mean-offset direct --query-jitter .002 --output reproduction/results/cached_queries/B-grid200-fixed',
        './reproduction/run_env.sh reproduction/summarize_cached_queries.py', '```', '',
        'For the full demo output check, run `applications.py` with and without',
        '`--cache-queries`, using the respective `--output` directories',
        '`reproduction/results/cached_queries/applications-cached` and',
        '`reproduction/results/cached_queries/applications-ordinary`, then run',
        '`reproduction/compare_application_cache.py`. Demo meshes/large arrays are',
        'regenerated rather than duplicated in Git; the numeric comparison is retained.', '',
        'GPU runs must be sequential. The report generator uses saved results only.',
        'Implementation and remaining scope: [issue #12](https://github.com/alecjacobson/fast-stochastic-psr/issues/12).', '']
    (ROOT / 'README.md').write_text('\n'.join(lines))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    events = np.load(ROOT / 'B-moving/events.npz')
    for name, label in [('reference_cached', 'released kernel'), ('compatible_cached', 'cached matched'), ('direct_cached', 'direct Fourier')]:
        axes[0].plot(events[name + '_collision'].mean(0), marker='.', label=label)
    axes[0].set(xlabel='Sphere near mean-surface crossing', ylabel='Collision probability', title='B: uncertain collision queries')
    axes[0].legend()
    for n in [100, 200, 400]:
        e = np.load(ROOT / f'B-grid{n}-fixed/events.npz')
        axes[1].plot(e['reference_cached_collision'].mean(0), marker='.', label=f'reference {n}³')
    axes[1].plot(e['direct_cached_collision'].mean(0), linestyle='--', color='black', label='direct Fourier')
    axes[1].set(xlabel='Sphere (fixed geometry and draws)', ylabel='Collision probability', title='Refining the kernel interpolation')
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(ROOT / 'comparison.png', dpi=160)


if __name__ == '__main__':
    main()
