"""Generate the paired application report from saved measurements (CPU only)."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent / 'results/paired_applications'


def main():
    lines = ['# Paired application evaluation', '',
        'These runs compare **prepared coherent posterior functions**, with the same',
        'random draws and analytic training covariance. They do not yet establish',
        'complete-application or massive-scale speedups. Work: [#3](https://github.com/alecjacobson/fast-stochastic-psr/issues/3),',
        '[#9](https://github.com/alecjacobson/fast-stochastic-psr/issues/9),',
        '[#10](https://github.com/alecjacobson/fast-stochastic-psr/issues/10),',
        '[#11](https://github.com/alecjacobson/fast-stochastic-psr/issues/11).', '',
        '## Controlled workload', '',
        '- A: 691 observations. B: 17,419. Same inputs as the matched scorpion figures.',
        '- 128 coherent posterior draws, seed 42, full analytic vector covariance,',
        '  finite 101³-mode scalar prior, observation noise variance 0.02.',
        '- 24 candidate rays × 256 points; 17 collision spheres × 256 surface points.',
        '- Every backend receives identical corrected pathwise weights and prior draws.',
        '  Omitted vector spectrum and observation noise are sampled. This is **not**',
        '  the unchanged upstream stochastic API, which uses different sampling choices.',
        '- Reference: released 50³ interpolated cross kernel, each block shared across',
        '  all 128 samples; the common prior already uses NUFFT for a stronger baseline.',
        '- Direct: underlying finite Fourier kernel. Compatible: 201³-mode representation',
        '  of the released interpolant. All use the same reference mean offset.',
        '- Queries include field evaluation, joint events, probabilities and view scores.',
        '  Warm times are medians of three synchronized GPU runs, including per-query',
        '  plan construction. Sample preparation, aggregation and kernel setup are separate.',
        '- Geometric scenarios are reconstructed on the released scorpion, not the',
        '  unpublished paper scenes. One seed and fixed query discretization limit validation.', '',
        '## Prepared-function query times after plan reuse', '',
        '| Case | Backend | Rays + next-view scores | Collisions | Ray ratio vs reference |',
        '| --- | --- | ---: | ---: | ---: |']
    for case in ['A', 'B']:
        r = json.loads((ROOT / (case + '-plans') / 'results.json').read_text())
        for backend, d in r['backends'].items():
            lines.append(f"| {case} | {backend} | {d['rays']['median_seconds']*1000:.2f} ms | {d['collision']['median_seconds']*1000:.2f} ms | {d['rays']['reference_over_backend_query_ratio']:.2f}× |")
    lines += ['', 'Ratios above are timings, **not accepted application speedups**. In B, direct',
        'Fourier fails the local probability gate, and collision probabilities all',
        'saturate at 0 or 1. The compatible ray result passes paired numerical gates',
        'but is close to the reference in speed. Model/Monte Carlo convergence and',
        'uncertain collision trajectories remain open.', '',
        '## Paired numerical quality', '',
        '| Case | Backend | Ray RMS / max error (percentage points) | Collision max (points) | View-score regret | Gate v2 |',
        '| --- | --- | ---: | ---: | ---: | --- |']
    for case in ['A', 'B']:
        r = json.loads((ROOT / (case + '-plans') / 'results.json').read_text())
        for backend, q in r['quality'].items():
            lines.append(f"| {case} | {backend} | {100*q['rays']['rms_probability_difference']:.3f} / {100*q['rays']['max_probability_difference']:.3f} | {100*q['collision']['max_probability_difference']:.3f} | {q['next_view']['reference_score_regret']:.4f} | {'pass' if q['compatibility_pass'] else 'FAIL'} |")
    lines += ['', 'Gate v2 requires ray RMS ≤1 percentage point, maximum ≤2 points, collision',
        'maximum ≤2 points and next-view regret ≤two ray steps. Initial `A/` and `B/`',
        'results used an RMS-only ray gate (v1); its passing flags are insufficient.',
        'The local maximum gate was added after the B diagnostic exposed the problem.',
        'A paired gate pass is not a posterior-distribution or sample-convergence proof.', '',
        '![Ray transmittance and collision diagnostic](diagnostic.png)', '',
        '## Plan reuse optimization', '',
        'Previously each batch of four functions rebuilt the same query plan. A single',
        'plan now handles all full batches in a query call, with a second plan for any',
        'partial batch. New query geometry still builds new plans. This optimization',
        'also applies to the reference prior evaluation.', '',
        '| Case | Direct ray query before → after | Direct collision before → after |',
        '| --- | ---: | ---: |']
    for case in ['A', 'B']:
        a = json.loads((ROOT / case / 'results.json').read_text())
        b = json.loads((ROOT / (case + '-plans') / 'results.json').read_text())
        assert a['input_sha256'] == b['input_sha256'] and a['seed'] == b['seed']
        row = []
        for kind in ['rays', 'collision']:
            x, y = [r['backends']['direct'][kind]['median_seconds'] for r in [a, b]]
            row.append(f'{1000*x:.2f} → {1000*y:.2f} ms ({x/y:.2f}×)')
        lines.append(f'| {case} | ' + ' | '.join(row) + ' |')
    lines += ['', 'The retained before/after runs use the same input hashes and seed. Warm timing',
        'variation remains; GPU clocks were not locked. This is an implementation',
        'improvement, not the much larger application speedup the overall goal seeks.', '',
        '## Preparation costs and exclusions', '',
        '| Case | Shared fit/mean setup | Shared corrected draws + direct aggregation |',
        '| --- | ---: | ---: |']
    for case in ['A', 'B']:
        r = json.loads((ROOT / (case + '-plans') / 'results.json').read_text())
        s = r['seconds']
        lines.append(f"| {case} | {s['common_fit_and_direct_mean_setup']:.2f} s | {s['common_corrected_draws_and_direct_aggregation']:.2f} s |")
    lines += ['', 'The controlled harness computes direct aggregation during shared preparation,',
        'even for the reference. Compatible aggregation is additional. Therefore these',
        'stages are diagnostic costs, **not independently optimized end-to-end totals**.',
        'The JSON records first query, setup, offset and all repeated query stages.',
        'Peak GPU memory was not instrumented in these initial runs. Geometry generation,',
        'file writes and rendering are excluded. Further preparation and memory profiling',
        'are required before complete-application claims.', '']
    sweep_paths = sorted(ROOT.glob('*-spectrum/results.json'))
    if sweep_paths:
        lines += ['## Experimental compression of centered samples', '',
            'Keep the full-resolution mean; truncate only the spectrum of each sample',
            'minus that mean. Conditioning and generation still use the full spectrum.',
            'Compare against the **full direct Fourier draws** to isolate this new error.',
            'These are exploratory finite-sample checks, not a validated cutoff policy.', '',
            '| Case | Fluctuation cutoff | Rays | Ray RMS / max error (points) | Paired gate vs full direct | Empirical variance relative error |',
            '| --- | ---: | ---: | ---: | --- | ---: |']
        for path in sweep_paths:
            r = json.loads(path.read_text())
            for name, q in r.get('quality_vs_full_direct', {}).items():
                t = r['backends'][name]['rays']['median_seconds']
                v = r['sample_spectrum_diagnostics'][name]['empirical_variance_relative_l2']
                lines.append(f"| {r['case']} | {name.split('_')[-1]} | {t*1000:.2f} ms | {100*q['rays']['rms_probability_difference']:.3f} / {100*q['rays']['max_probability_difference']:.3f} | {'pass' if q['compatibility_pass'] else 'FAIL'} | {v*100:.2f}% |")
        lines += ['', 'Passing against direct evaluation does not resolve direct-versus-interpolation',
                  'differences, population covariance error, or saturated collision coverage.', '']
    lines += ['## Reproduce', '', '```bash',
        './reproduction/run_env.sh reproduction/paired_applications.py --case A --rebuild-plans --output reproduction/results/paired_applications/A',
        './reproduction/run_env.sh reproduction/paired_applications.py --case B --rebuild-plans --output reproduction/results/paired_applications/B',
        './reproduction/run_env.sh reproduction/paired_applications.py --case A --output reproduction/results/paired_applications/A-plans',
        './reproduction/run_env.sh reproduction/paired_applications.py --case B --output reproduction/results/paired_applications/B-plans',
        './reproduction/run_env.sh reproduction/paired_applications.py --case B --backends direct --fluctuation-cutoffs 10 20 30 --output reproduction/results/paired_applications/B-spectrum',
        './reproduction/run_env.sh reproduction/paired_applications.py --case A --backends direct --fluctuation-cutoffs 10 20 30 --output reproduction/results/paired_applications/A-spectrum',
        './reproduction/run_env.sh reproduction/summarize_paired_applications.py', '```', '',
        'Run GPU jobs sequentially. Re-running overwrites the selected output directory.',
        'The report generator uses only the saved JSON/event arrays and CPU plotting.', '']
    (ROOT / 'README.md').write_text('\n'.join(lines))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    b = np.load(ROOT / 'B-plans/events.npz')
    ref, direct = b['reference_rays'].mean(0), b['direct_rays'].mean(0)
    ray = int(np.max(np.abs(ref - direct), axis=1).argmax())
    for name in ['reference', 'compatible', 'direct']:
        axes[0].plot(b['distance'], b[name + '_rays'].mean(0)[ray], label=name)
    peak = b['distance'][int(np.abs(ref[ray] - direct[ray]).argmax())]
    axes[0].set(xlabel='Distance along ray', ylabel='Transmittance',
                xlim=(peak - .1, peak + .1), title=f'B: ray {ray}, zoom at largest discrepancy')
    axes[0].legend()
    a = np.load(ROOT / 'A-plans/events.npz')
    for name in ['reference', 'compatible', 'direct']:
        axes[1].plot(a[name + '_collision'].mean(0), marker='.', label=name)
    axes[1].set(xlabel='Sphere along trajectory', ylabel='Collision probability', title='A: uncertain collision trajectory')
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(ROOT / 'diagnostic.png', dpi=160)


if __name__ == '__main__':
    main()
