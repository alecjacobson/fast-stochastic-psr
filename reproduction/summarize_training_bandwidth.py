"""Build the bandwidth-quality report from recorded measurements (CPU only)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent / 'results/training_bandwidth'


def load(name):
    return json.loads((ROOT / name / 'results.json').read_text())


def main():
    a, b, hard, scaling = [load(name) for name in ('A', 'B', 'B-short-low-noise', 'scaling')]
    lines = [
        '# Independent training bandwidth and massive-scale conditioning', '',
        'The scalar/cross Fourier cutoff stays at **50** (101³ modes); the vector',
        'training and prior cutoff increases independently. This reduces analytic-model',
        'error without enlarging the scalar sample/query representation. It does not',
        'make finite training identical to the analytic Matérn kernel.', '',
        'On the released default parameters, cutoff **150** passes the declared overlap',
        'gates at both 691 and 17,419 observations. A million-point mean fit takes',
        'about **5 seconds** at this cutoff and passes the **sampled** analytic-residual',
        'gate. The conditional-sampling pilot needs cutoff **200** to pass that gate',
        'on all eight draws (about **61 seconds** for the draws). Shorter length',
        'scales and lower noise remain a demonstrated failure',
        'regime. These are conditioning/sampling experiments, **not measured massive-scale',
        'application speedups**. No intractability claim for upstream SDD follows.', '',
        '![Training bandwidth, model error and fitting cost](convergence.png)', '',
        '## Overlap with full analytic conditioning', '',
        'A and B use the same released-camera scorpion inputs as the matched teaser.',
        'The reference uses dense **analytic** vector covariance plus noise variance',
        '0.02, with length scale 0.01 and per-axis variance 0.1. Scalar/cross kernels',
        'use direct cutoff-50 Fourier evaluation, so this study isolates training',
        'truncation rather than the released cross-kernel interpolant.', '',
        'All 20,000 slice means are checked. Deterministic variance, the full 256×256',
        'query covariance and inside probabilities use 256 probes: half selected',
        'by smallest absolute analytic mean, half randomly selected from the remainder.',
        'The analytic residual uses **every training row** at these feasible sizes.', '',
        'Gates set before these runs: mean relative L2 ≤0.2%; variance and query',
        'covariance relative L2 ≤0.02%; probability RMS ≤0.1 percentage points and',
        'maximum ≤2 points; analytic training residual ≤0.1%. NUFFT-CG must also',
        'converge at rtol 1e-5 and pass an independent finite-matrix residual check.', '',
        '| Case | Training cutoff | Mean error % | Variance error % | Covariance error % | Probability RMS / max (pp) | Analytic residual % | Pass |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |']
    for label, d in [('A', a), ('B', b)]:
        for row in d['rows']:
            e = row['errors']
            lines.append(f"| {label} | {row['training_cutoff']} | {100*e['mean_relative_l2']:.4f} | "
                         f"{100*e['variance_relative_l2']:.5f} | {100*e['covariance_relative_l2']:.5f} | "
                         f"{100*e['probability_rms']:.4f} / {100*e['probability_max']:.4f} | "
                         f"{100*e['analytic_residual']:.4f} | {'yes' if row['quality_pass'] else 'no'} |")
    lines += ['', 'The finite reference is independent of NUFFT: it evaluates the separable',
              'product of one-dimensional cosine sums, accumulating in float64, then',
              'uses dense Cholesky for probe covariance. This separates training-spectrum',
              'truncation from NUFFT and iterative-solver errors. Fitting on these small',
              'inputs is not necessarily faster than a cached dense factorization.', '',
              '## Distinct large inputs', '',
              'The 100,000 and 1,000,000 inputs are nested subsets of the previously generated',
              'area-uniform visible scorpion surface cloud, with coordinate/normal perturbations',
              'of standard deviation 0.02. This is **not** the camera-raster sampling used for',
              'A/B. Inputs, validation indices and hashes are fixed across bandwidths.', '',
              '512 randomly selected analytic covariance rows validate each fit. The empirical',
              'bootstrap intervals below describe variability over those sampled rows;',
              'they are **not global residual bounds or posterior-error guarantees**.',
              'The sampled gate requires the interval upper endpoint to be below 0.1%.', '',
              '| Observations | Training cutoff | Fit, first (s) | CG iterations | Analytic residual RMS % [bootstrap 95% interval] | Sampled gate |',
              '| ---: | ---: | ---: | ---: | --- | :---: |']
    for row in scaling['rows']:
        stats = row['analytic_residual_sample']
        lo, hi = stats['bootstrap_95_percent_interval']
        lines.append(f"| {row['N']:,} | {row['training_cutoff']} | {row['fit_first_seconds']:.3f} | "
                     f"{row['mean_solve']['iterations']} | {100*stats['relative_rms']:.4f} [{100*lo:.4f}, {100*hi:.4f}] | "
                     f"{'yes' if row['sampled_residual_gate_pass'] else 'no'} |")
    lines += ['', 'Fit time includes spectrum and GPU-plan creation, CG, scalar source aggregation',
              'and centering at the observations. Data generation, process startup, validation',
              'rows and query evaluation are excluded. First measurements are single runs;',
              'the overlap report separately records three repeated solves. All GPU work was',
              'serialized on an NVIDIA L40, using float32 and cuFINUFFT eps 1e-6. Training',
              'plans use oversampling 1.25; scalar plans retain 2.0.', '']
    sampling_path = ROOT / 'sampling/results.json'
    if sampling_path.exists():
        s = load('sampling')
        lines += ['## Coherent large-N sampling pilot', '',
                  'Vector Gaussian coefficients cover the entire training spectrum. Scalar',
                  'prior samples use the same low-frequency coefficients, and independent',
                  'observation noise is included before the conditional solve. There is no',
                  'dense training matrix or dense omitted-spectrum Cholesky factor.', '',
                  'An independent small-system test compares 4,096 draws to the exact finite',
                  'conditional mean/covariance and checks retained conditional right-hand sides.',
                  f"The million-point pilot uses only **{s['args']['samples']} draws per cutoff**; this is an execution",
                  'and cost check, not a Monte Carlo probability-accuracy result. Seeds alone',
                  'do not pair draws across bandwidths: the generated arrays have different shapes.', '',
                  '| Training cutoff | Fit (s) | All draws (s) | Worst sampled conditional residual RMS % | All sampled gates | Cached 20,000-query evaluation (ms) |',
                  '| ---: | ---: | ---: | ---: | :---: | ---: |']
        for row in s['rows']:
            c = row['training_cutoff']
            worst = max(r['relative_rms'] for r in row['conditional_analytic_residual_samples'])
            lines.append(f"| {c} | {s['seconds'][str(c)+'_fit']:.3f} | {s['seconds'][str(c)+'_draws']:.3f} | "
                         f"{100*worst:.4f} | {'yes' if row['all_sampled_residual_gates_pass'] else 'no'} | "
                         f"{1000*row['query_cached_median_seconds']:.3f} |")
        lines += ['', '**Cutoff 150 passes the mean-fit check but fails the conditional-sampling',
                  'check; cutoff 200 passes both for this pilot.** This is why observation',
                  'residuals alone are not enough for a sampling workload.', '',
                  'Conditional residuals use the actual generated sampling right-hand sides',
                  'and the same independent analytic rows as the mean-fit audit. The gate uses',
                  'the upper bootstrap endpoint for every draw. This tests the solve on those',
                  'right-hand sides; it does not validate the missing analytic prior tail or',
                  'replace deterministic posterior-covariance and application convergence checks.',
                  'Cached query times include all pilot draws, after separate cache preparation.',
                  'Repetitions use fixed queries and are kernel timings; an application could',
                  'cache a fixed answer. No cross-method speedup is inferred from this pilot.', '']
    memory_path = ROOT / 'memory-samples.csv'
    if memory_path.exists():
        values = [float(line.rsplit(',', 1)[1]) for line in memory_path.read_text().splitlines() if ',' in line]
        if values:
            lines += [f'The external 100 ms monitor observed a peak of **{max(values)/1024:.2f} GiB**',
                      'during the sampler validation/pilot queue, including runtime allocations',
                      'and retained validation rows/components. This is a sampled device-wide',
                      'measurement, not an exact live-workspace bound or minimum memory requirement.', '']
    lines += ['## Failure retained: shorter length scale and lower noise', '',
              'With B unchanged, length scale 0.005 and noise variance 0.0004, every tested',
              'bandwidth fails. Both parameters change in this stress case; separate sweeps',
              'are still needed to attribute sensitivity.', '',
              '| Training cutoff | Mean error % | Variance error % | Probability RMS / max (pp) | Analytic residual % |',
              '| ---: | ---: | ---: | ---: | ---: |']
    for row in hard['rows']:
        e = row['errors']
        lines.append(f"| {row['training_cutoff']} | {100*e['mean_relative_l2']:.3f} | {100*e['variance_relative_l2']:.3f} | "
                     f"{100*e['probability_rms']:.3f} / {100*e['probability_max']:.3f} | {100*e['analytic_residual']:.3f} |")
    lines += ['', 'This motivates [local/FMM omitted-spectrum correction (#14)](https://github.com/alecjacobson/fast-stochastic-psr/issues/14).',
              'A correction must preserve the joint GP and coherent sampling; a successful',
              'matvec alone is not sufficient. No FMM backend or speedup is claimed.', '',
              '## Remaining quality bridge and reproduction', '',
              '- Compare joint-event distributions, candidate-view decisions and hitboxes at converged sampling/geometric resolution.',
              '- Validate full-model posterior covariance at increasing N and parameters, beyond sampled mean/conditional residuals.',
              '- Benchmark the released iterative solver with documented convergence and time/memory budgets.',
              '- Measure full preparation plus application cost, rebuilds, and scaling in observations, queries and draws independently.', '',
              'Tracked by [#13](https://github.com/alecjacobson/fast-stochastic-psr/issues/13),',
              '[#6](https://github.com/alecjacobson/fast-stochastic-psr/issues/6) and',
              '[#8](https://github.com/alecjacobson/fast-stochastic-psr/issues/8).', '',
              '```bash',
              './reproduction/run_env.sh reproduction/training_bandwidth.py --case A',
              './reproduction/run_env.sh reproduction/training_bandwidth.py --case B',
              './reproduction/run_env.sh reproduction/training_bandwidth.py --case B --lengthscale .005 --noise .0004 --cutoffs 100 150 200 --output reproduction/results/training_bandwidth/B-short-low-noise',
              './reproduction/run_env.sh reproduction/scale_training_bandwidth.py',
              './reproduction/run_env.sh reproduction/scale_spectral_sampling.py',
              './reproduction/run_env.sh reproduction/summarize_training_bandwidth.py',
              '```', '',
              'If absent, the scaling script generates the million-point cloud using the',
              'released mesh and cameras. Raw measurements and compact field arrays are',
              'included in the adjacent directories; the large source cloud is regenerated.', '']
    (ROOT / 'README.md').write_text('\n'.join(lines))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), constrained_layout=True)
    for label, d in [('A, default', a), ('B, default', b), ('B, short/low noise', hard)]:
        xs = [r['training_cutoff'] for r in d['rows']]
        axes[0].semilogy(xs, [100*r['errors']['analytic_residual'] for r in d['rows']], 'o-', label=label)
        axes[1].semilogy(xs, [100*r['errors']['probability_max'] for r in d['rows']], 'o-', label=label)
    for n in sorted({r['N'] for r in scaling['rows']}):
        rows = [r for r in scaling['rows'] if r['N'] == n]
        xs = [r['training_cutoff'] for r in rows]
        axes[0].semilogy(xs, [100*r['analytic_residual_sample']['relative_rms'] for r in rows], 's--', label=f'{n:,}, sampled')
        axes[2].plot(xs, [r['fit_first_seconds'] for r in rows], 'o-', label=f'{n:,} observations')
    axes[0].axhline(.1, color='gray', ls=':', label='0.1% gate')
    axes[1].axhline(2, color='gray', ls=':', label='2 pp gate')
    for ax, title, ylabel in zip(axes, ['Analytic training residual', 'Overlap probability error', 'First fit, including plans'],
                               ['Relative RMS (%)', 'Maximum error (percentage points)', 'Seconds']):
        ax.set(xlabel='Vector training cutoff', ylabel=ylabel, title=title)
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.savefig(ROOT / 'convergence.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
