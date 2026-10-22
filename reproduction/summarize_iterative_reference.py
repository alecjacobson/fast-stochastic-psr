"""Generate the SDD/NUFFT conditioning report from saved measurements."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent / 'results/iterative_baseline'


def main():
    datasets = ['A', 'B', '100000', '1000000']
    pilots = {case: json.loads((ROOT / case / 'results.json').read_text()) for case in datasets}
    comparisons = {case: json.loads((ROOT / case / 'comparison.json').read_text()) for case in datasets
                   if (ROOT / case / 'comparison.json').exists()}
    lines = ['# A converged iterative conditioning reference', '',
             'The released stochastic dual descent (SDD) method **can fit the million-point',
             'input** within the tested time budget. Its use avoids a dense N×N matrix.',
             'Dense-factorization memory alone would give the wrong feasibility conclusion.',
             'Wider-band NUFFT reduces fitting time at large N; the compiled SDD reference',
             'wins repeated mean fitting on the smaller A/B inputs. Neither conclusion is',
             'a complete-application or full-posterior scaling result.', '',
             '![Iterative convergence and selected fit times](comparison.png)', '',
             '## Reference and noise convention', '',
             '- Pinned upstream: `2352f1ef98aab20e0303611c1fa2b0948acd2a05`.',
             '- Training reference: analytic product Matérn, length scale 0.01, per-axis variance 0.1, noise variance 0.02.',
             '- The untouched SDD function squares `sigma`; it receives `sqrt(0.02)` to match the dense covariance. The local API consistently accepts the variance.',
             '- A compiled `jax.lax.fori_loop` preserves the released gradient, momentum 0.9, Polyak averaging 0.01, sequential key folding and sampling without replacement.',
             '- Tests compare against the untouched Python-loop routine and verify chunked resumption and `jit(vmap(...))` through the public SDD API.',
             '- Tune effective coordinate step `eta = lr * N / batch` and batch size. The reference is not restricted to the poorly tuned default learning rate.', '',
             'A/B reuse the matched camera-raster inputs; larger inputs are nested subsets',
             'of the distinct area-uniform visible scorpion cloud. Both methods use the',
             'same scalar/cross cutoff 50 and centering convention. NUFFT uses vector',
             'training cutoff 200, previously validated in the',
             '[bandwidth audit](../training_bandwidth/README.md). The finite training',
             'approximation remains explicit.', '',
             '## Selection and independent confirmation', '',
             'SDD stops when the analytic training residual is <0.1%: all rows for A/B,',
             'or the upper endpoint of a 512-row empirical bootstrap 95% interval at',
             'larger N. A/B additionally require mean error <0.2% across 20,000 queries.',
             'The fastest passing configuration is selected from the documented pilot.', '',
             'Confirmation reruns its selected iteration count and checks **1,024 new',
             'analytic rows** at large N, independent of the rows used for tuning/stopping.',
             'Both methods use that same holdout. Such intervals describe sampled rows;',
             'they are not full-system bounds, posterior-error guarantees, or evidence',
             'that rare difficult observations are absent.', '',
             '| Input | Method | First fit incl. setup (s) | Warm fit median (s) | Analytic residual % | Confirmation gate |',
             '| --- | --- | ---: | ---: | ---: | :---: |']
    million = comparisons.get('1000000', {}).get('rows', [])
    if len(million) == 2:
        sdd, nufft = million
        position = lines.index('![Iterative convergence and selected fit times](comparison.png)')
        lines[position:position] = [
            f"The million-point first fit takes **{sdd['first_fit_seconds']:.1f} s for compiled SDD** versus",
            f"**{nufft['first_fit_seconds']:.2f} s for NUFFT**, a **{sdd['first_fit_seconds']/nufft['first_fit_seconds']:.1f}× fitting ratio** in one confirmation run.",
            'The table below records the independent accuracy checks and timing limitations.', '']
    for case, d in comparisons.items():
        label = f"{case} ({d['N']:,})" if case in ('A', 'B') else f"{d['N']:,}"
        for r in d['rows']:
            stats = r['analytic_validation']['stats']
            passed = r['analytic_validation']['quality_pass'] and r['errors'].get('quality_pass', True)
            warm = r['warm_fit_median_seconds']
            lines.append(f"| {label} | {r['method']} | {r['first_fit_seconds']:.3f} | "
                         f"{format(warm, '.3f') if warm is not None else 'not repeated'} | "
                         f"{100*stats['relative_rms']:.4f} | {'yes' if passed else 'no'} |")
    lines += ['', 'First fit includes JIT/spectrum/plan setup, solving from zero, scalar source',
              'aggregation and centering. Process/CUDA initialization, data generation,',
              'parameter search, validation and output queries are excluded. Method order',
              'in the confirmation harness is SDD then NUFFT. Each large first-fit number',
              'is one measurement, not a statistical estimate of runtime.', '',
              'Warm medians use three repetitions on A/B/100k, resetting the solution and',
              'random sequence while reusing compiled code or GPU plans. These repeats',
              'measure the solver workload, **not an application refit speedup**: identical',
              'inputs could cache the answer. The million-point confirmation is not repeated.',
              'An NVIDIA L40 executes the GPU workloads serially. Peak memory was not',
              'instrumented in this series; no memory-limit claim is made.', '',
              'For A/B, both confirmed fits pass mean and probability gates. Probability',
              'checks use the shared exact analytic variance at the 256 prior audit probes',
              '(RMS <0.1 percentage points, maximum <2 points). They isolate mean-fit',
              'error and do not benchmark each method computing a posterior variance.',
              'At large N, reported mean differences use the cutoff-200 solution as a',
              'diagnostic, **not analytic ground truth**.', '',
              '## Pilot configurations, including failures', '',
              'Budgets count update time, excluding compilation and validation, and may',
              'overshoot by one checkpoint chunk. A: 10 s / 10,000 iterations; B: 20 s /',
              '20,000 iterations; 100k: 30 s / 100,000 iterations; 1m: 240 s / 200,000',
              'iterations per configuration. Stopping at a budget is not a claim that',
              'the configuration cannot converge. Tuning/selection results use seed 0;',
              'broader seed, geometry and parameter coverage remains open.', '',
              '| Input | Batch | eta | Iterations | Updates (s) | Final analytic residual % | Status |',
              '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for case, d in pilots.items():
        for r in d['rows']:
            residual = r['history'][-1]['analytic_residual']['relative_rms']
            lines.append(f"| {case} | {r['batch']} | {r['eta']:.4g} | {r['iterations']:,} | "
                         f"{r['update_seconds']:.3f} | {100*residual:.4f} | {r['status']} |")
    lines += ['', 'The paper’s solver example uses visual comparisons; these numerical gates',
              'and reconstructed workloads do not reproduce its unpublished Figure 13 scene.', '',
              '## Use and next work', '',
              'The accelerated loop is optional in the existing API:', '',
              '```python',
              'alpha = sdd_solve(x, normals, kernel, noise_variance, key,',
              '                  bs=512, lr=chosen_rate, iterations=chosen_iterations,',
              '                  compiled_loop=True)',
              '```', '',
              'Existing GP entry points can pass `compiled_loop=True` inside `sdd_params`.',
              'The default Python loop remains available. Learning rate and iteration count',
              'must be checked for the particular data; the scorpion tuning is not a default',
              'accuracy guarantee. Compiled mode has no per-iteration progress bar.', '',
              '[#15](https://github.com/alecjacobson/fast-stochastic-psr/issues/15) tracks',
              'method selection by workload, memory and expected reuse. Dense Cholesky may',
              'win when reused across many draws, even where SDD wins a single mean fit.',
              'Next: equal-quality conditional-sampling and complete-application comparisons,',
              'held-out seeds/parameters and larger scans, including the short-scale/low-noise',
              'failure in [#14](https://github.com/alecjacobson/fast-stochastic-psr/issues/14).',
              'The full objective remains open in [#1](https://github.com/alecjacobson/fast-stochastic-psr/issues/1).', '',
              '```bash',
              './reproduction/run_env.sh reproduction/benchmark_iterative_reference.py --dataset A --include-default --time-budget 10 --iterations 10000',
              './reproduction/run_env.sh reproduction/benchmark_iterative_reference.py --dataset B --include-default --time-budget 20 --iterations 20000',
              './reproduction/run_env.sh reproduction/benchmark_iterative_reference.py --dataset 100000 --batches 128 512 2048 --etas 2 5 --include-default --time-budget 30 --iterations 100000 --chunk 256',
              './reproduction/run_env.sh reproduction/benchmark_iterative_reference.py --dataset 1000000 --batches 128 512 --etas 5 --time-budget 240 --iterations 200000 --chunk 1024',
              './reproduction/run_env.sh reproduction/compare_iterative_fits.py --dataset A',
              './reproduction/run_env.sh reproduction/compare_iterative_fits.py --dataset B',
              './reproduction/run_env.sh reproduction/compare_iterative_fits.py --dataset 100000',
              './reproduction/run_env.sh reproduction/compare_iterative_fits.py --dataset 1000000 --repeats 0',
              './reproduction/run_env.sh reproduction/summarize_iterative_reference.py',
              '```', '',
              'Run the matched and training-bandwidth inputs first. Raw configuration',
              'histories, compact fields and confirmation records accompany this report.', '']
    (ROOT / 'README.md').write_text('\n'.join(lines))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for case, d in pilots.items():
        passing = [r for r in d['rows'] if r['quality_pass']]
        if passing:
            best = min(passing, key=lambda r: r['update_seconds'])
            axes[0].loglog([r['update_seconds'] for r in best['history']],
                           [100*r['analytic_residual']['relative_rms'] for r in best['history']], label=f'{d["N"]:,} points')
    axes[0].axhline(.1, color='gray', ls=':', label='0.1% residual gate')
    axes[0].set(xlabel='SDD update time (s); setup excluded', ylabel='Analytic residual (%)', title='Selected SDD convergence')
    names = [case for case in datasets if len(comparisons.get(case, {}).get('rows', [])) == 2]
    for i, method in enumerate(['SDD', 'NUFFT']):
        values = [next(r['first_fit_seconds'] for r in comparisons[case]['rows'] if r['method'] == method) for case in names]
        axes[1].bar(np.arange(len(names)) + (i - .5) * .35, values, width=.35, label=method)
    axes[1].set(xticks=np.arange(len(names)), xticklabels=[f"{pilots[case]['N']:,}" for case in names],
                xlabel='Observations', ylabel='First fit (s), log scale', title='Selected mean fits, setup included', yscale='log')
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.savefig(ROOT / 'comparison.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
