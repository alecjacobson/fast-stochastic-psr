"""Report both robustness checks and the stronger sparse-control counterexample."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_rows(root):
    result = []
    for path in sorted(root.glob('*.json')):
        a = json.loads(path.read_text())
        if 'field' not in a:
            continue
        result.append(dict(path=str(path), method=a['arguments']['method'], n=a['arguments']['n'],
                           seed=a['arguments']['seed'], seconds=a['total_seconds'],
                           setup_seconds=a.get('setup_seconds'), cache_seconds=a.get('regional_cache_seconds'),
                           audit=a.get('independent_audit', a.get('quality', {})),
                           matrix_bytes=a.get('device_matrix_bytes', a.get('factor_bytes')),
                           sparse=a.get('sparse_diagnostics'), rank=a.get('max_h2_rank'),
                           mean_iterations=max(a.get('solve_info', {}).get('mean', {}).get('iterations', [0]))))
    return result


def summarize(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r['method'], r['n']), []).append(r)
    out = []
    for (method, n), rr in sorted(groups.items()):
        out.append(dict(method=method, n=n, runs=len(rr),
                        seconds=float(np.median([r['seconds'] for r in rr])),
                        min_seconds=min(r['seconds'] for r in rr), max_seconds=max(r['seconds'] for r in rr),
                        passes=all(r['audit'].get('passes', False) and r.get('paired_queries_pass', True) for r in rr),
                        max_cdf_error=max(r['audit'].get('max_probability_error', float('inf')) for r in rr),
                        max_variance_error_over_prior=max(r['audit'].get('max_variance_error_over_prior', float('inf')) for r in rr),
                        matrix_bytes=rr[0]['matrix_bytes'],
                        mean_neighbors=rr[0]['sparse']['mean_neighbors'] if rr[0]['sparse'] else None,
                        max_h2_rank=max((r['rank'] or 0) for r in rr)))
    return out


def best_sdd(path):
    if not path.exists():
        return [], []
    record = json.loads(path.read_text())
    choices = [(c, r) for c in record['configurations'] for r in c['history']
               if r.get('passes') and 'total_seconds' in r]
    if not choices:
        return [], [dict(path=str(path), seed=record['contract']['seed'],
                         configurations=record['configurations'],
                         interpretation='No passing observed checkpoint; no complete-runtime extrapolation.')]
    c, r = min(choices, key=lambda cr: cr[1]['total_seconds'])
    return [dict(path=c['passing_workload'], method='sdd', n=record['contract']['n'],
                 seed=record['contract']['seed'], seconds=r['total_seconds'], audit=r,
                 matrix_bytes=None, sparse=None, rank=None, eta=c['eta'],
                 iterations=r['iterations'], variant=r['selected_variant'])], []


def attach_comparisons(rows, comparisons):
    by_path = {r['candidate']: r for r in comparisons}
    for row in rows:
        check = by_path.get(row['path'])
        if check:
            row['paired_queries_pass'] = bool(check['max_probability_error'] <= 1e-3 and
                                              check['max_variance_error_over_prior'] <= 1e-4)
            row['paired_queries'] = check['queries']


def main(root):
    root = Path(root)
    locality = read_rows(root/'locality/scaling')+read_rows(root/'locality/large')
    partial, censored_sdd = [], []
    for folder in sorted((root/'partial').glob('seed*')):
        partial += read_rows(folder/'scaling')
        accepted, censored = best_sdd(folder/'sdd/77824.json')
        partial += accepted
        censored_sdd += censored
    paired = []
    for path in sorted((root/'partial').glob('seed*/application*.json')):
        paired.append(json.loads(path.read_text()))
    attach_comparisons(partial, paired)
    ls, ps = summarize(locality), summarize(partial)
    figure, grid = plt.subplots(2, 2, figsize=(11, 8))
    labels = {'h2-gpu': 'H² + AFN', 'sparse-gpu': 'GPU distance truncation + AFN',
              'dense-float32': 'GPU Cholesky', 'sdd': 'Best tested SDD'}
    colors = {'h2-gpu': 'C0', 'sparse-gpu': 'C2', 'dense-float32': 'C1', 'sdd': 'C3'}
    for summary, ax in [(ls, grid[0, 0]), (ps, grid[1, 0])]:
        for method, label in labels.items():
            rr = sorted([r for r in summary if r['method'] == method], key=lambda r: r['n'])
            if not rr:
                continue
            y = np.array([r['seconds'] for r in rr])
            ax.errorbar([r['n'] for r in rr], y,
                        yerr=[y-[r['min_seconds'] for r in rr], np.array([r['max_seconds'] for r in rr])-y],
                        marker='o', label=label, color=colors[method])
            failed = [r for r in rr if not r['passes']]
            if failed:
                ax.scatter([r['n'] for r in failed], [r['seconds'] for r in failed], marker='x', color='red', s=100)
        ax.set_yscale('log'); ax.set_ylabel('Seconds: setup + mean + variance + 1M queries')
    for method in ['h2-gpu', 'sparse-gpu']:
        rr = sorted([r for r in ls if r['method'] == method], key=lambda r: r['n'])
        grid[0, 1].plot([r['n'] for r in rr], [r['matrix_bytes']/2**30 for r in rr], 'o-',
                        label=labels[method], color=colors[method])
    grid[0, 1].set_yscale('log'); grid[0, 1].set_ylabel('Stored matrix / preconditioner GiB (not peak)')
    for method in ['h2-gpu', 'dense-float32', 'sparse-gpu']:
        rr = sorted([r for r in ps if r['method'] == method], key=lambda r: r['n'])
        short = {'h2-gpu': 'H²', 'dense-float32': 'Cholesky', 'sparse-gpu': 'Sparse'}[method]
        grid[1, 1].plot([r['n'] for r in rr], [r['max_cdf_error'] for r in rr], 'o-',
                        label=short+', tighter cross basis', color=colors[method])
    pilots = read_rows(root/'pilot/order4/seed21/scaling')
    rr = [r for r in pilots if r['method'] == 'h2-gpu']
    grid[1, 1].plot([r['n'] for r in rr], [r['audit']['max_probability_error'] for r in rr], 'x--',
                    color='gray', label='Retained loose cross-basis pilot')
    grid[1, 1].axhline(1e-3, color='k', linestyle=':', label='CDF accuracy gate')
    grid[1, 1].set_yscale('log'); grid[1, 1].set_ylabel('Worst checked probability error, with reference bound')
    titles = ['Full scan: 3 timing repetitions, cutoff fixed at 11', 'Full scan: stored representation scaling',
              'Noisy front scans: range over 3 observation seeds', 'Partial-scan cache accuracy']
    for ax, title in zip(grid.ravel(), titles):
        ax.set_title(title, fontsize=10); ax.set_xscale('log', base=2); ax.set_xlabel('Observation count N')
        ax.grid(True, alpha=.25); ax.legend(fontsize=8)
    figure.tight_layout(); figure.savefig(root/'scaling.png', dpi=170); figure.savefig(root/'scaling.pdf')
    slopes = {}
    for method in ['h2-gpu', 'sparse-gpu']:
        rr = [r for r in ls if r['method'] == method]
        if len(rr) > 1:
            slopes[method] = {quantity: float(np.polyfit(np.log([r['n'] for r in rr]),
                                                       np.log([r[quantity] for r in rr]), 1)[0])
                              for quantity in ['seconds', 'matrix_bytes']}
    result = dict(locality=ls, partial=ps, raw_locality=locality, raw_partial=partial,
                  censored_sdd=censored_sdd, paired_partial=paired,
                  finite_range_slopes=slopes,
                  interpretation='Timing ranges are repeats for locality, different scan seeds for partial. No asymptotic theorem or calibration certificate.')
    extension = read_rows(root/'extension/scaling')
    if extension:
        # Same nested seed-89 scan at every size; only the largest point has
        # three timing repeats. Other confirmation seeds are in the main plot.
        large_sdd, censored = best_sdd(root/'extension/sdd/196608.json')
        extension += large_sdd
        paired_extension = [json.loads(p.read_text()) for p in sorted((root/'extension').glob('application*.json'))]
        attach_comparisons(extension, paired_extension)
        extension_curve = [r for r in partial if r['seed'] == 89 and r['method'] in ['dense-float32', 'sdd']]+extension
        es = summarize(extension_curve)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for method in ['h2-gpu', 'sparse-gpu', 'dense-float32', 'sdd']:
            rr = sorted([r for r in es if r['method'] == method], key=lambda r: r['n'])
            if not rr:
                continue
            y = np.array([r['seconds'] for r in rr])
            ax[0].errorbar([r['n'] for r in rr], y,
                           yerr=[y-[r['min_seconds'] for r in rr], np.array([r['max_seconds'] for r in rr])-y],
                           marker='o', color=colors[method], label=labels[method])
            storage = [r for r in rr if r['matrix_bytes'] is not None]
            if storage:
                ax[1].plot([r['n'] for r in storage], [r['matrix_bytes']/2**30 for r in storage], 'o-',
                           color=colors[method], label=labels[method])
            failed = [r for r in rr if not r['passes']]
            if failed:
                ax[0].scatter([r['n'] for r in failed], [r['seconds'] for r in failed], marker='x', s=100, color='red')
        for failure in censored:
            record = json.loads(Path(failure['path']).read_text())
            elapsed = max(c['history'][-1]['solve_seconds'] for c in record['configurations'])
            ax[0].scatter([record['contract']['n']], [elapsed], marker='x', color='C3', s=65,
                          label='SDD solver budget; failed')
            ax[0].annotate('No passing checkpoint', (record['contract']['n'], elapsed),
                           xytext=(-8, -17), textcoords='offset points', ha='right', color='C3', fontsize=8)
        ax[0].set_ylabel('Seconds: setup + mean + variance + 1M queries')
        ax[1].set_ylabel('Stored matrix / preconditioner GiB (not peak)')
        for plot in ax:
            plot.set_xscale('log', base=2); plot.set_yscale('log'); plot.set_xlabel('Observation count N')
            plot.grid(True, alpha=.25); plot.legend(fontsize=8)
        fig.suptitle('Noisy front scan, seed 89: CG uses four variance steps; no dense extrapolation')
        fig.tight_layout(); fig.savefig(root/'extension/scaling.png', dpi=170); fig.savefig(root/'extension/scaling.pdf')
        result.update(extension=es, raw_extension=extension, censored_extension_sdd=censored, paired_extension=paired_extension)
    (root/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    for family, rows in [('locality', ls), ('partial', ps)]:
        print(family)
        for row in rows:
            print(row['method'], row['n'], round(row['seconds'], 3), row['passes'], row['max_cdf_error'])
    print('finite-range slopes', slopes)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='reproduction/results/hierarchical_robustness')
    main(p.parse_args().root)
