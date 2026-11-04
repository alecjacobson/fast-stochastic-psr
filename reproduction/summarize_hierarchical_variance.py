"""Summarize measured full-workload repetitions without runtime extrapolation."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main(root='reproduction/results/hierarchical_variance'):
    root = Path(root)
    groups = {}
    for directory in ['scaling', 'large', 'controls']:
        for path in (root/directory).glob('*.json'):
            row = json.loads(path.read_text())
            if 'field' not in row:
                continue
            key = row['arguments']['method'], row['arguments']['n']
            groups.setdefault(key, []).append(row)
    summary = []
    for (method, n), rows in sorted(groups.items()):
        times = [r['total_seconds'] for r in rows]
        audits = [r.get('independent_audit', {}) for r in rows]
        summary.append(dict(method=method, n=n, repetitions=len(rows),
                            seconds=float(np.median(times)), min_seconds=min(times), max_seconds=max(times),
                            median_setup_seconds=float(np.median([r['setup_seconds'] for r in rows])),
                            median_cache_seconds=float(np.median([r['regional_cache_seconds'] for r in rows])),
                            median_query_seconds=float(np.median([sum(r['query_seconds']) for r in rows])),
                            passes=all(a.get('passes', False) for a in audits),
                            max_variance_error_over_prior=(max(a['max_variance_error_over_prior'] for a in audits)
                                                          if all('passes' in a for a in audits) else None),
                            max_probability_error=(max(a['max_probability_error'] for a in audits)
                                                   if all('passes' in a for a in audits) else None),
                            matrix_bytes=rows[0].get('device_matrix_bytes', rows[0].get('factor_bytes')),
                            max_h2_rank=rows[0].get('max_h2_rank'), solve_rank=rows[0]['solve_rank'],
                            mean_iterations=max(rows[0].get('solve_info', {}).get('mean', {}).get('iterations', [0])),
                            variance_iterations=rows[0]['arguments']['variance_iterations']))
    for path in (root/'sdd').glob('*.json'):
        record = json.loads(path.read_text())
        if 'configurations' not in record:
            continue
        passing = [(config, row) for config in record['configurations'] for row in config['history']
                   if row.get('passes') and 'total_seconds' in row]
        if not passing:
            continue
        config, row = min(passing, key=lambda cr: cr[1]['total_seconds'])
        summary.append(dict(method='sdd', n=record['contract']['n'], repetitions=1,
                            seconds=row['total_seconds'], min_seconds=row['total_seconds'], max_seconds=row['total_seconds'],
                            passes=True, max_variance_error_over_prior=row['max_variance_error_over_prior'],
                            max_probability_error=row['max_probability_error'], matrix_bytes=None,
                            solve_rank=record['solve_rank'], sdd_iterations=row['iterations'],
                            eta=config['eta'], variant=row['selected_variant'],
                            interpretation='Best observed passing checkpoint; not an optimized stopping time or repeated timing.'))
    fig, grid = plt.subplots(2, 2, figsize=(11, 8))
    axes = grid.ravel()
    labels = {'h2-gpu': 'H² + AFN', 'dense-float32': 'GPU Cholesky (float32)',
              'cg-dense-float32': 'Dense GPU PCG + refinement', 'sdd': 'SDD (best tested checkpoint)'}
    for method, label in labels.items():
        rows = sorted([r for r in summary if r['method'] == method], key=lambda r: r['n'])
        if not rows:
            continue
        x, y = np.array([r['n'] for r in rows]), np.array([r['seconds'] for r in rows])
        axes[0].errorbar(x, y, yerr=[y-[r['min_seconds'] for r in rows],
                                    np.array([r['max_seconds'] for r in rows])-y],
                        marker='^' if method == 'sdd' else 'o', linestyle='--' if method == 'sdd' else '-', label=label)
        fail = [i for i, r in enumerate(rows) if not r['passes']]
        if fail:
            axes[0].scatter(x[fail], y[fail], marker='x', s=120, color='red', label='audit failed/missing')
        if method != 'sdd':
            axes[1].plot(x, [r['matrix_bytes']/2**30 for r in rows], 'o-', label=label)
            if method == 'cg-dense-float32':
                for r in rows:
                    if r['matrix_bytes'] < 4*r['n']**2:
                        axes[1].annotate('Row-tiled actions;\nno stored dense matrix',
                                         (r['n'], r['matrix_bytes']/2**30),
                                         xytext=(-115, 15), textcoords='offset points', fontsize=8)
    h2 = [r for r in summary if r['method'] == 'h2-gpu']
    axes[2].plot([r['n'] for r in h2], [r['max_h2_rank'] for r in h2], 'o-', label='Maximum H² rank')
    axes[2].plot([r['n'] for r in h2], [r['solve_rank'] for r in h2], '^-', label='Regional cross rank')
    axes[3].plot([r['n'] for r in h2], [r['mean_iterations'] for r in h2], 's-', label='H² mean CG')
    sdd = sorted([r for r in summary if r['method'] == 'sdd'], key=lambda r: r['n'])
    if sdd:
        axes[3].plot([r['n'] for r in sdd], [r['sdd_iterations'] for r in sdd], '^--', label='SDD first passing checkpoint')
    axes[0].set_ylabel('Seconds: setup + mean + variance + 1M queries')
    axes[1].set_ylabel('Stored matrix / factor GiB (not peak memory)')
    axes[2].set_ylabel('Observed rank')
    axes[3].set_ylabel('Iterations (different work per iteration)')
    for ax in axes:
        ax.set_xscale('log', base=2); ax.set_xlabel('Observation count N'); ax.grid(True, alpha=.25)
        ax.legend(fontsize=8)
    axes[0].set_yscale('log'); axes[1].set_yscale('log'); axes[3].set_yscale('log')
    example = next(iter(groups.values()))[0]['arguments']
    fig.suptitle(f"Nonperiodic contact workload: length scale {example['lengthscale']}, box halfwidth {example['halfwidth']}")
    fig.tight_layout(rect=[0, 0, 1, .97]); fig.savefig(root/'scaling.png', dpi=170); fig.savefig(root/'scaling.pdf')
    slopes = {}
    for method in ['h2-gpu', 'dense-float32']:
        rows = [r for r in summary if r['method'] == method and 32768 <= r['n'] <= 73728]
        slopes[method] = (float(np.polyfit(np.log([r['n'] for r in rows]), np.log([r['seconds'] for r in rows]), 1)[0])
                          if len(rows) > 1 else None)
    failures = [dict(path=str(path), **json.loads(path.read_text()))
                for directory in ['scaling', 'large', 'controls'] for path in (root/directory).glob('*.status.json')
                if json.loads(path.read_text()).get('returncode') != 0]
    (root/'summary.json').write_text(json.dumps(dict(rows=summary, failed_or_censored_runs=failures,
        observed_loglog_slopes_32768_to_73728=slopes,
        interpretation='Measured finite-range slopes, not asymptotic proofs or extrapolated runtimes.'), indent=2)+'\n')
    print('| N | H² seconds | Cholesky seconds | Dense PCG seconds | SDD seconds | H² / Cholesky matrix GiB |')
    print('|---:|---:|---:|---:|---:|---:|')
    for n in sorted({r['n'] for r in summary}):
        d = {r['method']: r for r in summary if r['n'] == n}
        times = [f"{d[m]['seconds']:.3f}" if m in d else '—' for m in labels]
        memory = f"{d['h2-gpu']['matrix_bytes']/2**30:.2f} / {4*n*n/2**30:.2f}"
        print(f"| {n:,} | {' | '.join(times)} | {memory} |")
    print(json.dumps(slopes))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='reproduction/results/hierarchical_variance')
    main(parser.parse_args().root)
