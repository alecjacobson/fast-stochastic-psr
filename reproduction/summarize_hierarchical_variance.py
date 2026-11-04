"""Summarize measured full-workload repetitions without runtime extrapolation."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path('reproduction/results/hierarchical_variance')
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
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    labels = {'h2-gpu': 'H² + AFN', 'dense-float32': 'GPU Cholesky (float32)',
              'cg-dense-float32': 'Dense GPU PCG + refinement'}
    for method, label in labels.items():
        rows = [r for r in summary if r['method'] == method]
        if not rows:
            continue
        x, y = np.array([r['n'] for r in rows]), np.array([r['seconds'] for r in rows])
        line = axes[0].errorbar(x, y, yerr=[y-[r['min_seconds'] for r in rows],
                                          np.array([r['max_seconds'] for r in rows])-y], marker='o', label=label)
        fail = [i for i, r in enumerate(rows) if not r['passes']]
        if fail:
            axes[0].scatter(x[fail], y[fail], marker='x', s=120, color='red', label='audit failed/missing')
        axes[1].plot(x, [r['matrix_bytes']/2**30 for r in rows], 'o-', label=label)
    h2 = [r for r in summary if r['method'] == 'h2-gpu']
    axes[2].plot([r['n'] for r in h2], [r['max_h2_rank'] for r in h2], 'o-', label='Maximum H² rank')
    axes[2].plot([r['n'] for r in h2], [r['mean_iterations'] for r in h2], 's-', label='Mean CG iterations')
    axes[2].plot([r['n'] for r in h2], [r['solve_rank'] for r in h2], '^-', label='Regional cross rank')
    axes[0].set_ylabel('Seconds: setup + mean + variance + 1M queries')
    axes[1].set_ylabel('Stored matrix / factor GiB (not peak memory)')
    axes[2].set_ylabel('Observed rank / iterations')
    for ax in axes:
        ax.set_xscale('log', base=2); ax.set_xlabel('Observation count N'); ax.grid(True, alpha=.25)
        ax.legend(fontsize=8)
    axes[0].set_yscale('log'); axes[1].set_yscale('log')
    fig.tight_layout(); fig.savefig(root/'scaling.png', dpi=170); fig.savefig(root/'scaling.pdf')
    slopes = {}
    for method in ['h2-gpu', 'dense-float32']:
        rows = [r for r in summary if r['method'] == method and 32768 <= r['n'] <= 73728]
        slopes[method] = float(np.polyfit(np.log([r['n'] for r in rows]), np.log([r['seconds'] for r in rows]), 1)[0])
    failures = [dict(path=str(path), **json.loads(path.read_text()))
                for directory in ['scaling', 'large', 'controls'] for path in (root/directory).glob('*.status.json')
                if json.loads(path.read_text()).get('returncode') != 0]
    (root/'summary.json').write_text(json.dumps(dict(rows=summary, failed_or_censored_runs=failures,
        observed_loglog_slopes_32768_to_73728=slopes,
        interpretation='Measured finite-range slopes, not asymptotic proofs or extrapolated runtimes.'), indent=2)+'\n')
    print('| N | H² seconds | Cholesky seconds | Dense PCG seconds | H² / Cholesky matrix GiB |')
    print('|---:|---:|---:|---:|---:|')
    for n in sorted({r['n'] for r in summary}):
        d = {r['method']: r for r in summary if r['n'] == n}
        times = [f"{d[m]['seconds']:.3f}" if m in d else '—' for m in labels]
        memory = f"{d['h2-gpu']['matrix_bytes']/2**30:.2f} / {4*n*n/2**30:.2f}"
        print(f"| {n:,} | {' | '.join(times)} | {memory} |")
    print(json.dumps(slopes))


if __name__ == '__main__':
    main()
