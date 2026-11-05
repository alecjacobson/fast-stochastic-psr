"""Plot explicitly selected audited real-scan runs; never infer acceptance from time."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main(root):
    root = Path(root)
    selection = json.loads((root/'selection.json').read_text())
    fig, panels = plt.subplots(2, 2, figsize=(11, 8))
    axes = panels.ravel()
    summary = []
    hardware = None
    colors = {'H2': '#1764a2', 'Sparse': '#c14c25', 'H2 cached': '#1764a2', 'H2 streamed': '#17a2a0',
              'Sparse cached': '#d18119', 'Sparse streamed': '#c14c25', 'Cholesky': '#777777'}
    for group in selection['series']:
        records = [json.loads((root/p).read_text()) for p in group['records']]
        if not all(d.get('independent_audit', {}).get('passes') for d in records):
            raise ValueError('Selected timing series contains an unaudited or failed record')
        for d in records:
            current = tuple(d[k] for k in ['device', 'device_memory_bytes', 'cpu_quota', 'host_memory_limit'])
            if hardware is None:
                hardware = current
            if current != hardware:
                raise ValueError('Selected measurements use different hardware contracts')
            if (d['arguments']['geometry'] != group['dataset'] or
                    d['arguments']['queries']*d['arguments']['batches'] != 1048576):
                raise ValueError('Selected dataset/query contract does not match the plot')
        records.sort(key=lambda d: d['arguments']['n'])
        n = np.array([d['arguments']['n'] for d in records])
        times = [d['total_seconds'] for d in records]
        storage = [d.get('device_matrix_bytes', d.get('factor_bytes'))/2**30 for d in records]
        color = colors[group['method']]
        label = group['dataset'].title()+' / '+group['method'].replace('H2', 'H²')
        style = '-' if group['dataset'] == 'eagle' else '--'
        axes[0].loglog(n, times, style+'o', label=label, color=color)
        entries = [d.get('interaction_nnz', d.get('sparse_diagnostics', {}).get('nnz', d['arguments']['n']**2)) for d in records]
        axes[1].loglog(n, entries, style+'o', label=label, color=color)
        axes[2].loglog(n, storage, style+'o', label=label, color=color)
        if 'solve_info' in records[0]:
            iterations = [max(d['solve_info']['mean']['iterations']) for d in records]
            axes[3].semilogx(n, iterations, style+'o', label=label, color=color)
        summary.extend(dict(dataset=group['dataset'], method=group['method'], n=d['arguments']['n'],
                            total_seconds=d['total_seconds'], storage_gib=s,
                            backend=d['arguments']['method'], order=d['arguments']['order'],
                            cross_rank=d['solve_rank'],
                            logical_entries=e,
                            workspace_gib=d.get('interaction_workspace_bytes', 0)/2**30,
                            cupy_pool_gib=d['cupy_pool_total_bytes']/2**30,
                            mean_iterations=max(d['solve_info']['mean']['iterations']) if 'solve_info' in d else None,
                            max_probability_error=d['independent_audit']['max_probability_error'],
                            max_variance_error_over_prior=d['independent_audit']['max_variance_error_over_prior'])
                       for d,s,e in zip(records, storage, entries))
    for ax,title,ylabel in zip(axes,
            ['Complete mean + variance workload', 'Represented kernel interactions', 'Stored GPU representation (cache policy varies)', 'Mean conditioning iterations'],
            ['Seconds, including 1,048,576 queries', 'Entries, including regenerated blocks', 'GiB; workspace reported separately', 'Maximum component PCG iterations']):
        ax.set(title=title, xlabel='Oriented observations N', ylabel=ylabel)
        ax.grid(True, which='both', alpha=.2)
    axes[0].legend(fontsize=7)
    fig.suptitle('PoissonRecon scans • matched sampled accuracy • single-run measurements', fontsize=12)
    fig.text(.5, .012, 'Accuracy settings and cached/streamed policies vary with N; see report. These curves do not prove an asymptotic exponent.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .035, 1, 1))
    fig.savefig(root/'scaling.png', dpi=180)
    fig.savefig(root/'scaling.pdf')
    (root/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='reproduction/results/poissonrecon_large')
    main(p.parse_args().root)
