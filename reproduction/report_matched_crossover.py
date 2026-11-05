"""Observed crossovers and resource limits for the coarse matched-width sweep."""
import argparse
import json
from pathlib import Path

import numpy as np
from reproduction.run_matched_lengthscales import digest
from reproduction.report_matched_lengthscales import LABELS, COLORS, save_figure
from reproduction.plot_asymptotic_regimes import slopes


def main(root):
    root = Path(root); base = Path('reproduction/results/matched_lengthscales')
    old = json.loads((base/'summary.json').read_text())
    paths = [Path(r['path']) for r in old['rows'] if r['lengthscale'] == .12]
    paths += [p for p in root.rglob('*.json') if p.name.split('-')[0] in ['dense', 'h2', 'sparse', 'original']
              and not any(x in p.name for x in ['.audit.', '.status.'])]
    rows, limits = [], []
    for path in sorted(paths):
        d = json.loads(path.read_text()); a = d['arguments']; method = a['method']
        if a['queries']*a['batches'] != 1048576: raise ValueError('Query count mismatch')
        if (a.get('lengthscale', a.get('original_lengthscale') or a.get('ell', 0)*2*np.pi)) != .12:
            raise ValueError('Lengthscale mismatch')
        # Enforce unchanged production settings within each family.
        template = json.loads((base/f'ell0.12/{method}-8192.json').read_text())['arguments']
        ignored = ['n', 'output']
        if path.parent == root/'refined' and method == 'h2-streamed':
            if a['variance_iterations'] not in [16, 32]: raise ValueError('Unexpected refined budget')
            if a['h2_tolerance'] not in [1e-5, 1e-6]: raise ValueError('Unexpected hierarchy tolerance')
            ignored.extend(['variance_iterations', 'h2_tolerance'])
        strip = lambda args: {k: v for k, v in args.items() if k not in ignored}
        if strip(a) != strip(template): raise ValueError('Changed settings '+str(path))
        quality = d.get('quality', d.get('independent_audit', {}))
        audit_path = path.with_name(path.stem+'.audit.json')
        audit = None
        if audit_path.exists():
            audit = json.loads(audit_path.read_text())
            if audit['source_sha256'] != digest(path): raise ValueError('Stale audit')
            quality = audit['quality']
        accepted = 'total_seconds' in d and (d.get('complete') if method == 'original' else quality.get('passes'))
        if not accepted:
            limits.append(dict(path=str(path), sha256=digest(path), n=a['n'], method=method,
                               status=d.get('failure', 'failed or missing numerical check'),
                               quality=quality, completed_production_seconds=d.get('total_seconds')))
            continue
        row = dict(path=str(path), sha256=digest(path), n=a['n'], method=method, total=d['total_seconds'],
                   preparation=d['preparation_seconds'], factor=d.get('stages', {}).get('cholesky_factorization', 0),
                   variance_iterations=a.get('variance_iterations'), mean_iterations=d.get('solve_info', {}).get('mean', {}).get('max_iterations'),
                   h2_tolerance=a.get('h2_tolerance'),
                   quality=quality, reused=path.parent == base/'ell0.12',
                   audit=str(audit_path) if audit is not None else None,
                   audit_sha256=digest(audit_path) if audit is not None else None,
                   offline_reference_failure=d.get('failure'),
                   hardware={k:d.get(k) for k in ['device','device_memory','device_memory_bytes','cpu_quota','host_limit','host_memory_limit']})
        if method == 'original': row['original_grid_sha256'] = digest(d['original_grid_file'])
        rows.append(row)
    groups = {m: sorted([r for r in rows if r['method'] == m], key=lambda r: r['n']) for m in LABELS}
    for method, group in groups.items():
        if len({r['n'] for r in group}) != len(group): raise ValueError('Duplicate accepted point '+method)
    h2 = {r['n']: r for r in groups['h2-streamed']}
    checks = []
    for r in groups['sparse-streamed']:
        if r['reused'] or r['n'] not in h2: continue
        from reproduction.compare_hierarchical_application import compare
        target = root/f"application-{r['n']}.json"
        compare(r['path'], h2[r['n']]['path'], str(target))
        check = json.loads(target.read_text())
        if not check['passes']: raise ValueError('Full-query check failed '+str(target))
        checks.append(dict(path=str(target), sha256=digest(target), passes=True))
    pairs = {m: [dict(n=r['n'], h2_seconds=h2[r['n']]['total'], competitor_seconds=r['total'],
                     competitor_over_h2=r['total']/h2[r['n']]['total']) for r in g if r['n'] in h2]
             for m, g in groups.items() if m != 'h2-streamed'}
    fits = {m: slopes([r for r in g if r['n'] >= 32768]) for m, g in groups.items()}
    result = dict(rows=rows, limits=limits, observed_pairs=pairs, total_slopes_from_32768=fits,
                  full_query_checks=checks,
                  dense_factor_slope=slopes([r for r in groups['dense'] if r['n'] >= 32768], 'factor'),
                  scope='Matched vector-kernel width; hierarchy tolerances and variance budgets selected offline to meet unchanged output gates; cross-model calibration not established',
                  analysis_source_sha256=digest(__file__))
    (root/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
    for method, group in groups.items():
        axes[0].loglog([r['n'] for r in group], [r['total'] for r in group], 'o-', color=COLORS[method], label=LABELS[method])
    for method in ['dense', 'h2-streamed']:
        group = groups[method]
        axes[1].loglog([r['n'] for r in group], [r['total'] for r in group], 'o-', color=COLORS[method], label=LABELS[method]+' total')
    group = groups['dense']
    axes[1].loglog([r['n'] for r in group], [r['factor'] for r in group], 's--', color=COLORS['dense'], label='Cholesky factorization stage')
    for limit in limits:
        if limit['method'] == 'dense':
            for ax in axes:
                ax.axvline(limit['n'], color=COLORS['dense'], linestyle=':', alpha=.6)
            reason = 'GPU memory limit' if 'OutOfMemoryError' in limit['status'] else 'no accepted result'
            axes[1].text(limit['n']*1.04, .8, f"Dense {limit['n']:,}:\n{reason}", fontsize=8,
                         transform=axes[1].get_xaxis_transform(), va='top')
        elif limit['method'] == 'h2-streamed' and limit['completed_production_seconds']:
            for ax in axes:
                ax.scatter(limit['n'], limit['completed_production_seconds'], marker='x', color='#666666', s=60)
    axes[0].set(title='All routes; extend competitors through observed limits', ylabel='Mean + variance workload (seconds)')
    axes[1].set(title='Dense GeoSPR versus H²: measured overlap')
    for ax in axes:
        ax.set_xlabel('Training points N'); ax.grid(True, which='both', alpha=.2)
    axes[1].legend(fontsize=7, loc='best')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .035), ncol=4, fontsize=8)
    fig.suptitle('Eagle, ℓ = .12, 1,048,576 marginal queries including preparation', fontsize=11)
    fig.text(.5, .01, '×: rejected accuracy settings. Periodic cutoff 50 not converged; cross-model calibration unmatched.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .15, 1, 1)); save_figure(fig, root, 'crossover'); plt.close(fig)
    print(json.dumps(dict(pairs=pairs, limits=limits), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='reproduction/results/matched_crossover')
    main(p.parse_args().root)
