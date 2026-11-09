"""Audit retained fields and summarize the PR118 stronger-baseline follow-up."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from geospsr.nonperiodic import FreeSpacePoissonMatern
from reproduction.partial_competitor_common import RESULTS, gates
from reproduction.partial_occupancy import ROOT


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect():
    historical = json.loads((RESULTS/'history/pr118-manifest.json').read_text())
    for relative, expected in historical['artifacts'].items():
        path = RESULTS/'history/pr118-README.md' if relative == 'README.md' else ROOT/relative
        assert digest(path) == expected, path
    model = FreeSpacePoissonMatern(.48, 1/3)
    fit = json.loads((ROOT/'controls.json').read_text())['fits']['spatial']
    rows = []
    for path in sorted(RESULTS.rglob('*.json')):
        record = json.loads(path.read_text())
        if 'arguments' not in record:
            continue
        args = record['arguments']
        n = args['n']
        field = path.with_suffix('.npz')
        if 'field_sha256' in record:
            assert digest(field) == record['field_sha256'], field
        contract = record.get('contract', {})
        for key, reference in [('reference_sha256', ROOT/'sparse_queries'/f'dense-{n}.npz'),
                               ('reference_record_sha256', ROOT/'sparse_queries'/f'dense-{n}.json'),
                               ('dataset_sha256', ROOT/'dataset.json'),
                               ('selected_sha256', ROOT/'selected.json'),
                               ('controls_sha256', ROOT/'controls.json'),
                               ('query_sha256', ROOT/'test_side-queries.npz')]:
            if key in contract:
                assert digest(reference) == contract[key], (path, key)
        for source, expected in record['source_sha256'].items():
            snapshot = RESULTS/'sources'/(expected+'.py')
            if not snapshot.exists():
                assert digest(source) == expected, source
                snapshot.write_bytes(Path(source).read_bytes())
            assert digest(snapshot) == expected, snapshot
        if field.exists():
            with np.load(field) as f, np.load(ROOT/'sparse_queries'/f'dense-{n}.npz') as truth:
                error = gates(f['mean'], f['variance'], truth, model, fit)
        else:
            error = record.get('errors')
            if error is None and 'selected' in record:
                selected = record['selected']
                error = selected['variants'][selected['selected_variant']]['errors']
            assert error is not None and not error['valid_moments'] and not record.get('accepted'), path
        for checkpoint in record.get('history', []):
            if 'field' not in checkpoint:
                continue
            with np.load(checkpoint['field']) as f, np.load(ROOT/'sparse_queries'/f'dense-{n}.npz') as truth:
                checked = gates(f['mean'], f['variance'], truth, model, fit)
            claimed = checkpoint['variants'][checkpoint['selected_variant']]['errors']
            assert checked.keys() == claimed.keys(), path
            for key in checked:
                assert np.isclose(checked[key], claimed[key], rtol=1e-12, atol=1e-15), (path, key)
        audit = record.get('gpu_audit', record.get('production_gpu_audit'))
        accepted = bool(error['passes'] and audit['sampled_isolation_pass'])
        if 'accepted' in record:
            assert accepted == record['accepted'], path
        if 'compression_seconds' in record:
            method = 'projected-'+args['method']
        elif 'batch' in args:
            method = 'sdd'
        elif 'dtype' in args:
            method = 'chol-'+args['dtype']+'-'+args['correction']
        else:
            method = args['method']
        total = record.get('total_seconds', record.get('selected', {}).get('total_seconds'))
        obs = audit['observations']
        row = dict(file=str(path.relative_to(RESULTS)), method=method, n=n,
                   seconds=total, accepted=accepted, isolated=audit['sampled_isolation_pass'],
                   gpu_gib=max((o['own_memory_bytes'] or 0) for o in obs)/2**30,
                   host_gib=max(o['own_host_rss_bytes'] for o in obs)/2**30,
                   **error)
        rows.append(row)
    return rows


def main():
    rows = collect()
    (RESULTS/'summary.json').write_text(json.dumps(rows, indent=2)+'\n')
    columns = ['file','method','n','seconds','accepted','isolated','gpu_gib','host_gib',
               'valid_moments','passes','mean','variance_over_prior','cdf','calibrated_cdf']
    with (RESULTS/'summary.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    series = [
        ('H² (direct)', '#176c91', lambda r: r['file'].startswith('h2-control-')),
        ('Cholesky (mixed precision)', '#cc6e1f', lambda r: r['file'].startswith('repeat-mixed-')),
        ('SDD (tuned checkpoints)', '#8064a2', lambda r: r['file'].startswith('repeat-sdd-8192-eta0.02-')
            or r['file'].startswith('sdd-large/n49152-b512-eta0.02-repeat')
            or r['file']=='sdd-large/n49152-b512-eta0.02-extended.json'),
    ]
    primary = {}
    for label, color, select in series:
        selected = [r for r in rows if select(r)]
        assert all(r['accepted'] for r in selected), label
        ns = sorted({r['n'] for r in selected})
        primary[label] = []
        for n in ns:
            rr = [r for r in selected if r['n'] == n]
            primary[label].append(dict(n=n, repeats=len(rr), seconds_median=float(np.median([r['seconds'] for r in rr])),
                seconds_min=min(r['seconds'] for r in rr), seconds_max=max(r['seconds'] for r in rr),
                gpu_gib_median=float(np.median([r['gpu_gib'] for r in rr])),
                host_gib_median=float(np.median([r['host_gib'] for r in rr])),
                max_cdf=max(r['cdf'] for r in rr), max_calibrated_cdf=max(r['calibrated_cdf'] for r in rr),
                max_variance_over_prior=max(r['variance_over_prior'] for r in rr)))
        times = [r['seconds_median'] for r in primary[label]]
        if ns:
            style = 'D:' if label.startswith('SDD') else 'o-'
            axes[0].loglog(ns, times, style, label=label, color=color)
            axes[0].fill_between(ns, [r['seconds_min'] for r in primary[label]],
                                [r['seconds_max'] for r in primary[label]], color=color, alpha=.15)
            axes[1].loglog(ns, [r['gpu_gib_median'] for r in primary[label]], style, label=label, color=color)
    # Historical float64 baseline is explicitly distinguished from fresh runs.
    old = [json.loads((ROOT/'sparse_queries'/f'dense-{n}.json').read_text()) for n in [8192,16384,32768,49152]]
    axes[0].loglog([8192,16384,32768,49152], [d['total_seconds'] for d in old], 's--',
                   color='gray', label='Float64 Cholesky (PR118)')
    for ax in axes:
        ax.set_xlabel('Normal observations N'); ax.grid(True, which='both', alpha=.2)
    axes[0].set_ylabel('Fresh setup + 128 mean/variance/CDF queries (s)')
    axes[1].set_ylabel('Sampled peak process GPU memory (GiB)')
    axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
    fig.savefig(RESULTS/'performance.png', dpi=180); fig.savefig(RESULTS/'performance.pdf'); plt.close(fig)
    (RESULTS/'primary.json').write_text(json.dumps(primary, indent=2)+'\n')

    fig, ax = plt.subplots(figsize=(8,4.5), layout='constrained')
    for path in sorted((RESULTS/'sdd-large').glob('*.json')):
        if 'repeat' in path.stem:
            continue
        d = json.loads(path.read_text()); args=d['arguments']; xx=[]; yy=[]
        # Correction pilots selected standard moments when neither variant passed;
        # plotting those as corrected completion times would be misleading.
        if args['correction'] != 'none':
            continue
        for h in d['history']:
            error = h['variants'][h['selected_variant']]['errors']
            if not error['valid_moments']: continue
            xx.append(h['total_seconds'])
            yy.append(max(error['cdf']/1e-3, error['calibrated_cdf']/1e-3, error['variance_over_prior']/1e-4))
        if xx:
            ax.loglog(xx, yy, 'o-', label=f"b={args['batch']}, η={args['eta']}")
    ax.axhline(1, color='black', linestyle='--', label='All output gates pass')
    ax.set_xlabel('Setup + updates + final output evaluation (s)')
    ax.set_ylabel('Largest output error / allowed error')
    ax.set_title('49,152 points: observed SDD checkpoints')
    ax.grid(True, which='both', alpha=.2); ax.legend(fontsize=7)
    fig.savefig(RESULTS/'sdd-checkpoints.png', dpi=180); plt.close(fig)


if __name__ == '__main__':
    main()
