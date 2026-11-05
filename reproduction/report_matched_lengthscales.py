"""Audit matched-width fields and regenerate timing/phenomenology figures."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr

from reproduction.benchmark_hierarchical_variance import read_field, query_batches

LABELS = {'original': 'Sellan–Jacobson (matched grid)', 'dense': 'GeoSPR Cholesky',
          'sdd': 'GeoSPR SDD', 'nufft-cg': 'NUFFT + analytic PCG',
          'nufft-hybrid': 'Cholesky–NUFFT variance', 'sparse-streamed': 'Streamed locality',
          'h2-streamed': 'H²'}
COLORS = dict(zip(LABELS, ['#777777', '#e69f00', '#cc79a7', '#56b4e9', '#9467bd', '#009e73', '#d55e00']))


def save_figure(fig, root, name):
    for ext in ['png', 'pdf', 'svg']:
        path = root/f'{name}.{ext}'
        fig.savefig(path, dpi=180)
        if ext == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')


def read_predict(record):
    if 'field' in record:
        return read_field(record['field']).predict
    import gpytoolbox as gpy
    a = record['grid']; arrays = np.load(record['original_grid_file'])
    mean, variance = arrays['mean'], arrays['variance']
    def predict(q):
        w = gpy.fd_interpolate(np.asarray(q), gs=np.array(a['shape']),
                               h=np.array(a['spacing']), corner=np.array(a['corner']))
        return np.asarray(w@mean).ravel(), np.asarray(w@variance).ravel()
    return predict


def numerical_checks(rows, root):
    import cupy as cp
    from cupyx.scipy.special import ndtr as gpu_ndtr
    from reproduction.compare_hierarchical_application import compare
    lookup = {(r['lengthscale'], r['n'], r['method']): r for r in rows}
    checks = []
    for row in rows:
        method = row['method']
        if method not in ['sparse-streamed', 'sdd', 'nufft-cg', 'nufft-hybrid'] or not row.get('complete'):
            continue
        base = lookup.get((row['lengthscale'], row['n'], 'h2-streamed' if method == 'sparse-streamed' else 'dense'))
        if base is None or not base.get('numerical_passes') or not row.get('numerical_passes'):
            continue
        target = root/'checks'/f"ell{row['lengthscale']:g}-{method}-{row['n']}.json"
        target.parent.mkdir(exist_ok=True)
        if method == 'sparse-streamed':
            compare(row['path'], base['path'], str(target))
            result = json.loads(target.read_text())
        else:
            c, b = [json.loads(Path(r['path']).read_text()) for r in [row, base]]
            for key in ['point_sha256', 'normal_sha256', 'queries_total', 'model']:
                if c[key] != b[key]: raise ValueError('Mismatched '+key)
            for key in ['seed', 'queries', 'batches', 'halfwidth']:
                if c['arguments'][key] != b['arguments'][key]: raise ValueError('Mismatched '+key)
            f, ref = read_field(c['field']), read_field(b['field'])
            pe = ve = 0.; positive = True
            for q in query_batches(SimpleNamespace(**c['arguments'])):
                m, v = f.predict_gpu(q); rm, rv = ref.predict_gpu(q)
                positive &= bool(cp.all(v > 0) and cp.all(rv > 0) and cp.isfinite(m).all() and
                                 cp.isfinite(v).all() and cp.isfinite(rm).all() and cp.isfinite(rv).all())
                p = gpu_ndtr((c['level']-m)/cp.sqrt(v)); rp = gpu_ndtr((b['level']-rm)/cp.sqrt(rv))
                pe = max(pe, float(cp.max(abs(p-rp))))
                ve = max(ve, float(cp.max(abs(v-rv)))/ref.prior_variance)
            result = dict(candidate=row['path'], reference=base['path'], candidate_sha256=row['sha256'],
                          reference_sha256=base['sha256'], queries=c['queries_total'],
                          probability_max_error=pe, variance_max_error_over_prior=ve,
                          passes=bool(positive and pe <= 1e-3 and ve <= 1e-4))
            target.write_text(json.dumps(result, indent=2)+'\n')
            del f, ref; cp.get_default_memory_pool().free_all_blocks()
        row['full_query_passes'] = result['passes']
        checks.append(dict(path=str(target), passes=result['passes']))
    return checks


def summarize(root, check=True):
    root = Path(root); manifest = json.loads((root/'manifest.json').read_text())
    rows = []
    for source in manifest['rows']:
        row = source.copy(); path = Path(row['path'])
        if path.exists():
            payload = path.read_bytes()
            if row['sha256'] != hashlib.sha256(payload).hexdigest():
                raise ValueError('Changed result '+str(path))
            d = json.loads(payload)
            row['complete'] = bool(d.get('complete', 'total_seconds' in d))
            if row['complete']:
                row.update(total_seconds=d['total_seconds'], preparation_seconds=d['preparation_seconds'])
                if 'original_grid_file' in d:
                    row['original_grid_sha256'] = hashlib.sha256(Path(d['original_grid_file']).read_bytes()).hexdigest()
                gate = d.get('quality', d.get('independent_audit'))
                row['numerical_passes'] = gate['passes'] if gate is not None else None
            else:
                row['status'] = d.get('failure', d.get('status', 'incomplete'))
        rows.append(row)
    checks = numerical_checks(rows, root) if check else []
    # These observations are excluded from every training set in this sweep.
    cloud_path = json.loads(Path(next(r['path'] for r in rows if r.get('complete'))).read_text())['arguments']['point_cloud']
    cloud = np.load(cloud_path, mmap_mode='r')
    mask = np.max(abs(cloud[32768:, :3]-[1, 0, 0]), axis=1) < .005-1e-5
    # The anchoring sample was already used to choose the mean isovalue.
    mask &= np.linalg.norm(cloud[32768:, :3]-[1, 0, 0], axis=1) > 1e-12
    heldout = np.array(cloud[32768:][mask])
    ids = np.flatnonzero(mask)+32768
    diagnostics = dict(scope='Held-out local surface geometry only; no ground-truth occupancy probabilities or calibration claim',
                       point_cloud=cloud_path, point_cloud_sha256=hashlib.sha256(Path(cloud_path).read_bytes()).hexdigest(),
                       anchor_excluded=True,
                       training_max_n=32768, heldout_indices=ids.tolist(), heldout_points=heldout.tolist(), rows=[])
    profile_q = np.column_stack((1+np.linspace(-.005, .005, 201), np.zeros((201, 2))))
    for row in rows:
        if not row.get('complete') or row['n'] != 8192: continue
        d = json.loads(Path(row['path']).read_text()); predict = read_predict(d)
        level = d['level']; q = heldout[:, :3]
        mean, variance = predict(q)
        gradient = np.column_stack([(predict(q+1e-6*e)[0]-predict(q-1e-6*e)[0])/2e-6 for e in np.eye(3)])
        norm = np.linalg.norm(gradient, axis=1)
        offset = (mean-level)/norm
        roots = []; missing = 0
        for point in q:
            def residual(t): return float(predict([[t, point[1], point[2]]])[0][0]-level)
            if residual(.995)*residual(1.005) <= 0:
                roots.append(brentq(residual, .995, 1.005)-point[0])
            else: missing += 1
        pm, pv = predict(profile_q)
        diagnostic = dict(method=row['method'], lengthscale=row['lengthscale'], n=row['n'],
                          source=row['path'], source_sha256=row['sha256'],
                          heldout_surface_rms_linearized_distance=float(np.sqrt(np.mean(offset**2))),
                          heldout_surface_rms_x_root_error=float(np.sqrt(np.mean(np.array(roots)**2))) if roots else None,
                          missing_x_roots=missing, heldout_count=len(q),
                          mean_outward_normal_alignment=float(np.mean(np.sum(gradient*heldout[:, 3:], axis=1)/norm)),
                          median_linearized_sigma_distance=float(np.median(np.sqrt(variance)/norm)),
                          mean_abs_surface_probability_minus_half=float(np.mean(abs(ndtr((level-mean)/np.sqrt(variance))-.5))),
                          profile_x_offset=(profile_q[:, 0]-1).tolist(),
                          profile_probability=ndtr((level-pm)/np.sqrt(pv)).tolist(),
                          profile_variance=pv.tolist(), profile_mean_minus_level=(pm-level).tolist(),
                          profile_linearized_normal_sigma=(np.sqrt(pv)/abs(np.gradient(pm, profile_q[:, 0]))).tolist())
        diagnostics['rows'].append(diagnostic)
    (root/'phenomenology.json').write_text(json.dumps(diagnostics, indent=2)+'\n')
    summary = dict(rows=rows, full_query_checks=checks,
                   analysis_source_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                       ['reproduction/report_matched_lengthscales.py', 'reproduction/compare_hierarchical_application.py']})
    (root/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    plot(root, rows, diagnostics)
    return summary


def plot(root, rows, diagnostics):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for method, label in LABELS.items():
        for ax, selected, key in [(axes[0], [r for r in rows if r['n'] == 8192], 'lengthscale'),
                                  (axes[1], [r for r in rows if r['lengthscale'] == .12], 'n')]:
            good = [r for r in selected if r['method'] == method and r.get('complete') and
                    r.get('numerical_passes') is not False and r.get('full_query_passes') is not False]
            good.sort(key=lambda r: 1/r[key] if key == 'lengthscale' else r[key])
            ax.loglog([1/r[key] if key == 'lengthscale' else r[key] for r in good],
                      [r['total_seconds'] for r in good], 'o-', color=COLORS[method], label=label)
        bad = [r for r in rows if r['method'] == method and r['n'] == 8192 and r.get('complete') and
               (r.get('numerical_passes') is False or r.get('full_query_passes') is False)]
        axes[0].scatter([1/r['lengthscale'] for r in bad], [r['total_seconds'] for r in bad],
                        marker='x', color=COLORS[method], s=70)
    axes[0].set(xlabel='Inverse physical lengthscale 1/ℓ', ylabel='Mean + variance workload (seconds)',
                title='Fixed N = 8,192; matched vector-kernel width')
    axes[1].set(xlabel='Training points N', title='Fixed ℓ = 0.12; original grid 14 × 22 × 17')
    limits = []
    for r in rows:
        if r['method'] != 'original' or r['n'] != 8192 or r.get('complete'): continue
        if 'lower_bound_bytes' in r:
            reason = f"≥{np.ceil(r['lower_bound_bytes']/2**30):.0f} GiB for one covariance"
        elif 'MemoryError' in r.get('status', ''):
            reason = 'memory guard'
        else:
            reason = 'incomplete; see raw status'
        limits.append(f"Original ℓ = {r['lengthscale']:g}: {reason}")
    axes[0].text(.03, .85, '\n'.join(limits), transform=axes[0].transAxes, va='top', fontsize=8)
    for ax in axes: ax.grid(True, which='both', alpha=.2)
    axes[1].legend(fontsize=8, loc='best')
    fig.suptitle('Eagle: 1,048,576 marginal queries including preparation; matched width ≠ matched posterior quality', fontsize=10)
    fig.text(.5, .01, 'Periodic curves use finite cutoff 50; bandwidth convergence checks fail.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, 1))
    save_figure(fig, root, 'timing')
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for col, ell in enumerate([.12, .06, .03]):
        for d in diagnostics['rows']:
            if d['lengthscale'] != ell: continue
            method = d['method']; x = np.array(d['profile_x_offset'])*1000
            axes[0, col].plot(x, d['profile_probability'], color=COLORS[method], label=LABELS[method])
            axes[1, col].semilogy(x, d['profile_linearized_normal_sigma'], color=COLORS[method])
        axes[0, col].set(title=f'ℓ = {ell:g}', ylim=(-.02, 1.02))
        axes[1, col].set(xlabel='Contact-normal offset (10⁻³ units)', ylim=(1e-3, .03))
        for ax in axes[:, col]: ax.grid(alpha=.2)
    axes[0, 0].set_ylabel('Marginal occupancy probability')
    axes[1, 0].set_ylabel('Linearized normal displacement σ')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, loc='lower center', ncol=4)
    fig.suptitle('Local phenomenology at N = 8,192; contact mean anchored to zero; no calibration equivalence', fontsize=10)
    fig.tight_layout(rect=(0, .09, 1, 1))
    save_figure(fig, root, 'phenomenology')
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='reproduction/results/matched_lengthscales')
    a = p.parse_args(); summarize(a.root)
