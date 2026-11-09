"""Reference quality of inside probabilities on actual partial range scans."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.special import ndtr, log_ndtr

from geospsr.nonperiodic import FreeSpacePoissonMatern, DenseFreeSpacePosterior
from geospsr.scalar_contacts import ContactConditionedDensePosterior
from reproduction.stanford_partial_data import CACHE, digest

ROOT = Path('reproduction/results/real_partial')


def farthest_contacts(points, count=32):
    """Deterministic spatial coverage using only the input scan."""
    points = np.asarray(points)
    chosen = [int(np.argmin(np.sum((points-points.mean(0))**2, axis=1)))]
    distance = np.full(len(points), np.inf)
    for _ in range(min(count, len(points))-1):
        distance = np.minimum(distance, np.sum((points-points[chosen[-1]])**2, axis=1))
        chosen.append(int(np.argmax(distance)))
    return points[chosen]


def score(mean, variance, inside):
    if not np.isfinite(mean).all() or not np.isfinite(variance).all() or np.any(variance <= 0):
        raise ValueError('nonpositive or nonfinite marginal variance; no clipping permitted')
    z = -mean/np.sqrt(variance)
    p = ndtr(z); y = np.asarray(inside, dtype=bool)
    bins = np.minimum((p*10).astype(int), 9)
    calibration = []
    for i in range(10):
        mask = bins == i
        calibration.append(dict(count=int(mask.sum()),
            probability=float(p[mask].mean()) if mask.any() else None,
            frequency=float(y[mask].mean()) if mask.any() else None))
    ece = sum(b['count']*abs(b['probability']-b['frequency']) for b in calibration if b['count'])/len(p)
    return dict(count=len(p), inside_fraction=float(y.mean()), brier=float(np.mean((p-y)**2)),
        log_loss=float(-np.where(y, log_ndtr(z), log_ndtr(-z)).mean()),
        classification_accuracy=float(((p >= .5) == y).mean()), calibration=calibration,
        ece=float(ece), probability_quantiles=np.quantile(p, [0, .1, .5, .9, 1]).tolist(),
        standard_deviation_quantiles=np.quantile(np.sqrt(variance), [0, .1, .5, .9, 1]).tolist())


def stratified(mean, variance, panel):
    masks = dict(all=np.ones(len(mean), bool), observed=panel['observed'], occluded=panel['occluded'],
                 near_surface=panel['near_surface'], volume=~panel['near_surface'])
    return {name: score(mean[mask], np.broadcast_to(variance, mean.shape)[mask], panel['inside'][mask])
            for name, mask in masks.items() if mask.any()}


def load(group, n, query_stride=1):
    metadata = json.loads((ROOT/'dataset.json').read_text())
    path = CACHE/(group+'.npy')
    if digest(path) != metadata['groups'][group]['sha256']: raise ValueError('input checksum mismatch')
    rows = np.load(path, mmap_mode='r')
    if not 1 <= n <= len(rows): raise ValueError('invalid observation count')
    qpath = ROOT/(group+'-queries.npz')
    if digest(qpath) != metadata['groups'][group]['query_sha256']: raise ValueError('query checksum mismatch')
    panel = {k: v[::query_stride] for k, v in np.load(qpath).items()}
    return np.array(rows[:n, :3]), np.array(rows[:n, 3:]), farthest_contacts(rows[:, :3]), panel


def exact(group, n, ell, sigma, contact_sigma=.01, query_stride=1):
    x, normal, contacts, panel = load(group, n, query_stride)
    model = FreeSpacePoissonMatern(ell, 1/3)
    start = perf_counter()
    base = DenseFreeSpacePosterior(model, x, normal, sigma**2)
    posterior = ContactConditionedDensePosterior(base, contacts, np.zeros(len(contacts)), contact_sigma**2)
    mean, variance = posterior.predict(panel['queries'])
    seconds = perf_counter()-start
    return mean, variance, panel, model, seconds


def select(n=768):
    out = ROOT/'selection'; out.mkdir(exist_ok=True)
    protocol = dict(n=n, group='validation', query_stride=2, amplitude=1/3,
                    ell_grid=[.06, .12, .24, .48, .96], normal_sigma_grid=[.05, .2, .5],
                    contact_sigma=.01, contact_count=32, scalar_targets=0.,
                    criterion='minimum all-query Brier on validation view group only',
                    likelihood_scope='effective working-model noise, not measured scanner standard deviation',
                    source_sha256={p: digest(p) for p in [__file__, 'geospsr/scalar_contacts.py']})
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    records = []
    for ell in protocol['ell_grid']:
        for sigma in protocol['normal_sigma_grid']:
            path = out/f'ell{ell}-sigma{sigma}.json'
            mean, variance, panel, model, seconds = exact('validation', n, ell, sigma, query_stride=2)
            record = dict(ell=ell, sigma=sigma, seconds=seconds,
                posterior=stratified(mean, variance, panel), prior_variance=stratified(mean, model.prior_variance, panel))
            # A strong cheap control: choose a scalar variance on validation labels.
            grid = np.geomspace(1e-7, 10*model.prior_variance, 161)
            bs = [score(mean, np.full(len(mean), s), panel['inside'])['brier'] for s in grid]
            best = int(np.argmin(bs)); record['best_constant_variance'] = float(grid[best])
            record['best_constant_brier'] = bs[best]
            record['constant_search_boundary'] = best in (0, len(grid)-1)
            path.write_text(json.dumps(record, indent=2)+'\n'); records.append(record)
            print(ell, sigma, record['posterior']['all']['brier'], bs[best], seconds, flush=True)
    chosen = min(records, key=lambda r: r['posterior']['all']['brier'])
    # Also allow the constant-variance competitor to choose its own kernel/noise.
    constant = min(records, key=lambda r: r['best_constant_brier'])
    chosen = dict(chosen, constant_control_configuration=constant,
                  constant_inside_probability=float(load('validation', n, 2)[3]['inside'].mean()))
    (ROOT/'selected.json').write_text(json.dumps(chosen, indent=2)+'\n')


def test(group, n):
    selected = json.loads((ROOT/'selected.json').read_text())
    mean, variance, panel, model, seconds = exact(group, n, selected['ell'], selected['sigma'])
    const = selected['constant_control_configuration']
    if (const['ell'], const['sigma']) == (selected['ell'], selected['sigma']): cm = mean
    else: cm, _, _, _, _ = exact(group, n, const['ell'], const['sigma'])
    record = dict(group=group, n=n, ell=selected['ell'], normal_sigma=selected['sigma'], seconds=seconds,
        posterior=stratified(mean, variance, panel),
        prior_variance=stratified(mean, model.prior_variance, panel),
        constant_same_mean=stratified(mean, selected['best_constant_variance'], panel),
        independently_tuned_constant=stratified(cm, const['best_constant_variance'], panel),
        hard_mean_brier=float(np.mean(((mean < 0)-panel['inside'].astype(float))**2)),
        constant_class_probability=selected['constant_inside_probability'],
        constant_class_brier=float(np.mean((selected['constant_inside_probability']-panel['inside'])**2)),
        cdf_max_change_prior=float(np.max(abs(ndtr(-mean/np.sqrt(variance))-ndtr(-mean/np.sqrt(model.prior_variance))))),
        cdf_max_change_constant=float(np.max(abs(ndtr(-mean/np.sqrt(variance))-ndtr(-mean/np.sqrt(selected['best_constant_variance']))))),
        source_sha256={p: digest(p) for p in [__file__, 'geospsr/scalar_contacts.py', 'geospsr/nonperiodic.py']})
    output = ROOT/'exact'; output.mkdir(exist_ok=True)
    (output/f'{group}-{n}.json').write_text(json.dumps(record, indent=2)+'\n')
    np.savez_compressed(output/f'{group}-{n}.npz', mean=mean, variance=variance)
    print(json.dumps({k: record[k] for k in ['group', 'n', 'seconds', 'hard_mean_brier', 'constant_class_brier', 'cdf_max_change_prior']}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['select', 'test']); p.add_argument('--n', type=int, default=768)
    p.add_argument('--group', default='test_side')
    args = p.parse_args()
    if args.action == 'select': select(args.n)
    else: test(args.group, args.n)
