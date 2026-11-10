"""Frozen PR118 inputs and common numerical gates for stronger competitors."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtr

from geospsr.nonperiodic import FreeSpacePoissonMatern
from geospsr.scalar_contacts import ScalarContactUpdate
from reproduction.partial_occupancy import ROOT, load
from reproduction.stanford_partial_data import digest

RESULTS = Path('reproduction/results/partial_competitors')


def problem(n, group='test_side', query_limit=128):
    x, y, contacts, panel = load(group, n)
    ids = np.linspace(0, len(panel['queries'])-1, query_limit, dtype=int)
    panel = {k: v[ids] for k, v in panel.items()}
    selected = json.loads((ROOT/'selected.json').read_text())
    model = FreeSpacePoissonMatern(selected['ell'], 1/3)
    reference = ROOT/'sparse_queries'/f'dense-{n}.npz'
    reference_record = json.loads(reference.with_suffix('.json').read_text())
    assert reference_record['arguments']['group'] == group
    assert reference_record['queries'] == query_limit
    assert reference_record['arguments']['query_limit'] == query_limit
    truth = dict(np.load(reference))
    fit = json.loads((ROOT/'controls.json').read_text())['fits']['spatial']
    contract = dict(n=n, group=group, queries=query_limit, contacts=len(contacts),
        lengthscale=selected['ell'], amplitude=model.amplitude,
        normal_noise_variance=selected['sigma']**2, scalar_noise_variance=.01**2,
        reference_sha256=digest(reference), reference_record_sha256=digest(reference.with_suffix('.json')),
        dataset_sha256=digest(ROOT/'dataset.json'), selected_sha256=digest(ROOT/'selected.json'),
        controls_sha256=digest(ROOT/'controls.json'), query_sha256=digest(ROOT/f'{group}-queries.npz'),
        query_indices=ids.tolist())
    return x, y, contacts, panel, model, selected['sigma']**2, truth, fit, contract


def source_snapshot(paths):
    out = RESULTS/'sources'; out.mkdir(parents=True, exist_ok=True)
    result = {}
    for path in paths:
        data = Path(path).read_bytes(); h = hashlib.sha256(data).hexdigest()
        snapshot = out/(h+'.py')
        if not snapshot.exists(): snapshot.write_bytes(data)
        result[str(path)] = h
    return result


def gates(mean, variance, reference, model, fit):
    valid = bool(np.isfinite(mean).all() and np.isfinite(variance).all() and np.all(variance > 0))
    result = dict(valid_moments=valid, passes=False)
    if not valid: return result
    rm, rv = reference['mean'], reference['variance']
    z, rz = -mean/np.sqrt(variance), -rm/np.sqrt(rv)
    scale, bias = np.exp(fit['log_scale']), fit['bias']
    result.update(mean=float(np.max(abs(mean-rm))),
        variance_over_prior=float(np.max(abs(variance-rv))/model.prior_variance),
        cdf=float(np.max(abs(ndtr(z)-ndtr(rz)))),
        calibrated_cdf=float(np.max(abs(ndtr(scale*z+bias)-ndtr(scale*rz+bias)))))
    result['passes'] = bool(result['variance_over_prior'] <= 1e-4
                            and result['cdf'] <= 1e-3 and result['calibrated_cdf'] <= 1e-3)
    return result


def conditioned_moments(model, contacts, queries, cross, values, xp, *, whitened=False,
                        residual=None, scalar_variance=.01**2):
    """Project only needed scalar moments, then condition the scalar contacts.

    Columns are the three components of [contacts, queries], followed by
    the three normal-data RHS. ``values`` is L^-1 B or A^-1 B. For an
    approximate solve, residual-corrected moments are B.T Z + Z.T (B-AZ).
    This corrects the represented full kernel, without clipping variance.
    """
    nc, nq = len(contacts), len(queries); width = nc+nq
    if whitened and residual is not None: raise ValueError('residual needs solved, not whitened columns')
    mean = xp.zeros(width, dtype=xp.float64)
    correction = xp.zeros(width, dtype=xp.float64)
    cc = xp.zeros((nc, nc), dtype=xp.float64)
    qc = xp.zeros((nq, nc), dtype=xp.float64)
    for d in range(3):
        z = values[:, d*width:(d+1)*width]
        b = z if whitened else cross[:, d*width:(d+1)*width]
        mean += b.T@values[:, 3*width+d]
        correction += xp.sum(b*z, axis=0)
        cc += b[:, :nc].T@z[:, :nc]
        qc += b[:, nc:].T@z[:, :nc]
        if residual is not None:
            r = residual[:, d*width:(d+1)*width]
            mean += z.T@residual[:, 3*width+d]
            correction += xp.sum(z*r, axis=0)
            cc += z[:, :nc].T@r[:, :nc]
            qc += z[:, nc:].T@r[:, :nc]
    host = np.asarray if xp is np else xp.asnumpy
    mean, correction, cc, qc = map(host, (mean, correction, cc, qc))
    update = ScalarContactUpdate(mean[:nc], model.scalar_covariance(contacts, contacts)-.5*(cc+cc.T),
                                 np.zeros(nc), scalar_variance)
    return update.apply(mean[nc:], model.prior_variance-correction[nc:],
                        model.scalar_covariance(queries, contacts)-qc)


def store_result(path, record, mean=None, variance=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if mean is not None:
        np.savez_compressed(path.with_suffix('.npz'), mean=mean, variance=variance)
        record['field_sha256'] = digest(path.with_suffix('.npz'))
    path.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')


def projected_moments(model, contacts, queries, coefficients, gram, projected_mean, *, scalar_variance=.01**2):
    """Scalar contact/query moments from a shared compressed cross basis.

    This evaluates the requested discrete locations directly, with no spatial
    polynomial interpolation. Every producer can use the same compression.
    """
    nc, width = len(contacts), len(contacts)+len(queries)
    blocks = [coefficients[:, d*width:(d+1)*width] for d in range(3)]
    mean = sum(b.T@projected_mean[:, d] for d, b in enumerate(blocks))
    correction = sum(b.T@gram@b for b in blocks)
    update = ScalarContactUpdate(mean[:nc], model.scalar_covariance(contacts, contacts)-correction[:nc,:nc],
                                 np.zeros(nc), scalar_variance)
    return update.apply(mean[nc:], model.prior_variance-np.diag(correction)[nc:],
                        model.scalar_covariance(queries, contacts)-correction[nc:,:nc])
