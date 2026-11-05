"""Compare complete saved probability workloads, including decision regret."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import cupy as cp
from cupyx.scipy.special import ndtr

from reproduction.benchmark_hierarchical_variance import query_batches, read_field


def compare(candidate, baseline, output):
    payloads = [Path(p).read_bytes() for p in [candidate, baseline]]
    c, b = [json.loads(payload) for payload in payloads]
    for key in ['n', 'seed', 'geometry', 'lengthscale', 'amplitude', 'relative_noise',
                'halfwidth', 'queries', 'batches']:
        if c['arguments'][key] != b['arguments'][key]:
            raise ValueError(f'comparison contract mismatch: {key}')
    for key, default in [('scan', 'full'), ('normal_noise', 0.)]:
        if c['arguments'].get(key, default) != b['arguments'].get(key, default):
            raise ValueError(f'comparison contract mismatch: {key}')
    if c['arguments'].get('point_cloud_sha256') != b['arguments'].get('point_cloud_sha256'):
        raise ValueError('comparison point cloud mismatch')
    cf, bf = read_field(c['field']), read_field(b['field'])
    result = dict(candidate=candidate, baseline=baseline, max_mean_error=0.,
                  candidate_sha256=hashlib.sha256(payloads[0]).hexdigest(),
                  baseline_sha256=hashlib.sha256(payloads[1]).hexdigest(),
                  max_variance_error_over_prior=0., max_probability_error=0.,
                  max_decision_regret=0., exact_winner_agreements=0,
                  max_error_if_prior_variance_is_used=0.,
                  max_error_if_constant_contact_variance_is_used=0., batches=[],
                  max_constant_variance_error_over_prior=0.,
                  interpretation='Same-model numerical and decision comparison; marginal probabilities, not joint collision events or calibration evidence.')
    brier_sums = dict(candidate=0., baseline=0.)
    known_shape = not b['arguments'].get('point_cloud')
    axes = cp.asarray([1., .8, .6] if b['arguments']['geometry'] == 'ellipsoid' else [1., 1., 1.])
    contact_variance = float(bf.predict([[1, 0, 0]])[1][0])
    for q in query_batches(SimpleNamespace(**b['arguments'])):
        cm, cv = cf.predict_gpu(q); bm, bv = bf.predict_gpu(q)
        if not bool(cp.isfinite(cm).all() and cp.isfinite(bm).all() and
                    cp.isfinite(cv).all() and cp.isfinite(bv).all() and
                    (cv > 0).all() and (bv > 0).all()):
            raise ValueError('Full-query comparison requires finite means and positive finite variances')
        cpred = ndtr((c['level']-cm)/cp.sqrt(cv)); bpred = ndtr((b['level']-bm)/cp.sqrt(bv))
        if known_shape:
            inside = cp.sum((cp.asarray(q)/axes)**2, axis=1) <= 1
            brier_sums['candidate'] += float(cp.sum((cpred-inside)**2))
            brier_sums['baseline'] += float(cp.sum((bpred-inside)**2))
        if not bool(cp.isfinite(cpred).all() and cp.isfinite(bpred).all()):
            raise RuntimeError('nonfinite probability')
        cs = (4*cpred*(1-cpred)).reshape(-1, 64).mean(axis=1)
        bs = (4*bpred*(1-bpred)).reshape(-1, 64).mean(axis=1)
        ci, bi = int(cp.argmax(cs)), int(cp.argmax(bs))
        regret = float(bs[bi]-bs[ci])
        result['batches'].append(dict(candidate_winner=ci, baseline_winner=bi, regret=regret))
        result['exact_winner_agreements'] += int(ci == bi)
        values = dict(max_mean_error=float(cp.max(abs(cm-bm))),
                      max_variance_error_over_prior=float(cp.max(abs(cv-bv)))/bf.prior_variance,
                      max_probability_error=float(cp.max(abs(cpred-bpred))),
                      max_decision_regret=regret,
                      max_constant_variance_error_over_prior=float(cp.max(abs(bv-contact_variance)))/bf.prior_variance,
                      max_error_if_prior_variance_is_used=float(cp.max(abs(bpred-ndtr((b['level']-bm)/bf.prior_variance**.5)))),
                      max_error_if_constant_contact_variance_is_used=float(cp.max(abs(bpred-ndtr((b['level']-bm)/contact_variance**.5)))))
        for key, value in values.items():
            result[key] = max(result[key], value)
    result['independent_audits'] = dict(candidate=bool(c.get('independent_audit', {}).get('passes')),
                                        baseline=bool(b.get('independent_audit', {}).get('passes')))
    result['passes'] = bool(all(result['independent_audits'].values()) and
                            result['max_probability_error'] <= 1e-3 and
                            result['max_variance_error_over_prior'] <= 1e-4)
    result['queries'] = b['arguments']['queries']*b['arguments']['batches']
    result['shape_brier'] = ({key: value/result['queries'] for key, value in brier_sums.items()} if known_shape else None)
    result['shape_brier_scope'] = ('Known synthetic ellipsoid/sphere inside labels, at anchored local contact; not a repeated-scan calibration certificate.'
                                   if known_shape else 'Unavailable: oriented point cloud has no ground-truth inside labels.')
    Path(output).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'batches'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate'); parser.add_argument('baseline'); parser.add_argument('--output', required=True)
    a = parser.parse_args(); compare(a.candidate, a.baseline, a.output)
