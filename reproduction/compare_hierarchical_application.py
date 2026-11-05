"""Compare complete saved probability workloads, including decision regret."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import cupy as cp
from cupyx.scipy.special import ndtr

from reproduction.benchmark_hierarchical_variance import query_batches, read_field


def compare(candidate, baseline, output):
    c, b = [json.loads(Path(p).read_text()) for p in [candidate, baseline]]
    for key in ['n', 'seed', 'geometry', 'lengthscale', 'amplitude', 'relative_noise',
                'halfwidth', 'queries', 'batches']:
        if c['arguments'][key] != b['arguments'][key]:
            raise ValueError(f'comparison contract mismatch: {key}')
    for key, default in [('scan', 'full'), ('normal_noise', 0.)]:
        if c['arguments'].get(key, default) != b['arguments'].get(key, default):
            raise ValueError(f'comparison contract mismatch: {key}')
    cf, bf = read_field(c['field']), read_field(b['field'])
    result = dict(candidate=candidate, baseline=baseline, max_mean_error=0.,
                  max_variance_error_over_prior=0., max_probability_error=0.,
                  max_decision_regret=0., exact_winner_agreements=0,
                  max_error_if_prior_variance_is_used=0.,
                  max_error_if_constant_contact_variance_is_used=0., batches=[],
                  max_constant_variance_error_over_prior=0.,
                  interpretation='Same-model numerical and decision comparison; marginal probabilities, not joint collision events or calibration evidence.')
    brier_sums = dict(candidate=0., baseline=0.)
    axes = cp.asarray([1., .8, .6] if b['arguments']['geometry'] == 'ellipsoid' else [1., 1., 1.])
    contact_variance = float(bf.predict([[1, 0, 0]])[1][0])
    for q in query_batches(SimpleNamespace(**b['arguments'])):
        cm, cv = cf.predict_gpu(q); bm, bv = bf.predict_gpu(q)
        cpred = ndtr((c['level']-cm)/cp.sqrt(cv)); bpred = ndtr((b['level']-bm)/cp.sqrt(bv))
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
    result['queries'] = b['arguments']['queries']*b['arguments']['batches']
    result['shape_brier'] = {key: value/result['queries'] for key, value in brier_sums.items()}
    result['shape_brier_scope'] = 'Known synthetic ellipsoid/sphere inside labels, at anchored local contact; not a repeated-scan calibration certificate.'
    Path(output).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'batches'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate'); parser.add_argument('baseline'); parser.add_argument('--output', required=True)
    a = parser.parse_args(); compare(a.candidate, a.baseline, a.output)
