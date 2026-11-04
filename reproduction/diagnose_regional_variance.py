"""Quantify prior/constant-variance shortcuts on a saved regional workload."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import cupy as cp
from cupyx.scipy.special import ndtr

from reproduction.benchmark_hierarchical_variance import query_batches, read_field


def diagnose(path, output):
    record = json.loads(Path(path).read_text())
    if not record.get('independent_audit', {}).get('passes'):
        raise ValueError('Use an independently audited field')
    field = read_field(record['field'])
    anchor_variance = float(field.predict([[1, 0, 0]])[1][0])
    result = dict(record=str(path), queries=0, minimum_variance=float('inf'), maximum_variance=0.,
                  prior_probability_error=0., constant_probability_error=0.,
                  prior_decision_regret=0., constant_decision_regret=0.,
                  scope='Shortcut differences from the audited approximate field, not an independent uniform accuracy or calibration certificate.')
    for queries in query_batches(SimpleNamespace(**record['arguments'])):
        mean, variance = field.predict_gpu(queries)
        probability = ndtr((record['level']-mean)/cp.sqrt(variance))
        score = (4*probability*(1-probability)).reshape(-1, 64).mean(axis=1)
        result['queries'] += len(queries)
        result['minimum_variance'] = min(result['minimum_variance'], float(variance.min()))
        result['maximum_variance'] = max(result['maximum_variance'], float(variance.max()))
        for name, v in [('prior', field.prior_variance), ('constant', anchor_variance)]:
            shortcut = ndtr((record['level']-mean)/v**.5)
            shortcut_score = (4*shortcut*(1-shortcut)).reshape(-1, 64).mean(axis=1)
            error = float(cp.max(abs(probability-shortcut)))
            regret = float(score.max()-score[cp.argmax(shortcut_score)])
            result[name+'_probability_error'] = max(result[name+'_probability_error'], error)
            result[name+'_decision_regret'] = max(result[name+'_decision_regret'], regret)
    result.update(prior_variance=field.prior_variance, contact_variance=anchor_variance)
    Path(output).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('record'); p.add_argument('--output', required=True)
    a = p.parse_args(); diagnose(a.record, a.output)
