"""Replay whole-box bounds for the six selected Eagle pairs and one failure."""
import argparse
import hashlib
import json
from pathlib import Path

from reproduction.bound_hierarchical_application import compare


PAIRS = ['application-sparse-gpu-32768', 'application-dense-float32-32768',
         'application-sparse-gpu-77824', 'application-dense-float32-77824',
         'application-196608-cut15', 'application-796825',
         'application-variance-steps-796825']


def main(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in PAIRS:
        source = Path(args.source)/'eagle'/(name+'.json')
        payload = source.read_bytes()
        observed = json.loads(payload)
        result = compare(observed['candidate'], observed['baseline'], output/(name+'.json'),
                         max_leaves=args.max_leaves)
        for key in ('candidate', 'baseline'):
            if key+'_sha256' in observed and observed[key+'_sha256'] != result[key+'_sha256']:
                raise ValueError(f'stale observed comparison: {source} ({key})')
        bound = result['bounds']
        if (bound['probability_error_bound'] is not None and
                bound['probability_error_bound'] < observed['max_probability_error']):
            raise AssertionError(f'bound below observed error: {source}')
        rows.append(dict(pair=name, observed_source=str(source),
                         observed_source_sha256=hashlib.sha256(payload).hexdigest(),
                         observed_query_pass=observed['passes'],
                         observed_max_probability_error=observed['max_probability_error'],
                         bound_status=bound['status'], probability_error_bound=bound['probability_error_bound'],
                         variance_error_over_prior_bound=bound['variance_error_over_prior_bound'],
                         leaves=bound['leaves'], seconds=result['seconds']))
    summary = dict(comparisons=rows, all_selected_pairs_bounded=all(r['bound_status'] == 'bounded' for r in rows[:-1]),
                   known_failure_not_accepted=rows[-1]['bound_status'] == 'inconclusive',
                   interpretation='Whole-box cache agreement, not uniform error to the exact GP. '
                                  'Inconclusive bounds alone do not prove a tolerance violation.')
    (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    if not summary['all_selected_pairs_bounded'] or not summary['known_failure_not_accepted']:
        raise RuntimeError('regression expectations not satisfied; inspect saved results')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='reproduction/results/poissonrecon_large')
    parser.add_argument('--output', default='reproduction/results/regional_accuracy/bounds')
    parser.add_argument('--max-leaves', type=int, default=256)
    main(parser.parse_args())
