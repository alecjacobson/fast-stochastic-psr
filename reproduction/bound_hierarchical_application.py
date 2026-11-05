"""CPU-only whole-box comparison of saved regional probability polynomials.

This bounds agreement between caches, not error to the exact GP. The separate
independent audits are reported as provenance and are not promoted to uniform
accuracy guarantees. Exit code 2 denotes an inconclusive bound, not a proven
tolerance violation. No query generation or GPU is required.
"""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

from geospsr.regional_bounds import compare_moment_fields


def compare(candidate, baseline, output, max_leaves=256, max_depth=24):
    payloads = [Path(p).read_bytes() for p in (candidate, baseline)]
    c, b = [json.loads(p) for p in payloads]
    for key in ['n', 'seed', 'geometry', 'lengthscale', 'amplitude', 'relative_noise', 'halfwidth']:
        if c['arguments'][key] != b['arguments'][key]:
            raise ValueError(f'comparison contract mismatch: {key}')
    for key, default in [('scan', 'full'), ('normal_noise', 0.), ('point_cloud_sha256', None)]:
        if c['arguments'].get(key, default) != b['arguments'].get(key, default):
            raise ValueError(f'comparison contract mismatch: {key}')
    start = perf_counter()
    bounds = compare_moment_fields(SimpleNamespace(**c['field']), SimpleNamespace(**b['field']),
                                  candidate_level=c['level'], reference_level=b['level'],
                                  max_leaves=max_leaves, max_depth=max_depth)
    result = dict(candidate=str(candidate), baseline=str(baseline),
                  candidate_sha256=hashlib.sha256(payloads[0]).hexdigest(),
                  baseline_sha256=hashlib.sha256(payloads[1]).hexdigest(),
                  candidate_level=c['level'], baseline_level=b['level'],
                  seconds=perf_counter()-start, max_leaves=max_leaves, max_depth=max_depth,
                  bounds=bounds,
                  independent_point_audits={k: r.get('independent_audit') for k, r in [('candidate', c), ('baseline', b)]},
                  source_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                                 ['geospsr/regional_bounds.py', 'reproduction/bound_hierarchical_application.py']})
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=bounds['status'], seconds=result['seconds'],
                          probability_error_bound=bounds['probability_error_bound'],
                          variance_error_over_prior_bound=bounds['variance_error_over_prior_bound'],
                          leaves=bounds['leaves'], output=str(path))), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate')
    parser.add_argument('baseline')
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-leaves', type=int, default=256)
    parser.add_argument('--max-depth', type=int, default=24)
    args = parser.parse_args()
    result = compare(**vars(args))
    raise SystemExit(0 if result['bounds']['status'] == 'bounded' else 2)
