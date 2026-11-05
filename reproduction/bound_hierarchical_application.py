"""CPU-only whole-box comparison of saved regional probability polynomials.

This bounds agreement between caches, not error to the exact GP. The separate
independent audits are reported as provenance and are not promoted to uniform
accuracy guarantees. Exit code 2 denotes an inconclusive bound, not a proven
tolerance violation. No query generation or GPU is required.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from scipy.special import ndtr

from geospsr.regional_bounds import compare_moment_fields


def inspect_leaf_probes(bounds, candidate, baseline):
    """27 numerical probes in the worst box, separate from interval bounds."""
    center, width = np.zeros(3), np.ones(3)
    for axis, side in bounds['worst_leaf']['subdivision_path']:
        width[axis] *= .5
        center[axis] += side*width[axis]
    probes = center + width*np.array(list(itertools.product([-1., 0., 1.], repeat=3)))
    probabilities = []
    for record in (candidate, baseline):
        field = record['field']
        mean = np.polynomial.chebyshev.chebval3d(*probes.T, np.asarray(field['mean']))
        variance = field['prior_variance']-np.polynomial.chebyshev.chebval3d(*probes.T, np.asarray(field['correction']))
        valid = np.isfinite(mean) & np.isfinite(variance) & (variance > 0)
        p = np.full(len(probes), np.nan)
        p[valid] = ndtr((record['level']-mean[valid])/np.sqrt(variance[valid]))
        probabilities.append(p)
    valid = np.isfinite(probabilities).all(axis=0)
    differences = np.abs(probabilities[0]-probabilities[1])
    index = int(np.argmax(np.where(valid, differences, -1.))) if valid.any() else None
    point = probes[index] if index is not None else None
    physical = (np.asarray(candidate['field']['center']) + point*np.asarray(candidate['field']['halfwidth'])
                if point is not None else None)
    difference = float(differences[index]) if index is not None else None
    return dict(queries=len(probes), invalid_queries=int((~valid).sum()),
                normalized_probe=point.tolist() if point is not None else None,
                physical_probe=physical.tolist() if physical is not None else None,
                candidate_probability=float(probabilities[0][index]) if index is not None else None,
                baseline_probability=float(probabilities[1][index]) if index is not None else None,
                observed_probability_difference=difference,
                observed_above_tolerance=difference is not None and difference > bounds['probability_tolerance'],
                scope='Floating-point observations on a 3x3x3 grid in the worst bound leaf; '
                      'not an interval certificate or an exact-GP reference.')


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
                  worst_leaf_probes=inspect_leaf_probes(bounds, c, b),
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
