"""Independent full-kernel, mixed-precision audit of saved regional predictions.

The dense factor is only a preconditioner for reference refinement. Residuals
use the complete float64 kernel, without H2 or regional cross interpolation.
Reference construction is excluded from benchmark timings.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.special import ndtr

from geospsr.nonperiodic import FreeSpacePoissonMatern
from geospsr.nonperiodic_dense_gpu import GPUDenseMaternSystem, full_kernel_matmul
from reproduction.benchmark_hierarchical_variance import data, read_field


def audit(path, reference_path, random_queries=32):
    record = json.loads(Path(path).read_text())
    a = record['arguments']
    contract = {k: a[k] for k in ['n', 'seed', 'geometry', 'lengthscale', 'amplitude',
                                 'relative_noise', 'halfwidth']}
    contract['random_queries'] = random_queries
    refpath = Path(reference_path)
    if refpath.exists():
        reference = json.loads(refpath.read_text())
        if reference['contract'] != contract:
            raise ValueError('reference contract mismatch')
    else:
        start = perf_counter()
        x, y, _ = data(a['n'], a['seed'], a['geometry'])
        model = FreeSpacePoissonMatern(a['lengthscale'], a['amplitude'])
        noise = model.amplitude*a['relative_noise']
        rng = np.random.default_rng(a['seed']+200)
        q = [1, 0, 0]+a['halfwidth']*np.vstack((rng.uniform(-1, 1, (random_queries, 3)),
                    np.array(list(itertools.product([-1., 1.], repeat=3))), np.zeros((1, 3))))
        c = model.cross_covariance(q, x)
        rhs = c.transpose(1, 2, 0).reshape(len(x), -1)
        if a['n'] <= 73728:
            system = GPUDenseMaternSystem(model, x, noise, dtype='float64' if a['n'] <= 16384 else 'float32')
            z = system.refined_solve(rhs, tolerance=1e-10, max_steps=6)
            refinement = system.refinement_info
            producer = 'dense '+str(system.dtype)
        else:
            from geospsr.hierarchical import HierarchicalMaternSystem
            from geospsr.hierarchical_gpu import GPUHierarchicalSystem
            system = GPUHierarchicalSystem(HierarchicalMaternSystem(
                model, x, noise, tolerance=1e-7, leaf_size=100, preconditioner_rank=50, neighbors=40))
            z = system.solve(rhs, tolerance=1e-9)
            norms = np.linalg.norm(rhs, axis=0)
            for step in range(6):
                r = rhs-full_kernel_matmul(model, x, noise, z)
                relative = np.linalg.norm(r, axis=0)/norms
                if np.all(relative < 1e-10):
                    break
                z += system.solve(r, tolerance=1e-9)
            else:
                raise RuntimeError('full-kernel reference refinement failed')
            refinement = dict(steps=step, relative_residuals=relative.tolist())
            producer = 'H2-initialized; refined and audited with full float64 kernel'
        residual = rhs-full_kernel_matmul(model, x, noise, z)
        # Q - (2 C^T Z - Z^T A Z) = R^T A^-1 R >= 0.
        # A >= noise*I gives a computable absolute variance error bound.
        correction = np.sum(z*(rhs+residual), axis=0).reshape(3, len(q)).sum(axis=0)
        variance = model.prior_variance-correction
        variance_bound = (np.sum(residual*residual, axis=0)/noise).reshape(3, len(q)).sum(axis=0)
        mean = np.sum(z.reshape(len(x), 3, len(q))*y[:, :, None], axis=(0, 1))
        # Cauchy-Schwarz in the A^-1 norm, using A >= noise*I.
        mean_bound = (np.linalg.norm(residual, axis=0).reshape(3, len(q))*
                      np.linalg.norm(y, axis=0)[:, None]/noise).sum(axis=0)
        reference = dict(contract=contract, queries=q.tolist(), mean=mean.tolist(),
                         variance=variance.tolist(), variance_error_bound=variance_bound.tolist(),
                         mean_error_bound=mean_bound.tolist(), prior_variance=model.prior_variance,
                         refinement=refinement, producer=producer,
                         residual_dtype='float64', seconds=perf_counter()-start,
                         source_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                           ['reproduction/audit_hierarchical_variance.py', 'geospsr/nonperiodic_dense_gpu.py',
                            'geospsr/nonperiodic.py']})
        refpath.parent.mkdir(parents=True, exist_ok=True)
        refpath.write_text(json.dumps(reference, indent=2)+'\n')
    field = read_field(record['field'])
    mean, variance = field.predict(reference['queries'])
    mr, vr = np.asarray(reference['mean']), np.asarray(reference['variance'])
    mb, vb = np.asarray(reference['mean_error_bound']), np.asarray(reference['variance_error_bound'])
    if np.any(vr-vb <= 0):
        raise RuntimeError('reference variance interval is not positive')
    p = ndtr((record['level']-mean)/np.sqrt(variance))
    pref = ndtr((mr[-1]-mr)/np.sqrt(vr))
    # Monotonicity in variance changes with numerator sign: test both endpoints.
    candidates = [ndtr((mr[-1]-mr+s*(mb[-1]+mb))/np.sqrt(vr-t*vb))
                  for s in [-1, 1] for t in [0, 1]]
    probability_bound = np.max(np.abs(np.asarray(candidates)-pref), axis=0)
    result = dict(reference=str(refpath), reference_sha256=hashlib.sha256(refpath.read_bytes()).hexdigest(),
                  queries=len(mr), max_mean_error=float(np.max(abs(mean-mr)+mb)),
                  max_variance_error_over_prior=float(np.max(abs(variance-vr)+vb)/reference['prior_variance']),
                  max_probability_error=float(np.max(abs(p-pref)+probability_bound)),
                  max_reference_mean_bound=float(mb.max()),
                  max_reference_variance_bound=float(vb.max()),
                  max_reference_probability_bound=float(probability_bound.max()))
    result['passes'] = bool(result['max_variance_error_over_prior'] <= 1e-4 and
                            result['max_probability_error'] <= 1e-3 and np.all(variance > 0))
    record['independent_audit'] = result
    Path(path).write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('record')
    parser.add_argument('--reference', required=True)
    parser.add_argument('--random-queries', type=int, default=32)
    args = parser.parse_args()
    audit(args.record, args.reference, args.random_queries)
