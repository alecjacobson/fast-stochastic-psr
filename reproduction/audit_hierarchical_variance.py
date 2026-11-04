"""Independent full-kernel, mixed-precision audit of saved regional predictions.

The dense factor is only a preconditioner for reference refinement. Residuals
use the complete float64 kernel, without H2 or regional cross interpolation.
Reference construction is excluded from benchmark timings.
"""
import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.special import ndtr

from geospsr.nonperiodic import FreeSpacePoissonMatern
from geospsr.nonperiodic_dense_gpu import GPUDenseMaternSystem, full_kernel_matmul
from reproduction.benchmark_hierarchical_variance import data_from_arguments, read_field


def corrected_reference_moments(rhs, solution, residual, prior_variance, noise):
    """Full-kernel residual bounds for variance and two-sided mean correction.

    For R=C-AZ and S=Y-AW, mean error is RᵀA⁻¹S. This avoids
    multiplying the cross residual by the much larger unconditioned Y norm.
    """
    count = (rhs.shape[1]-3)//3
    c, z, r = rhs[:, :-3], solution[:, :-3], residual[:, :-3]
    w, ry = solution[:, -3:], residual[:, -3:]
    reduction = (np.einsum('ij,ij->j', z, c)+np.einsum('ij,ij->j', z, r)).reshape(3, count).sum(axis=0)
    variance = prior_variance-reduction
    r2 = np.einsum('ij,ij->j', r, r).reshape(3, count)
    variance_bound = r2.sum(axis=0)/noise
    moments = c.T@w+z.T@ry
    mean = sum(moments[d*count:(d+1)*count, d] for d in range(3))
    mean_bound = (np.sqrt(r2)*np.linalg.norm(ry, axis=0)[:, None]/noise).sum(axis=0)
    return mean, variance, mean_bound, variance_bound


def probability_allowance(mean, variance, mean_bound, variance_bound):
    if np.any(variance-variance_bound <= 0):
        return np.full(len(mean), float('inf'))
    reference = ndtr((mean[-1]-mean)/np.sqrt(variance))
    endpoints = [ndtr((mean[-1]-mean+sign*(mean_bound[-1]+mean_bound))/np.sqrt(variance-side*variance_bound))
                 for sign in [-1, 1] for side in [0, 1]]
    return np.max(abs(np.asarray(endpoints)-reference), axis=0)


def audit(path, reference_path, random_queries=32, reference_preconditioner='native'):
    record = json.loads(Path(path).read_text())
    a = record['arguments']
    contract = {k: a[k] for k in ['n', 'seed', 'geometry', 'lengthscale', 'amplitude',
                                 'relative_noise', 'halfwidth']}
    contract['random_queries'] = random_queries
    if a.get('point_cloud'):
        contract.update(point_cloud=a['point_cloud'], point_cloud_sha256=a['point_cloud_sha256'])
    for key, default in [('scan', 'full'), ('normal_noise', 0.)]:
        if a.get(key, default) != default:
            contract[key] = a[key]
    refpath = Path(reference_path)
    if refpath.exists():
        reference = json.loads(refpath.read_text())
        if reference['contract'] != contract:
            raise ValueError('reference contract mismatch')
    else:
        start = perf_counter()
        x, y, _ = data_from_arguments(a)
        model = FreeSpacePoissonMatern(a['lengthscale'], a['amplitude'])
        noise = model.amplitude*a['relative_noise']
        rng = np.random.default_rng(a['seed']+200)
        q = [1, 0, 0]+a['halfwidth']*np.vstack((rng.uniform(-1, 1, (random_queries, 3)),
                    np.array(list(itertools.product([-1., 1.], repeat=3))), np.zeros((1, 3))))
        # Direct cross kernels, without regional interpolation/compression;
        # bounded temporary storage also works for multi-million-point inputs.
        rhs = np.empty((len(x), 3*len(q)+3))
        for first in range(0, len(x), 8192):
            last = min(first+8192, len(x))
            cross = model.cross_covariance(q, x[first:last])
            rhs[first:last, :-3] = cross.transpose(1, 2, 0).reshape(last-first, -1)
        rhs[:, -3:] = y
        if a['n'] <= 73728:
            system = GPUDenseMaternSystem(model, x, noise, dtype='float64' if a['n'] <= 16384 else 'float32')
            solve = system.solve
            producer = 'dense '+str(system.dtype)
        else:
            from geospsr.hierarchical import HierarchicalMaternSystem
            from geospsr.hierarchical_gpu import GPUHierarchicalSystem
            system = GPUHierarchicalSystem(HierarchicalMaternSystem(
                model, x, noise, tolerance=1e-7, leaf_size=100, preconditioner_rank=50, neighbors=40,
                preconditioner_backend=reference_preconditioner), interaction_budget_bytes=256*2**20,
                rhs_block_size=16 if a['n'] > 262144 else None, release_native=True)
            solve = lambda b: system.solve(b, tolerance=1e-7)
            producer = 'Streamed H2 initializer; full-float64-kernel residual certification; '+reference_preconditioner
        z = solve(rhs)
        norms = np.linalg.norm(rhs, axis=0)
        history = []
        for step in range(6):
            residual = rhs-full_kernel_matmul(model, x, noise, z)
            mean, variance, mean_bound, variance_bound = corrected_reference_moments(
                rhs, z, residual, model.prior_variance, noise)
            allowance = probability_allowance(mean, variance, mean_bound, variance_bound)
            history.append(dict(step=step, max_probability_bound=float(allowance.max()),
                                max_variance_bound_over_prior=float(variance_bound.max()/model.prior_variance)))
            print(json.dumps(dict(reference_step=step, elapsed_seconds=perf_counter()-start, **history[-1])),
                  file=sys.stderr, flush=True)
            if allowance.max() <= 1e-5 and variance_bound.max()/model.prior_variance <= 1e-7:
                break
            if step == 5:
                raise RuntimeError('full-kernel reference output bounds failed')
            z += solve(residual)
        refinement = dict(steps=step, relative_residuals=(np.linalg.norm(residual, axis=0)/norms).tolist(),
                          history=history, mean_formula='C^T W + Z^T (Y-AW)',
                          mean_error_bound='||C-AZ|| * ||Y-AW|| / noise, summed over components')
        reference = dict(contract=contract, queries=q.tolist(), mean=mean.tolist(),
                         variance=variance.tolist(), variance_error_bound=variance_bound.tolist(),
                         mean_error_bound=mean_bound.tolist(), prior_variance=model.prior_variance,
                         refinement=refinement, producer=producer,
                         residual_dtype='float64', seconds=perf_counter()-start,
                         source_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                           ['reproduction/audit_hierarchical_variance.py', 'geospsr/nonperiodic_dense_gpu.py',
                            'geospsr/nonperiodic.py', 'reproduction/benchmark_hierarchical_variance.py']})
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
    parser.add_argument('--reference-preconditioner', choices=['native', 'prefix-kd'], default='native')
    args = parser.parse_args()
    audit(args.record, args.reference, args.random_queries, args.reference_preconditioner)
