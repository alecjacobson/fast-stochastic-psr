"""Grant compiled SDD the same regional compression and moment cache.

Checkpoint results are measured prefixes, never extrapolated completion times.
This is a same-model algorithm comparison, not the original 2022 SPSR method.
"""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import jax.numpy as jnp
import numpy as np
from scipy.linalg import svd
from scipy.special import ndtr

from geospsr.iterative import StochasticDualDescent
from geospsr.nonperiodic import FreeSpacePoissonMatern, RegionalVarianceCache
from geospsr.nonperiodic_dense_gpu import full_kernel_matmul
from geospsr.regional_field import PolynomialMomentField
from reproduction.benchmark_hierarchical_variance import data


def run(args):
    import cupy as cp
    reference = json.loads(Path(args.reference).read_text())
    a = reference['contract']
    x, y, _ = data(a['n'], a['seed'], a['geometry'])
    model = FreeSpacePoissonMatern(a['lengthscale'], a['amplitude'])
    q = np.array(reference['queries'])
    mr, vr = np.array(reference['mean']), np.array(reference['variance'])
    pref = ndtr((mr[-1]-mr)/np.sqrt(vr))
    mb, vb = np.array(reference['mean_error_bound']), np.array(reference['variance_error_bound'])
    probability_allowance = np.max(np.abs(np.array([
        ndtr((mr[-1]-mr+sign*(mb[-1]+mb))/np.sqrt(vr-endpoint*vb))
        for sign in [-1, 1] for endpoint in [0, 1]])-pref), axis=0)
    cache = RegionalVarianceCache.__new__(RegionalVarianceCache)
    cache._initialize_box([1, 0, 0], a['halfwidth'], 4)
    cache.prior_variance = model.prior_variance
    begin = perf_counter()
    c = model.cross_covariance(cache.nodes, x)
    rhs = c.transpose(1, 2, 0).reshape(len(x), -1)
    u, s, vt = svd(rhs, full_matrices=False, check_finite=False)
    rank = int(np.count_nonzero(s > 1e-4*s[0]))
    u, coefficients = u[:, :rank], s[:rank, None]*vt[:rank]
    compression_seconds = perf_counter()-begin
    full_rhs = np.column_stack((u, y))
    rhs = full_rhs.astype(np.float32)
    def kernel(x, z):
        r = jnp.sqrt(jnp.sum((x-z)**2))*model.kappa
        return (1+r)*jnp.exp(-r)
    record = dict(arguments=vars(args), contract=a, solve_rank=rank,
                  compression_seconds=compression_seconds,
                  device=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
                  reference_sha256=hashlib.sha256(Path(args.reference).read_bytes()).hexdigest(),
                  description='Compiled original SDD update on the same isotropic nonperiodic kernel; same regional SVD and cache as H2/Cholesky',
                  source_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                    ['geospsr/iterative.py', 'geospsr/nonperiodic.py', 'geospsr/nonperiodic_dense_gpu.py',
                     'geospsr/regional_field.py', 'reproduction/benchmark_hierarchical_sdd.py',
                     'reproduction/benchmark_hierarchical_variance.py']}, configurations=[])
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    for eta in args.eta:
        begin = perf_counter()
        solver = StochasticDualDescent(jnp.asarray(x, dtype=jnp.float32), jnp.asarray(rhs), kernel,
                    noise_variance=a['relative_noise'], batch=args.batch,
                    lr=eta*args.batch/len(x), seed=a['seed'])
        solver.compile()
        compile_seconds = perf_counter()-begin
        config = dict(eta=eta, batch=args.batch, compile_seconds=compile_seconds, history=[])
        record['configurations'].append(config)
        elapsed = 0.
        for checkpoint in args.checkpoints:
            start = perf_counter()
            while solver.iterations < checkpoint:
                value = solver.advance(min(256, checkpoint-solver.iterations))
                if elapsed+perf_counter()-start >= args.budget:
                    break
            z = np.asarray(value, dtype=float)/model.amplitude
            elapsed += perf_counter()-start
            start = perf_counter()
            residual = full_rhs-full_kernel_matmul(model, x, model.amplitude*a['relative_noise'], z)
            moment = full_rhs.T@z+z.T@residual
            moment = .5*(moment+moment.T)
            correction_seconds = perf_counter()-start
            gram = moment[:rank, :rank]
            projected_mean = moment[:rank, rank:]
            r = 4**3
            start = perf_counter()
            cache.correction = sum(coefficients[:, d*r:(d+1)*r].T@gram@coefficients[:, d*r:(d+1)*r] for d in range(3))
            cache.mean_coefficients = sum(coefficients[:, d*r:(d+1)*r].T@projected_mean[:, d] for d in range(3))
            field = PolynomialMomentField(cache)
            finalization_seconds = perf_counter()-start
            mean, variance = field.predict(q)
            level = float(field.predict([[1, 0, 0]])[0][0])
            valid = bool(np.all(variance > 0) and np.isfinite(variance).all() and np.isfinite(mean).all())
            pe = float(np.max(abs(ndtr((level-mean)/np.sqrt(variance))-pref)+probability_allowance)) if valid else None
            ve = float(np.max(abs(variance-vr)+vb)/model.prior_variance) if np.isfinite(variance).all() else None
            row = dict(iterations=solver.iterations, solve_seconds=elapsed,
                       preparation_seconds=compression_seconds+compile_seconds+elapsed+correction_seconds+finalization_seconds,
                       residual_correction_seconds=correction_seconds,
                       residual_correction='full float64 kernel; joint mean/variance bilinear correction',
                       max_probability_error=pe, max_variance_error_over_prior=ve,
                       minimum_variance=float(variance.min()) if np.isfinite(variance).all() else None,
                       passes=bool(valid and pe <= 1e-3 and ve <= 1e-4))
            config['history'].append(row)
            out.write_text(json.dumps(record, indent=2)+'\n')
            print(eta, json.dumps(row), flush=True)
            if row['passes'] or elapsed >= args.budget or not np.isfinite(z).all():
                break
        config['status'] = ('passing preparation checkpoint (queries not timed)' if row['passes'] else
                            'no passing observed checkpoint within budget')
        out.write_text(json.dumps(record, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--eta', nargs='+', type=float, default=[.03, .1])
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--checkpoints', nargs='+', type=int, default=[128, 512, 2048, 8192])
    parser.add_argument('--budget', type=float, default=45)
    run(parser.parse_args())
