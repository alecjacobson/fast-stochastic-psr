"""Grant compiled SDD the same regional compression and moment cache.

Checkpoint results are measured prefixes, never extrapolated completion times.
This is a same-model algorithm comparison, not the original 2022 SPSR method.
"""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
from scipy.linalg import svd
from scipy.special import ndtr

from geospsr.iterative import StochasticDualDescent
from geospsr.nonperiodic import FreeSpacePoissonMatern, RegionalVarianceCache
from geospsr.nonperiodic_dense_gpu import full_kernel_matmul
from geospsr.regional_field import PolynomialMomentField
from reproduction.benchmark_hierarchical_variance import data, query_batches, save_field


def run(args):
    import cupy as cp
    from cupyx.scipy.special import ndtr as gpu_ndtr
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
    query_args = SimpleNamespace(seed=a['seed'], halfwidth=a['halfwidth'], queries=args.queries, batches=args.batches)
    start = perf_counter()
    cache.correction = np.zeros((64, 64)); cache.mean_coefficients = np.zeros(64)
    warm = PolynomialMomentField(cache)
    wm, wv = warm.predict_gpu(np.tile([1., 0, 0], (args.queries, 1)))
    wp = gpu_ndtr(wm/cp.sqrt(wv)); ws = (4*wp*(1-wp)).reshape(-1, 64).mean(axis=1)
    cp.argmax(ws).item(); cp.sum(wp).item()
    runtime_initialization = perf_counter()-start
    del warm, wm, wv, wp, ws
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
                  common_runtime_initialization_seconds=runtime_initialization,
                  device=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
                  reference_sha256=hashlib.sha256(Path(args.reference).read_bytes()).hexdigest(),
                  description='Compiled original SDD update on the same isotropic nonperiodic kernel; same regional SVD and cache as H2/Cholesky',
                  source_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                    ['geospsr/iterative.py', 'geospsr/nonperiodic.py', 'geospsr/nonperiodic_dense_gpu.py',
                     'geospsr/regional_field.py', 'reproduction/benchmark_hierarchical_sdd.py',
                     'reproduction/benchmark_hierarchical_variance.py']}, configurations=[])
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)

    def evaluate(gram, projected_mean):
        start = perf_counter()
        r = 4**3
        cache.correction = sum(coefficients[:, d*r:(d+1)*r].T@gram@coefficients[:, d*r:(d+1)*r] for d in range(3))
        cache.mean_coefficients = sum(coefficients[:, d*r:(d+1)*r].T@projected_mean[:, d] for d in range(3))
        field = PolynomialMomentField(cache)
        field.prepare_gpu(); cp.cuda.get_current_stream().synchronize()
        seconds = perf_counter()-start
        mean, variance = field.predict(q)
        level = float(field.predict([[1, 0, 0]])[0][0])
        valid = bool(np.all(variance > 0) and np.isfinite(variance).all() and np.isfinite(mean).all())
        pe = float(np.max(abs(ndtr((level-mean)/np.sqrt(variance))-pref)+probability_allowance)) if valid else None
        ve = float(np.max(abs(variance-vr)+vb)/model.prior_variance) if np.isfinite(variance).all() else None
        quality = dict(max_probability_error=pe, max_variance_error_over_prior=ve,
                       minimum_variance=float(variance.min()) if np.isfinite(variance).all() else None,
                       passes=bool(valid and pe <= 1e-3 and ve <= 1e-4))
        return field, level, quality, seconds

    for eta in args.eta:
        begin = perf_counter()
        solver = StochasticDualDescent(jnp.asarray(x, dtype=jnp.float32), jnp.asarray(rhs), kernel,
                    noise_variance=a['relative_noise'], batch=args.batch,
                    lr=eta*args.batch/len(x), seed=a['seed'], momentum=.9, polyak=.01)
        solver.compile()
        compile_seconds = perf_counter()-begin
        config = dict(eta=eta, batch=args.batch, learning_rate=eta*args.batch/len(x),
                      momentum=.9, polyak=.01, compile_seconds=compile_seconds, history=[])
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
            gram = .5*(u.T@z[:, :rank]+z[:, :rank].T@u)
            projected_mean = u.T@z[:, rank:]
            projection_seconds = perf_counter()-start
            standard = evaluate(gram, projected_mean)
            field, level, quality, finalization_seconds = standard
            variants = dict(standard=dict(quality, projection_seconds=projection_seconds,
                                          field_seconds=finalization_seconds))
            selected = 'standard'
            # Correction is optional: it can be more conservative and slower
            # for stochastic iterates with large errors orthogonal to U.
            if not quality['passes']:
                start = perf_counter()
                residual = full_rhs-full_kernel_matmul(model, x, model.amplitude*a['relative_noise'], z)
                moment = full_rhs.T@z+z.T@residual
                moment = .5*(moment+moment.T)
                corrected_seconds = perf_counter()-start
                corrected = evaluate(moment[:rank, :rank], moment[:rank, rank:])
                variants['corrected'] = dict(corrected[2], projection_seconds=corrected_seconds,
                                              field_seconds=corrected[3])
                if corrected[2]['passes']:
                    field, level, quality, finalization_seconds = corrected
                    projection_seconds = corrected_seconds
                    selected = 'corrected'
            row = dict(iterations=solver.iterations, solve_seconds=elapsed,
                       preparation_seconds=compression_seconds+compile_seconds+elapsed+projection_seconds+finalization_seconds,
                       selected_variant=selected, variants=variants, **quality)
            config['history'].append(row)
            if row['passes']:
                times, decisions = [], []
                for queries in query_batches(query_args):
                    start = perf_counter()
                    qm, qv = field.predict_gpu(queries)
                    probability = gpu_ndtr((level-qm)/cp.sqrt(qv))
                    if float(cp.min(qv)) <= 0 or not bool(cp.isfinite(probability).all()):
                        raise RuntimeError('SDD query batch has invalid variance/probability')
                    scores = (4*probability*(1-probability)).reshape(-1, 64).mean(axis=1)
                    winner = int(cp.argmax(scores)); cp.sum(probability).item()
                    cp.cuda.get_current_stream().synchronize()
                    times.append(perf_counter()-start)
                    decisions.append(dict(winner=winner, score=float(scores[winner])))
                row.update(query_seconds=times, total_seconds=row['preparation_seconds']+sum(times))
                workload = dict(arguments=dict(a, queries=args.queries, batches=args.batches, method='sdd'),
                                level=level, field=save_field(field), decisions=decisions,
                                total_seconds=row['total_seconds'], preparation_seconds=row['preparation_seconds'],
                                query_seconds=times, quality=row, source_sha256=record['source_sha256'])
                passing = out.with_name(out.stem+f'-eta{eta}-passing.json')
                passing.write_text(json.dumps(workload, indent=2)+'\n')
                config['passing_workload'] = str(passing)
            out.write_text(json.dumps(record, indent=2)+'\n')
            print(eta, json.dumps(row), flush=True)
            if row['passes'] or elapsed >= args.budget or not np.isfinite(z).all():
                break
        config['status'] = ('passing complete workload' if row['passes'] else
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
    parser.add_argument('--queries', type=int, default=65536)
    parser.add_argument('--batches', type=int, default=16)
    arguments = parser.parse_args()
    if arguments.queries <= 0 or arguments.queries % 64 or arguments.batches < 1:
        parser.error('queries must be a positive multiple of 64 and batches positive')
    if any(t <= 0 for t in arguments.checkpoints) or any(b <= a for a, b in zip(arguments.checkpoints, arguments.checkpoints[1:])):
        parser.error('checkpoints must be positive and strictly increasing')
    run(arguments)
