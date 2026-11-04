"""End-to-end non-periodic contact-region mean/variance benchmark.

Each method gets the same cross compression and scalar-polynomial query cache.
The known contact point fixes a deterministic posterior-mean isovalue; variance
is unchanged. Probabilities are marginal occupancy predictions, not joint events.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import itertools
import json
import platform
import resource
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.special import ndtr

from geospsr.nonperiodic import FreeSpacePoissonMatern, RegionalVarianceCache
from geospsr.regional_field import PolynomialMomentField


def data(n, seed, geometry):
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(n, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    axes = np.array([1., .8, .6]) if geometry == 'ellipsoid' else np.ones(3)
    points = directions*axes
    vectors = directions/axes
    vectors /= np.linalg.norm(vectors, axis=1)[:, None]
    return points, vectors, axes


def query_batches(args):
    # New candidate trajectories in every batch, all inside the stated box.
    rng = np.random.default_rng(args.seed+100)
    for _ in range(args.batches):
        trajectories = args.queries//64
        yz = rng.uniform(-args.halfwidth, args.halfwidth, (trajectories, 2))
        q = np.empty((trajectories, 64, 3))
        q[:, :, 0] = 1+np.linspace(args.halfwidth, -args.halfwidth, 64)
        q[:, :, 1:] = yz[:, None, :]
        yield q.reshape(-1, 3)


def save_field(field):
    return dict(center=field.center.tolist(), halfwidth=field.halfwidth.tolist(),
                prior_variance=field.prior_variance, order=field.order,
                mean=field.mean.tolist(), correction=field.correction.tolist())


def read_field(record):
    field = PolynomialMomentField.__new__(PolynomialMomentField)
    for key, value in record.items():
        setattr(field, key, np.asarray(value) if key in ['center', 'halfwidth', 'mean', 'correction'] else value)
    field._gpu = None
    return field


def run(args):
    import cupy as cp
    from cupyx.scipy.special import ndtr as gpu_ndtr
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    runtime_start = perf_counter()
    # Warm the identical query/CDF pipeline for every backend. Algorithm-specific
    # setup and first solver actions remain inside preparation timings.
    warm = PolynomialMomentField.__new__(PolynomialMomentField)
    warm.center, warm.halfwidth = np.array([1., 0, 0]), np.full(3, args.halfwidth)
    warm.order, warm.prior_variance = 2*args.order-1, 1.
    warm.mean = np.zeros((warm.order,)*3); warm.correction = np.zeros_like(warm.mean)
    warm._gpu = None
    wm, wv = warm.predict_gpu(np.tile([1., 0, 0], (args.queries, 1)))
    wp = gpu_ndtr(wm/cp.sqrt(wv))
    ws = (4*wp*(1-wp)).reshape(-1, 64).mean(axis=1)
    cp.argmax(ws).item(); cp.sum(wp).item()
    runtime_seconds = perf_counter()-runtime_start
    del warm, wm, wv, wp, ws
    cp.get_default_memory_pool().free_all_blocks()
    x, vectors, axes = data(args.n, args.seed, args.geometry)
    model = FreeSpacePoissonMatern(args.lengthscale, args.amplitude)
    noise = args.amplitude*args.relative_noise
    provenance = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [
        'geospsr/hierarchical.py', 'geospsr/hierarchical_gpu.py',
        'geospsr/nonperiodic.py', 'geospsr/nonperiodic_dense_gpu.py',
        'geospsr/nonperiodic_dense_cg.py', 'geospsr/native/higp_snapshot.inc',
        'reproduction/setup_higp.py',
        'geospsr/regional_field.py', 'reproduction/benchmark_hierarchical_variance.py']}
    record = dict(arguments=vars(args), source_sha256=provenance,
                  python=platform.python_version(), numpy=np.__version__,
                  scipy=importlib.metadata.version('scipy'), cupy=cp.__version__,
                  device=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
                  device_memory_bytes=cp.cuda.runtime.memGetInfo()[1],
                  cpu_quota=Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
                  host_memory_limit=Path('/sys/fs/cgroup/memory.max').read_text().strip(),
                  native_build=json.loads((Path(importlib.util.find_spec('psr_higp_cext').origin).parent/'psr_higp_build.json').read_text()),
                  model='nonperiodic independent vector Matern-3/2, Poisson scalar projection',
                  application='marginal contact-ray probabilities and maximum mean Bernoulli ambiguity',
                  level_policy='posterior mean at the known surface contact (1,0,0); variance unchanged')
    record['common_runtime_initialization_seconds'] = runtime_seconds
    begin = perf_counter()
    if args.method.startswith('dense'):
        from geospsr.nonperiodic_dense_gpu import GPUDenseMaternSystem
        system = GPUDenseMaternSystem(model, x, noise, dtype=args.method.split('-')[1])
        project = system.project
    elif args.method.startswith('cg-dense'):
        from geospsr.nonperiodic_dense_cg import GPUDenseCGSystem
        system = GPUDenseCGSystem(model, x, noise, dtype=args.method.split('-')[-1],
                                  rank=args.rank, neighbors=args.neighbors, threads=args.threads, seed=args.seed)
        project = (lambda u, y: system.project(u, y, variance_iterations=args.variance_iterations,
                                               mean_tolerance=args.solve_tolerance)) if args.variance_iterations else None
    else:
        from geospsr.hierarchical import HierarchicalMaternSystem
        system = HierarchicalMaternSystem(model, x, noise, tolerance=args.h2_tolerance,
                                          preconditioner_rank=args.rank, neighbors=args.neighbors,
                                          leaf_size=args.leaf_size, threads=args.threads, seed=args.seed)
        if args.method == 'h2-gpu':
            from geospsr.hierarchical_gpu import GPUHierarchicalSystem
            system = GPUHierarchicalSystem(system)
        project = None
        if args.method == 'h2-gpu' and args.variance_iterations:
            project = lambda u, y: system.project(u, y, variance_iterations=args.variance_iterations,
                                                  mean_tolerance=args.solve_tolerance)
    cp.cuda.get_current_stream().synchronize()
    setup = perf_counter()-begin
    start = perf_counter()
    solve = (system.solve if args.method.startswith('dense') else
             lambda b: system.solve(b, tolerance=args.solve_tolerance))
    cache = RegionalVarianceCache.from_solver(model, x, vectors, solve, [1, 0, 0],
                                              args.halfwidth, args.order,
                                              compression_tolerance=args.cross_tolerance,
                                              project=project)
    cp.cuda.get_current_stream().synchronize()
    cache_seconds = perf_counter()-start
    start = perf_counter()
    field = PolynomialMomentField(cache)
    field.prepare_gpu()
    level = float(field.predict([[1, 0, 0]])[0][0])
    field_seconds = perf_counter()-start
    preparation = perf_counter()-begin
    decisions, times, checksums, minimum_variances = [], [], [], []
    for q in query_batches(args):
        start = perf_counter()
        mean, variance = field.predict_gpu(q)
        probability = gpu_ndtr((level-mean)/cp.sqrt(variance))
        minimum = float(cp.min(variance).item())
        if minimum <= 0 or not bool(cp.isfinite(probability).all()):
            raise RuntimeError('nonpositive variance or invalid application probability')
        scores = (4*probability*(1-probability)).reshape(-1, 64).mean(axis=1)
        winner = int(cp.argmax(scores).item())
        checksum = float(cp.sum(probability).item())
        cp.cuda.get_current_stream().synchronize()
        times.append(perf_counter()-start)
        decisions.append(dict(winner=winner, score=float(scores[winner].item())))
        checksums.append(checksum)
        minimum_variances.append(minimum)
    record.update(setup_seconds=setup, regional_cache_seconds=cache_seconds,
                  field_seconds=field_seconds, preparation_seconds=preparation,
                  query_seconds=times, total_seconds=preparation+sum(times),
                  solve_rank=cache.solve_rank, cross_relative_frobenius_error=cache.cross_relative_frobenius_error,
                  level=level, decisions=decisions, probability_checksums=checksums,
                  minimum_batch_variances=minimum_variances,
                  field=save_field(field), peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                  cupy_pool_total_bytes=cp.get_default_memory_pool().total_bytes())
    if args.method.startswith('dense'):
        record.update(factor_bytes=system.factor_bytes, factor_workspace_bytes=system.factor_workspace_bytes)
    else:
        record['solve_info'] = system.last_solve
        if args.method == 'h2-gpu':
            record.update(device_matrix_bytes=system.device_matrix_bytes,
                          max_h2_rank=system.max_rank, sum_h2_ranks=int(system.ranks.sum()),
                          interaction_nnz=system.interaction_nnz)
        elif args.method.startswith('cg-dense'):
            record.update(device_matrix_bytes=system.device_matrix_bytes, matrix_cached=system.matrix_cached)
    out.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({k: record[k] for k in ['total_seconds', 'preparation_seconds', 'setup_seconds',
                                             'regional_cache_seconds', 'solve_rank']}), flush=True)
    # Numerical oracle checks are outside timing and reported separately.
    if args.validate:
        validation_start = perf_counter()
        rng = np.random.default_rng(args.seed+200)
        q = [1, 0, 0]+args.halfwidth*np.vstack((rng.uniform(-1, 1, (128, 3)),
                         np.array(list(itertools.product([-1., 1.], repeat=3))), np.zeros((1, 3))))
        if not args.method.startswith('dense'):
            del system
            gc.collect(); cp.get_default_memory_pool().free_all_blocks()
            from geospsr.nonperiodic_dense_gpu import GPUDenseMaternSystem
            system = GPUDenseMaternSystem(model, x, noise, dtype='float64' if args.n <= 32768 else 'float32')
        mean_ref, variance_ref = [], []
        # Whiten cross kernels and the three vector observations together.
        from cupyx.scipy.linalg import solve_triangular
        y = solve_triangular(system.factor, cp.asarray(vectors, dtype=system.dtype), lower=True, check_finite=False).astype(cp.float64)
        for start in range(0, len(q), 16):
            c = model.cross_covariance(q[start:start+16], x)
            rhs = c.transpose(1, 2, 0).reshape(len(x), -1)
            w = solve_triangular(system.factor, cp.asarray(rhs, dtype=system.dtype), lower=True, check_finite=False).astype(cp.float64)
            w = w.reshape(len(x), 3, len(c))
            mean_ref.extend(cp.asnumpy(cp.sum(w*y[:, :, None], axis=(0, 1))).tolist())
            variance_ref.extend(cp.asnumpy(model.prior_variance-cp.sum(w*w, axis=(0, 1))).tolist())
        mean_ref, variance_ref = np.array(mean_ref), np.array(variance_ref)
        mean, variance = field.predict(q)
        p_ref = ndtr((mean_ref[-1]-mean_ref)/np.sqrt(variance_ref))
        p = ndtr((level-mean)/np.sqrt(variance))
        errors = dict(max_mean_error=float(np.max(abs(mean-mean_ref))),
                      max_variance_error_over_prior=float(np.max(abs(variance-variance_ref))/model.prior_variance),
                      max_probability_error=float(np.max(abs(p-p_ref))),
                      minimum_variance=float(variance.min()),
                      oracle_dtype=str(system.dtype), seconds=perf_counter()-validation_start)
        errors['provisional'] = system.dtype == np.dtype('float32')
        errors['passes'] = bool(errors['max_variance_error_over_prior'] <= 1e-4 and
                                errors['max_probability_error'] <= 1e-3 and errors['minimum_variance'] > 0)
        record['validation'] = errors
        out.write_text(json.dumps(record, indent=2)+'\n')
        print(json.dumps(errors), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, required=True)
    parser.add_argument('--method', choices=['h2-cpu', 'h2-gpu', 'dense-float32', 'dense-float64',
                                            'cg-dense-float32', 'cg-dense-float64'], required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--geometry', choices=['sphere', 'ellipsoid'], default='ellipsoid')
    parser.add_argument('--lengthscale', type=float, default=.3)
    parser.add_argument('--amplitude', type=float, default=.0025)
    parser.add_argument('--relative-noise', type=float, default=.01)
    parser.add_argument('--h2-tolerance', type=float, default=1e-5)
    parser.add_argument('--solve-tolerance', type=float, default=1e-6)
    parser.add_argument('--variance-iterations', type=int, default=0)
    parser.add_argument('--cross-tolerance', type=float, default=1e-4)
    parser.add_argument('--leaf-size', type=int, default=100)
    parser.add_argument('--rank', type=int, default=50)
    parser.add_argument('--neighbors', type=int, default=40)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--order', type=int, default=4)
    parser.add_argument('--halfwidth', type=float, default=.05)
    parser.add_argument('--queries', type=int, default=65536)
    parser.add_argument('--batches', type=int, default=16)
    parser.add_argument('--validate', action='store_true')
    arguments = parser.parse_args()
    if arguments.queries <= 0 or arguments.queries % 64:
        parser.error('--queries must be a positive multiple of 64')
    run(arguments)
