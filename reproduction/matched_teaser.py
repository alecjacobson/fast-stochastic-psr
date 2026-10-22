"""Matched analytic-GP scorpion comparisons against the released GeoSPSR API.

Unlike million_teaser.py, this retains the full analytic training covariance
and computes deterministic marginal variances. No posterior Monte Carlo draws.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import cupy as cp
import cupyx.scipy.linalg as cla
import igl
import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import ndtr

from geospsr.spectral import FourierPoisson

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = Path(os.environ.get('GEOSPSR_RUNTIME', '/opt/geospsr-runtime')) / 'upstream'
spec = importlib.util.spec_from_file_location('released_geospsr', UPSTREAM / 'geospsr/__init__.py',
                                            submodule_search_locations=[str(UPSTREAM / 'geospsr')])
released = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = released
spec.loader.exec_module(released)
from released_geospsr import gp as original
from released_geospsr.kernels import Matern32Kernel, ProductKernel
from released_geospsr.utils import periodic_stationary_interpolator, trunc_Zd
sys.path.insert(0, str(ROOT / 'experiments'))
from utils import normalize_points, simulate_scan


def synchronize():
    cp.cuda.runtime.deviceSynchronize()


def measure(fn):
    synchronize()
    start = time.perf_counter()
    value = fn()
    jax.block_until_ready(value)
    synchronize()
    return value, time.perf_counter() - start


def scan(density):
    read = getattr(igl, 'read_obj', None) or igl.readOBJ
    v, _, _, f, _, _ = read(str(ROOT / 'experiments/meshes/scorpion.obj'))
    v = normalize_points(v) * np.pi
    points, normals = simulate_scan(v, f, np.array([[0, -2, 0], [1, 1, 1]]) * np.pi,
                                   np.array([[0, 1, 0], [-1, -1, -1]]) * np.pi / 2,
                                   np.pi / 3, density)
    clean = points.copy()
    rng = np.random.default_rng(0)
    points += rng.standard_normal(points.shape) * .02
    normals += rng.standard_normal(normals.shape) * .02
    return points.astype('f'), normals.astype('f'), clean, v, f


def queries(resolution):
    axis = np.linspace(-np.pi / 2, np.pi / 2, resolution, dtype='f')
    xx, zz = np.meshgrid(axis, axis, indexing='xy')
    heights = np.array([-.35, -.75], dtype='f')
    q = np.concatenate([np.stack([xx.ravel(), np.full(xx.size, h), zz.ravel()], -1) for h in heights])
    return q, axis, heights


def own_jax_array(array):
    """Copy a JAX result before its allocation can be recycled on another stream."""
    array.block_until_ready()
    owned = cp.from_dlpack(array).copy()
    cp.cuda.get_current_stream().synchronize()
    return owned


def conditional_variance(chol, prior, q, cross_at, batch=64, progress=None):
    """Exact diagonal: prior - ||L^{-1} k_vf||², streamed over queries."""
    values = []
    for i in range(0, len(q), batch):
        cross = cross_at(q[i:i + batch])
        rhs = cross.transpose(1, 0, 2).reshape(len(chol), -1)
        z = cla.solve_triangular(chol, rhs, lower=True)
        values.append(prior - (z * z).sum(0).reshape(-1, 3).sum(1))
        if progress and i % 2048 == 0:
            synchronize()
            progress(i)
    return cp.concatenate(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=['A', 'B'], default='A')
    parser.add_argument('--density', type=int, default=200)
    parser.add_argument('--resolution', type=int, default=100)
    parser.add_argument('--backend', choices=['original', 'nufft', 'compatible'], required=True)
    parser.add_argument('--batch', type=int, default=1024)
    parser.add_argument('--interpolant-cutoff', type=int, default=100)
    parser.add_argument('--literal-api', action='store_true')
    parser.add_argument('--reuse-literal', action='store_true')
    args = parser.parse_args()
    out = ROOT / 'reproduction/results/matched' / args.case
    out.mkdir(parents=True, exist_ok=True)
    x, normals, clean, v, faces = scan(args.density)
    q, axis, heights = queries(args.resolution)
    np.savez_compressed(out / 'input.npz', points=x, normals=normals, clean=clean,
                        vertices=v, faces=faces, query=q, axis=axis, heights=heights)
    meta = {'args': vars(args), 'N': len(x), 'M': len(q), 'device': cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
            'points_sha256': hashlib.sha256(x.tobytes()).hexdigest(),
            'normals_sha256': hashlib.sha256(normals.tobytes()).hexdigest(),
            'queries_sha256': hashlib.sha256(q.tobytes()).hexdigest(),
            'upstream_commit': '2352f1ef98aab20e0303611c1fa2b0948acd2a05',
            'noise_variance': .02, 'coordinate_and_normal_noise_std': .02,
            'lengthscale': .01, 'per_axis_variance': .1, 'cutoff': 50,
            'training': 'full analytic product-Matern, identical to released Cholesky API',
            'variance': 'deterministic diagonal of conditioned covariance, no Monte Carlo',
            'timing': 'synchronized GPU wall times; first invocation and repeated stage measured separately',
            'seconds': {}}
    if args.reuse_literal:
        prior_meta = json.loads((out / 'original.json').read_text())
        assert prior_meta['points_sha256'] == meta['points_sha256'] and prior_meta['M'] == len(q)
        meta['seconds'].update(prior_meta['seconds'])
    def save():
        (out / (args.backend + '.json')).write_text(json.dumps(meta, indent=2))
    def timed(name, fn):
        value, seconds = measure(fn)
        meta['seconds'][name] = seconds
        save()
        print(name, seconds, flush=True)
        return value
    print(args.case, args.backend, 'N', len(x), 'M', len(q), flush=True)
    k = Matern32Kernel(.01, .1)
    kv = ProductKernel(k, k, k)
    xj, yj, qj = map(jnp.asarray, (x, normals, q))
    op = FourierPoisson()
    norm2 = (op.ns ** 2).sum(0)
    prior = float(cp.sum(cp.where(norm2 > 0, op.rho / cp.where(norm2 > 0, norm2, 1), 0)))
    meta['prior_variance'] = prior
    if args.backend == 'original':
        from functools import partial
        freq = trunc_Zd(50)
        ev = jax.vmap(kv.spectral_density)(freq) ** .5
        cross = partial(original.poisson_cross_covariances, eigenvectors=freq, eigenvalues=ev)
        kfv = timed('kernel_setup_first', lambda: periodic_stationary_interpolator(cross, 3, 50, exponent=5))
        if args.literal_api and not args.reuse_literal:
            offset = timed('data_mean_first', lambda: original.compute_mean(xj, xj, yj, kv, kfv, .02, bs=args.batch)).mean()
            mean = timed('mean_first', lambda: original.compute_mean(qj, xj, yj, kv, kfv, .02, bs=args.batch)) - offset
            mean = timed('mean_repeat', lambda: original.compute_mean(qj, xj, yj, kv, kfv, .02, bs=args.batch)) - offset
            var = timed('variance_first', lambda: original.compute_variance(qj, xj, kv, kfv, prior, .02, bs=args.batch))
            var = timed('variance_repeat', lambda: original.compute_variance(qj, xj, kv, kfv, prior, .02, bs=args.batch))
            np.savez_compressed(out / 'literal_api.npz', mean=np.asarray(mean), variance=np.asarray(var), offset=float(offset))
        # Stronger baseline: same released kernels, one cached factorization.
        cross_rows = jax.jit(jax.vmap(jax.vmap(kfv, (None, 0)), (0, None)))
    def fit():
        K = jax.jit(jax.vmap(jax.vmap(kv, (None, 0)), (0, None)))(xj, xj)
        Kcp = own_jax_array(K)
        Kcp.flat[::len(x)+1] += .02
        L = cp.linalg.cholesky(Kcp)
        alpha = cla.solve_triangular(L.T, cla.solve_triangular(L, cp.asarray(normals), lower=True), lower=False)
        return L, alpha
    L, alpha = timed('fit_first', fit)
    if args.backend != 'original':
        if args.backend == 'compatible':
            from reproduction.interpolated_nufft import InterpolatedFourierPoisson
            op = timed('interpolant_spectrum_setup', lambda: InterpolatedFourierPoisson(args.interpolant_cutoff))
            meta['cross_kernel'] = 'Fourier representation of released 50^3 trilinear interpolant'
            meta['interpolant_cutoff'] = args.interpolant_cutoff
        def mean_at(points):
            coeff = op.aggregate(x, alpha)
            return op.evaluate(points, coeff)[0]
        offset = timed('offset_first', lambda: mean_at(x).mean())
        mean = timed('mean_first', lambda: mean_at(q)) - offset
        warm = []
        for i in range(3):
            mean = timed('mean_repeat_' + str(i), lambda: mean_at(q)) - offset
            warm.append(meta['seconds']['mean_repeat_' + str(i)])
        meta['mean_warm_median'] = float(np.median(warm))
        def cross_at(query):
            diff = (cp.asarray(query)[:, None] - cp.asarray(x)[None]).reshape(-1, 3)
            return op.cross_values(diff).reshape(len(query), len(x), 3)
    else:
        def cross_at(query):
            z = cross_rows(jnp.asarray(query), xj)
            return own_jax_array(z)
        def mean_at(points):
            return cp.concatenate([(cross_at(points[i:i+args.batch]) * alpha).sum((1, 2))
                                   for i in range(0, len(points), args.batch)])
        offset = timed('offset_first', lambda: mean_at(x).mean())
        mean = timed('cached_mean_first', lambda: mean_at(q)) - offset
        warm = []
        for i in range(3):
            mean = timed('cached_mean_repeat_' + str(i), lambda: mean_at(q)) - offset
            warm.append(meta['seconds']['cached_mean_repeat_' + str(i)])
        meta['mean_warm_median'] = float(np.median(warm))
    def variance():
        return conditional_variance(L, prior, q, cross_at, args.batch,
                                    lambda i: print('variance queries', i, '/', len(q), flush=True))
    var = timed('cached_variance_first', variance)
    var = timed('cached_variance_repeat', variance)
    mean, var = cp.asnumpy(mean), cp.asnumpy(var)
    assert np.isfinite(mean).all() and np.isfinite(var).all() and var.min() > 0
    prob = ndtr(-mean / np.sqrt(var))
    meta['ranges'] = {key: [float(a.min()), float(a.max())] for key, a in [('mean', mean), ('variance', var), ('probability', prob)]}
    if args.backend == 'original' and args.literal_api:
        literal = np.load(out / 'literal_api.npz')
        meta['cached_vs_literal_api'] = {key: {'max_abs': float(np.max(np.abs(a - literal[key]))),
                                              'relative_l2': float(np.linalg.norm(a - literal[key]) / np.linalg.norm(literal[key]))}
                                          for key, a in [('mean', mean), ('variance', var)]}
        assert all(v['relative_l2'] < 3e-6 for v in meta['cached_vs_literal_api'].values()), meta['cached_vs_literal_api']
    shape = (2, args.resolution, args.resolution)
    np.savez_compressed(out / (args.backend + '.npz'), mean=mean.reshape(shape), variance=var.reshape(shape),
                        probability=prob.reshape(shape), offset=float(offset), prior=prior)
    save()
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == '__main__':
    main()
