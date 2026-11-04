"""Reproducible CPU rank/variance pilot; dense setup is intentionally explicit.

Run with OPENBLAS_NUM_THREADS=1 python -m reproduction.nonperiodic_pilot.
Use --original to also run gpytoolbox's unmodified full-covariance SPSR.
"""
import argparse
import hashlib
import importlib.metadata
import inspect
import itertools
import json
import os
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
from scipy.linalg import svdvals
from scipy.special import ndtr

from geospsr.nonperiodic import (
    DenseFreeSpacePosterior, FreeSpacePoissonMatern, RegionalVarianceCache,
)


def sphere_cloud(n, seed):
    rng = np.random.default_rng(seed)
    normals = rng.normal(size=(n, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return normals.copy(), normals


def original_reference(out):
    import gpytoolbox as gpy
    function = gpy.stochastic_poisson_surface_reconstruction
    package = Path(inspect.getfile(function)).parent
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(package.glob('*.py'))}
    x, normals = sphere_cloud(128, 415)
    x *= .7
    query = np.column_stack((np.linspace(-1.2, 1.2, 121), np.zeros((121, 2))))
    rows = []
    for grid_size in [8, 12, 16]:
        gs = np.full(3, grid_size)
        h = np.full(3, 3./(grid_size-1))
        corner = np.full(3, -1.5)
        start = perf_counter()
        mean, variance, _ = function(
            x, normals, gs=gs, h=h, corner=corner,
            output_variance=True, solve_subspace_dim=0, sigma_n=.01, sigma=.05,
        )
        elapsed = perf_counter()-start
        interpolation = gpy.fd_interpolate(query, gs=gs, h=h, corner=corner)
        mq = np.asarray(interpolation @ mean).ravel()
        vq = np.asarray(interpolation @ variance).ravel()
        rows.append(dict(grid_size=grid_size, grid_unknowns=grid_size**3,
                         seconds=elapsed, h=h.tolist(), corner=corner.tolist(),
                         minimum_variance=float(np.min(variance)),
                         maximum_variance=float(np.max(variance)),
                         ray_mean=mq.tolist(), ray_variance=vq.tolist(),
                         ray_inside_probability=ndtr(-mq/np.sqrt(vq)).tolist()))
        print(f'original full covariance {grid_size}³: {elapsed:.3f}s', flush=True)
    result = dict(gpytoolbox_version=importlib.metadata.version('gpytoolbox'),
                  source_sha256=hashes, n=128, seed=415, sphere_radius=.7,
                  sigma_n=.01, sigma=.05, solve_subspace_dim=0,
                  source_url='https://gpytoolbox.org/latest/stochastic_poisson_surface_reconstruction/',
                  query=query.tolist(), rows=rows,
                  interpretation='Unmodified published implementation; grid changes also change its kernel width. Not a same-model convergence test or a calibrated probability benchmark.')
    (out/'original_reference.json').write_text(json.dumps(result, indent=2)+'\n')


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    metadata = dict(python=platform.python_version(), numpy=np.__version__,
                    scipy=importlib.metadata.version('scipy'),
                    platform=platform.platform(), processor=platform.processor(),
                    openblas_num_threads=os.environ.get('OPENBLAS_NUM_THREADS'),
                    seed=args.seed, lengthscale=args.lengthscale,
                    near_halfwidth=args.near_halfwidth,
                    amplitude=1., noise_variance=args.noise_variance,
                    n=args.n, orders=args.orders, heldout_queries=args.queries+14,
                    uniform_random_queries=args.queries, boundary_queries=14,
                    source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in [Path('geospsr/nonperiodic.py'), Path(__file__).relative_to(Path.cwd())]},
                    variance_gate_prior_relative=1e-4,
                    geometry='nested uniform samples of the unit sphere, fixed domain',
                    probability_gate=.001,
                    probability_interpretation='CDF at the fixed scalar threshold zero; numerical diagnostic only, not calibrated occupancy')
    (out/'environment.json').write_text(json.dumps(metadata, indent=2)+'\n')
    model = FreeSpacePoissonMatern(args.lengthscale)
    all_x, all_y = sphere_cloud(max(args.n), args.seed)
    regions = [('separated', [2., 0, 0], .25),
               ('near_surface', [1., 0, 0], args.near_halfwidth),
               ('whole_object', [0., 0, 0], 1.25)]
    rng = np.random.default_rng(args.seed+1)
    unit_q = np.vstack([rng.uniform(-1, 1, (args.queries, 3)),
                        np.array(list(itertools.product([-1., 1.], repeat=3))),
                        np.eye(3), -np.eye(3)])
    with (out/'regional.jsonl').open('w') as f:
        for n in args.n:
            start = perf_counter()
            posterior = DenseFreeSpacePosterior(model, all_x[:n], all_y[:n], args.noise_variance)
            setup_seconds = perf_counter()-start
            for name, center, halfwidth in regions:
                q = np.array(center) + halfwidth*unit_q
                start = perf_counter()
                mean_ref, var_ref = posterior.predict(q)
                direct_seconds = perf_counter()-start
                w_query = posterior.whitened_cross(q)
                for order in args.orders:
                    start = perf_counter()
                    cache = RegionalVarianceCache(posterior, center, halfwidth, order)
                    cache_seconds = perf_counter()-start
                    # Best of three warmed timings; construction and diagnostics excluded.
                    online_times = []
                    for _ in range(3):
                        start = perf_counter()
                        mean, var = cache.predict(q)
                        online_times.append(perf_counter()-start)
                    phi = cache.basis(q)
                    w_nodes = posterior.whitened_cross(cache.nodes)
                    w_approx = w_nodes @ phi.T
                    en = np.linalg.norm(w_query-w_approx, axis=0)
                    bound = 2*np.linalg.norm(w_approx, axis=0)*en + en*en
                    singular = svdvals(w_nodes)
                    # Spectral relative rank of a dense diagnostic, not a
                    # claimed out-of-sample variance-accuracy guarantee.
                    ranks = {str(tol): int(np.count_nonzero(singular > singular[0]*tol))
                             for tol in (1e-2, 1e-4, 1e-6)}
                    max_var_error = float(np.max(np.abs(var-var_ref)))
                    valid = bool(np.all(var > 0) and np.all(var_ref > 0))
                    probability_error = (float(np.max(np.abs(
                        ndtr(-mean/np.sqrt(var))-ndtr(-mean_ref/np.sqrt(var_ref)))))
                        if valid else None)
                    row = dict(n=n, region=name, center=center, halfwidth=halfwidth,
                               order=order, basis_size=order**3,
                               dense_setup_seconds=setup_seconds,
                               direct_queries_seconds=direct_seconds,
                               cache_setup_seconds=cache_seconds,
                               cache_queries_seconds=min(online_times),
                               cache_bytes=cache.correction.nbytes+cache.mean_coefficients.nbytes,
                               whitened_node_spectral_ranks=ranks,
                               max_mean_error=float(np.max(np.abs(mean-mean_ref))),
                               max_variance_error=max_var_error,
                               max_variance_error_over_prior=max_var_error/model.prior_variance,
                               max_variance_relative_error=float(np.max(np.abs(var-var_ref)/var_ref)),
                               min_reference_variance=float(var_ref.min()),
                               min_cache_variance=float(var.min()),
                               max_energy_error_bound=float(bound.max()),
                               max_energy_bound_violation=float(np.max(np.abs(var-var_ref)-bound)),
                               max_zero_threshold_cdf_error=probability_error,
                               passes=bool(valid and max_var_error/model.prior_variance <= 1e-4
                                           and probability_error <= .001))
                    f.write(json.dumps(row)+'\n'); f.flush()
                    print(f'N={n} {name} order={order}: var/prior error={row["max_variance_error_over_prior"]:.3g}, pass={row["passes"]}', flush=True)
    if args.original:
        original_reference(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='reproduction/results/nonperiodic/pilot')
    parser.add_argument('--n', nargs='+', type=int, default=[128, 256, 512, 1024])
    parser.add_argument('--orders', nargs='+', type=int, default=[3, 5, 7])
    parser.add_argument('--queries', type=int, default=256)
    parser.add_argument('--seed', type=int, default=414)
    parser.add_argument('--lengthscale', type=float, default=.3)
    parser.add_argument('--noise-variance', type=float, default=.01)
    parser.add_argument('--near-halfwidth', type=float, default=.25)
    parser.add_argument('--original', action='store_true')
    run(parser.parse_args())
