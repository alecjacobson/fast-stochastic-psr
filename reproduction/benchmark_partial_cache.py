"""Whole-object tiled scalar-contact cache with a shared conditioning backend.

Small-N exact query checks and full-query comparisons are mandatory evidence;
timing alone is not an accepted accuracy result. This is a marginal benchmark.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.special import ndtr

from geospsr.nonperiodic import FreeSpacePoissonMatern, DenseFreeSpacePosterior
from geospsr.scalar_contacts import conditioned_regional_cache, ContactConditionedDensePosterior
from geospsr.regional_field import PolynomialMomentField
from reproduction.benchmark_hierarchical_variance import save_field
from reproduction.partial_occupancy import ROOT, load, stratified
from reproduction.stanford_partial_data import CACHE, digest


def run(args):
    x, vectors, contacts, panel = load(args.group, args.n)
    selected = json.loads((ROOT/'selected.json').read_text())
    model = FreeSpacePoissonMatern(selected['ell'], 1/3); noise = selected['sigma']**2
    mesh = np.load(CACHE/'reference.npz')['vertices']
    lower, upper = mesh.min(0)-.15, mesh.max(0)+.15
    step = (upper-lower)/args.tiles
    ids = np.minimum(((panel['queries']-lower)/step).astype(int), args.tiles-1)
    # Every region is prepared, independent of test labels or query occupancy.
    records, fields = [], []
    mean, variance = np.empty(len(ids)), np.empty(len(ids))
    start = perf_counter()
    if args.method == 'cpu-dense':
        from scipy.linalg import cho_solve
        system = DenseFreeSpacePosterior(model, x, vectors, noise)
        project = lambda u, y: (u.T@cho_solve((system.factor, True), u), u.T@system.alpha)
    else:
        import cupy as cp
        cp.cuda.Device().synchronize()
        if args.method == 'dense':
            from geospsr.nonperiodic_dense_gpu import GPUDenseMaternSystem
            system = GPUDenseMaternSystem(model, x, noise, dtype='float64'); project = system.project
        else:
            if args.method == 'h2':
                from geospsr.hierarchical import HierarchicalMaternSystem
                from geospsr.hierarchical_gpu import GPUHierarchicalSystem
                native = HierarchicalMaternSystem(model, x, noise, tolerance=args.h2_tolerance,
                    leaf_size=40, preconditioner_rank=50, neighbors=40, threads=4,
                    seed=9101, preconditioner_backend='prefix-kd')
                system = GPUHierarchicalSystem(native, interaction_budget_bytes=256*2**20,
                                               rhs_block_size=72, release_native=True)
            else:
                from geospsr.nonperiodic_sparse import GPUSparseMaternSystem
                system = GPUSparseMaternSystem(model, x, noise, cutoff=args.cutoff,
                    rank=50, neighbors=40, threads=4, seed=9101, graph_backend='gpu',
                    preconditioner_backend='prefix-kd', rhs_block_size=72)
            project = lambda u, y: system.project(u, y, variance_iterations=args.iterations, mean_tolerance=1e-10)
    setup = perf_counter()-start
    print('setup', setup, flush=True)
    cache_start = perf_counter()
    for index in np.ndindex((args.tiles,)*3):
        begin = perf_counter(); center = lower+(np.array(index)+.5)*step
        cache = conditioned_regional_cache(model, x, vectors, contacts, np.zeros(len(contacts)), .01**2,
            project, center, step/2, order=args.order, compression_tolerance=args.cross_tolerance)
        field = PolynomialMomentField(cache)
        fields.append(field)
        records.append(dict(index=index, seconds=perf_counter()-begin, rank=cache.solve_rank,
                            solve_info=getattr(system, 'last_solve', None)))
        print('region', index, 'rank', cache.solve_rank, 'seconds', records[-1]['seconds'], flush=True)
    preparation = perf_counter()-cache_start
    query_start = perf_counter()
    for index, field in zip(np.ndindex((args.tiles,)*3), fields):
        mask = (ids == index).all(axis=1)
        mean[mask], variance[mask] = field.predict(panel['queries'][mask])
    query_seconds = perf_counter()-query_start
    valid = bool(np.isfinite(mean).all() and np.isfinite(variance).all() and np.all(variance > 0))
    result = dict(arguments=vars(args), setup_seconds=setup, cache_seconds=preparation,
                  finite_panel_query_seconds=query_seconds, finite_panel_total_seconds=setup+preparation+query_seconds,
                  records=records, quality=stratified(mean, variance, panel) if valid else None,
                  valid_moments=valid, minimum_panel_variance=float(variance.min()),
                  field_box=[lower.tolist(), upper.tolist()],
                  stored_backend_bytes=getattr(system, 'device_matrix_bytes', getattr(system, 'factor_bytes', None)),
                  interaction_entries=getattr(system, 'interaction_nnz', None),
                  source_sha256={p: digest(p) for p in [__file__, 'geospsr/scalar_contacts.py',
                      'geospsr/nonperiodic.py', 'geospsr/hierarchical_gpu.py', 'geospsr/regional_field.py']})
    if args.method != 'cpu-dense' and valid:
        from cupyx.scipy.special import ndtr as gpu_ndtr
        # Warm shared query kernels before timing; field uploads still included.
        fields[0].prepare_gpu(); cp.cuda.Device().synchronize()
        rng = np.random.default_rng(9201)
        queries = rng.uniform(lower, upper, (args.queries, 3))
        begin = perf_counter()
        qids = np.minimum(((queries-lower)/step).astype(int), args.tiles-1)
        checksum = 0.; minimum = np.inf
        for index, field in zip(np.ndindex((args.tiles,)*3), fields):
            mask = (qids == index).all(axis=1)
            m, v = field.predict_gpu(queries[mask])
            if len(v): minimum = min(minimum, float(v.min().item()))
            checksum += float(gpu_ndtr(-m/cp.sqrt(v)).sum().item())
        cp.cuda.Device().synchronize()
        if minimum <= 0 or not np.isfinite(checksum): raise RuntimeError('invalid query probabilities')
        result.update(million_query_seconds=perf_counter()-begin, queries=args.queries,
                      probability_checksum=checksum, minimum_query_variance=minimum)
        result['total_seconds'] = setup+preparation+result['million_query_seconds']
    exact_path = ROOT/'exact'/f'{args.group}-{args.n}.npz'
    if exact_path.exists():
        ref = np.load(exact_path)
        result['exact_panel_error'] = dict(mean=float(np.max(abs(mean-ref['mean']))),
            variance_over_prior=float(np.max(abs(variance-ref['variance']))/model.prior_variance),
            cdf=float(np.max(abs(ndtr(-mean/np.sqrt(variance))-ndtr(-ref['mean']/np.sqrt(ref['variance']))))) if valid else None)
        result['passes_exact_panel'] = (valid and result['exact_panel_error']['variance_over_prior'] <= 1e-4
                                       and result['exact_panel_error']['cdf'] <= 1e-3)
    else:
        result['passes_exact_panel'] = None
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output.with_suffix('.npz'), mean=mean, variance=variance)
    result['fields'] = [save_field(f) for f in fields]
    output.write_text(json.dumps(result, indent=2)+'\n')
    print('RESULT', {k:v for k,v in result.items() if k in ['setup_seconds','cache_seconds','total_seconds','exact_panel_error','passes_exact_panel']}, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--group', default='test_side'); p.add_argument('--n', type=int, default=768)
    p.add_argument('--method', choices=['cpu-dense', 'dense', 'h2', 'locality'], default='cpu-dense')
    p.add_argument('--tiles', type=int, default=4); p.add_argument('--order', type=int, default=4)
    p.add_argument('--queries', type=int, default=1000000)
    p.add_argument('--h2-tolerance', type=float, default=1e-7)
    p.add_argument('--cross-tolerance', type=float, default=1e-9)
    p.add_argument('--iterations', type=int, default=32); p.add_argument('--cutoff', type=float, default=12.)
    p.add_argument('--output', required=True)
    run(p.parse_args())
