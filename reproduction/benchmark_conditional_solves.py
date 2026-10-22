"""Shared-RHS dense/SDD/NUFFT conditional solves and paired application fields.

At feasible N, use corrected full-analytic prior draws. At large N the common
prior is explicitly truncated: this checks conditional solves, not full-model
posterior equivalence. SDD shares analytic rows across every sample RHS.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import cupy as cp
import jax
import jax.numpy as jnp
import numpy as np

from geospsr.iterative import StochasticDualDescent
from geospsr.spectral import FourierKernel
from geospsr.training_spectrum import FourierCovariance
from reproduction.matched_teaser import ROOT, measure, own_jax_array, Matern32Kernel, ProductKernel
from reproduction.posterior import Posterior
from reproduction.spectral_posterior import SpectralPosterior
from reproduction.paired_applications import application_geometry, uncertain_objects, events, quality, field_quality


def residual_panel(residual, rhs, full=False):
    """Per-draw residuals; bootstrap columns share the sampled observation rows."""
    r, b = cp.asnumpy(residual).astype('d'), cp.asnumpy(rhs).astype('d')
    errors = (r.reshape(len(r), -1, 3) ** 2).sum(2)
    scales = (b.reshape(len(b), -1, 3) ** 2).sum(2)
    ratios = np.sqrt(errors.sum(0) / scales.sum(0))
    result = {'per_draw_relative_rms': ratios.tolist(), 'max_relative_rms': float(ratios.max()),
              'all_rows': full}
    if not full:
        counts = np.random.default_rng(18).multinomial(len(r), np.full(len(r), 1 / len(r)), size=2000).astype('d')
        boot = np.sqrt((counts @ errors) / (counts @ scales))
        intervals = np.quantile(boot, [.025, .975], axis=0).T
        result.update({'per_draw_bootstrap_95_percent_intervals': intervals.tolist(),
                       'max_upper_endpoint': float(intervals[:, 1].max()),
                       'qualification': 'marginal empirical intervals, not a simultaneous or full-system bound'})
    else:
        result['max_upper_endpoint'] = result['max_relative_rms']
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['A', 'B', '100000', '1000000'], default='B')
    p.add_argument('--samples', type=int, default=128)
    p.add_argument('--cutoff', type=int, default=200)
    p.add_argument('--batch', type=int, default=512, help='SDD analytic row batch shared across all RHS')
    p.add_argument('--eta', type=float, default=5)
    p.add_argument('--chunk', type=int, default=256)
    p.add_argument('--time-budget', type=float, default=600)
    p.add_argument('--rtol', type=float, default=.0005)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    out = a.output or ROOT / 'reproduction/results/conditional_solves' / a.dataset
    out.mkdir(parents=True, exist_ok=True)
    small = a.dataset in ('A', 'B')
    if small:
        d = np.load(ROOT / 'reproduction/results/matched' / a.dataset / 'input.npz')
        x, y = d['points'], d['normals']
    else:
        d = np.load(ROOT / 'reproduction/results/teaser/cloud_1000000.npz')
        ids = np.random.default_rng(19).permutation(len(d['points']))[:int(a.dataset)]
        x, y = d['points'][ids], d['normals'][ids]
    n = len(x)
    result = {'args': {**vars(a), 'output': str(out)}, 'N': n, 'seconds': {}, 'solvers': {},
              'device': cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
              'sample_seed': 42, 'sdd_seed': 0, 'rng_backend': 'gpu',
              'hashes': {name: hashlib.sha256(v.tobytes()).hexdigest() for name,v in [('points',x),('normals',y)]},
              'scope': 'conditional solve comparison with common RHS, scalar priors and centering; not complete application timing',
              'prior': 'full analytic vector / cutoff50 scalar, corrected noise' if small else f'cutoff{a.cutoff} vector / cutoff50 scalar, noise variance .02; omitted analytic tail NOT sampled',
              'gates': {'analytic_residual': .001, 'error_over_fluctuation_l2': .002,
                        'empirical_variance_relative_l2': .0002, 'empirical_covariance_relative_l2': .0002,
                        'ray_rms': .01, 'ray_max': .02, 'collision_max': .02}}
    def save():
        (out / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    def timed(key, fn):
        value, seconds = measure(fn)
        result['seconds'][key] = seconds
        save()
        print(key, seconds, flush=True)
        return value
    model = timed('common_mean_fit', lambda: Posterior(x, y, upsampfac=1.25) if small else
                  SpectralPosterior(x, y, training_cutoff=a.cutoff))
    if small:
        timed('common_exact_draw_preparation', lambda: model.draw(a.samples, seed=42, batch=4,
              rng_backend='gpu', joint_solve=True, retain_components=True, retain_rhs=True, profile=True))
        result['draw_profile'] = model.draw_profile
    else:
        def progress(records):
            print('NUFFT draw', records[-1]['samples_done'], records[-1]['seconds'], flush=True)
        timed('finite_prior_and_nufft_draws', lambda: model.draw(a.samples, seed=42, batch=1,
              retain_components=True, retain_rhs=True, progress=progress))
        result['nufft_draw_records'] = model.sample_solve_records
    op, offset, prior = model.op, model.offset, model.sample_prior_coeff
    rhs = cp.ascontiguousarray(model.sample_rhs.transpose(1, 0, 2).reshape(n, 3 * a.samples))
    coefficients = {'dense' if small else 'nufft': model.sample_coeff}
    nufft_weights = None if small else cp.ascontiguousarray(model.sample_weights.transpose(1, 0, 2).reshape(n, -1))
    if small:
        covariance = model.K
        dense_times = []
        for repeat in range(3):
            timed(f'dense_shared_solve_{repeat}', lambda: model.solve(rhs))
            dense_times.append(result['seconds'][f'dense_shared_solve_{repeat}'])
        result['solvers']['dense'] = {'median_solve_seconds': float(np.median(dense_times))}
    rays, objects, distance = application_geometry(x)
    norm2 = (op.ns ** 2).sum(0)
    std = float(cp.sqrt(cp.sum(cp.where(norm2 > 0, op.rho / cp.maximum(norm2, 1), 0))))
    objects = timed('common_uncertain_objects', lambda: uncertain_objects(model.mean, objects, std))
    del model
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    k = Matern32Kernel(.01, .1)
    kv = ProductKernel(k, k, k)
    xj = jnp.asarray(x)
    exact_rows = jax.jit(jax.vmap(jax.vmap(kv, (None, 0)), (0, None)))
    row_ids = np.arange(n) if small else np.random.default_rng(73).choice(n, 512, replace=False)
    if not small:
        covariance = timed('analytic_stop_rows', lambda: cp.concatenate([
            own_jax_array(exact_rows(jnp.asarray(x[row_ids[i:i + 64]]), xj)) for i in range(0, len(row_ids), 64)]))
    def audit(weights):
        residual = covariance @ weights - rhs[row_ids]
        if not small:
            residual += .02 * weights[row_ids]
        return residual_panel(residual, rhs[row_ids], full=small)
    def setup_sdd():
        solver = StochasticDualDescent(xj, jnp.asarray(cp.asnumpy(rhs)), kv,
                                      batch=a.batch, lr=a.eta * a.batch / n)
        solver.compile()
        return solver
    sdd = timed('sdd_setup', setup_sdd)
    elapsed, history = 0., []
    while elapsed < a.time_budget:
        begin = time.perf_counter()
        solution = sdd.advance(a.chunk)
        elapsed += time.perf_counter() - begin
        weights = own_jax_array(solution)
        stats = audit(weights)
        history.append({'iterations': sdd.iterations, 'update_seconds': elapsed, 'analytic_residual': stats})
        result['solvers']['sdd'] = {'history': history, 'update_seconds': elapsed,
                                  'converged': stats['max_upper_endpoint'] < a.rtol}
        save()
        print('SDD', sdd.iterations, elapsed, stats['max_upper_endpoint'], flush=True)
        if stats['max_upper_endpoint'] < a.rtol or not np.isfinite(stats['max_upper_endpoint']):
            break
    def aggregate(weights):
        w = weights.reshape(n, a.samples, 3).transpose(1, 0, 2)
        return cp.concatenate([op.aggregate(x, w[i:i + 4]) + prior[i:i + 4] for i in range(0, a.samples, 4)])
    coefficients['sdd'] = timed('sdd_scalar_aggregation', lambda: aggregate(weights))
    if not small:
        holdout = np.random.default_rng(917).choice(n, 1024, replace=False)
        def heldout(weights):
            residual = []
            for i in range(0, len(holdout), 64):
                ids = holdout[i:i + 64]
                block = own_jax_array(exact_rows(jnp.asarray(x[ids]), xj))
                residual.append(block @ weights + .02 * weights[ids] - rhs[ids])
            return residual_panel(cp.concatenate(residual), rhs[holdout])
        result['solvers']['sdd']['holdout'] = timed('sdd_holdout', lambda: heldout(weights))
    del sdd, solution, weights
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    if small:
        training = FourierCovariance(a.cutoff)
        kernel = timed('nufft_plan_setup', lambda: FourierKernel(x, training, .02))
        weights, records = [], []
        for i in range(a.samples):
            (w, info), seconds = measure(lambda: kernel.solve(rhs[:, 3*i:3*i+3], rtol=1e-5, maxiter=1000))
            if not info['converged']:
                raise RuntimeError(info)
            weights.append(w)
            records.append({**info, 'seconds': seconds})
            if i % 8 == 0:
                print('NUFFT conditional solve', i + 1, seconds, flush=True)
        nufft_weights = cp.concatenate(weights, axis=1)
        coefficients['nufft'] = timed('nufft_scalar_aggregation', lambda: aggregate(nufft_weights))
        result['solvers']['nufft'] = {'records': records, 'solve_seconds': sum(r['seconds'] for r in records)}
        del kernel, training, weights
    else:
        result['solvers']['nufft'] = {'solve_seconds': sum(r['solve_seconds'] for r in result['nufft_draw_records']),
                                     'holdout': timed('nufft_holdout', lambda: heldout(nufft_weights))}
    result['solvers']['nufft']['analytic_residual'] = audit(nufft_weights)
    del covariance, rhs, nufft_weights
    cp.get_default_memory_pool().free_all_blocks()
    probe = np.random.default_rng(975).uniform(-np.pi, np.pi, (257, 3)).astype('f')
    fields, event_outputs = {}, {}
    for name, coeff in coefficients.items():
        fields[name] = op.evaluate_batched(probe, coeff, batch=4) - offset
        event_outputs[name] = {kind: cp.asnumpy(events(op.evaluate_batched(pos.reshape(-1, 3), coeff, batch=4) - offset,
                                                    pos.shape[:2], kind))
                               for kind, pos in [('rays', rays), ('collision', objects)]}
    reference = 'dense' if small else 'sdd'
    result['reference'] = reference
    result['quality'] = {}
    for name in coefficients:
        if name == reference:
            continue
        f = field_quality(fields[reference], fields[name])
        f.pop('pass')
        f['pass'] = all(f[key] < result['gates'][key] for key in
                        ['error_over_fluctuation_l2', 'empirical_variance_relative_l2', 'empirical_covariance_relative_l2'])
        result['quality'][name] = {'fields': f, 'events': quality(event_outputs[reference], event_outputs[name], distance)}
    save()
    np.savez_compressed(out / 'fields.npz', query=probe, rays=rays, objects=objects,
                        **{name: cp.asnumpy(value) for name, value in fields.items()})
    np.savez_compressed(out / 'events.npz', **{name+'_'+kind: value for name, pair in event_outputs.items() for kind, value in pair.items()})
    print('quality', result['quality'], flush=True)


if __name__ == '__main__':
    main()
