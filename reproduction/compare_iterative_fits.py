"""Confirm selected SDD versus NUFFT fits, with preparation and quality checks.

Repetitions reset weights at fixed data/seed and exclude compiled-plan setup.
These are solver benchmarks, not acquisition/refit or complete applications.
"""
import argparse
import gc
import json

import cupy as cp
import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import ndtr

from geospsr.kernels import Matern32Kernel, ProductKernel
from geospsr.spectral import FourierPoisson
from reproduction.iterative_reference import StochasticDualDescent
from reproduction.matched_teaser import ROOT, measure, own_jax_array, queries
from reproduction.spectral_posterior import SpectralPosterior
from reproduction.scale_training_bandwidth import residual_statistics


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['A', 'B', '100000', '1000000'], default='B')
    p.add_argument('--cutoff', type=int, default=200)
    p.add_argument('--repeats', type=int, default=3)
    a = p.parse_args()
    if a.repeats < 0:
        p.error('repeats must be nonnegative')
    folder = ROOT / 'reproduction/results/iterative_baseline' / a.dataset
    pilot = json.loads((folder / 'results.json').read_text())
    candidates = [r for r in pilot['rows'] if r['quality_pass']]
    if not candidates:
        p.error('no quality-passing SDD configuration exists in this pilot')
    best = min(candidates, key=lambda r: r['update_seconds'])
    if a.dataset in ('A', 'B'):
        data = np.load(ROOT / 'reproduction/results/matched' / a.dataset / 'input.npz')
        x, y, q = data['points'], data['normals'], data['query']
        fields = np.load(ROOT / 'reproduction/results/training_bandwidth' / a.dataset / 'fields.npz')
        ref_mean = fields['analytic_mean']
        probe_ids = fields['query_ids']
        variance = np.diag(fields['analytic_covariance'])
        ref_probability = fields['analytic_probability']
    else:
        n = int(a.dataset)
        data = np.load(ROOT / 'reproduction/results/teaser/cloud_1000000.npz')
        ids = np.random.default_rng(19).permutation(len(data['points']))[:n]
        x, y = data['points'][ids], data['normals'][ids]
        q, _, _ = queries(100)
        ref_mean = np.load(ROOT / 'reproduction/results/training_bandwidth/scaling' / f'fields_{n}.npz')['200_mean']
    result = {'args': vars(a), 'N': len(x), 'selected_sdd': best,
              'reference': pilot['reference'], 'hashes': pilot['hashes'],
              'scope': 'selected mean-fit comparison; warm repetitions reset at fixed data and seed; no complete application speedup',
              'seconds': {}, 'rows': []}
    def save():
        (folder / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    def timed(name, fn):
        value, seconds = measure(fn)
        result['seconds'][name] = seconds
        print(name, seconds, flush=True)
        save()
        return value
    def check(mean):
        mean = cp.asnumpy(mean)
        errors = {'mean_relative_error': float(np.linalg.norm(mean - ref_mean) / np.linalg.norm(ref_mean))}
        if a.dataset in ('A', 'B'):
            probability = ndtr(-mean[probe_ids] / np.sqrt(variance))
            errors.update({'probability_rms': float(np.sqrt(np.mean((probability - ref_probability) ** 2))),
                           'probability_max': float(np.max(np.abs(probability - ref_probability))),
                           'probability_scope': 'shared exact analytic variance; isolates fitting mean error, not a full method probability benchmark'})
            errors['quality_pass'] = errors['mean_relative_error'] < .002 and errors['probability_rms'] < .001 and errors['probability_max'] < .02
        return errors
    k = Matern32Kernel(.01, .1)
    kv = ProductKernel(k, k, k)
    # This holdout sample is independent of the rows used to stop/tune SDD.
    validation_ids = (np.arange(len(x)) if a.dataset in ('A', 'B') else
                      np.random.default_rng(917).choice(len(x), 1024, replace=False))
    result['validation_rows'] = validation_ids.tolist()
    exact = jax.jit(jax.vmap(jax.vmap(kv, (None, 0)), (0, None)))
    xj = jnp.asarray(x)
    def validate(alpha):
        residual = []
        for start in range(0, len(validation_ids), 64):
            rows = validation_ids[start:start + 64]
            block = own_jax_array(exact(jnp.asarray(x[rows]), xj))
            residual.append(block @ alpha + .02 * alpha[rows] - cp.asarray(y[rows]))
        error = cp.concatenate(residual)
        target = cp.asarray(y[validation_ids])
        if len(validation_ids) == len(x):
            stats = {'relative_rms': float(cp.linalg.norm(error) / cp.linalg.norm(target)), 'all_training_rows': True}
            passed = stats['relative_rms'] < .001
        else:
            stats = residual_statistics(error, target)
            passed = stats['bootstrap_95_percent_interval'][1] < .001
        return {'stats': stats, 'quality_pass': passed,
                'scope': 'all analytic rows' if len(validation_ids) == len(x) else '1024 holdout analytic rows, independent of tuning/stopping rows'}
    def sdd_setup():
        solver = StochasticDualDescent(x, y, kv, batch=best['batch'], lr=best['lr'], seed=pilot['args']['seed'])
        initial = solver.state
        solver.compile()
        return solver, initial, FourierPoisson(50)
    solver, initial, op = timed('sdd_setup', sdd_setup)
    def sdd_fit():
        solver.state = initial
        alpha = own_jax_array(solver.advance(best['iterations']))
        coeff = op.aggregate(x, alpha)
        offset = op.evaluate(x, coeff).mean()
        return alpha, coeff, offset
    alpha, coeff, offset = timed('sdd_fit_first_after_setup', sdd_fit)
    sdd_mean = timed('sdd_mean_queries', lambda: op.evaluate(q, coeff)[0] - offset)
    errors = check(sdd_mean)
    validation = timed('sdd_independent_analytic_validation', lambda: validate(alpha))
    repeat_times = []
    for i in range(a.repeats):
        alpha, coeff, offset = timed(f'sdd_fit_repeat_{i}', sdd_fit)
        repeat_times.append(result['seconds'][f'sdd_fit_repeat_{i}'])
    result['rows'].append({'method': 'SDD', 'errors': errors, 'analytic_validation': validation,
                           'first_fit_seconds': result['seconds']['sdd_setup'] + result['seconds']['sdd_fit_first_after_setup'],
                           'warm_fit_median_seconds': float(np.median(repeat_times)) if repeat_times else None})
    save()
    arrays = {'query': q, 'reference_mean': ref_mean, 'sdd_mean': cp.asnumpy(sdd_mean)}
    del solver, initial, op, alpha, coeff, offset, sdd_mean
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    model = timed('nufft_fit_first', lambda: SpectralPosterior(x, y, training_cutoff=a.cutoff))
    mean = timed('nufft_mean_queries', lambda: model.mean(q))
    errors = check(mean)
    validation = timed('nufft_independent_analytic_validation', lambda: validate(model.alpha))
    arrays['nufft_mean'] = cp.asnumpy(mean)
    def nufft_fit():
        alpha, info = model.K.solve(model.normals, rtol=model.rtol, maxiter=1000)
        if not info['converged']:
            raise RuntimeError(info)
        coeff = model.op.aggregate(model.x, alpha)
        offset = model.op.evaluate(model.x, coeff).mean()
        return coeff, offset
    repeat_times = []
    for i in range(a.repeats):
        timed(f'nufft_fit_repeat_{i}', nufft_fit)
        repeat_times.append(result['seconds'][f'nufft_fit_repeat_{i}'])
    result['rows'].append({'method': 'NUFFT', 'errors': errors, 'analytic_validation': validation, 'finite_solve': model.mean_solve,
                           'first_fit_seconds': result['seconds']['nufft_fit_first'],
                           'warm_fit_median_seconds': float(np.median(repeat_times)) if repeat_times else None})
    result['paired_mean_relative_l2'] = float(np.linalg.norm(arrays['sdd_mean'] - arrays['nufft_mean']) /
                                              np.linalg.norm(arrays['nufft_mean']))
    save()
    np.savez_compressed(folder / 'comparison_fields.npz', **arrays)


if __name__ == '__main__':
    main()
