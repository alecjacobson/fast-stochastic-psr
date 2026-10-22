"""SDD convergence and cost against full/sampled analytic training residuals.

Keep failed and budget-limited configurations; neither is an intractability proof.
The stronger baseline compiles the released update loop and tunes its parameters.
"""
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import cupy as cp
import jax
import jax.numpy as jnp
import numpy as np

from geospsr.kernels import Matern32Kernel, ProductKernel
from geospsr.spectral import FourierPoisson
from reproduction.iterative_reference import StochasticDualDescent
from reproduction.matched_teaser import ROOT, measure, own_jax_array, queries
from reproduction.scale_training_bandwidth import residual_statistics


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['A', 'B', '100000', '1000000'], default='A')
    p.add_argument('--batches', nargs='+', type=int, default=[128, 512])
    p.add_argument('--etas', nargs='+', type=float, default=[.5, 2, 5, 8])
    p.add_argument('--include-default', action='store_true', help='also use released lr=.001 at each batch size')
    p.add_argument('--time-budget', type=float, default=30, help='per configuration update time, excludes compilation/validation')
    p.add_argument('--iterations', type=int, default=100000)
    p.add_argument('--chunk', type=int, default=128)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--rows', type=int, default=512)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    if a.time_budget <= 0 or a.iterations < 1 or a.chunk < 1 or a.rows < 1:
        p.error('time budget, iterations, chunk and validation row count must be positive')
    out = a.output or ROOT / 'reproduction/results/iterative_baseline' / a.dataset
    out.mkdir(parents=True, exist_ok=True)
    if a.dataset in ('A', 'B'):
        data = np.load(ROOT / 'reproduction/results/matched' / a.dataset / 'input.npz')
        x, y, q = data['points'], data['normals'], data['query']
        reference_mean = np.load(ROOT / 'reproduction/results/training_bandwidth' / a.dataset / 'fields.npz')['analytic_mean']
        ids = np.arange(len(x))
        reference_type = 'dense analytic; all training rows and all 20000 mean queries'
    else:
        n = int(a.dataset)
        cloud = np.load(ROOT / 'reproduction/results/teaser/cloud_1000000.npz')
        permutation = np.random.default_rng(19).permutation(len(cloud['points']))[:n]
        x, y = cloud['points'][permutation], cloud['normals'][permutation]
        q, _, _ = queries(100)
        reference_mean = np.load(ROOT / 'reproduction/results/training_bandwidth/scaling' / f'fields_{n}.npz')['200_mean']
        ids = np.random.default_rng(73).choice(n, a.rows, replace=False)
        reference_type = 'sampled analytic training rows; cutoff200 mean diagnostic is NOT analytic truth'
    x, y = x.astype('f'), y.astype('f')
    result = {'args': {**vars(a), 'output': str(out)}, 'N': len(x),
              'reference': reference_type,
              'model': 'full analytic product Matern, lengthscale .01, per-axis variance .1, noise variance .02',
              'algorithm': 'compiled released SDD update, momentum .9, Polyak .01, same sequential fold_in and no-replacement choice; eta=lr*N/batch',
              'provenance': 'released 2352f1ef98aab20e0303611c1fa2b0948acd2a05; untouched-code equivalence test uses sigma=sqrt(.02)',
              'gate': 'full analytic relative residual <.001 at A/B; sampled bootstrap 95% upper endpoint <.001 at larger N; full mean relative L2 <.002 at A/B',
              'budget_policy': 'update time per configuration; excludes compilation, validation and field queries; may overshoot by one chunk',
              'device': cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
              'hashes': {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in [('points', x), ('normals', y), ('query', q)]},
              'validation_rows': ids.tolist(), 'seconds': {}, 'rows': []}
    def save():
        (out / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    def timed(name, fn):
        value, seconds = measure(fn)
        result['seconds'][name] = seconds
        save()
        return value
    k = Matern32Kernel(.01, .1)
    kv = ProductKernel(k, k, k)
    exact = jax.jit(jax.vmap(jax.vmap(kv, (None, 0)), (0, None)))
    xj, yj = jnp.asarray(x), jnp.asarray(y)
    def analytic_rows():
        return cp.concatenate([own_jax_array(exact(jnp.asarray(x[ids[i:i + 64]]), xj))
                               for i in range(0, len(ids), 64)])
    reference = timed('analytic_validation_rows', analytic_rows)
    target = cp.asarray(y[ids])
    op = timed('shared_scalar_spectrum', lambda: FourierPoisson(50))
    qcp = cp.asarray(q)
    means = {'query': q, 'reference_mean': reference_mean}
    for batch in a.batches:
        rates = [(f'eta{eta:g}', eta * batch / len(x)) for eta in a.etas]
        if a.include_default:
            rates.insert(0, ('released_lr', .001))
        for label, rate in rates:
            tag = f'batch{batch}_{label}'
            solver = StochasticDualDescent(xj, yj, kv, batch=batch, lr=rate, seed=a.seed)
            timed(tag + '_compile', solver.compile)
            history, elapsed, status = [], 0., 'iteration_budget'
            while solver.iterations < a.iterations:
                steps = min(a.chunk, a.iterations - solver.iterations)
                begin = time.perf_counter()
                alpha = solver.advance(steps)
                elapsed += time.perf_counter() - begin
                weights = own_jax_array(alpha)
                error = reference @ weights + .02 * weights[ids] - target
                if len(ids) == len(x):
                    rms = float(cp.linalg.norm(error) / cp.linalg.norm(target))
                    stats = {'relative_rms': rms, 'all_training_rows': True}
                    passed = rms < .001
                else:
                    stats = residual_statistics(error, target)
                    rms = stats['relative_rms']
                    passed = stats['bootstrap_95_percent_interval'][1] < .001
                history.append({'iterations': solver.iterations, 'update_seconds': elapsed, 'analytic_residual': stats})
                print(tag, solver.iterations, f'{elapsed:.3f}s', f'residual {rms:.6g}', flush=True)
                if not np.isfinite(rms) or rms > 1e4:
                    status = 'diverged'
                    break
                if passed:
                    status = 'residual_gate_pass'
                    break
                if elapsed >= a.time_budget:
                    status = 'time_budget'
                    break
            mean_error, aggregation_seconds, query_seconds = None, None, None
            if np.isfinite(rms) and rms < 1e4:
                def aggregate():
                    coefficients = op.aggregate(x, weights)
                    offset = op.evaluate(x, coefficients).mean()
                    return coefficients, offset
                (coeff, offset), aggregation_seconds = measure(aggregate)
                mean, query_seconds = measure(lambda: op.evaluate(qcp, coeff)[0] - offset)
                mean = cp.asnumpy(mean)
                mean_error = float(np.linalg.norm(mean - reference_mean) / np.linalg.norm(reference_mean))
                means[tag] = mean
                del coeff, offset
            quality_pass = status == 'residual_gate_pass' and (a.dataset not in ('A', 'B') or mean_error < .002)
            row = {'tag': tag, 'batch': batch, 'lr': rate, 'eta': rate * len(x) / batch,
                   'status': status, 'quality_pass': quality_pass,
                   'iterations': solver.iterations, 'update_seconds': elapsed,
                   'compile_seconds': result['seconds'][tag + '_compile'],
                   'scalar_aggregation_and_centering_seconds': aggregation_seconds,
                   'mean_query_seconds': query_seconds, 'mean_relative_error': mean_error,
                   'history': history,
                   'scope': 'mean-fit check only; no covariance, sampling or application quality claim'}
            result['rows'].append(row)
            save()
            np.savez_compressed(out / 'fields.npz', **means)
            del solver, weights, error, alpha
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
            print('completed', tag, status, 'mean error', mean_error, flush=True)


if __name__ == '__main__':
    main()
