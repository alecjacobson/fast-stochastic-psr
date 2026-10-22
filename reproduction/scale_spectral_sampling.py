"""Large-N coherent-sampling pilot with independent analytic residual rows.

Small draw counts exercise the implementation and cost; they do not establish
Monte Carlo convergence or agreement of downstream joint-event probabilities.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cupy as cp
import jax
import jax.numpy as jnp
import numpy as np

from reproduction.matched_teaser import ROOT, measure, own_jax_array, Matern32Kernel, ProductKernel, queries
from reproduction.scale_training_bandwidth import residual_statistics
from reproduction.spectral_posterior import SpectralPosterior


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--points', type=int, default=1000000)
    p.add_argument('--cutoffs', type=int, nargs='+', default=[150, 200])
    p.add_argument('--samples', type=int, default=8)
    p.add_argument('--rows', type=int, default=512)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', type=Path, default=ROOT / 'reproduction/results/training_bandwidth/sampling')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    cloud = np.load(ROOT / 'reproduction/results/teaser/cloud_1000000.npz')
    if not a.rows <= a.points <= len(cloud['points']):
        p.error('point count must cover validation rows and fit the available cloud')
    permutation = np.random.default_rng(19).permutation(len(cloud['points']))[:a.points]
    x, y = cloud['points'][permutation].astype('f'), cloud['normals'][permutation].astype('f')
    ids = np.random.default_rng(73).choice(len(x), a.rows, replace=False)
    q, _, _ = queries(100)
    result = {'args': {**vars(a), 'output': str(a.output)},
              'scope': 'coherent finite-training sampling pilot; sampled analytic conditional residuals; no Monte Carlo accuracy or application speedup claim',
              'scene': 'distinct visible surface cloud, area-uniform rather than camera raster',
              'model': 'vector training cutoff varies, scalar/cross cutoff50, lengthscale .01, noise variance .02',
              'device': cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
              'points_sha256': hashlib.sha256(x.tobytes()).hexdigest(),
              'normals_sha256': hashlib.sha256(y.tobytes()).hexdigest(),
              'query_sha256': hashlib.sha256(q.tobytes()).hexdigest(),
              'validation_rows': ids.tolist(), 'seconds': {}, 'rows': []}
    def save():
        (a.output / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    def timed(name, fn):
        value, secs = measure(fn)
        result['seconds'][name] = secs
        print(name, secs, flush=True)
        save()
        return value
    k = Matern32Kernel(.01, .1)
    kv = ProductKernel(k, k, k)
    exact_rows = jax.jit(jax.vmap(jax.vmap(kv, (None, 0)), (0, None)))
    def analytic_rows():
        xj = jnp.asarray(x)
        return cp.concatenate([own_jax_array(exact_rows(jnp.asarray(x[ids[i:i + 64]]), xj))
                               for i in range(0, len(ids), 64)])
    reference = timed('analytic_validation_rows', analytic_rows)
    for cutoff in a.cutoffs:
        prefix = str(cutoff) + '_'
        model = timed(prefix + 'fit', lambda: SpectralPosterior(x, y, training_cutoff=cutoff))
        def progress(records):
            (a.output / (prefix + 'sample_progress.json')).write_text(json.dumps(records, indent=2) + '\n')
            print(prefix + 'draw', records[-1]['samples_done'], records[-1]['seconds'], flush=True)
        timed(prefix + 'draws', lambda: model.draw(a.samples, seed=a.seed, batch=1,
              retain_components=True, validation_rows=ids, progress=progress))
        checks = []
        for i in range(a.samples):
            w, rhs = model.sample_weights[i], model.sample_validation_rhs[i]
            stats = residual_statistics(reference @ w + .02 * w[ids] - rhs, rhs)
            checks.append(stats)
        values = timed(prefix + 'sample_queries', lambda: model.samples(q))
        cache = timed(prefix + 'cache_setup', lambda: model.prepare_query_cache())
        cached = timed(prefix + 'cached_sample_queries_first', lambda: model.samples(q))
        repeats = []
        for i in range(3):
            cached = timed(prefix + 'cached_sample_queries_repeat_' + str(i), lambda: model.samples(q))
            repeats.append(result['seconds'][prefix + 'cached_sample_queries_repeat_' + str(i)])
        error = float(cp.linalg.norm(cached - values) / cp.linalg.norm(values))
        row = {'training_cutoff': cutoff, 'mean_solve': model.mean_solve,
               'draw_solves': model.sample_solve_records,
               'conditional_analytic_residual_samples': checks,
               'all_sampled_residual_gates_pass': all(r['bootstrap_95_percent_interval'][1] < .001 for r in checks),
               'query_cache': cache, 'query_cache_relative_error': error,
               'query_cached_median_seconds': float(np.median(repeats)),
               'quality_bridge_complete': False}
        result['rows'].append(row)
        np.savez_compressed(a.output / (prefix + 'fields.npz'), query=q, samples=cp.asnumpy(values),
                            mean=cp.asnumpy(model.mean(q)))
        save()
        print(prefix + 'maximum_sampled_residual', max(c['relative_rms'] for c in checks), flush=True)
        del model, values, cached, w, rhs
        cp.get_default_memory_pool().free_all_blocks()


if __name__ == '__main__':
    main()
