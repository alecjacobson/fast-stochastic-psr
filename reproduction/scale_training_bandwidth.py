"""Scale bandwidth-controlled training with independent analytic residual rows."""
import argparse
import hashlib
import json
from pathlib import Path

import cupy as cp
import jax
import jax.numpy as jnp
import numpy as np

from reproduction.matched_teaser import ROOT, measure, own_jax_array, Matern32Kernel, ProductKernel, queries
from reproduction.spectral_posterior import SpectralPosterior


def residual_statistics(residual, rhs, seed=18):
    residual, rhs = cp.asnumpy(residual), cp.asnumpy(rhs)
    error2, rhs2 = (residual.astype('d') ** 2).sum(1), (rhs.astype('d') ** 2).sum(1)
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, len(rhs), (2000, len(rhs)))
    ratios = np.sqrt(error2[ids].sum(1) / rhs2[ids].sum(1))
    return {'relative_rms': float(np.sqrt(error2.sum() / rhs2.sum())),
            'bootstrap_95_percent_interval': np.quantile(ratios, [.025, .975]).tolist(),
            'max_absolute_row_residual_norm': float(np.sqrt(error2.max())),
            'row_residual_norm_quantiles': np.quantile(np.sqrt(error2), [.5, .95, .99]).tolist(),
            'qualification': 'random row estimate and empirical bootstrap interval, not a full-system error bound'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--points', type=int, nargs='+', default=[100000, 1000000])
    p.add_argument('--cutoffs', type=int, nargs='+', default=[100, 150, 200])
    p.add_argument('--rows', type=int, default=512)
    p.add_argument('--output', type=Path, default=ROOT / 'reproduction/results/training_bandwidth/scaling')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    cloud_path = ROOT / 'reproduction/results/teaser/cloud_1000000.npz'
    if not cloud_path.exists():
        from reproduction.million_teaser import visible_cloud
        x, normals, clean, v, f = visible_cloud(1000000)
        np.savez_compressed(cloud_path, points=x, normals=normals, clean=clean, vertices=v, faces=f)
    cloud = np.load(cloud_path)
    permutation = np.random.default_rng(19).permutation(len(cloud['points']))
    result = {'args': {**vars(a), 'output': str(a.output)},
              'scene': 'distinct area-uniform surface points visible from released cameras, coordinate/normal perturbations std .02; not the camera raster sampler',
              'model': 'training Fourier bandwidth varies; scalar/cross cutoff50, lengthscale .01, noise variance .02',
              'device': cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
              'scope': 'training and mean scaling with sampled analytic residuals; not a complete-application or exact full-model proof',
              'rows': [], 'seconds': {}}
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
    q, _, _ = queries(100)
    for n in a.points:
        if n > len(permutation) or n < a.rows:
            p.error('point count must fit the available cloud and cover the residual sample')
        x = cloud['points'][permutation[:n]].astype('f')
        y = cloud['normals'][permutation[:n]].astype('f')
        ids = np.random.default_rng(73).choice(n, a.rows, replace=False)
        def analytic_rows():
            xj = jnp.asarray(x)
            return cp.concatenate([own_jax_array(exact_rows(jnp.asarray(x[ids[i:i + 64]]), xj))
                                   for i in range(0, len(ids), 64)])
        reference = timed(str(n) + '_analytic_validation_rows', analytic_rows)
        fields = {'query': q, 'validation_rows': ids}
        for cutoff in a.cutoffs:
            prefix = f'{n}_{cutoff}_'
            model = timed(prefix + 'fit_first', lambda: SpectralPosterior(x, y, training_cutoff=cutoff))
            residual = reference @ model.alpha + .02 * model.alpha[ids] - cp.asarray(y[ids])
            stats = residual_statistics(residual, cp.asarray(y[ids]))
            mean = timed(prefix + 'mean_queries', lambda: model.mean(q))
            fields[str(cutoff) + '_mean'] = cp.asnumpy(mean)
            row = {'N': n, 'training_cutoff': cutoff, 'mean_solve': model.mean_solve,
                   'analytic_residual_sample': stats,
                   'sampled_residual_gate_pass': stats['bootstrap_95_percent_interval'][1] < .001,
                   'quality_bridge_complete': False,
                   'points_sha256': hashlib.sha256(x.tobytes()).hexdigest(),
                   'normals_sha256': hashlib.sha256(y.tobytes()).hexdigest(),
                   'validation_rows': ids.tolist(),
                   'fit_first_seconds': result['seconds'][prefix + 'fit_first']}
            result['rows'].append(row)
            print(json.dumps(row), flush=True)
            save()
            np.savez_compressed(a.output / f'fields_{n}.npz', **fields)
            del model, residual, mean
            cp.get_default_memory_pool().free_all_blocks()
        del reference
        cp.get_default_memory_pool().free_all_blocks()


if __name__ == '__main__':
    main()
