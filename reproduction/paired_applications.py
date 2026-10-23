"""Paired pathwise ray/collision queries with shared conditioning and randomness.

This compares evaluation backends, not the untouched stochastic API: all backends
receive corrected full-analytic-v/finite-f draws, including observation noise.
The reference shares released interpolated cross-kernel rows across all samples.
Direct Fourier is a diagnostic until convergence to the intended model is audited.
"""
import argparse
from functools import partial
import hashlib
import json
from pathlib import Path

import cupy as cp
import jax
import jax.numpy as jnp
import numpy as np

from reproduction.matched_teaser import (ROOT, original, own_jax_array, measure,
    Matern32Kernel, ProductKernel, periodic_stationary_interpolator, trunc_Zd)
from reproduction.posterior import Posterior
from reproduction.interpolated_nufft import InterpolatedFourierPoisson
from geospsr.spectral import FourierPoisson
from geospsr.cached_spectral import CachedFourierEvaluator


def application_geometry(points, ray_count=24, ray_steps=256, sphere_points=256):
    center = points.mean(0)
    angles = np.arange(ray_count) * 2 * np.pi / ray_count
    origins = center + 2.5 * np.stack([np.cos(angles), np.sin(angles), np.full(ray_count, .2)], -1)
    directions = center - origins
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    distance = np.linspace(0, 5, ray_steps, dtype='f')
    rays = (origins[:, None] + directions[:, None] * distance[None, :, None]).astype('f')
    unit = np.random.default_rng(1).normal(size=(sphere_points, 3))
    unit /= np.linalg.norm(unit, axis=1)[:, None]
    centers = center + np.stack([np.linspace(-1.5, 1.5, 17), np.zeros(17), np.zeros(17)], -1)
    objects = (centers[:, None] + .12 * unit).astype('f')
    return rays, objects, distance


def events(samples, shape, kind):
    values = samples.reshape(len(samples), *shape)
    if kind == 'rays':
        return cp.cumprod(values > 0, axis=2).astype(cp.float32)
    return (values <= 0).any(axis=2).astype(cp.float32)


def uncertain_objects(mean_at, objects, prior_std, iterations=14):
    """Place a common sphere trajectory around a mean-surface crossing.

    Calibration uses only the deterministic mean, never the realized sample
    events. All evaluation backends receive these same resulting coordinates.
    Sphere surface discretization is preserved and remains an explicit error.
    """
    count, samples, _ = objects.shape
    initial = cp.asnumpy(mean_at(objects.reshape(-1, 3))).reshape(count, samples).min(1)
    crossings = np.flatnonzero(initial[:-1] * initial[1:] < 0)
    if not len(crossings):
        raise ValueError('no mean-surface crossing on the common sphere trajectory')
    index = int(crossings[0])
    left, right = objects[index], objects[index + 1]
    a, b = np.zeros(count), np.ones(count)
    span = min(2 * prior_std, .8 * min(abs(initial[index]), abs(initial[index + 1])))
    targets = np.linspace(-span, span, count)
    increasing = initial[index + 1] > initial[index]
    for _ in range(iterations):
        t = (a + b) / 2
        probes = (left[None] + t[:, None, None] * (right - left)[None]).astype('f')
        values = cp.asnumpy(mean_at(probes.reshape(-1, 3))).reshape(count, samples).min(1)
        move_left = (values < targets) if increasing else (values > targets)
        a, b = np.where(move_left, t, a), np.where(move_left, b, t)
    result = (left[None] + ((a + b) / 2)[:, None, None] * (right - left)[None]).astype('f')
    return result


def quality(reference, candidate, distance):
    result = {}
    for key in ['rays', 'collision']:
        delta = candidate[key] - reference[key]
        average = delta.mean(0)
        result[key] = {
            'rms_probability_difference': float(np.sqrt(np.mean(average ** 2))),
            'max_probability_difference': float(np.max(np.abs(average))),
            'max_paired_standard_error': float(np.max(delta.std(0, ddof=1) / np.sqrt(len(delta)))),
            'event_disagreement_fraction': float(np.mean(delta != 0)),
        }
    ref_t, candidate_t = reference['rays'].mean(0), candidate['rays'].mean(0)
    score = lambda t: np.trapezoid(((t >= .05) & (t <= .95)).astype(float), distance, axis=1)
    a, b = score(ref_t), score(candidate_t)
    result['next_view'] = {'reference_best': int(a.argmax()), 'candidate_best': int(b.argmax()),
                           'reference_score_regret': float(a.max() - a[b.argmax()]),
                           'max_score_difference': float(np.max(np.abs(a - b)))}
    # Version 2 adds a local ray gate after v1 exposed a small RMS hiding a
    # large local error. These gates do not certify sampling or model error.
    result['compatibility_pass'] = bool(
        result['rays']['rms_probability_difference'] <= .01
        and result['rays']['max_probability_difference'] <= .02
        and result['collision']['max_probability_difference'] <= .02
        and result['next_view']['reference_score_regret'] <= 2 * float(distance[1] - distance[0]))
    # Version 3 fixes Boolean trapezoidal addition in the physical ray length.
    result['quality_gate_version'] = 3
    result['next_view']['score_convention'] = 'numeric trapezoidal integral of 5%-95% uncertainty mask'
    result['uncertainty_coverage_pass'] = bool(np.any((reference['collision'].mean(0) > 0)
                                                    & (reference['collision'].mean(0) < 1)))
    result['application_validation_complete'] = False
    return result


def field_quality(reference, candidate):
    """Coupled mean/variance/covariance checks on independent query locations."""
    a, b = cp.asnumpy(reference), cp.asnumpy(candidate)
    var_a, var_b = a.var(0, ddof=1), b.var(0, ddof=1)
    cov_a, cov_b = np.cov(a[:, :32].T), np.cov(b[:, :32].T)
    relative = lambda error, scale: float(np.linalg.norm(error) / max(np.linalg.norm(scale), 1e-30))
    result = {'field_relative_l2': relative(b - a, a),
              'error_over_fluctuation_l2': relative(b - a, a - a.mean(0)),
              'empirical_variance_relative_l2': relative(var_b - var_a, var_a),
              'empirical_covariance_relative_l2': relative(cov_b - cov_a, cov_a),
              'queries': a.shape[1], 'covariance_queries': min(32, a.shape[1])}
    result['pass'] = bool(result['error_over_fluctuation_l2'] < 5e-4
                          and result['empirical_variance_relative_l2'] < 1e-4
                          and result['empirical_covariance_relative_l2'] < 1e-4)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=['A', 'B'], default='A')
    parser.add_argument('--samples', type=int, default=128)
    parser.add_argument('--sample-batch', type=int, default=4)
    parser.add_argument('--query-batch', type=int, default=1024)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--rebuild-plans', action='store_true', help='reproduce pre-optimization query plan construction')
    parser.add_argument('--cached-grids', action='store_true')
    parser.add_argument('--fast-sampling', action='store_true', help='GPU RNG, shared triangular solve, validated 1.25 source FFT oversampling')
    parser.add_argument('--grid-query-batch', type=int, default=16, help='packed complex grids per interpolation batch')
    parser.add_argument('--uncertain-collisions', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ray-steps', type=int, default=256)
    parser.add_argument('--sphere-points', type=int, default=256)
    parser.add_argument('--reference-grid', type=int, default=50,
                        help='released grid is 50; larger values study interpolation convergence')
    parser.add_argument('--mean-offset', choices=['reference', 'direct'], default='reference')
    parser.add_argument('--query-jitter', type=float, default=0.,
                        help='deterministic translation of queries between repeats; every backend gets the same translations')
    parser.add_argument('--backends', nargs='+', choices=['compatible', 'direct'], default=['compatible', 'direct'])
    parser.add_argument('--fluctuation-cutoffs', nargs='*', type=int, default=[],
                        help='experimental centered-sample compression; full mean retained')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.samples < 2 or args.repeats < 1:
        parser.error('need at least two samples and one repeat')
    if min(args.sample_batch, args.query_batch, args.grid_query_batch, args.sphere_points) < 1 or args.ray_steps < 2:
        parser.error('batch sizes and sphere count must be positive; need at least two ray steps')
    if args.reference_grid < 2 or (args.reference_grid != 50 and 'compatible' in args.backends):
        parser.error('refined reference needs grid >=2 and --backends direct (compatibility spectrum is fixed at grid 50)')
    if any(c < 1 or c > 50 for c in args.fluctuation_cutoffs):
        parser.error('fluctuation cutoffs must be between 1 and 50')
    data = np.load(ROOT / 'reproduction/results/matched' / args.case / 'input.npz')
    x, normals = data['points'], data['normals']
    rays, objects, distance = application_geometry(x, ray_steps=args.ray_steps, sphere_points=args.sphere_points)
    query = {'rays': rays.reshape(-1, 3), 'collision': objects.reshape(-1, 3)}
    shapes = {'rays': rays.shape[:2], 'collision': objects.shape[:2]}
    out = args.output or ROOT / 'reproduction/results/paired_applications' / args.case
    out.mkdir(parents=True, exist_ok=True)
    record = {'case': args.case, 'N': len(x), 'samples': args.samples, 'seed': args.seed,
              'query_counts': {k: len(q) for k, q in query.items()},
              'device': cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
              'args': {**vars(args), 'output': str(out)}, 'seconds': {},
              'input_sha256': {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in
                               {'points': x, 'normals': normals, **query}.items()},
              'reference': 'released interpolated cross kernel; shared corrected pathwise draws; cached analytic Cholesky; kernel blocks shared across samples',
              'model': 'full analytic product-Matern vector covariance, finite 101^3-mode scalar prior, noise variance .02; independent omitted-vector-spectrum plus noise draws',
              'scope': 'GPU-synchronized stages; query includes sample fields, joint-event reductions, probabilities and ray scores; excludes geometry generation, disk writes and rendering',
              'status': 'paired evaluation benchmark; not untouched stochastic API or massive-scale model validation'}
    def save():
        (out / 'results.json').write_text(json.dumps(record, indent=2) + '\n')
    def timed(name, fn):
        value, elapsed = measure(fn)
        record['seconds'][name] = elapsed
        print(name, elapsed, flush=True)
        save()
        return value
    model = timed('common_fit_and_direct_mean_setup', lambda: Posterior(x, normals, upsampfac=1.25 if args.fast_sampling else 2.))
    timed('common_corrected_draws_and_direct_aggregation', lambda:
          model.draw(args.samples, seed=args.seed, batch=args.sample_batch, retain_components=True,
                     rng_backend='gpu' if args.fast_sampling else 'cpu', joint_solve=args.fast_sampling))
    k = Matern32Kernel(.01, .1)
    kv = ProductKernel(k, k, k)
    frequencies = trunc_Zd(50)
    ev = jax.vmap(kv.spectral_density)(frequencies) ** .5
    cross = partial(original.poisson_cross_covariances, eigenvectors=frequencies, eigenvalues=ev)
    if args.reference_grid == 50:
        kfv = timed('reference_kernel_setup', lambda: periodic_stationary_interpolator(cross, 3, 50, exponent=5))
    else:
        from jax.scipy.interpolate import RegularGridInterpolator
        def refined_kernel():
            knots = (jnp.mgrid[-1:1:complex(0, args.reference_grid)] ** 5) * jnp.pi
            coordinates = np.asarray(knots)
            grid = np.stack(np.meshgrid(coordinates, coordinates, coordinates, indexing='ij'), -1).reshape(-1, 3)
            table = cp.asnumpy(model.op.cross_values(grid)).reshape((args.reference_grid,) * 3 + (3,))
            interpolator = RegularGridInterpolator((knots,) * 3, jnp.asarray(table))
            return lambda a, b: interpolator((a - b + jnp.pi) % (2 * jnp.pi) - jnp.pi).squeeze()
        kfv = timed('reference_kernel_setup', refined_kernel)
        record['reference'] = 'refined released trilinear algorithm, table sampled with independently validated finite-Fourier NUFFT; shared corrected draws and cached analytic Cholesky'
        record['reference_table_builder'] = 'NUFFT values; kernel direct-sum validation in matched/direct_fourier_validation.json'
    rows = jax.jit(jax.vmap(jax.vmap(kfv, (None, 0)), (0, None)))
    xj = jnp.asarray(x)
    weights = cp.ascontiguousarray(model.sample_weights.reshape(args.samples, -1).T)
    def reference_offset():
        total = cp.float32(0)
        for start in range(0, len(x), args.query_batch):
            block = own_jax_array(rows(xj[start:start + args.query_batch], xj))
            total += (block * model.alpha).sum()
        return total / len(x)
    offset = (timed('reference_mean_offset', reference_offset)
              if args.mean_offset == 'reference' else model.offset)
    record['mean_offset'] = {'source': args.mean_offset, 'value': float(offset)}
    if args.uncertain_collisions:
        norm2 = (model.op.ns ** 2).sum(0)
        prior_std = float(cp.sqrt(cp.sum(cp.where(norm2 > 0, model.op.rho / cp.maximum(norm2, 1), 0))))
        objects = timed('common_collision_calibration', lambda: uncertain_objects(
            lambda q: model.op.evaluate(q, model.mean_coeff)[0] - offset, objects, prior_std))
        query['collision'] = objects.reshape(-1, 3)
        record['input_sha256']['collision'] = hashlib.sha256(query['collision'].tobytes()).hexdigest()
        record['collision_calibration'] = {'uses': 'common deterministic mean only, no sample outcomes',
            'centroids': objects.mean(1).tolist(), 'prior_std': prior_std,
            'surface_points_per_object': args.sphere_points}
        save()
    def evaluate_many(op, q, coefficients):
        if args.rebuild_plans:
            return cp.concatenate([op.evaluate(q, coefficients[i:i + args.sample_batch])
                                   for i in range(0, args.samples, args.sample_batch)])
        return op.evaluate_batched(q, coefficients, args.sample_batch)
    def evaluate_prior(q):
        return evaluate_many(model.op, q, model.sample_prior_coeff)
    def reference_correction(q):
        blocks = []
        for start in range(0, len(q), args.query_batch):
            block = own_jax_array(rows(jnp.asarray(q[start:start + args.query_batch]), xj))
            blocks.append(block.reshape(len(block), -1) @ weights)
        return cp.concatenate(blocks).T
    def reference(q):
        return reference_correction(q) + evaluate_prior(q) - offset
    # All backends use the same reference mean offset to isolate evaluation.
    def direct(q):
        return evaluate_many(model.op, q, model.sample_coeff) - offset
    outputs = {}
    validation_q = np.random.default_rng(975).uniform(-np.pi, np.pi, (257, 3)).astype('f')
    def validate_cache(name, ordinary, cached):
        result = field_quality(ordinary(validation_q), cached(validation_q))
        record.setdefault('field_checks', {})[name] = result
        save()
        assert result['pass'], (name, result)
    def benchmark(name, evaluator):
        outputs[name] = {}
        record.setdefault('backends', {})[name] = {}
        for kind, q in query.items():
            def run(positions=q):
                event = events(evaluator(positions), shapes[kind], kind)
                probability = event.mean(0)
                score = (cp.trapz(((probability >= .05) & (probability <= .95)).astype('f'),
                                 cp.asarray(distance), axis=1) if kind == 'rays' else probability)
                return event, probability, score
            values = timed(name + '_' + kind + '_first', run)
            repetitions = []
            for repeat in range(args.repeats):
                shift = args.query_jitter * np.array([np.sin(repeat + 1), np.cos(repeat + 1),
                                                     np.sin(2 * repeat + 1)], dtype='f')
                positions = (q + shift).astype('f')
                values = timed(name + '_' + kind + '_repeat_' + str(repeat), lambda: run(positions))
                repetitions.append(record['seconds'][name + '_' + kind + '_repeat_' + str(repeat)])
            outputs[name][kind] = cp.asnumpy(values[0])
            probability = cp.asnumpy(values[1])
            record['backends'][name][kind] = {
                'median_seconds': float(np.median(repetitions)),
                'probability_min': float(probability.min()), 'probability_max': float(probability.max()),
                'uncertain_query_count': int(np.sum((probability > 0) & (probability < 1))),
                'max_monte_carlo_standard_error': float(np.sqrt((probability * (1 - probability)).max() / args.samples))}
        save()
    benchmark('reference', reference)
    if args.cached_grids:
        prior_grid = timed('reference_prior_grid_setup', lambda:
                           CachedFourierEvaluator(model.sample_prior_coeff, batch=1))
        record.setdefault('cache_bytes', {})['reference_prior'] = prior_grid.cache_bytes
        validate_cache('reference_prior', evaluate_prior,
                       lambda q: prior_grid.evaluate(q, batch=args.grid_query_batch))
        benchmark('reference_cached', lambda q: reference_correction(q) + prior_grid.evaluate(q, batch=args.grid_query_batch) - offset)
        record.setdefault('quality', {})['reference_cached'] = quality(outputs['reference'], outputs['reference_cached'], distance)
        del prior_grid
        cp.get_default_memory_pool().free_all_blocks()
    if 'direct' in args.backends or args.fluctuation_cutoffs:
        benchmark('direct', direct)
        record.setdefault('quality', {})['direct'] = quality(outputs['reference'], outputs['direct'], distance)
        save()
        if args.cached_grids:
            grid = timed('direct_grid_setup', lambda: CachedFourierEvaluator(model.sample_coeff, batch=1))
            record.setdefault('cache_bytes', {})['direct'] = grid.cache_bytes
            validate_cache('direct', direct, lambda q: grid.evaluate(q, batch=args.grid_query_batch) - offset)
            benchmark('direct_cached', lambda q: grid.evaluate(q, batch=args.grid_query_batch) - offset)
            record.setdefault('quality', {})['direct_cached'] = quality(outputs['reference'], outputs['direct_cached'], distance)
            record.setdefault('quality_vs_full_direct', {})['direct_cached'] = quality(outputs['direct'], outputs['direct_cached'], distance)
            del grid
            cp.get_default_memory_pool().free_all_blocks()
    for cutoff in args.fluctuation_cutoffs:
        name = 'fluctuation_' + str(cutoff)
        low_op = timed(name + '_operator_setup', lambda: FourierPoisson(cutoff))
        sl = slice(50 - cutoff, 51 + cutoff)
        low_coeff = timed(name + '_coefficient_compression', lambda:
                         cp.ascontiguousarray(model.sample_coeff[:, sl, sl, sl]
                                              - model.mean_coeff[:, sl, sl, sl]))
        def compressed(q):
            return (evaluate_many(low_op, q, low_coeff)
                    + model.op.evaluate(q, model.mean_coeff)[0] - offset)
        benchmark(name, compressed)
        record.setdefault('quality_vs_full_direct', {})[name] = quality(outputs['direct'], outputs[name], distance)
        record.setdefault('quality', {})[name] = quality(outputs['reference'], outputs[name], distance)
        # Coupled finite-sample moment diagnostics on a deterministic ray subset.
        check_q = query['rays'][np.linspace(0, len(query['rays']) - 1, 256, dtype=int)]
        full, small = cp.asnumpy(direct(check_q)), cp.asnumpy(compressed(check_q))
        mean_q = cp.asnumpy(model.op.evaluate(check_q, model.mean_coeff)[0] - offset)
        var_full, var_small = full.var(0, ddof=1), small.var(0, ddof=1)
        record.setdefault('sample_spectrum_diagnostics', {})[name] = {
            'centered_field_relative_l2': float(np.linalg.norm(small - full) / np.linalg.norm(full - mean_q)),
            'empirical_variance_relative_l2': float(np.linalg.norm(var_small - var_full) / np.linalg.norm(var_full)),
            'coefficient_storage_bytes': int(low_coeff.nbytes),
            'full_coefficient_storage_bytes': int(model.sample_coeff.nbytes),
            'check_queries': len(check_q),
            'qualification': 'paired finite-sample diagnostics; not a bound on population covariance'}
        del low_coeff, low_op
        save()
    if 'compatible' in args.backends:
        op = timed('compatible_kernel_setup', InterpolatedFourierPoisson)
        def aggregate():
            coefficients = []
            for i in range(0, args.samples, args.sample_batch):
                c = op.aggregate(x, model.sample_weights[i:i + args.sample_batch])
                pad = (op.shape[0] - model.op.shape[0]) // 2
                c += cp.pad(model.sample_prior_coeff[i:i + args.sample_batch],
                            ((0, 0), (pad, pad), (pad, pad), (pad, pad)))
                coefficients.append(c)
            return cp.concatenate(coefficients)
        coeff = timed('compatible_sample_aggregation', aggregate)
        def compatible(q):
            return evaluate_many(op, q, coeff) - offset
        benchmark('compatible', compatible)
        record.setdefault('quality', {})['compatible'] = quality(outputs['reference'], outputs['compatible'], distance)
        if args.cached_grids:
            cp.get_default_memory_pool().free_all_blocks()
            grid = timed('compatible_grid_setup', lambda: CachedFourierEvaluator(coeff, batch=1))
            record.setdefault('cache_bytes', {})['compatible'] = grid.cache_bytes
            validate_cache('compatible', compatible, lambda q: grid.evaluate(q, batch=args.grid_query_batch) - offset)
            benchmark('compatible_cached', lambda q: grid.evaluate(q, batch=args.grid_query_batch) - offset)
            record.setdefault('quality', {})['compatible_cached'] = quality(outputs['reference'], outputs['compatible_cached'], distance)
            record.setdefault('quality_vs_full_compatible', {})['compatible_cached'] = quality(outputs['compatible'], outputs['compatible_cached'], distance)
            del grid
    common = record['seconds']['common_fit_and_direct_mean_setup'] + record['seconds']['common_corrected_draws_and_direct_aggregation']
    record['accounting'] = {
        'common_preparation_seconds': common,
        'limitation': 'Common draw preparation includes direct aggregation even for the reference. Stage sums are this controlled harness, not independently optimized end-to-end implementations. Reference uses accelerated prior evaluation for a stronger baseline.',
        'speedups_are_query_only': True}
    for backend in record['backends']:
        for kind in query:
            value = record['backends'][backend][kind]
            value['reference_over_backend_query_ratio'] = record['backends']['reference'][kind]['median_seconds'] / value['median_seconds']
    np.savez_compressed(out / 'events.npz', distance=distance,
                        **{name + '_' + k: a for name, result in outputs.items() for k, a in result.items()})
    save()
    print(json.dumps(record.get('quality', {}), indent=2), flush=True)


if __name__ == '__main__':
    main()
