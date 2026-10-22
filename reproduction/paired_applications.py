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
    score = lambda t: np.trapezoid((t >= .05) & (t <= .95), distance, axis=1)
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
    result['quality_gate_version'] = 2
    result['uncertainty_coverage_pass'] = bool(np.any((reference['collision'].mean(0) > 0)
                                                    & (reference['collision'].mean(0) < 1)))
    result['application_validation_complete'] = False
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=['A', 'B'], default='A')
    parser.add_argument('--samples', type=int, default=128)
    parser.add_argument('--sample-batch', type=int, default=4)
    parser.add_argument('--query-batch', type=int, default=1024)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--rebuild-plans', action='store_true', help='reproduce pre-optimization query plan construction')
    parser.add_argument('--backends', nargs='+', choices=['compatible', 'direct'], default=['compatible', 'direct'])
    parser.add_argument('--fluctuation-cutoffs', nargs='*', type=int, default=[],
                        help='experimental centered-sample compression; full mean retained')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.samples < 2 or args.repeats < 1:
        parser.error('need at least two samples and one repeat')
    if any(c < 1 or c > 50 for c in args.fluctuation_cutoffs):
        parser.error('fluctuation cutoffs must be between 1 and 50')
    data = np.load(ROOT / 'reproduction/results/matched' / args.case / 'input.npz')
    x, normals = data['points'], data['normals']
    rays, objects, distance = application_geometry(x)
    query = {'rays': rays.reshape(-1, 3), 'collision': objects.reshape(-1, 3)}
    shapes = {'rays': rays.shape[:2], 'collision': objects.shape[:2]}
    out = args.output or ROOT / 'reproduction/results/paired_applications' / args.case
    out.mkdir(parents=True, exist_ok=True)
    record = {'case': args.case, 'N': len(x), 'samples': args.samples, 'seed': 42,
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
    model = timed('common_fit_and_direct_mean_setup', lambda: Posterior(x, normals))
    timed('common_corrected_draws_and_direct_aggregation', lambda:
          model.draw(args.samples, seed=42, batch=args.sample_batch, retain_components=True))
    k = Matern32Kernel(.01, .1)
    kv = ProductKernel(k, k, k)
    frequencies = trunc_Zd(50)
    ev = jax.vmap(kv.spectral_density)(frequencies) ** .5
    cross = partial(original.poisson_cross_covariances, eigenvectors=frequencies, eigenvalues=ev)
    kfv = timed('reference_kernel_setup', lambda: periodic_stationary_interpolator(cross, 3, 50, exponent=5))
    rows = jax.jit(jax.vmap(jax.vmap(kfv, (None, 0)), (0, None)))
    xj = jnp.asarray(x)
    weights = cp.ascontiguousarray(model.sample_weights.reshape(args.samples, -1).T)
    def reference_offset():
        total = cp.float32(0)
        for start in range(0, len(x), args.query_batch):
            block = own_jax_array(rows(xj[start:start + args.query_batch], xj))
            total += (block * model.alpha).sum()
        return total / len(x)
    offset = timed('reference_mean_offset', reference_offset)
    def evaluate_many(op, q, coefficients):
        if args.rebuild_plans:
            return cp.concatenate([op.evaluate(q, coefficients[i:i + args.sample_batch])
                                   for i in range(0, args.samples, args.sample_batch)])
        return op.evaluate_batched(q, coefficients, args.sample_batch)
    def evaluate_prior(q):
        return evaluate_many(model.op, q, model.sample_prior_coeff)
    def reference(q):
        blocks = []
        for start in range(0, len(q), args.query_batch):
            block = own_jax_array(rows(jnp.asarray(q[start:start + args.query_batch]), xj))
            blocks.append(block.reshape(len(block), -1) @ weights)
        return cp.concatenate(blocks).T + evaluate_prior(q) - offset
    # All backends use the same reference mean offset to isolate evaluation.
    def direct(q):
        return evaluate_many(model.op, q, model.sample_coeff) - offset
    outputs = {}
    def benchmark(name, evaluator):
        outputs[name] = {}
        record.setdefault('backends', {})[name] = {}
        for kind, q in query.items():
            def run():
                event = events(evaluator(q), shapes[kind], kind)
                probability = event.mean(0)
                score = (cp.trapz(((probability >= .05) & (probability <= .95)).astype('f'),
                                 cp.asarray(distance), axis=1) if kind == 'rays' else probability)
                return event, probability, score
            values = timed(name + '_' + kind + '_first', run)
            repetitions = []
            for repeat in range(args.repeats):
                values = timed(name + '_' + kind + '_repeat_' + str(repeat), run)
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
    if 'direct' in args.backends or args.fluctuation_cutoffs:
        benchmark('direct', direct)
        record.setdefault('quality', {})['direct'] = quality(outputs['reference'], outputs['direct'], distance)
        save()
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
