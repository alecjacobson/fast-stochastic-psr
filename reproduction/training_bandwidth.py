"""Analytic-model overlap audit for an independent vector training bandwidth."""
import argparse
import hashlib
import json
from pathlib import Path

import cupy as cp
import cupyx.scipy.linalg as cla
import numpy as np
from scipy.special import ndtr

from geospsr.spectral import FourierKernel
from geospsr.training_spectrum import FourierCovariance, finite_covariance_rows
from reproduction.matched_teaser import ROOT, measure
from reproduction.posterior import Posterior


def conditioned_covariance(chol, cross, prior):
    count, n, _ = cross.shape
    rhs = cross.transpose(1, 0, 2).reshape(n, count * 3)
    z = cla.solve_triangular(chol, rhs, lower=True).reshape(n, count, 3)
    return prior - cp.einsum('nqd,nrd->qr', z, z)


def relative(error, reference):
    return float(cp.linalg.norm(error) / cp.maximum(cp.linalg.norm(reference), 1e-30))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=['A', 'B'], default='A')
    parser.add_argument('--cutoffs', nargs='+', type=int, default=[50, 100, 150, 200])
    parser.add_argument('--probes', type=int, default=256)
    parser.add_argument('--lengthscale', type=float, default=.01)
    parser.add_argument('--noise', type=float, default=.02)
    parser.add_argument('--rtol', type=float, default=1e-5)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if min(args.cutoffs) < 50 or args.probes < 2:
        parser.error('training cutoffs must cover scalar cutoff 50; need at least two probes')
    out = args.output or ROOT / 'reproduction/results/training_bandwidth' / args.case
    out.mkdir(parents=True, exist_ok=True)
    data = np.load(ROOT / 'reproduction/results/matched' / args.case / 'input.npz')
    x, y, query = data['points'], data['normals'], data['query']
    record = {'args': {**vars(args), 'output': str(out)}, 'N': len(x), 'M': len(query),
              'scope': 'independent training bandwidth; scalar/cross spectrum fixed at 101^3; dense full analytic overlap reference',
              'device': cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
              'hashes': {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in
                         [('points', x), ('normals', y), ('query', query)]},
              'quality_gates': {'mean_relative_l2': .002, 'variance_relative_l2': .0002,
                                'covariance_relative_l2': .0002, 'probability_rms': .001,
                                'probability_max': .02, 'analytic_residual': .001},
              'seconds': {}, 'rows': []}
    def save():
        (out / 'results.json').write_text(json.dumps(record, indent=2) + '\n')
    def timed(name, fn):
        value, seconds = measure(fn)
        record['seconds'][name] = seconds
        print(name, seconds, flush=True)
        save()
        return value
    model = timed('analytic_fit_and_scalar_setup', lambda:
                  Posterior(x, y, lengthscale=args.lengthscale, noise_variance=args.noise))
    full_mean = timed('analytic_mean_queries', lambda: model.mean(query))
    # Half deliberately near the zero set, half uniformly sampled from the rest.
    hard = np.argsort(np.abs(cp.asnumpy(full_mean)))[:args.probes // 2]
    remaining = np.setdiff1d(np.arange(len(query)), hard)
    ids = np.r_[hard, np.random.default_rng(192).choice(remaining, args.probes - len(hard), replace=False)]
    q = query[ids]
    cross = timed('shared_cross_rows', lambda: model.op.cross_values(
        (cp.asarray(q)[:, None] - cp.asarray(x)[None]).reshape(-1, 3)).reshape(len(q), len(x), 3))
    norm2 = (model.op.ns ** 2).sum(0)
    scalar_rho = cp.where(norm2 > 0, model.op.rho / cp.maximum(norm2, 1), 0)
    prior = timed('shared_scalar_prior', lambda: model.op.evaluate(
        (q[:, None] - q[None]).reshape(-1, 3), scalar_rho)[0].reshape(len(q), len(q)))
    full_cov = timed('analytic_probe_covariance', lambda: conditioned_covariance(model.L, cross, prior))
    full_var = cp.diag(full_cov)
    full_prob = ndtr(-cp.asnumpy(full_mean[ids]) / np.sqrt(cp.asnumpy(full_var)))
    arrays = {'query_ids': ids, 'probe_query': q, 'analytic_mean': cp.asnumpy(full_mean),
              'analytic_covariance': cp.asnumpy(full_cov), 'analytic_probability': full_prob}
    target = cp.asarray(y)
    for cutoff in args.cutoffs:
        prefix = str(cutoff) + '_'
        vector = timed(prefix + 'spectrum_setup', lambda:
                       FourierCovariance(cutoff, args.lengthscale, .1, upsampfac=1.25))
        dense = timed(prefix + 'independent_finite_covariance', lambda: finite_covariance_rows(x, x, vector))
        dense.flat[::len(x) + 1] += args.noise
        chol = timed(prefix + 'finite_cholesky', lambda: cp.linalg.cholesky(dense))
        finite_cov = timed(prefix + 'finite_probe_covariance', lambda: conditioned_covariance(chol, cross, prior))
        kernel = timed(prefix + 'plan_setup', lambda: FourierKernel(x, vector, args.noise))
        alpha, info = timed(prefix + 'solve_first', lambda: kernel.solve(target, rtol=args.rtol))
        repeats = []
        for repeat in range(3):
            alpha, info = timed(prefix + 'solve_repeat_' + str(repeat), lambda: kernel.solve(target, rtol=args.rtol))
            repeats.append(record['seconds'][prefix + 'solve_repeat_' + str(repeat)])
        analytic_residual = relative(model.K @ alpha - target, target)
        finite_residual = relative(dense @ alpha - target, target)
        coeff = timed(prefix + 'scalar_aggregation', lambda: model.op.aggregate(x, alpha))
        offset = timed(prefix + 'mean_offset', lambda: model.op.evaluate(x, coeff).mean())
        mean = timed(prefix + 'mean_queries', lambda: model.op.evaluate(query, coeff)[0] - offset)
        var = cp.diag(finite_cov)
        probability = ndtr(-cp.asnumpy(mean[ids]) / np.sqrt(cp.asnumpy(var)))
        errors = {'mean_relative_l2': relative(mean - full_mean, full_mean),
                  'variance_relative_l2': relative(var - full_var, full_var),
                  'covariance_relative_l2': relative(finite_cov - full_cov, full_cov),
                  'probability_rms': float(np.sqrt(np.mean((probability - full_prob) ** 2))),
                  'probability_max': float(np.max(np.abs(probability - full_prob))),
                  'analytic_residual': analytic_residual}
        row = {'training_cutoff': cutoff, 'scalar_cutoff': 50, 'finite_solve': info,
               'independent_finite_residual': finite_residual,
               'analytic_diagonal_fraction': float(vector.rho.sum() / .001),
               'median_solve_seconds': float(np.median(repeats)), 'errors': errors,
               'quality_pass': bool(info['converged'] and finite_residual < 2 * args.rtol
                                    and all(errors[k] <= v for k, v in record['quality_gates'].items()))}
        record['rows'].append(row)
        arrays[prefix + 'mean'] = cp.asnumpy(mean)
        arrays[prefix + 'covariance'] = cp.asnumpy(finite_cov)
        arrays[prefix + 'probability'] = probability
        print(json.dumps(row), flush=True)
        save()
        np.savez_compressed(out / 'fields.npz', **arrays)
        del kernel, vector, dense, chol, alpha, coeff, finite_cov, mean
        cp.get_default_memory_pool().free_all_blocks()


if __name__ == '__main__':
    main()
