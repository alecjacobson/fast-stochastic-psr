"""Dense-validated posterior covariance bracket and large-N tail diagnostic."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import cupy as cp
import cupyx.scipy.linalg as cla
import numpy as np

from geospsr.spectral import FourierPoisson, FourierKernel
from geospsr.training_spectrum import FourierCovariance, finite_covariance_rows
from geospsr.spectral_tail import (analytic_covariance_rows64, sample_tail_frequencies,
                                   tail_quadratic_samples)

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['A', 'B', '100000', '1000000'], default='A')
    parser.add_argument('--cutoff', type=int, default=200)
    parser.add_argument('--lengthscale', type=float, default=.01)
    parser.add_argument('--noise', type=float, default=.02)
    parser.add_argument('--probes', type=int, default=16)
    parser.add_argument('--frequencies', type=int, default=16384)
    parser.add_argument('--seeds', type=int, nargs='+', default=[71, 72, 73])
    parser.add_argument('--rtol', type=float, default=1e-6)
    parser.add_argument('--eps', type=float, default=2e-7)
    parser.add_argument('--precision', choices=['float32', 'float64'], default='float32',
                        help='large-N arithmetic; dense overlap always uses float64')
    parser.add_argument('--probe-indices', type=int, nargs='+',
                        help='optional subset of the reproducibly selected panel')
    parser.add_argument('--training-upsampfac', type=float, choices=[1.25, 2.0], default=1.25)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.probes < 2 or args.frequencies < 2 or args.cutoff < 50:
        parser.error('need at least two probes/frequencies and training cutoff >= 50')
    if min(args.lengthscale, args.noise, args.rtol, args.eps) <= 0:
        parser.error('lengthscale, noise variance and solver/transform tolerances must be positive')
    if args.probe_indices is not None and (len(set(args.probe_indices)) != len(args.probe_indices)
            or min(args.probe_indices) < 0 or max(args.probe_indices) >= args.probes):
        parser.error('probe indices must be distinct members of the selected panel')
    out = args.output or ROOT / 'reproduction/results/spectral_tail' / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    small = args.dataset in ('A', 'B')
    if small:
        data = np.load(ROOT / 'reproduction/results/matched' / args.dataset / 'input.npz')
        x = data['points'].astype('f')
        panel = np.load(ROOT / 'reproduction/results/training_bandwidth' / args.dataset / 'fields.npz')
        all_q, mean = data['query'], panel['analytic_mean']
    else:
        data = np.load(ROOT / 'reproduction/results/teaser/cloud_1000000.npz')
        ids = np.random.default_rng(19).permutation(len(data['points']))[:int(args.dataset)]
        x = data['points'][ids].astype('f')
        panel = np.load(ROOT / f'reproduction/results/training_bandwidth/scaling/fields_{args.dataset}.npz')
        all_q, mean = panel['query'], panel['200_mean']
    # Query panel fixed before drawing tail frequencies; no selection by estimated error.
    rng = np.random.default_rng(993)
    hard_pool = np.argsort(np.abs(mean))[:256]
    hard = rng.choice(hard_pool, args.probes // 2, replace=False)
    ids = np.r_[hard, rng.choice(np.setdiff1d(np.arange(len(all_q)), hard_pool),
                                args.probes - len(hard), replace=False)]
    if args.probe_indices is not None:
        ids = ids[args.probe_indices]
    q = all_q[ids].astype('f')
    n, count = len(x), len(q)
    result = {'args': {**vars(args), 'output': str(out)}, 'N': n,
              'device': cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
              'hashes': {key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in [('points', x), ('queries', q)]},
              'scope': 'training-tail posterior covariance diagnostic at a fixed finite scalar/cross cutoff50; not posterior event or geometry convergence',
              'query_selection': 'seed993; half from 256 smallest absolute default-parameter mean slice values, half from remaining slice positions',
              'dense_reference': small, 'normalization':'analytic',
              'complete': False, 'seconds': {}, 'seeds': []}
    result['training_nufft_effective_eps'] = max(args.eps, float(np.finfo(args.precision).eps)) if not small else None
    arrays = {'query': q, 'query_ids': ids}
    def save():
        (out / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    def timed(name, fn):
        cp.cuda.runtime.deviceSynchronize()
        start = time.perf_counter()
        value = fn()
        cp.cuda.runtime.deviceSynchronize()
        result['seconds'][name] = time.perf_counter() - start
        print(name, result['seconds'][name], flush=True)
        save()
        return value
    # Double scalar kernel/conditioning in overlap; production float32 at large N.
    dtype = 'float64' if small else args.precision
    # Oversampling 2 supports a wider spreading kernel for the float64 control.
    upsampfac = 2.0 if dtype == 'float64' else 1.25
    op = FourierPoisson(50, args.lengthscale, dtype=dtype,
                        eps=1e-11 if small else args.eps, upsampfac=upsampfac)
    vector = FourierCovariance(args.cutoff, args.lengthscale, eps=args.eps,
                               dtype=dtype, upsampfac=args.training_upsampfac)
    xg = cp.asarray(x)
    cross = timed('cross_rows', lambda: cp.stack([op.cross_values(qi[None] - xg) for qi in cp.asarray(q, dtype=op.dtype)], axis=1))
    # Layout N,Q,3, shared by exact matrices and sampled feature quadratics.
    norm2 = (op.ns ** 2).sum(0)
    rho_f = cp.where(norm2 > 0, op.rho / cp.maximum(norm2, 1), 0)
    q_precise = q.astype(dtype)
    prior = timed('scalar_prior', lambda: op.evaluate((q_precise[:, None] - q_precise[None]).reshape(-1, 3), rho_f)[0].reshape(count, count).astype('d'))
    if small:
        analytic = timed('dense_analytic', lambda: analytic_covariance_rows64(x, x, args.lengthscale))
        finite = timed('dense_finite', lambda: finite_covariance_rows(x, x, vector, dtype='d'))
        delta = analytic - finite
        analytic.flat[::n+1] += args.noise
        finite.flat[::n+1] += args.noise
        def solve(matrix):
            l = cp.linalg.cholesky(matrix)
            c = cross.reshape(n, -1)
            return cla.solve_triangular(l.T, cla.solve_triangular(l, c, lower=True)).reshape(n, count, 3)
        a = timed('dense_finite_solve', lambda: solve(finite))
        afull = timed('dense_analytic_solve', lambda: solve(analytic))
        full_cov = prior - cp.einsum('iqd,ikd->qk', cross, afull)
        exact_h = timed('exact_tail_quadratic', lambda:
                        cp.einsum('iqd,ikd->qk', a, (delta @ a.reshape(n, -1)).reshape(n, count, 3)))
        del analytic, finite, delta, afull
    else:
        kernel = timed('plans', lambda: FourierKernel(x, vector, args.noise))
        solved, info = [], []
        def solve_all():
            for j in range(count):
                value, record = kernel.solve(cross[:, j], rtol=args.rtol, maxiter=1000)
                if not record['converged']:
                    raise RuntimeError(record)
                solved.append(value)
                info.append(record)
                print('cross solve', j + 1, record['iterations'], flush=True)
            return cp.stack(solved, axis=1)
        a = timed('finite_cross_solves', solve_all)
        result['solves'] = info
        del kernel, solved
    finite_cov = prior - cp.einsum('iqd,ikd->qk', cross.astype('d'), a.astype('d'))
    finite_cov = (finite_cov + finite_cov.T) / 2
    scale = float(cp.linalg.norm(finite_cov))
    variance = cp.asnumpy(cp.diag(finite_cov))
    if variance.min() <= 0:
        raise ArithmeticError('nonpositive posterior variance')
    arrays['finite_covariance'] = cp.asnumpy(finite_cov)
    if small:
        gap = full_cov - finite_cov
        gap, exact_h = (gap + gap.T) / 2, (exact_h + exact_h.T) / 2
        result['exact'] = {
            'covariance_gap_relative_frobenius': float(cp.linalg.norm(gap)) / scale,
            'upper_bound_relative_frobenius': float(cp.linalg.norm(exact_h)) / scale,
            'trace_upper_bound_over_covariance_frobenius': float(cp.trace(exact_h)) / scale,
            'max_variance_gap_fraction': float(cp.max(cp.diag(gap) / cp.asarray(variance))),
            'max_variance_bound_fraction': float(cp.max(cp.diag(exact_h) / cp.asarray(variance))),
            'min_gap_eigenvalue': float(cp.linalg.eigvalsh(gap).min()),
            'min_bound_minus_gap_eigenvalue': float(cp.linalg.eigvalsh(exact_h-gap).min()),
            'min_finite_covariance_eigenvalue': float(cp.linalg.eigvalsh(finite_cov).min())}
        if min(result['exact']['min_gap_eigenvalue'],
               result['exact']['min_bound_minus_gap_eigenvalue']) < -1e-10 * scale:
            raise ArithmeticError('dense posterior covariance bracket failed')
        # A permissive roundoff test must not certify a gap too small to
        # resolve by subtracting the two posterior covariance matrices.
        result['exact']['bracket_within_numerical_tolerance'] = True
        result['exact']['gap_roundoff_indicator'] = 64*np.finfo(float).eps
        result['exact']['covariance_gap_resolved_above_roundoff'] = bool(
            result['exact']['covariance_gap_relative_frobenius'] > 64*np.finfo(float).eps)
        arrays.update(analytic_covariance=cp.asnumpy(full_cov), exact_bound=cp.asnumpy(exact_h))
        print('exact', json.dumps(result['exact']), flush=True)
    # A is fixed before frequency sampling. This is a stochastic estimate of H,
    # not a deterministic bound; uncertainty applies to all reported ratios.
    for seed in args.seeds:
        modes, mass = sample_tail_frequencies(args.frequencies, args.cutoff, args.lengthscale, seed=seed)
        draws = timed(f'tail_features_{seed}', lambda: tail_quadratic_samples(xg, a, modes, mass,
                       progress=lambda done: print('tail frequencies', seed, done, flush=True) if done % 4096 == 0 else None))
        h = draws.mean(0)
        diag = np.diagonal(draws, axis1=1, axis2=2)
        diag_se = diag.std(0, ddof=1) / np.sqrt(len(draws))
        traces = np.trace(draws, axis1=1, axis2=2)
        trace_se = traces.std(ddof=1) / np.sqrt(len(draws))
        row = {'seed': seed, 'tail_mass': mass, 'max_absolute_frequency': int(np.abs(modes).max()),
               'trace_bound_estimate_over_covariance_frobenius': float(traces.mean() / scale),
               'trace_bound_standard_error_over_covariance_frobenius': float(trace_se / scale),
               'max_variance_bound_estimate_fraction': float(np.max(np.diag(h) / variance)),
               'max_variance_bound_marginal_normal_95_upper_fraction': float(np.max((np.diag(h)+1.96*diag_se) / variance)),
               'qualification': 'empirical standard errors and marginal normal intervals; not rigorous or simultaneous certificates; finite solver/NUFFT errors separate'}
        if small:
            eh = cp.asnumpy(exact_h)
            row['bound_estimate_relative_frobenius_error'] = float(np.linalg.norm(h-eh)/np.linalg.norm(eh))
            row['trace_error_standard_errors'] = float((traces.mean()-np.trace(eh))/trace_se)
            row['max_diagonal_error_standard_errors'] = float(np.max(np.abs(np.diag(h-eh))/diag_se))
        result['seeds'].append(row)
        arrays[f'{seed}_bound_estimate'] = h
        arrays[f'{seed}_diagonal_standard_error'] = diag_se
        arrays[f'{seed}_frequencies'] = modes
        arrays[f'{seed}_diagonal_samples'] = diag
        arrays[f'{seed}_trace_samples'] = traces
        print('tail estimate', json.dumps(row), flush=True)
        save()
        np.savez_compressed(out / 'fields.npz', **arrays)
    result['complete'] = True
    save()


if __name__ == '__main__':
    main()
