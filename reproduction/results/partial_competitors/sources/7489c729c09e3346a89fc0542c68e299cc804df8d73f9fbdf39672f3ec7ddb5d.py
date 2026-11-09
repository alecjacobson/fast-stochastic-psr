"""Precision-tuned Cholesky on the frozen real partial-scan probability task."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.special import ndtr

from reproduction.partial_competitor_common import (
    problem, source_snapshot, conditioned_moments, gates, store_result)


def run(args):
    import cupy as cp
    from cupyx.scipy.linalg import solve_triangular
    from geospsr.nonperiodic_dense_gpu import GPUDenseMaternSystem, full_kernel_matmul
    from reproduction.gpu_audit import GpuProcessAudit
    from reproduction.partial_direct_gpu import cross_rhs
    output = Path(args.output)
    if output.exists(): raise FileExistsError(output)
    x, y, contacts, panel, model, noise, truth, fit, contract = problem(args.n)
    sources = source_snapshot([__file__, 'reproduction/partial_competitor_common.py',
        'reproduction/partial_direct_gpu.py', 'geospsr/nonperiodic_dense_gpu.py',
        'geospsr/scalar_contacts.py', 'geospsr/nonperiodic.py'])
    cross_rhs(model, contacts[:2], x[:2]); cp.cuda.Device().synchronize()
    audit = GpuProcessAudit(cp.cuda.Device().pci_bus_id, interval=.1, cuda_device=cp.cuda.Device().id).start()
    record = dict(arguments=vars(args), contract=contract, source_sha256=sources,
                  scope='Fresh setup + all mean/variance/inside-CDF outputs; common input and CUDA initialization excluded')
    begin = perf_counter(); mean = variance = None
    try:
        system = GPUDenseMaternSystem(model, x, noise, dtype=args.dtype)
        cp.cuda.Device().synchronize(); record['factor_seconds'] = perf_counter()-begin
        start = perf_counter()
        cross = cross_rhs(model, np.concatenate((contacts, panel['queries'])), x)
        rhs = cp.column_stack((cross, cp.asarray(y)))
        cp.cuda.Device().synchronize(); record['cross_seconds'] = perf_counter()-start
        start = perf_counter()
        w = solve_triangular(system.factor, rhs.astype(system.dtype), lower=True, check_finite=False)
        if args.correction == 'none':
            values, residual, whitened = w.astype(cp.float64), None, True
        else:
            z = solve_triangular(system.factor, w, lower=True, trans='T', check_finite=False).astype(cp.float64)
            residual = rhs-cp.asarray(full_kernel_matmul(model, x, noise, cp.asnumpy(z)))
            for _ in range(args.refinements):
                z += cp.asarray(system.solve(cp.asnumpy(residual)))
                residual = rhs-cp.asarray(full_kernel_matmul(model, x, noise, cp.asnumpy(z)))
            values, whitened = z, False
        cp.cuda.Device().synchronize(); record['solve_seconds'] = perf_counter()-start
        start = perf_counter()
        mean, variance = conditioned_moments(model, contacts, panel['queries'], rhs,
                                             values, cp, whitened=whitened, residual=residual)
        if not (np.isfinite(mean).all() and np.isfinite(variance).all() and np.all(variance > 0)):
            raise ValueError('invalid predicted moments; no clipping')
        record['probability_checksum'] = float(ndtr(-mean/np.sqrt(variance)).sum())
        cp.cuda.Device().synchronize(); record['moment_and_cdf_seconds'] = perf_counter()-start
        record['total_seconds'] = perf_counter()-begin
        record['status'] = 'completed'
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        record.update(status='failed', error=f'{type(exc).__name__}: {exc}', wall_seconds=perf_counter()-begin)
    finally:
        record['gpu_audit'] = audit.stop()
    if mean is not None:
        record['errors'] = gates(mean, variance, truth, model, fit)
    record['accepted'] = bool(record.get('errors', {}).get('passes') and record['gpu_audit']['sampled_isolation_pass'])
    store_result(output, record, mean, variance)
    print(json.dumps({k: v for k, v in record.items() if k not in ['gpu_audit','source_sha256','contract']}, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n', type=int, default=49152)
    p.add_argument('--dtype', choices=['float32','float64'], default='float32')
    p.add_argument('--correction', choices=['none','residual'], default='none')
    p.add_argument('--refinements', type=int, default=0)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    if args.refinements < 0 or (args.refinements and args.correction == 'none'): p.error('refinements require residual mode')
    run(args)
