"""Compiled SDD checkpoints for the frozen PR118 full marginal task.

The three normal-data RHS and all contact/query cross RHS share stochastic
kernel rows. Output gates determine acceptance; no residual tolerance chosen
for a different solver is imposed. Checkpoint selection is favorable to SDD.
"""
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
    import jax
    import jax.numpy as jnp
    from geospsr.iterative import StochasticDualDescent
    from geospsr.nonperiodic_dense_gpu import full_kernel_matmul
    from reproduction.gpu_audit import GpuProcessAudit
    from reproduction.partial_direct_gpu import cross_rhs
    output = Path(args.output)
    if output.exists(): raise FileExistsError(output)
    x, y, contacts, panel, model, noise, truth, fit, contract = problem(args.n)
    sources = source_snapshot([__file__, 'reproduction/partial_competitor_common.py',
        'reproduction/partial_direct_gpu.py', 'geospsr/iterative.py',
        'geospsr/nonperiodic_dense_gpu.py', 'geospsr/scalar_contacts.py', 'geospsr/nonperiodic.py'])
    jax.devices(); cross_rhs(model, contacts[:2], x[:2]); cp.cuda.Device().synchronize()
    audit = GpuProcessAudit(cp.cuda.Device().pci_bus_id, interval=.1, cuda_device=cp.cuda.Device().id).start()
    wall_start = perf_counter()
    start = perf_counter()
    cross = cross_rhs(model, np.concatenate((contacts, panel['queries'])), x)
    rhs = cp.column_stack((cross, cp.asarray(y)))
    host_rhs = cp.asnumpy(rhs).astype(np.float32)
    cross_seconds = perf_counter()-start
    def kernel(a, b):
        z = jnp.sqrt(jnp.sum((a-b)**2))*model.kappa
        return (1+z)*jnp.exp(-z)
    start = perf_counter()
    solver = StochasticDualDescent(jnp.asarray(x, dtype=jnp.float32), jnp.asarray(host_rhs),
        kernel, noise_variance=noise/model.amplitude, batch=args.batch,
        lr=args.eta*args.batch/len(x), seed=args.seed, momentum=.9, polyak=.01,
        sampling=args.sampling)
    solver.compile()
    compile_seconds = perf_counter()-start
    record = dict(arguments=vars(args), contract=contract, source_sha256=sources,
        cross_seconds=cross_seconds, compile_and_state_seconds=compile_seconds,
        rhs_count=host_rhs.shape[1], history=[],
        stopping_scope='First observed passing output checkpoint; earlier checks excluded from selected total, favoring SDD. Wall time also retained.',
        operator='Unit-amplitude isotropic Matern32 with noise/amplitude; final solutions divided by amplitude')
    elapsed = 0.; accepted = False; selected_mean = selected_variance = None

    def evaluate(z, residual=None):
        start = perf_counter()
        try:
            m, v = conditioned_moments(model, contacts, panel['queries'], rhs, z, cp, residual=residual)
            valid = bool(np.isfinite(m).all() and np.isfinite(v).all() and np.all(v > 0))
            checksum = float(ndtr(-m/np.sqrt(v)).sum()) if valid else None
            cp.cuda.Device().synchronize()
            seconds = perf_counter()-start
            error = gates(m, v, truth, model, fit)
            return m, v, dict(errors=error, moment_and_cdf_seconds=seconds, probability_checksum=checksum)
        except (ValueError, np.linalg.LinAlgError) as exc:
            cp.cuda.Device().synchronize()
            return None, None, dict(errors=dict(passes=False, valid_moments=False),
                moment_and_cdf_seconds=perf_counter()-start, error=f'{type(exc).__name__}: {exc}')

    for checkpoint in args.checkpoints:
        while solver.iterations < checkpoint and elapsed < args.budget:
            start = perf_counter()
            value = solver.advance(min(256, checkpoint-solver.iterations))
            elapsed += perf_counter()-start
        start = perf_counter()
        z = cp.asarray(np.asarray(value, dtype=float)/model.amplitude)
        cp.cuda.Device().synchronize(); transfer = perf_counter()-start
        row = dict(iterations=solver.iterations, updates_seconds=elapsed, transfer_seconds=transfer, variants={})
        m, v, standard = evaluate(z)
        row['variants']['standard'] = standard
        selected = 'standard'; correction_seconds = 0.
        if not standard['errors']['passes'] and args.correction == 'residual' and bool(cp.isfinite(z).all()):
            start = perf_counter()
            residual = rhs-cp.asarray(full_kernel_matmul(model, x, noise, cp.asnumpy(z)))
            cp.cuda.Device().synchronize(); correction_seconds = perf_counter()-start
            cm, cv, corrected = evaluate(z, residual)
            corrected['residual_action_seconds'] = correction_seconds
            row['variants']['residual'] = corrected
            if corrected['errors']['passes']:
                selected = 'residual'; m, v = cm, cv
        variant = row['variants'][selected]
        row.update(selected_variant=selected, passes=variant['errors']['passes'],
            total_seconds=cross_seconds+compile_seconds+elapsed+transfer+variant['moment_and_cdf_seconds']
                          +(correction_seconds if selected == 'residual' else 0.))
        record['history'].append(row)
        record['wall_seconds'] = perf_counter()-wall_start
        if m is not None:
            field_path = output.with_name(output.stem+f'-step{solver.iterations}.npz')
            field_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(field_path, mean=m, variance=v)
            row['field'] = str(field_path)
        store_result(output, record)
        print(json.dumps(row), flush=True)
        selected_mean, selected_variance = m, v
        if row['passes']:
            accepted = True
            break
        if elapsed >= args.budget or not bool(cp.isfinite(z).all()): break
    record['gpu_audit'] = audit.stop()
    record['accepted'] = bool(accepted and record['gpu_audit']['sampled_isolation_pass'])
    record['status'] = 'passing checkpoint' if accepted else 'no passing observed checkpoint'
    record['selected'] = row
    store_result(output, record, selected_mean, selected_variance)
    print('FINAL', record['status'], 'isolated', record['gpu_audit']['sampled_isolation_pass'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n', type=int, default=49152)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--eta', type=float, default=.01)
    p.add_argument('--seed', type=int, default=9101)
    p.add_argument('--sampling', choices=['permutation','rejection'], default='rejection')
    p.add_argument('--checkpoints', type=int, nargs='+', default=[128,512,2048,8192,32768,131072])
    p.add_argument('--budget', type=float, default=90.)
    p.add_argument('--correction', choices=['none','residual'], default='none')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    if args.eta <= 0 or args.budget <= 0: p.error('positive learning rate and budget required')
    if not args.checkpoints or args.checkpoints[0] < 1 or any(b<=a for a,b in zip(args.checkpoints,args.checkpoints[1:])):
        p.error('strictly increasing positive checkpoints required')
    run(args)
