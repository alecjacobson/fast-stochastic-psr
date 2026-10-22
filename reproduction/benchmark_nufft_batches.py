"""Replay common conditional RHS to isolate NUFFT batching from random draws.

Large arrays live in an explicit scratch directory, not the public repository.
Preparation uses the existing cutoff200 vector/cutoff50 scalar joint prior.
This is a solver-batching experiment, not full-analytic posterior validation.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
from shutil import copyfile
import time

import cupy as cp
import numpy as np

from geospsr.spectral import FourierKernel, FourierPoisson
from geospsr.training_spectrum import FourierCovariance
from reproduction.spectral_posterior import SpectralPosterior, solve_centered_fluctuations
from reproduction.gpu_audit import GpuProcessAudit
from reproduction.paired_applications import (application_geometry, uncertain_objects,
                                             events, field_quality, quality)

ROOT = Path(__file__).resolve().parents[1]


def measure(fn):
    cp.cuda.runtime.deviceSynchronize()
    start = time.perf_counter()
    value = fn()
    cp.cuda.runtime.deviceSynchronize()
    return value, time.perf_counter() - start


def prior_rhs(model, rng, root_rho, denominator):
    """One draw, preserving SpectralPosterior.draw(batch=1)'s random sequence."""
    vector = rng.standard_normal((3, *model.training.shape), dtype='f').astype('F')
    vector.imag = -rng.standard_normal(vector.shape, dtype='f')
    vector *= root_rho[None]
    pad = (model.training.shape[0] - model.op.shape[0]) // 2
    sl = slice(pad, pad + model.op.shape[0])
    scalar = -1j * (vector[:, sl, sl, sl] * model.op.ns).sum(0) / denominator
    values = model.K.backward.execute(vector).real.T
    noise = rng.standard_normal(values.shape, dtype='f') * np.sqrt(model.noise_variance)
    return model.normals - values - noise, scalar


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['prepare', 'solve', 'sdd'])
    p.add_argument('--scratch', type=Path, required=True)
    p.add_argument('--dataset', choices=['B', '1000000'], default='1000000')
    p.add_argument('--samples', type=int, default=8)
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--rtol', type=float, default=1e-5, help='NUFFT-CG relative residual tolerance')
    p.add_argument('--initial-mean', action='store_true', help='start either solver from the common finite posterior mean weights')
    p.add_argument('--solve-cutoff',type=int,default=200)
    p.add_argument('--block-core',type=int,help='experimental positive-block core for centered inverse; solve-cutoff is its extent')
    p.add_argument('--centered-fluctuations',action='store_true',help='experimental inverse on centered fluctuations; preserve cutoff200 mean/prior')
    p.add_argument('--time-budget', type=float, default=1800, help='SDD update budget, seconds')
    p.add_argument('--chunk', type=int, default=512, help='SDD steps between residual checks')
    p.add_argument('--sdd-row-batch', type=int, default=512)
    p.add_argument('--eta', type=float, default=5.)
    p.add_argument('--stop-tolerance', type=float, default=.0005)
    p.add_argument('--stop-rows', type=int, default=512)
    p.add_argument('--stream-stop-rows',action='store_true',help='SDD: regenerate small analytic validation blocks instead of retaining an N-by-stop_rows matrix')
    p.add_argument('--holdout-rows', type=int, default=1024)
    p.add_argument('--holdout-seed', type=int, default=917)
    p.add_argument('--retain-weights', action='store_true', help='save final weights under scratch for independent diagnostics')
    p.add_argument('--audit-analytic', action='store_true')
    p.add_argument('--reference', type=Path, help='compact output from a common-RHS reference run')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if not 50 <= a.solve_cutoff <= 200 or (a.solve_cutoff != 200 and not a.centered_fluctuations):
        p.error('a narrower solve cutoff requires centered fluctuations and must cover scalar cutoff50')
    if a.centered_fluctuations and (a.mode != 'solve' or a.initial_mean):
        p.error('centered fluctuations require solve mode without --initial-mean')
    if a.block_core is not None and (not a.centered_fluctuations or not 50 <= a.block_core <= a.solve_cutoff):
        p.error('block core requires centered fluctuations and 50 <= core <= solve cutoff <= 200')
    work_started = time.perf_counter()
    if min(a.samples, a.batch, a.repeats, a.chunk, a.sdd_row_batch,a.stop_rows,a.holdout_rows) < 1 or min(a.rtol,a.time_budget,a.eta,a.stop_tolerance) <= 0:
        p.error('counts, tolerance and time budget must be positive')
    a.output.mkdir(parents=True, exist_ok=True)
    record = {'args': {**vars(a), 'scratch': str(a.scratch), 'reference': str(a.reference) if a.reference else None,
                       'output': str(a.output)}, 'complete': False, 'seconds': {}, 'device_used_bytes_at_stage': {},
              'scope': 'common finite-vector/cutoff50-scalar priors; conditional solve batching only',
              'device': cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}
    if a.centered_fluctuations:
        record['scope'] = ('experimental centered sampler: cutoff200 prior/mean and '
                           f'cutoff{a.solve_cutoff} fluctuation inverse; not an analytic posterior sampler')
        if a.block_core is not None:
            record['scope'] += f'; inverse retains only core{a.block_core} plus one-high-axis slabs'
    def save():
        (a.output / 'results.json').write_text(json.dumps(record, indent=2)+'\n')
    def timed(name, fn):
        value, elapsed = measure(fn)
        record['seconds'][name] = elapsed
        free,total = cp.cuda.runtime.memGetInfo()
        record['device_used_bytes_at_stage'][name] = total-free
        save()
        print(name, elapsed, flush=True)
        return value
    save()
    audit = GpuProcessAudit(cp.cuda.Device().pci_bus_id).start()
    try:
        if a.mode == 'prepare':
            a.scratch.mkdir(parents=True, exist_ok=True)
            if a.dataset == 'B':
                data = np.load(ROOT / 'reproduction/results/matched/B/input.npz')
                x, y = data['points'].astype('f'), data['normals'].astype('f')
            else:
                data = np.load(ROOT / 'reproduction/results/teaser/cloud_1000000.npz')
                ids = np.random.default_rng(19).permutation(len(data['points']))
                x, y = data['points'][ids].astype('f'), data['normals'][ids].astype('f')
            record['N'] = len(x)
            record['hashes'] = {k: hashlib.sha256(v.tobytes()).hexdigest() for k,v in [('points',x),('normals',y)]}
            np.save(a.scratch / 'points.npy', x)
            np.save(a.scratch / 'normals.npy', y)
            model = timed('mean_fit', lambda: SpectralPosterior(x, y, training_cutoff=200))
            np.save(a.scratch / 'mean_alpha.npy',cp.asnumpy(model.alpha))
            rhs_file = np.lib.format.open_memmap(a.scratch / 'rhs.npy', mode='w+', dtype='f', shape=(a.samples,len(x),3))
            prior_file = np.lib.format.open_memmap(a.scratch / 'prior.npy', mode='w+', dtype='F', shape=(a.samples,*model.op.shape))
            rng = cp.random.RandomState(a.seed)
            root = cp.sqrt(model.training.rho)
            denominator = cp.maximum((model.op.ns**2).sum(0),1)
            record['prior_generation_seconds'] = []
            digest = hashlib.sha256()
            for i in range(a.samples):
                (rhs, prior), elapsed = measure(lambda: prior_rhs(model, rng, root, denominator))
                record['prior_generation_seconds'].append(elapsed)
                host = cp.asnumpy(rhs)
                digest.update(host.tobytes())
                rhs_file[i], prior_file[i] = host, cp.asnumpy(prior)
                if (i+1) % 8 == 0:
                    print('prepared', i+1, flush=True)
            rhs_file.flush(); prior_file.flush()
            record['hashes']['all_rhs'] = digest.hexdigest()
            rays, objects, distance = application_geometry(x)
            norm2 = (model.op.ns**2).sum(0)
            std = float(cp.sqrt(cp.where(norm2 > 0, model.op.rho / cp.maximum(norm2,1),0).sum()))
            objects = timed('uncertain_geometry', lambda: uncertain_objects(model.mean, objects, std))
            probe = np.random.default_rng(975).uniform(-np.pi,np.pi,(257,3)).astype('f')
            np.savez(a.scratch / 'geometry.npz', rays=rays, objects=objects, distance=distance,
                     query=probe, offset=cp.asnumpy(model.offset))
            record['geometry_sha256'] = hashlib.sha256((a.scratch / 'geometry.npz').read_bytes()).hexdigest()
            copyfile(a.scratch / 'geometry.npz',a.output / 'geometry.npz')
            record['complete'] = True
            (a.scratch / 'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
        else:
            manifest = json.loads((a.scratch / 'manifest.json').read_text())
            if not manifest['complete'] or a.samples > manifest['args']['samples']:
                raise ValueError('incomplete or insufficient prepared RHS')
            if a.dataset != manifest['args']['dataset']:
                raise ValueError('dataset does not match prepared inputs')
            record['source_manifest'] = manifest
            if a.reference:
                ref_record = json.loads((a.reference / 'results.json').read_text())
                if not ref_record['complete'] or (ref_record['args']['mode'] == 'sdd' and
                        (not ref_record['sdd_converged'] or
                         ref_record['analytic_holdout']['max_upper_endpoint'] >= .001)):
                    raise ValueError('reference is incomplete or fails convergence/analytic holdout')
            x = np.load(a.scratch / 'points.npy')
            source = np.load(a.scratch / 'rhs.npy', mmap_mode='r')
            prior = np.load(a.scratch / 'prior.npy', mmap_mode='r')
            geometry = np.load(a.scratch / 'geometry.npz')
            record['geometry_sha256'] = hashlib.sha256((a.scratch / 'geometry.npz').read_bytes()).hexdigest()
            record['N'] = len(x)
            rhs = timed('rhs_transfer', lambda: cp.asarray(np.asarray(source[:a.samples]).transpose(1,0,2).reshape(len(x),-1)))
            if a.mode == 'sdd' or a.audit_analytic:
                import jax
                import jax.numpy as jnp
                from geospsr.iterative import StochasticDualDescent
                from reproduction.matched_teaser import own_jax_array, Matern32Kernel, ProductKernel
                from reproduction.benchmark_conditional_solves import residual_panel
                k = Matern32Kernel(.01,.1)
                kv = ProductKernel(k,k,k)
                xj = jnp.asarray(x)
                exact_rows = jax.jit(jax.vmap(jax.vmap(kv,(None,0)),(0,None)))
                def analytic_rows(ids):
                    return cp.concatenate([own_jax_array(exact_rows(jnp.asarray(x[ids[i:i+64]]),xj))
                                           for i in range(0,len(ids),64)])
                def analytic_residual(ids,weights):
                    residual=[]
                    for i in range(0,len(ids),64):
                        block=ids[i:i+64]
                        residual.append(analytic_rows(block)@weights+.02*weights[block]-rhs[block])
                    return residual_panel(cp.concatenate(residual),rhs[ids])
            if a.mode == 'sdd':
                if a.repeats != 1:
                    raise ValueError('SDD confirmation currently requires --repeats 1')
                record['sdd_parameters'] = {'seed':0,'row_batch':a.sdd_row_batch,'eta':a.eta,'momentum':.9,
                                            'polyak':.01,'stop_max_bootstrap_upper':a.stop_tolerance,
                                            'stream_stop_rows':a.stream_stop_rows}
                ids = np.random.default_rng(73).choice(len(x),a.stop_rows,replace=False)
                stop = None if a.stream_stop_rows else timed('analytic_stop_rows', lambda: analytic_rows(ids))
                # Release temporary concatenation blocks before JAX allocates
                # its update workspace; the retained stopping matrix stays live.
                gc.collect(); cp.get_default_memory_pool().free_all_blocks()
                def setup():
                    initial = None
                    if a.initial_mean:
                        alpha = np.load(a.scratch / 'mean_alpha.npy')
                        record['initial_mean_sha256'] = hashlib.sha256(alpha.tobytes()).hexdigest()
                        initial = jnp.tile(jnp.asarray(alpha),(1,a.samples))
                    sdd = StochasticDualDescent(xj,jnp.asarray(cp.asnumpy(rhs)),kv,batch=a.sdd_row_batch,
                                               lr=a.eta*a.sdd_row_batch/len(x),initial=initial)
                    sdd.compile()
                    return sdd
                sdd = timed('sdd_setup',setup)
                elapsed, validation_elapsed, history = 0., 0., []
                while elapsed < a.time_budget:
                    value, duration = measure(lambda: sdd.advance(a.chunk))
                    elapsed += duration
                    weights = own_jax_array(value)
                    check_started=time.perf_counter()
                    stats = analytic_residual(ids,weights) if stop is None else residual_panel(
                        stop @ weights + .02*weights[ids] - rhs[ids],rhs[ids])
                    check_elapsed=time.perf_counter()-check_started
                    validation_elapsed+=check_elapsed
                    history.append({'iterations':sdd.iterations,'update_seconds':elapsed,
                                    'stopping_check_seconds':check_elapsed,'analytic_residual':stats})
                    record['sdd_history'] = history
                    save()
                    print('SDD',sdd.iterations,elapsed,stats['max_upper_endpoint'],flush=True)
                    if stats['max_upper_endpoint'] < a.stop_tolerance or not np.isfinite(stats['max_upper_endpoint']):
                        break
                record['sdd_converged'] = stats['max_upper_endpoint'] < a.stop_tolerance
                record['seconds']['sdd_updates'] = elapsed
                record['seconds']['sdd_stopping_checks'] = validation_elapsed
                del sdd,value,stop
            else:
                if a.block_core is None:
                    vector = timed('spectrum', lambda: FourierCovariance(a.solve_cutoff))
                    kernel_type = FourierKernel
                else:
                    from geospsr.block_spectral import BlockSpectrum,BlockKernel
                    vector = timed('spectrum',lambda:BlockSpectrum(a.block_core,a.solve_cutoff))
                    kernel_type = BlockKernel
                sizes = sorted(set(min(a.batch,a.samples-i) for i in range(0,a.samples,a.batch)))
                kernels = timed('plans', lambda: {size: kernel_type(x,vector,.02,rhs_count=3*size) for size in sizes})
                initial = None
                if a.initial_mean or a.centered_fluctuations:
                    alpha = np.load(a.scratch / 'mean_alpha.npy')
                    key = 'held_mean_sha256' if a.centered_fluctuations else 'initial_mean_sha256'
                    record[key] = hashlib.sha256(alpha.tobytes()).hexdigest()
                if a.initial_mean:
                    initial = {size:cp.tile(cp.asarray(alpha),(1,size)) for size in sizes}
                if a.centered_fluctuations:
                    normals = cp.asarray(np.load(a.scratch / 'normals.npy'))
                    mean_alpha = cp.asarray(alpha)
                record['iterations'] = []
                def solve_all():
                    result = cp.empty_like(rhs)
                    info = []
                    for start in range(0,a.samples,a.batch):
                        size = min(a.batch,a.samples-start)
                        sl = slice(3*start,3*(start+size))
                        if a.centered_fluctuations:
                            value,stats = solve_centered_fluctuations(
                                kernels[size],rhs[:,sl],normals,mean_alpha,rtol=a.rtol)
                        else:
                            value, stats = kernels[size].solve(rhs[:,sl],rtol=a.rtol,maxiter=1000,
                                                              x0=None if initial is None else initial[size])
                        if not stats['converged']:
                            raise RuntimeError(stats)
                        result[:,sl] = value
                        info.append(stats)
                        print('solved samples',start+size,flush=True)
                    record['iterations'].append(info)
                    return result
                for repeat in range(a.repeats):
                    weights = timed(f'solve_{repeat}', solve_all)
                record['median_solve_seconds'] = float(np.median([record['seconds'][f'solve_{r}'] for r in range(a.repeats)]))
                del kernels, vector, initial
                if a.centered_fluctuations:
                    del normals,mean_alpha
            gc.collect(); cp.get_default_memory_pool().free_all_blocks()
            if a.mode == 'sdd' or a.audit_analytic:
                # Draw from a fresh seed and exclude the fixed stopping panel.
                # This is still an empirical sample, not a full-system bound.
                stop_ids = np.random.default_rng(73).choice(len(x),a.stop_rows,replace=False)
                pool = np.setdiff1d(np.arange(len(x)),stop_ids,assume_unique=True)
                ids = np.random.default_rng(a.holdout_seed).choice(pool,a.holdout_rows,replace=False)
                record['holdout_excludes_stopping_rows'] = True
                record['holdout_indices_sha256'] = hashlib.sha256(ids.tobytes()).hexdigest()
                record['analytic_holdout'] = timed('analytic_holdout',lambda:analytic_residual(ids,weights))
            if a.retain_weights:
                target = a.scratch / (a.output.name + '-weights.npy')
                np.save(target,cp.asnumpy(weights))
                record['retained_weights'] = str(target)
                with target.open('rb') as stream:
                    record['retained_weights_sha256'] = hashlib.file_digest(stream,'sha256').hexdigest()
            del rhs
            gc.collect(); cp.get_default_memory_pool().free_all_blocks()
            op = FourierPoisson(50)
            coeff = timed('scalar_aggregation', lambda: cp.concatenate([
                op.aggregate(x,weights[:,3*i:3*min(i+4,a.samples)].reshape(len(x),-1,3).transpose(1,0,2))
                + cp.asarray(np.asarray(prior[i:min(i+4,a.samples)])) for i in range(0,a.samples,4)]))
            del weights
            fields = timed('probe_fields', lambda: cp.asnumpy(op.evaluate_batched(geometry['query'],coeff,batch=4)-float(geometry['offset'])))
            outputs = {}
            for key,pos in [('rays',geometry['rays']),('collision',geometry['objects'])]:
                outputs[key] = timed(key, lambda: cp.asnumpy(events(op.evaluate_batched(pos.reshape(-1,3),coeff,batch=4)-float(geometry['offset']),pos.shape[:2],key)))
            if a.reference:
                if ref_record['source_manifest']['hashes'] != manifest['hashes']:
                    raise ValueError('reference inputs/RHS differ')
                if ref_record.get('geometry_sha256') != record['geometry_sha256']:
                    raise ValueError('reference query geometry or offset differs')
                ref = np.load(a.reference / 'fields.npz')
                if ref['fields'].shape != fields.shape:
                    raise ValueError('reference sample/query count differs')
                record['quality'] = {'fields': field_quality(cp.asarray(ref['fields']),cp.asarray(fields)),
                                     'events': quality({k:ref[k] for k in outputs},outputs,geometry['distance'])}
                # A core equal to its extent is the complete cube, so it
                # requires the stricter same-model numerical gates.
                candidate_core = a.block_core if a.block_core is not None and a.block_core<a.solve_cutoff else None
                reference_core = ref_record['args'].get('block_core')
                if reference_core == ref_record['args'].get('solve_cutoff',200):
                    reference_core = None
                model_comparison = ((a.mode == 'sdd') != (ref_record['args']['mode'] == 'sdd')
                                    or a.solve_cutoff != ref_record['args'].get('solve_cutoff',200)
                                    or candidate_core != reference_core)
                record['quality']['comparison_kind'] = 'different inverse models' if model_comparison else 'same model numerical comparison'
                gates = {'error_over_fluctuation_l2':.002 if model_comparison else 5e-4,
                         'empirical_variance_relative_l2':.0002 if model_comparison else 1e-4,
                         'empirical_covariance_relative_l2':.0002 if model_comparison else 1e-4}
                f = record['quality']['fields']
                f['pass'] = all(f[key] < value for key,value in gates.items())
                record['quality']['field_gates'] = gates
            np.savez_compressed(a.output / 'fields.npz',fields=fields,**outputs)
            record['complete'] = True
        record['benchmark_work_wall_seconds'] = time.perf_counter()-work_started
        record['gpu_process_audit'] = audit.stop()
        save()
    except Exception as exc:
        record['failure'] = f'{type(exc).__name__}: {exc}'
        record['benchmark_work_wall_seconds'] = time.perf_counter()-work_started
        record['gpu_process_audit'] = audit.stop()
        save()
        raise


if __name__ == '__main__':
    main()
