"""Precision/storage controls and larger-N continuation of the PR120 workload."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.special import ndtr

from geospsr.nonperiodic import FreeSpacePoissonMatern
from reproduction.partial_occupancy import ROOT, load
from reproduction.partial_competitor_common import conditioned_moments, gates, store_result
from reproduction.stanford_partial_data import digest

RESULTS = Path('reproduction/results/mixed_h2')


def task(n):
    x, y, contacts, panel = load('test_side', n)
    ids = np.linspace(0, len(panel['queries'])-1, 128, dtype=int)
    panel = {k: v[ids] for k, v in panel.items()}
    selected = json.loads((ROOT/'selected.json').read_text())
    model = FreeSpacePoissonMatern(selected['ell'], 1/3)
    fit = json.loads((ROOT/'controls.json').read_text())['fits']['spatial']
    ref = ROOT/'sparse_queries'/f'dense-{n}.npz'
    if not ref.exists(): ref = RESULTS/f'reference-{n}.npz'
    truth = dict(np.load(ref)) if ref.exists() else None
    contract = dict(n=n, group='test_side', contacts=len(contacts), queries=128,
        query_indices=ids.tolist(), lengthscale=model.lengthscale, amplitude=model.amplitude,
        normal_noise_variance=selected['sigma']**2, scalar_noise_variance=.0001,
        dataset_sha256=digest(ROOT/'dataset.json'), selected_sha256=digest(ROOT/'selected.json'),
        controls_sha256=digest(ROOT/'controls.json'), query_sha256=digest(ROOT/'test_side-queries.npz'))
    if truth is not None:
        metadata = json.loads(ref.with_suffix('.json').read_text())
        if 'field_sha256' in metadata: assert digest(ref) == metadata['field_sha256']
        if ref.parent == RESULTS:
            assert metadata['arguments']['method'] == 'reference'
            assert metadata['accepted'] and metadata['reference_converged']
            for key in contract:
                assert metadata['contract'][key] == contract[key], key
        contract.update(reference=str(ref), reference_sha256=digest(ref),
                        reference_record_sha256=digest(ref.with_suffix('.json')))
    return x, y, contacts, panel, model, selected['sigma']**2, truth, fit, contract


def sources():
    paths = [__file__, 'reproduction/partial_competitor_common.py', 'reproduction/partial_direct_gpu.py',
        'reproduction/partial_occupancy.py', 'reproduction/gpu_audit.py', 'geospsr/nonperiodic.py',
        'geospsr/hierarchical.py', 'geospsr/hierarchical_gpu.py', 'geospsr/hierarchical_interactions.py',
        'geospsr/nonperiodic_dense_gpu.py', 'geospsr/scalar_contacts.py']
    folder = RESULTS/'sources'; folder.mkdir(parents=True, exist_ok=True)
    result = {}
    for p in paths:
        data = Path(p).read_bytes(); h = hashlib.sha256(data).hexdigest()
        (folder/(h+'.py')).write_bytes(data); result[p] = h
    return result


def run(args):
    import cupy as cp
    from cupyx.scipy.linalg import solve_triangular
    from reproduction.gpu_audit import GpuProcessAudit
    from reproduction.partial_direct_gpu import cross_rhs, DirectContactQueries
    from geospsr.nonperiodic_dense_gpu import GPUDenseMaternSystem, full_kernel_matmul
    output = Path(args.output)
    if output.exists(): raise FileExistsError(output)
    x, y, contacts, panel, model, noise, truth, fit, contract = task(args.n)
    if truth is None and args.method != 'reference': raise ValueError('Generate an independent reference first')
    record = dict(arguments=vars(args), contract=contract, source_sha256=sources())
    cross_rhs(model, contacts[:2], x[:2]); cp.cuda.Device().synchronize()
    audit = GpuProcessAudit(cp.cuda.Device().pci_bus_id, interval=.1, cuda_device=cp.cuda.Device().id).start()
    begin = perf_counter(); mean = variance = None
    try:
        if args.method == 'h2':
            from geospsr.hierarchical import HierarchicalMaternSystem
            from geospsr.hierarchical_gpu import GPUHierarchicalSystem
            native = HierarchicalMaternSystem(model, x, noise, tolerance=args.h2_tolerance,
                leaf_size=40, preconditioner_rank=100, neighbors=80, threads=4, seed=9101,
                preconditioner_backend='prefix-kd')
            system = GPUHierarchicalSystem(native, dtype=args.dtype,
                interaction_budget_bytes=256*2**20 if args.storage=='streamed' else None,
                rhs_block_size=72, release_native=True)
            cp.cuda.Device().synchronize(); record['setup_seconds']=perf_counter()-begin
            start=perf_counter()
            posterior=DirectContactQueries(model,x,y,contacts,system,iterations=args.iterations,solve_tolerance=1e-9)
            cp.cuda.Device().synchronize();record['mean_contact_seconds']=perf_counter()-start
            start=perf_counter();mean,variance=posterior.predict(panel['queries'],batch_size=96)
            record['probability_checksum']=float(ndtr(-mean/np.sqrt(variance)).sum())
            cp.cuda.Device().synchronize();record['query_seconds']=perf_counter()-start
            record['total_seconds']=perf_counter()-begin
            record.update(mean_solve=posterior.mean_solve_info,contact_solve=posterior.contact_solve_info,
                variance_iterations=posterior.solve_iterations,stored_bytes=system.device_matrix_bytes,
                interaction_workspace_bytes=system.interaction_workspace_bytes,
                reference_workspace_bytes=system.reference_workspace_bytes,interaction_entries=system.interaction_nnz)
        else:
            system=GPUDenseMaternSystem(model,x,noise,dtype=args.dtype)
            cp.cuda.Device().synchronize();record['setup_seconds']=perf_counter()-begin
            start=perf_counter();cross=cross_rhs(model,np.concatenate((contacts,panel['queries'])),x)
            rhs=cp.column_stack((cross,cp.asarray(y)));cp.cuda.Device().synchronize()
            record['cross_seconds']=perf_counter()-start
            start=perf_counter()
            w=solve_triangular(system.factor,rhs.astype(system.dtype),lower=True,check_finite=False)
            z=solve_triangular(system.factor,w,lower=True,trans='T',check_finite=False).astype(cp.float64)
            residual=rhs-cp.asarray(full_kernel_matmul(model,x,noise,cp.asnumpy(z)))
            if args.method == 'reference':
                record['refinement_history']=[];previous=None;converged=False
                norms=cp.linalg.norm(rhs,axis=0)
                for stage in range(6):
                    mean,variance=conditioned_moments(model,contacts,panel['queries'],rhs,z,cp,residual=residual)
                    relative=float(cp.max(cp.linalg.norm(residual,axis=0)/norms).item())
                    entry_bound=float(cp.max(cp.sum(residual*residual,axis=0)).item()/noise)
                    change=gates(mean,variance,previous,model,fit) if previous is not None else None
                    record['refinement_history'].append(dict(stage=stage,max_relative_residual=relative,
                        joint_reduction_entry_error_bound=entry_bound,change_from_previous=change))
                    print('REFERENCE',record['refinement_history'][-1],flush=True)
                    if relative<=1e-9 and change and change['cdf']<=1e-6 and change['calibrated_cdf']<=1e-6:
                        converged=True;break
                    previous=dict(mean=mean.copy(),variance=variance.copy())
                    z+=cp.asarray(system.solve(cp.asnumpy(residual)))
                    residual=rhs-cp.asarray(full_kernel_matmul(model,x,noise,cp.asnumpy(z)))
                record['reference_converged']=converged
                if not converged: raise RuntimeError('full-kernel reference refinement did not converge')
            cp.cuda.Device().synchronize();record['solve_seconds']=perf_counter()-start
            start=perf_counter()
            mean,variance=conditioned_moments(model,contacts,panel['queries'],rhs,z,cp,residual=residual)
            record['probability_checksum']=float(ndtr(-mean/np.sqrt(variance)).sum())
            cp.cuda.Device().synchronize();record['moment_seconds']=perf_counter()-start
            record['total_seconds']=perf_counter()-begin
        record['status']='completed'
    except (ValueError,RuntimeError,MemoryError,np.linalg.LinAlgError) as exc:
        record.update(status='capacity failure' if isinstance(exc,MemoryError) else 'failed',
                      error=f'{type(exc).__name__}: {exc}',wall_seconds=perf_counter()-begin)
    finally:
        record['gpu_audit']=audit.stop()
    if mean is not None and truth is not None:record['errors']=gates(mean,variance,truth,model,fit)
    valid=mean is not None and bool(np.isfinite(mean).all() and np.isfinite(variance).all() and np.all(variance>0))
    accurate=record.get('errors',{}).get('passes',args.method=='reference' and record.get('reference_converged',False))
    record['accepted']=bool(record['status']=='completed' and valid and accurate and record['gpu_audit']['sampled_isolation_pass'])
    store_result(output,record,mean,variance)
    print('FINAL',record.get('total_seconds'),record['accepted'],record.get('errors'),record.get('error'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=['h2','dense','reference'],required=True)
    p.add_argument('--n',type=int,default=49152)
    p.add_argument('--dtype',choices=['float32','float64'],default='float32')
    p.add_argument('--storage',choices=['cached','streamed'],default='cached')
    p.add_argument('--iterations',type=int,default=8)
    p.add_argument('--h2-tolerance',type=float,default=1e-6)
    p.add_argument('--output',required=True)
    run(p.parse_args())
