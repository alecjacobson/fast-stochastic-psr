"""Shared finite-query cross compression for H2, Cholesky and SDD controls."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.special import ndtr

from reproduction.partial_competitor_common import (
    problem, source_snapshot, projected_moments, gates, store_result)


def run(args):
    import cupy as cp
    from geospsr.nonperiodic import compress_cross_covariance
    from reproduction.gpu_audit import GpuProcessAudit
    from reproduction.partial_direct_gpu import cross_rhs
    output=Path(args.output)
    if output.exists(): raise FileExistsError(output)
    x,y,contacts,panel,model,noise,truth,fit,contract=problem(args.n)
    sources=source_snapshot([__file__,'reproduction/partial_competitor_common.py',
        'geospsr/nonperiodic.py','geospsr/hierarchical.py','geospsr/hierarchical_gpu.py',
        'geospsr/nonperiodic_dense_gpu.py','geospsr/iterative.py','geospsr/scalar_contacts.py'])
    cross_rhs(model,contacts[:2],x[:2]);cp.cuda.Device().synchronize()
    if args.method == 'sdd':
        import jax
        jax.devices()
    audit=GpuProcessAudit(cp.cuda.Device().pci_bus_id,interval=.1,cuda_device=cp.cuda.Device().id).start()
    record=dict(arguments=vars(args),contract=contract,source_sha256=sources)
    wall_start=perf_counter()
    start=perf_counter()
    u,coef,error=compress_cross_covariance(model,x,np.concatenate((contacts,panel['queries'])),args.cross_tolerance)
    record.update(compression_seconds=perf_counter()-start,rank=u.shape[1],cross_relative_error=error)
    print('COMPRESSION',record['compression_seconds'],record['rank'],error,flush=True)

    def evaluate(gram,projected):
        start=perf_counter()
        try:
            m,v=projected_moments(model,contacts,panel['queries'],coef,gram,projected)
            valid=np.isfinite(m).all() and np.isfinite(v).all() and np.all(v>0)
            checksum=float(ndtr(-m/np.sqrt(v)).sum()) if valid else None
            seconds=perf_counter()-start
            return m,v,dict(moment_and_cdf_seconds=seconds,probability_checksum=checksum,
                            errors=gates(m,v,truth,model,fit))
        except (ValueError,np.linalg.LinAlgError) as exc:
            return None,None,dict(moment_and_cdf_seconds=perf_counter()-start,
                errors=dict(passes=False,valid_moments=False),error=str(exc))

    if args.method=='sdd':
        import jax.numpy as jnp
        from geospsr.iterative import StochasticDualDescent
        def kernel(a,b):
            z=jnp.sqrt(jnp.sum((a-b)**2))*model.kappa
            return (1+z)*jnp.exp(-z)
        start=perf_counter()
        solver=StochasticDualDescent(jnp.asarray(x,dtype=jnp.float32),
            jnp.asarray(np.column_stack((u,y)),dtype=jnp.float32),kernel,
            noise_variance=noise/model.amplitude,batch=args.batch,lr=args.eta*args.batch/len(x),
            seed=args.seed,momentum=.9,polyak=.01,sampling='rejection')
        solver.compile();ug=cp.asarray(u);cp.cuda.Device().synchronize()
        record['setup_seconds']=perf_counter()-start
        elapsed=0.;record['history']=[]
        for checkpoint in args.checkpoints:
            while solver.iterations<checkpoint and elapsed<args.budget:
                start=perf_counter();value=solver.advance(min(256,checkpoint-solver.iterations));elapsed+=perf_counter()-start
            start=perf_counter();z=cp.asarray(np.asarray(value,dtype=float)/model.amplitude)
            gram=ug.T@z[:,:u.shape[1]];gram=.5*(gram+gram.T)
            projected=ug.T@z[:,u.shape[1]:]
            gram,projected=cp.asnumpy(gram),cp.asnumpy(projected)
            projection=perf_counter()-start
            m,v,row=evaluate(gram,projected)
            row.update(iterations=solver.iterations,updates_seconds=elapsed,projection_seconds=projection,
                total_seconds=record['compression_seconds']+record['setup_seconds']+elapsed+projection+row['moment_and_cdf_seconds'])
            record['history'].append(row);store_result(output,record)
            print(json.dumps(row),flush=True)
            if row['errors']['passes'] or elapsed>=args.budget or not np.isfinite(gram).all(): break
        record['selected']=row
        record['total_seconds']=row['total_seconds'];record['errors']=row['errors']
        record['stopping_scope']='Earlier checkpoint checks excluded from selected total, favoring SDD.'
    else:
        start=perf_counter()
        if args.method=='h2':
            from geospsr.hierarchical import HierarchicalMaternSystem
            from geospsr.hierarchical_gpu import GPUHierarchicalSystem
            native=HierarchicalMaternSystem(model,x,noise,tolerance=1e-6,leaf_size=40,
                preconditioner_rank=100,neighbors=80,threads=4,seed=9101,preconditioner_backend='prefix-kd')
            system=GPUHierarchicalSystem(native,interaction_budget_bytes=256*2**20,rhs_block_size=72,release_native=True)
        else:
            from geospsr.nonperiodic_dense_gpu import GPUDenseMaternSystem
            system=GPUDenseMaternSystem(model,x,noise,dtype=args.dtype)
        cp.cuda.Device().synchronize();record['setup_seconds']=perf_counter()-start
        start=perf_counter()
        if args.method=='h2':gram,projected=system.project(u,y,variance_iterations=args.iterations,mean_tolerance=1e-9)
        else:gram,projected=system.project(u,y)
        cp.cuda.Device().synchronize();record['solve_seconds']=perf_counter()-start
        record['solve_info']=getattr(system,'last_solve',None)
        m,v,row=evaluate(gram,projected);record.update(row)
        record['total_seconds']=sum(record[k] for k in ['compression_seconds','setup_seconds','solve_seconds','moment_and_cdf_seconds'])
    record['wall_seconds']=perf_counter()-wall_start
    record['gpu_audit']=audit.stop()
    record['accepted']=bool(record['errors']['passes'] and record['gpu_audit']['sampled_isolation_pass'])
    store_result(output,record,m,v)
    print('FINAL',record['total_seconds'],record['errors'],'accepted',record['accepted'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=['h2','dense','sdd'],required=True)
    p.add_argument('--n',type=int,default=49152)
    p.add_argument('--cross-tolerance',type=float,default=1e-7)
    p.add_argument('--dtype',choices=['float32','float64'],default='float32')
    p.add_argument('--iterations',type=int,default=8)
    p.add_argument('--batch',type=int,default=128);p.add_argument('--eta',type=float,default=.01)
    p.add_argument('--seed',type=int,default=9101);p.add_argument('--budget',type=float,default=90.)
    p.add_argument('--checkpoints',type=int,nargs='+',default=[128,512,2048,8192,32768,131072])
    p.add_argument('--output',required=True)
    args=p.parse_args()
    if args.eta <= 0 or args.budget <= 0 or args.batch < 1: p.error('positive tuning parameters required')
    if not args.checkpoints or args.checkpoints[0]<1 or any(b<=a for a,b in zip(args.checkpoints,args.checkpoints[1:])):
        p.error('strictly increasing positive checkpoints required')
    run(args)
