"""Retest the fixed 128-query task after freezing quality-selected parameters."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.special import ndtr

from geospsr.nonperiodic import FreeSpacePoissonMatern
from reproduction.partial_quality import OUT, snapshot
from reproduction.partial_occupancy import ROOT, load
from reproduction.partial_competitor_common import conditioned_moments, gates, store_result
from reproduction.stanford_partial_data import digest


def dense_moments(model,x,y,contacts,q,noise,method):
    import cupy as cp
    from cupyx.scipy.linalg import solve_triangular
    from reproduction.partial_direct_gpu import cross_rhs
    start=perf_counter()
    from geospsr.nonperiodic_dense_gpu import GPUDenseMaternSystem, full_kernel_matmul
    from geospsr.scalar_contacts import ScalarContactUpdate
    system=GPUDenseMaternSystem(model,x,noise,dtype='float32')
    cp.get_default_memory_pool().free_all_blocks()
    cp.cuda.Device().synchronize();setup=perf_counter()-start
    allq=np.concatenate((contacts,q));width=len(allq);nc=len(contacts)
    if method=='plain':
        # Avoid retaining a duplicate float64 cross matrix beside the factor.
        rhs=cp.empty((len(x),3*width+3),dtype=cp.float32)
        for first in range(0,width,96):
            qq=allq[first:first+96];b=cross_rhs(model,qq,x);m=len(qq)
            for d in range(3):rhs[:,d*width+first:d*width+first+m]=b[:,d*m:(d+1)*m]
            del b
        rhs[:,3*width:]=cp.asarray(y,dtype=cp.float32)
        cp.get_default_memory_pool().free_all_blocks()
        w=solve_triangular(system.factor,rhs,lower=True,check_finite=False)
        del rhs;cp.get_default_memory_pool().free_all_blocks()
        values=w.astype(cp.float64);del w;cp.get_default_memory_pool().free_all_blocks()
        mean,variance=conditioned_moments(model,contacts,q,None,values,cp,whitened=True)
    elif method=='corrected-wide':
        rhs=cp.empty((len(x),3*width+3),dtype=cp.float64)
        for first in range(0,width,96):
            qq=allq[first:first+96];b=cross_rhs(model,qq,x);m=len(qq)
            for d in range(3):rhs[:,d*width+first:d*width+first+m]=b[:,d*m:(d+1)*m]
            del b
        rhs[:,3*width:]=cp.asarray(y)
        cp.get_default_memory_pool().free_all_blocks()
        z=cp.asarray(system.solve(cp.asnumpy(rhs)))
        cp.get_default_memory_pool().free_all_blocks()
        residual=rhs-cp.asarray(full_kernel_matmul(model,x,noise,cp.asnumpy(z),workspace_bytes=128*2**20))
        mean,variance=conditioned_moments(model,contacts,q,rhs,z,cp,residual=residual)
    else:
        # One vector component at a time bounds RHS memory at large C.
        bm=cp.zeros(width);red=cp.zeros(width);cc=cp.zeros((nc,nc));qc=cp.zeros((len(q),nc))
        for d in range(3):
            b=cp.empty((len(x),width+1),dtype=cp.float64)
            for first in range(0,width,96):
                qq=allq[first:first+96];raw=cross_rhs(model,qq,x);m=len(qq)
                b[:,first:first+m]=raw[:,d*m:(d+1)*m];del raw
            b[:,-1]=cp.asarray(y[:,d]);cp.get_default_memory_pool().free_all_blocks()
            z=cp.asarray(system.solve(cp.asnumpy(b)))
            r=b-cp.asarray(full_kernel_matmul(model,x,noise,cp.asnumpy(z)))
            bm+=b[:,:width].T@z[:,-1]+z[:,:width].T@r[:,-1]
            red+=cp.sum(b[:,:width]*z[:,:width]+z[:,:width]*r[:,:width],axis=0)
            cc+=b[:,:nc].T@z[:,:nc]+z[:,:nc].T@r[:,:nc]
            qc+=b[:,nc:width].T@z[:,:nc]+z[:,nc:width].T@r[:,:nc]
            del b,z,r;cp.get_default_memory_pool().free_all_blocks()
        bm,red,cc,qc=map(cp.asnumpy,(bm,red,cc,qc))
        update=ScalarContactUpdate(bm[:nc],model.scalar_covariance(contacts,contacts)-(cc+cc.T)*.5,np.zeros(nc),.0001)
        mean,variance=update.apply(bm[nc:],model.prior_variance-red[nc:],model.scalar_covariance(q,contacts)-qc)
    return mean,variance,setup


def run(args):
    import cupy as cp
    from cupyx.scipy.linalg import solve_triangular
    from reproduction.gpu_audit import GpuProcessAudit
    from reproduction.partial_direct_gpu import cross_rhs, DirectContactQueries
    path=Path(args.output)
    if path.exists():raise FileExistsError(path)
    selected=json.loads((OUT/'selected.json').read_text())
    x,y,_,panel=load('test_side',args.n)
    contacts=np.load(OUT/'test_side-panel.npz')['contacts'][:selected['contacts']]
    ids=np.linspace(0,len(panel['queries'])-1,128,dtype=int);q=panel['queries'][ids]
    model=FreeSpacePoissonMatern(selected['ell'],1/3);noise=selected['sigma']**2
    fit=json.loads((ROOT/'controls.json').read_text())['fits']['spatial']
    sources=snapshot()
    for p in [__file__,'reproduction/partial_quality_large.py','reproduction/partial_competitor_common.py',
        'geospsr/hierarchical.py','geospsr/hierarchical_gpu.py','geospsr/hierarchical_interactions.py','reproduction/gpu_audit.py']:
        h=digest(p);(OUT/'sources'/(h+'.py')).write_bytes(Path(p).read_bytes());sources[p]=h
    contract=dict(n=args.n,group='test_side',contacts=len(contacts),query_indices=ids.tolist(),
        controls_sha256=digest(ROOT/'controls.json'),selected_sha256=digest(OUT/'selected.json'),panel_sha256=digest(OUT/'test_side-panel.npz'),
        query_sha256=digest(ROOT/'test_side-queries.npz'),dataset_sha256=digest(ROOT/'dataset.json'))
    record=dict(arguments=vars(args),contract=contract,source_sha256=sources,
        scope='Fresh setup + all contacts + 128 marginal mean/variance/raw-CDF queries; shared data and CUDA initialization excluded')
    refpath=Path(args.reference) if args.reference else OUT/f'benchmark-reference-{args.n}.json'
    if args.method!='reference':
        ref=json.loads(refpath.read_text())
        assert ref['field_sha256']==digest(refpath.with_suffix('.npz'))
        fields=dict(np.load(refpath.with_suffix('.npz')))
        if 'benchmark_query_indices' in ref:
            assert ref['status']=='completed' and ref['gpu_audit']['sampled_isolation_pass']
            assert ref['arguments']['method']=='reference' and ref['arguments']['group']=='test_side' and ref['arguments']['n']==args.n
            for key in ['ell','sigma','contacts']:assert ref['arguments'][key]==selected[key]
            assert ref['benchmark_query_indices']==ids.tolist()
            assert ref['panel_sha256']==contract['panel_sha256']
            truth=dict(mean=fields['benchmark_mean'],variance=fields['benchmark_variance'])
        else:
            assert ref['accepted'] and ref['contract']==contract
            truth=fields
        record['reference_sha256']=digest(refpath);record['reference_path']=str(refpath)
    cross_rhs(model,contacts[:2],x[:2]);cp.cuda.Device().synchronize()
    audit=GpuProcessAudit(cp.cuda.Device().pci_bus_id,interval=.1,cuda_device=cp.cuda.Device().id).start()
    start=perf_counter();mean=variance=None
    try:
        if args.method in ['plain','corrected','corrected-wide']:
            mean,variance,record['setup_seconds']=dense_moments(model,x,y,contacts,q,noise,args.method)
        else:
            if args.method=='reference':
                from reproduction.partial_quality_large import RefinedDenseSystem
                system=RefinedDenseSystem(model,x,noise)
            else:
                from geospsr.hierarchical import HierarchicalMaternSystem
                from geospsr.hierarchical_gpu import GPUHierarchicalSystem
                native=HierarchicalMaternSystem(model,x,noise,tolerance=args.tolerance,leaf_size=40,
                    preconditioner_rank=100,neighbors=80,threads=4,seed=9101,preconditioner_backend='prefix-kd')
                system=GPUHierarchicalSystem(native,dtype='float32',rhs_block_size=72,release_native=True)
            cp.cuda.Device().synchronize();record['setup_seconds']=perf_counter()-start
            tick=perf_counter();posterior=DirectContactQueries(model,x,y,contacts,system,iterations=args.iterations,solve_tolerance=1e-9)
            record['mean_contact_seconds']=perf_counter()-tick
            record.update(mean_solve=posterior.mean_solve_info,contact_solve=posterior.contact_solve_info)
            mean,variance=posterior.predict(q,batch_size=96)
            record['variance_iterations']=posterior.solve_iterations
            if args.method=='reference':record['refinement_history']=system.histories
        if not(np.isfinite(mean).all() and np.isfinite(variance).all() and np.all(variance>0)):raise ValueError('invalid moments')
        record['probability_checksum']=float(ndtr(-mean/np.sqrt(variance)).sum())
        cp.cuda.Device().synchronize();record.update(status='completed',total_seconds=perf_counter()-start)
    except Exception as exc:
        record.update(status='capacity failure' if isinstance(exc,MemoryError) else 'failed',
                      error=f'{type(exc).__name__}: {exc}',wall_seconds=perf_counter()-start)
    finally:record['gpu_audit']=audit.stop()
    if mean is not None and args.method!='reference':record['errors']=gates(mean,variance,truth,model,fit)
    record['accepted']=bool(record['status']=='completed' and record['gpu_audit']['sampled_isolation_pass'] and
        (args.method=='reference' or record.get('errors',{}).get('passes',False)))
    store_result(path,record,mean,variance)
    print('FINAL',args.method,args.n,record.get('total_seconds'),record['accepted'],record.get('errors'),record.get('error'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--method',choices=['plain','corrected','corrected-wide','reference','h2'],required=True)
    p.add_argument('--reference');p.add_argument('--n',type=int,default=70652);p.add_argument('--iterations',type=int,default=10)
    p.add_argument('--tolerance',type=float,default=1e-6);p.add_argument('--output',required=True)
    run(p.parse_args())
