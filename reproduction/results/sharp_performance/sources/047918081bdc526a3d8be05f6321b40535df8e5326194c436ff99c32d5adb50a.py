"""Fresh performance measurements for the frozen PR127/129 posterior."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.special import ndtr

from geospsr.nonperiodic import FreeSpacePoissonMatern
from reproduction.partial_occupancy import ROOT, load
from reproduction.partial_competitor_common import gates, store_result
from reproduction.stanford_partial_data import digest

OUT=Path('reproduction/results/sharp_performance')
FROZEN=Path('reproduction/results/observed_fidelity')


def snapshot(extra_paths=()):
    folder=OUT/'sources';folder.mkdir(exist_ok=True);result={}
    paths=[__file__,'reproduction/partial_competitor_common.py','reproduction/partial_direct_gpu.py',
           'reproduction/observed_fidelity_large.py','reproduction/partial_quality_large.py',
           'reproduction/gpu_audit.py','reproduction/partial_occupancy.py',
           'geospsr/nonperiodic.py','geospsr/nonperiodic_dense_gpu.py','geospsr/scalar_contacts.py',
           'geospsr/hierarchical.py','geospsr/hierarchical_gpu.py','geospsr/hierarchical_interactions.py',
           'geospsr/nonperiodic_local_gpu.py','geospsr/nonperiodic_dense_cg.py','geospsr/coupled_gpu.py']
    for path in dict.fromkeys([*paths,*extra_paths]):
        sha=digest(path);(folder/(sha+'.py')).write_bytes(Path(path).read_bytes());result[path]=sha
    return result


def problem(n,group='test_side',validation_panel=False,query_count=128):
    cfg=json.loads((FROZEN/'selected.json').read_text())['configuration']
    x,y,_,old=load(group,n);panel=np.load(FROZEN/(group+'-panel.npz'))
    source=panel['queries'] if validation_panel else old['queries']
    if not 1<=query_count<=len(source):raise ValueError('invalid query count')
    ids=np.linspace(0,len(source)-1,query_count,dtype=int);q=source[ids]
    c=panel['contacts'][:cfg['contacts']];model=FreeSpacePoissonMatern(cfg['ell'],1/3)
    contract=dict(n=n,group=group,configuration=cfg,queries=query_count,query_indices=ids.tolist(),validation_panel=validation_panel,
                  selected_sha256=digest(FROZEN/'selected.json'),panel_sha256=digest(FROZEN/(group+'-panel.npz')),
                  dataset_sha256=digest(ROOT/'dataset.json'),query_sha256=digest(ROOT/(group+'-queries.npz')),
                  protocol_sha256=digest(OUT/'PROTOCOL.md'))
    return x,y,c,q,model,cfg,contract


def reference(args):
    import cupy as cp
    from reproduction.observed_fidelity_large import StreamingReference
    from reproduction.gpu_audit import GpuProcessAudit
    path=Path(args.output)
    if args.query_count!=128:raise ValueError('reference driver retains the frozen128-query task')
    if path.exists():raise FileExistsError(path)
    x,y,c,q,model,cfg,contract=problem(args.n,args.group,args.validation_panel,args.query_count)
    check=args.n==70652 and args.group=='test_side'
    if check:
        oldpanel=np.load(FROZEN/'test_side-panel.npz');ids=np.linspace(0,511,32,dtype=int)
        q=np.concatenate((q,oldpanel['queries'][ids]))
    record=dict(arguments=vars(args),contract=contract,source_sha256=snapshot(),scope='Independent full-kernel reference, excluded from competitive timings')
    audit=GpuProcessAudit(cp.cuda.Device().pci_bus_id,interval=.5,cuda_device=cp.cuda.Device().id).start()
    start=perf_counter();m=v=None
    try:
        if args.reference_kind=='float64':
            from geospsr.nonperiodic_dense_gpu import GPUDenseMaternSystem
            system=GPUDenseMaternSystem(model,x,cfg['sigma']**2,dtype='float64')
            m,v,record['phases']=component_moments(system,model,x,y,c,q,cfg['scalar_variance'],'float64')
            record['reference_backend']='full float64 factor and whitening'
        else:
            posterior=StreamingReference(model,x,y,c,cfg['sigma']**2,cfg['scalar_variance'],block=192)
            m,v=posterior.predict(q)
            record['refinement_history']=posterior.system.histories
            record['reference_backend']='full float64-kernel refinement'
        assert np.isfinite(m).all() and np.isfinite(v).all() and np.all(v>0)
        if check:
            old=np.load(FROZEN/'test-side-selected.npz')
            error=dict(mean_max=float(np.max(abs(m[128:]-old['mean'][ids]))),
                       variance_over_prior_max=float(np.max(abs(v[128:]-old['variance'][ids]))/model.prior_variance))
            assert error['mean_max']<=1e-9 and error['variance_over_prior_max']<=1e-7
            record['retained_agreement']=error
        m,v=m[:128],v[:128];record['status']='completed'
    except Exception as exc:
        record.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        record.update(seconds=perf_counter()-start,gpu_audit=audit.stop())
        record['accepted']=record['status']=='completed' and record['gpu_audit']['sampled_isolation_pass']
        store_result(path,record,m,v)
    print('REFERENCE',args.n,record['seconds'],record['accepted'],flush=True)


def component_moments(system,model,x,y,contacts,q,scalar_variance,method,*,iterations=14,block=288,refinements=0):
    """Reuse the vector factor/preconditioner with bounded component workspace."""
    import cupy as cp
    from cupyx.scipy.linalg import solve_triangular
    from reproduction.partial_direct_gpu import cross_rhs
    from geospsr.nonperiodic_dense_gpu import full_kernel_matmul
    from geospsr.scalar_contacts import ScalarContactUpdate
    nc,nq=len(contacts),len(q);width=nc+nq;allq=np.concatenate((contacts,q))
    bm=cp.zeros(width);red=cp.zeros(width);cc=cp.zeros((nc,nc));qc=cp.zeros((nq,nc))
    phases=dict(cross_seconds=0.,solve_seconds=0.,residual_seconds=0.,projection_seconds=0.)
    infos=[];whitened=method in ['plain','plain-assembled64','float64']
    iterative=method in ['h2','locality','dense-cg']
    if iterative:
        tick=perf_counter();alpha=cp.asarray(system.solve(y,tolerance=1e-9))
        normal_info=system.last_solve;normal_residual=cp.asarray(y)-system.accurate_matmul(alpha)
        cp.cuda.Device().synchronize();phases['solve_seconds']+=perf_counter()-tick
    for d in range(3):
        tick=perf_counter();dtype=system.factor.dtype if whitened else cp.float64
        b=cp.empty((len(x),width+1),dtype=dtype)
        for first in range(0,width,96):
            b[:,first:min(first+96,width)]=cross_rhs(model,allq[first:first+96],x,component=d)
        b[:,-1]=cp.asarray(y[:,d]);cp.get_default_memory_pool().free_all_blocks()
        cp.cuda.Device().synchronize();phases['cross_seconds']+=perf_counter()-tick
        tick=perf_counter()
        if whitened:
            z=solve_triangular(system.factor,b,lower=True,check_finite=False).astype(cp.float64)
            del b;cp.get_default_memory_pool().free_all_blocks();b=z;r=None
        elif iterative:
            z=cp.empty_like(b);r=cp.empty_like(b);z[:,-1]=alpha[:,d];r[:,-1]=normal_residual[:,d]
            for first in range(0,width,block):
                last=min(first+block,width)
                if method=='dense-cg':
                    from geospsr.hierarchical import batched_pcg
                    zz,info,rr=batched_pcg(system.matmul,system.precondition,b[:,first:last],tolerance=1e-10,
                        max_iterations=iterations,xp=cp,return_residual=True)
                else:zz,info,rr=system.variance_solve(b[:,first:last],iterations)
                z[:,first:last]=zz;r[:,first:last]=rr;infos.append(info)
                del zz,rr
        else:
            z=cp.asarray(system.solve(cp.asnumpy(b)))
        cp.cuda.Device().synchronize();phases['solve_seconds']+=perf_counter()-tick
        if method=='dense-cg':
            tick=perf_counter()
            # Share one full-kernel residual action over this entire component,
            # rather than rebuilding kernel tiles for each small PCG block.
            del r;cp.get_default_memory_pool().free_all_blocks()
            r=b-system.accurate_matmul(z)
            phases.setdefault('full_residual_relative_max',[]).append(float(cp.max(cp.linalg.norm(r,axis=0)/cp.linalg.norm(b,axis=0)).item()))
            cp.cuda.Device().synchronize();phases['residual_seconds']+=perf_counter()-tick
        if not whitened and not iterative:
            tick=perf_counter()
            r=b-cp.asarray(full_kernel_matmul(model,x,system.noise_variance,cp.asnumpy(z),workspace_bytes=128*2**20))
            for _ in range(refinements):
                z+=cp.asarray(system.solve(cp.asnumpy(r)))
                r=b-cp.asarray(full_kernel_matmul(model,x,system.noise_variance,cp.asnumpy(z),workspace_bytes=128*2**20))
            cp.cuda.Device().synchronize();phases['residual_seconds']+=perf_counter()-tick
        tick=perf_counter()
        cp.get_default_memory_pool().free_all_blocks()
        bm+=b[:,:width].T@z[:,-1]
        for first in range(0,width,96):
            last=min(first+96,width);red[first:last]+=cp.sum(b[:,first:last]*z[:,first:last],axis=0)
        cc+=b[:,:nc].T@z[:,:nc];qc+=b[:,nc:width].T@z[:,:nc]
        if r is not None:
            bm+=z[:,:width].T@r[:,-1]
            for first in range(0,width,96):
                last=min(first+96,width);red[first:last]+=cp.sum(z[:,first:last]*r[:,first:last],axis=0)
            cc+=z[:,:nc].T@r[:,:nc];qc+=z[:,nc:width].T@r[:,:nc]
        del b,z,r;cp.get_default_memory_pool().free_all_blocks()
        cp.cuda.Device().synchronize();phases['projection_seconds']+=perf_counter()-tick
        print('COMPONENT',method,d,phases,flush=True)
    tick=perf_counter();bm,red,cc,qc=map(cp.asnumpy,(bm,red,cc,qc))
    update=ScalarContactUpdate(bm[:nc],model.scalar_covariance(contacts,contacts)-(cc+cc.T)*.5,np.zeros(nc),scalar_variance)
    m,v=update.apply(bm[nc:],model.prior_variance-red[nc:],model.scalar_covariance(q,contacts)-qc)
    phases['scalar_conditioning_seconds']=perf_counter()-tick
    if iterative:phases.update(normal_solve=normal_info,cross_solves=infos)
    return m,v,phases


def run(args):
    import cupy as cp
    from reproduction.partial_direct_gpu import cross_rhs
    from reproduction.gpu_audit import GpuProcessAudit
    path=Path(args.output)
    if path.exists():raise FileExistsError(path)
    x,y,c,q,model,cfg,contract=problem(args.n,args.group,args.validation_panel,args.query_count)
    fit=json.loads((ROOT/'controls.json').read_text())['fits']['spatial']
    if args.group=='validation' and args.n==16384 and args.validation_panel:
        refpath=FROZEN/'screen-n16384-ell0.015-sigma0.02.json';ref=json.loads(refpath.read_text());fields=np.load(refpath.with_suffix('.npz'))
        assert ref['status']=='completed' and ref['field_sha256']==digest(refpath.with_suffix('.npz'))
        column=next(i for i,r in enumerate(ref['records']) if r['contacts']==cfg['contacts'] and r['scalar_variance']==cfg['scalar_variance'])
        ids=contract['query_indices'];truth={k:fields[k][ids,column] for k in ['mean','variance']}
    elif args.group=='test_side' and args.n==70652 and args.validation_panel:
        refpath=FROZEN/'test-side-selected.json';ref=json.loads(refpath.read_text());fields=np.load(refpath.with_suffix('.npz'))
        assert ref['status']=='completed' and ref['configuration']==cfg and ref['gpu_audit']['sampled_isolation_pass']
        assert ref['field_sha256']==digest(refpath.with_suffix('.npz'))
        truth={k:fields[k][contract['query_indices']] for k in ['mean','variance']}
    else:
        refpath=Path(args.reference) if args.reference else OUT/f'reference-{args.n}.json'
        ref=json.loads(refpath.read_text());assert ref['accepted'] and ref['contract']==contract
        assert ref['field_sha256']==digest(refpath.with_suffix('.npz'));truth=dict(np.load(refpath.with_suffix('.npz')))
    record=dict(arguments=vars(args),contract=contract,source_sha256=snapshot(),reference_path=str(refpath),reference_sha256=digest(refpath),
                scope=f'Fresh setup + all{len(c)} contacts +{len(q)} marginal mean/variance/CDF queries; shared data/CUDA initialization excluded')
    if args.method=='sdd':
        from reproduction.partial_sdd import run as sdd_run
        sdd_run(args,problem_factory=lambda n:(x,y,c,dict(queries=q),model,cfg['sigma']**2,truth,fit,contract),
                scalar_variance=cfg['scalar_variance'],snapshot_factory=snapshot,
                record_metadata={k:record[k] for k in ['reference_path','reference_sha256','scope']})
        return
    cross_rhs(model,c[:2],x[:2])
    for d in range(3):cross_rhs(model,c[:2],x[:2],component=d)
    cp.cuda.Device().synchronize()
    audit=GpuProcessAudit(cp.cuda.Device().pci_bus_id,interval=.1,cuda_device=cp.cuda.Device().id).start()
    start=perf_counter();m=v=None
    try:
        if args.method in ['plain','plain-assembled64','corrected','float64']:
            from geospsr.nonperiodic_dense_gpu import GPUDenseMaternSystem
            system=GPUDenseMaternSystem(model,x,cfg['sigma']**2,dtype='float64' if args.method=='float64' else 'float32',
                kernel_dtype='float64' if args.method=='plain-assembled64' else None)
        elif args.method in ['h2','coupled-h2']:
            from geospsr.hierarchical import HierarchicalMaternSystem
            from geospsr.hierarchical_gpu import GPUHierarchicalSystem
            native=HierarchicalMaternSystem(model,x,cfg['sigma']**2,tolerance=args.tolerance,leaf_size=40,
                preconditioner_rank=args.rank,neighbors=args.neighbors,threads=4,seed=9101,preconditioner_backend='prefix-kd')
            system=GPUHierarchicalSystem(native,dtype='float32',rhs_block_size=72,release_native=True)
        elif args.method=='locality':
            from geospsr.nonperiodic_local_gpu import GPUStreamedLocalSystem
            system=GPUStreamedLocalSystem(model,x,cfg['sigma']**2,cutoff=args.cutoff,rank=args.rank,neighbors=args.neighbors,
                threads=4,seed=9101,preconditioner_backend='prefix-kd',rhs_block_size=72)
        elif args.method=='dense-cg':
            import geospsr.nonperiodic_dense_cg as dense_cg
            from geospsr.hierarchical import HierarchicalMaternSystem
            dense_cg.HierarchicalMaternSystem=lambda *a,**kw:HierarchicalMaternSystem(*a,preconditioner_backend='prefix-kd',**kw)
            system=dense_cg.GPUDenseCGSystem(model,x,cfg['sigma']**2,dtype='float32',rank=args.rank,neighbors=args.neighbors,threads=4,
                seed=9101,cache=True,block_size=512)
            from geospsr.nonperiodic_dense_gpu import full_kernel_matmul
            system.mixed_actions=True
            system.accurate_matmul=lambda b:cp.asarray(full_kernel_matmul(model,x,cfg['sigma']**2,cp.asnumpy(b),workspace_bytes=128*2**20))
        cp.get_default_memory_pool().free_all_blocks();cp.cuda.Device().synchronize();record['setup_seconds']=perf_counter()-start
        if args.method=='coupled-h2':
            from geospsr.coupled_gpu import CoupledObservationSystem
            tick=perf_counter();coupled=CoupledObservationSystem(system,model,x,c,cfg['scalar_variance'])
            cp.cuda.Device().synchronize();record['coupled_setup_seconds']=perf_counter()-tick
            m,v=coupled.predict(y,q,iterations=args.iterations,refinements=args.refinements)
            phases=dict(coupled_solve=coupled.last_solve,coupled_extra_device_bytes=coupled.extra_device_bytes)
        else:
            m,v,phases=component_moments(system,model,x,y,c,q,cfg['scalar_variance'],args.method,
                iterations=args.iterations,block=args.block,refinements=args.refinements)
        record['phases']=phases
        if not(np.isfinite(m).all() and np.isfinite(v).all() and np.all(v>0)):raise ValueError('invalid moments')
        record['probability_checksum']=float(ndtr(-m/np.sqrt(v)).sum())
        cp.cuda.Device().synchronize();record.update(status='completed',total_seconds=perf_counter()-start,
            locality_diagnostics=getattr(system,'diagnostics',None),stored_bytes=getattr(system,'device_matrix_bytes',None),
            hierarchy_max_rank=getattr(system,'max_rank',None),interaction_entries=getattr(system,'interaction_nnz',None),
            dense_cg_matrix_cached=getattr(system,'matrix_cached',None))
    except Exception as exc:
        record.update(status='capacity failure' if isinstance(exc,MemoryError) else 'failed',error=f'{type(exc).__name__}: {exc}',wall_seconds=perf_counter()-start)
    finally:record['gpu_audit']=audit.stop()
    if m is not None:record['errors']=gates(m,v,truth,model,fit)
    record['accepted']=record['status']=='completed' and record['gpu_audit']['sampled_isolation_pass'] and record.get('errors',{}).get('passes',False)
    store_result(path,record,m,v)
    print('FINAL',args.method,args.n,record.get('total_seconds'),record['accepted'],record.get('errors'),record.get('error'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--method',choices=['reference','plain','plain-assembled64','corrected','float64','h2','coupled-h2','locality','dense-cg','sdd'],required=True)
    p.add_argument('--n',type=int,default=70652);p.add_argument('--group',default='test_side')
    p.add_argument('--validation-panel',action='store_true');p.add_argument('--output',required=True)
    p.add_argument('--query-count',type=int,default=128)
    p.add_argument('--reference');p.add_argument('--iterations',type=int,default=14);p.add_argument('--block',type=int,default=288)
    p.add_argument('--tolerance',type=float,default=1e-6);p.add_argument('--rank',type=int,default=100);p.add_argument('--neighbors',type=int,default=80)
    p.add_argument('--cutoff',type=float,default=12.);p.add_argument('--refinements',type=int,default=0)
    p.add_argument('--reference-kind',choices=['refined','float64'],default='refined')
    p.add_argument('--batch',type=int,default=128);p.add_argument('--eta',type=float,default=.01);p.add_argument('--seed',type=int,default=9101)
    p.add_argument('--sampling',choices=['permutation','rejection'],default='rejection')
    p.add_argument('--checkpoints',type=int,nargs='+',default=[128,512,2048,8192,32768,131072])
    p.add_argument('--budget',type=float,default=120.);p.add_argument('--correction',choices=['none','residual'],default='residual')
    a=p.parse_args();reference(a) if a.method=='reference' else run(a)
