"""Large-N quality controls; full-kernel refinement is independent of H2."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from geospsr.nonperiodic import FreeSpacePoissonMatern
from reproduction.partial_quality import OUT, occupancy, roots, snapshot
from reproduction.partial_occupancy import load
from reproduction.stanford_partial_data import digest


class RefinedDenseSystem:
    """Bound RHS memory while refining against every float64 kernel entry."""
    def __init__(self,model,x,noise,block=576):
        import cupy as cp
        from geospsr.nonperiodic_dense_gpu import GPUDenseMaternSystem
        self.model,self.points,self.noise,self.block=model,x,noise,block
        self.system=GPUDenseMaternSystem(model,x,noise,dtype='float32')
        cp.get_default_memory_pool().free_all_blocks()
        self.histories=[];self.last_solve=None

    def _solve(self,b):
        import cupy as cp
        from geospsr.nonperiodic_dense_gpu import full_kernel_matmul
        b=cp.asarray(b);z=cp.asarray(self.system.solve(cp.asnumpy(b)))
        norm=cp.linalg.norm(b,axis=0);history=[]
        for stage in range(12):
            residual=b-cp.asarray(full_kernel_matmul(self.model,self.points,self.noise,cp.asnumpy(z),workspace_bytes=256*2**20))
            relative=float(cp.max(cp.linalg.norm(residual,axis=0)/cp.where(norm>0,norm,1.)).item())
            bound=float(cp.max(cp.sum(residual*residual,axis=0)).item()/self.noise)
            history.append(dict(stage=stage,relative_residual=relative,reduction_entry_error_bound=bound))
            if relative<=1e-10:break
            z+=cp.asarray(self.system.solve(cp.asnumpy(residual)))
        else:raise RuntimeError('full-kernel iterative refinement failed')
        self.histories.append(history)
        return z,residual,history

    def solve(self,b,tolerance=None):
        import cupy as cp
        b=np.asarray(b);outputs=[];histories=[]
        for first in range(0,b.shape[1],self.block):
            z,_,h=self._solve(b[:,first:first+self.block]);outputs.append(cp.asnumpy(z));histories.append(h)
            del z;cp.get_default_memory_pool().free_all_blocks()
        self.last_solve=dict(converged=True,blocks=histories)
        return np.column_stack(outputs)

    def variance_solve(self,b,iterations):
        import cupy as cp
        zs=[];rs=[];hs=[]
        for first in range(0,b.shape[1],self.block):
            z,r,h=self._solve(b[:,first:first+self.block]);zs.append(z);rs.append(r);hs.append(h)
        return cp.column_stack(zs),dict(max_iterations=max(len(h) for h in hs)),cp.column_stack(rs)


def run(args):
    import cupy as cp
    from reproduction.gpu_audit import GpuProcessAudit
    from reproduction.partial_direct_gpu import DirectContactQueries, cross_rhs
    from reproduction.compute_partial_figures import mean_coefficients, mean_field
    from reproduction.evaluate_partial_controls import contact_only_variance
    path=Path(args.output)
    if path.exists():raise FileExistsError(path)
    panel=dict(np.load(OUT/(args.group+'-panel.npz')))
    x,y,_,original_panel=load(args.group,args.n);contacts=panel['contacts'][:args.contacts]
    benchmark_ids=np.linspace(0,len(original_panel['queries'])-1,128,dtype=int) if args.group=='test_side' else np.array([],dtype=int)
    queries=np.concatenate((panel['queries'],original_panel['queries'][benchmark_ids]))
    model=FreeSpacePoissonMatern(args.ell,1/3);noise=args.sigma**2
    source=snapshot()
    for p in [__file__,'reproduction/compute_partial_figures.py','geospsr/hierarchical.py','geospsr/hierarchical_gpu.py','geospsr/hierarchical_interactions.py','reproduction/gpu_audit.py']:
        h=digest(p);(OUT/'sources'/(h+'.py')).write_bytes(Path(p).read_bytes());source[p]=h
    record=dict(arguments=vars(args),source_sha256=source,panel_sha256=digest(OUT/(args.group+'-panel.npz')),
                dataset_sha256=digest(OUT/'inputs.json'),
                scope='Quality evaluation, not the 128-query performance benchmark',benchmark_query_indices=benchmark_ids.tolist())
    cross_rhs(model,contacts[:2],x[:2]);cp.cuda.Device().synchronize()
    audit=GpuProcessAudit(cp.cuda.Device().pci_bus_id,interval=.5,cuda_device=cp.cuda.Device().id).start()
    start=perf_counter()
    try:
        if args.method=='reference':system=RefinedDenseSystem(model,x,noise)
        elif args.method=='float64':
            from geospsr.nonperiodic_dense_gpu import GPUDenseMaternSystem
            system=GPUDenseMaternSystem(model,x,noise,dtype='float64')
        else:
            from geospsr.hierarchical import HierarchicalMaternSystem
            from geospsr.hierarchical_gpu import GPUHierarchicalSystem
            native=HierarchicalMaternSystem(model,x,noise,tolerance=args.tolerance,leaf_size=40,
                preconditioner_rank=100,neighbors=80,threads=4,seed=9101,preconditioner_backend='prefix-kd')
            system=GPUHierarchicalSystem(native,dtype='float32',rhs_block_size=72,release_native=True)
        posterior=DirectContactQueries(model,x,y,contacts,system,dense=args.method=='float64',iterations=args.iterations,solve_tolerance=1e-9)
        record['mean_solve']=posterior.mean_solve_info;record['contact_solve']=posterior.contact_solve_info
        print('CONTACTS',args.method,args.group,args.n,args.contacts,perf_counter()-start,flush=True)
        parts=[]
        for first in range(0,len(queries),192):
            parts.append(posterior.predict(queries[first:first+192],batch_size=96))
            print('QUERY PROGRESS',args.group,min(first+192,len(queries)),len(queries),perf_counter()-start,flush=True)
        allm,allv=(np.concatenate([p[k] for p in parts]) for k in range(2))
        if not(np.isfinite(allm).all() and np.isfinite(allv).all() and np.all(allv>0)):raise ValueError('invalid posterior moments')
        m,v=allm[:512],allv[:512]
        print('QUERIES',args.method,args.group,perf_counter()-start,flush=True)
        coefficients=mean_coefficients(posterior)
        folded=mean_field(posterior,panel['queries'],coefficients)
        record['folded_mean_max_difference']=float(np.max(abs(folded-m)))
        if record['folded_mean_max_difference']>1e-10:raise ValueError('folded mean mismatch')
        probe=mean_field(posterior,panel['probes'].reshape(-1,3),coefficients)
        geo,error,missing=roots(probe,panel)
        cv=contact_only_variance(model,contacts,panel['queries'])
        record.update(geometry=geo,occupancy=occupancy(m,v,panel),contact_only=occupancy(m,cv,panel),prior_variance=model.prior_variance,
            variance_iterations=posterior.solve_iterations)
        if args.method=='reference':record['refinement_history']=system.histories
        np.savez_compressed(path.with_suffix('.npz'),mean=m,variance=v,probe_mean=probe,root_error=error,missing=missing,
            coefficients=cp.asnumpy(coefficients[0]),contact_weights=coefficients[1],benchmark_mean=allm[512:],benchmark_variance=allv[512:])
        record.update(status='completed',field_sha256=digest(path.with_suffix('.npz')))
    except Exception as exc:
        record.update(status='capacity failure' if isinstance(exc,MemoryError) else 'failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        record.update(seconds=perf_counter()-start,gpu_audit=audit.stop())
        path.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print('DONE',args.group,args.method,record['seconds'],record['occupancy']['all']['brier'],geo,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--group',default='validation');p.add_argument('--n',type=int,default=70652)
    p.add_argument('--ell',type=float,required=True);p.add_argument('--sigma',type=float,required=True)
    p.add_argument('--contacts',type=int,choices=[32,128,512],required=True);p.add_argument('--method',choices=['reference','float64','h2'],default='reference')
    p.add_argument('--iterations',type=int,default=10);p.add_argument('--tolerance',type=float,default=1e-6)
    p.add_argument('--output',required=True);run(p.parse_args())
