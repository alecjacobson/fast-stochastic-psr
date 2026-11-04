"""Matched analytic-model mean solves: dense, compiled SDD, and NUFFT refinement.

Nested distinct scan observations; full analytic residuals and independent
direct rows. All methods use the same Fourier mean evaluator, isolating solve
scaling even when SDD benefits from NUFFT query acceleration.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import cupy as cp
import cupyx.scipy.linalg as cla
import jax
import jax.numpy as jnp
import numpy as np

from geospsr.analytic_refinement import refine_analytic
from geospsr.iterative import StochasticDualDescent
from geospsr.kernels import Matern32Kernel, ProductKernel
from geospsr.packed_spectral import PackedFourierKernel
from geospsr.periodic_leaf_cache import LeafCachedRangeKernel
from geospsr.spectral import FourierPoisson
from geospsr.spectral_tail import analytic_covariance_rows64
from geospsr.training_spectrum import FourierCovariance
from reproduction.analytic_posterior import _AnalyticCG
from reproduction.benchmark_variance_reuse import own_jax,timed
from reproduction.gpu_audit import GpuProcessAudit
from reproduction.dense_covariance import analytic_covariance_rows32


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--n',type=int,required=True)
    p.add_argument('--cutoff',type=int,default=100)
    p.add_argument('--rtol',type=float,default=.0001)
    p.add_argument('--dense-max',type=int,default=16384)
    p.add_argument('--analytic-cache-gib',type=float,default=4.)
    p.add_argument('--batches',type=int,nargs='+',default=[128,512])
    p.add_argument('--etas',type=float,nargs='+',default=[2.,5.])
    p.add_argument('--budget',type=float,default=120.)
    p.add_argument('--maxiter',type=int,default=200000)
    p.add_argument('--chunk',type=int,default=256)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--adaptive-confirm',action='store_true')
    p.add_argument('--selected-sdd',type=Path,help='reuse only the pilot-selected configuration from this result')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    x=np.load(a.source/'points.npy',mmap_mode='r')[:a.n].copy()
    y=np.load(a.source/'normals.npy',mmap_mode='r')[:a.n].copy()
    if len(x)!=a.n:raise ValueError('source has fewer observations than requested')
    q=np.load(a.source/'clean_points.npy',mmap_mode='r')[-4096:].copy()
    q+=np.random.default_rng(700).normal(0,.02,q.shape).astype('f')
    b=cp.asarray(y,dtype='d');target_norm=cp.linalg.norm(b)
    r={'complete':False,'n':len(x),'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
       'model':{'training':'full analytic periodic product Matern32','ell':.01,'per_axis_variance':.1,'noise_variance':.02,'cross_cutoff':50},
       'scope':'mean preparation and common NUFFT queries; no marginal/joint uncertainty speedup',
       'gates':{'full_analytic_relative_residual':a.rtol,'mean_relative_l2':.002,'direct_action_relative_error':1e-9},
       'seconds':{},'methods':{},'hashes':{k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in [('points',x),('normals',y),('queries',q)]},
       'device':cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),'device_memory':cp.cuda.runtime.memGetInfo()[1],
       'versions':{'cupy':cp.__version__,'jax':jax.__version__,'numpy':np.__version__}}
    # Hash source files, including uncommitted benchmark implementation.
    root=Path(__file__).resolve().parents[1]
    r['source_sha256']={str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest()
                        for folder in ('geospsr','reproduction')
                        for path in sorted((root/folder).glob('*.py'))}
    def save():(a.output/'results.json').write_text(json.dumps(r,indent=2)+'\n')
    def stage(name,fn):
        v,t=timed(fn);r['seconds'][name]=t;save();print(name,t,flush=True);return v
    monitor=GpuProcessAudit(cp.cuda.Device().pci_bus_id,cuda_device=cp.cuda.Device().id).start()
    try:
        action=stage('analytic_geometry_setup',lambda:LeafCachedRangeKernel(x,.01,.1,.02,leaf_size=128,
            max_cache_bytes=int(a.analytic_cache_gib*2**30),max_entries=150_000_000))
        r['analytic_geometry']=action.record
        # This independent reference and its validation are outside candidate
        # timing. NUFFT refinement must separately pay for its analytic geometry.
        ref_solver=_AnalyticCG(action)
        ref,ref_info=stage('reference_analytic_solve',lambda:ref_solver.solve(b,rtol=1e-10,maxiter=2000))
        if not ref_info['converged']:raise RuntimeError(('reference failed',ref_info))
        r['reference_solve']=ref_info
        ids=np.random.default_rng(1237).choice(len(x),min(len(x),64),replace=False)
        rows=stage('reference_direct_rows',lambda:analytic_covariance_rows64(x[ids],x))
        discrepancy=rows@ref+.02*ref[ids]-action(ref)[ids]
        r['direct_action_relative_error']=float(cp.linalg.norm(discrepancy)/cp.linalg.norm(b[ids]))
        if r['direct_action_relative_error']>1e-9:raise RuntimeError('analytic action independent check failed')
        op=stage('shared_cross_spectrum',lambda:FourierPoisson(50,dtype='float64',eps=1e-11))
        def field(w):
            coeff=op.aggregate(x,w);offset=op.evaluate(x,coeff).mean()
            return op.evaluate(q,coeff)[0]-offset
        ref_field=stage('reference_mean_queries',lambda:field(ref))
        def check(w):
            res=float(cp.linalg.norm(action(w)-b)/target_norm)
            f,seconds=timed(lambda:field(w))
            error=float(cp.linalg.norm(f-ref_field)/cp.linalg.norm(ref_field))
            return {'analytic_relative_residual':res,'mean_relative_l2':error,
                    'common_mean_prepare_query_seconds':seconds,'pass':bool(res<=a.rtol and error<=.002)}
        # Ablation: the refinement machinery is itself a fast exact action.
        # Compare it without NUFFT at the same tolerance before attributing gains.
        control={'prepare_seconds':r['seconds']['analytic_geometry_setup'],'runs':[]}
        r['methods']['analytic_cg']=control
        for repeat in range(a.repeats):
            (w,info),seconds=timed(lambda:ref_solver.solve(b,rtol=a.rtol/2,maxiter=2000))
            control['runs'].append({'repeat':repeat,'solve_seconds':seconds,
                                    'solver':info,'quality':check(w)})
            save()
        if len(x)<=a.dense_max:
            for dtype,name in [('f','cholesky'),('d','cholesky_float64')]:
                if dtype=='d' and len(x)>16384:
                    r['methods'][name]={'status':'not run beyond float64 cap 16384; earlier 32768 attempt exceeded workspace memory'}
                    continue
                def dense_prepare():
                    matrix=(analytic_covariance_rows32 if dtype=='f' else analytic_covariance_rows64)(x,x)
                    matrix.flat[::len(x)+1]+=.02
                    return matrix
                matrix=stage(name+'_matrix',dense_prepare)
                chol=stage(name+'_factor',lambda:cp.linalg.cholesky(matrix))
                weights,seconds=timed(lambda:cla.solve_triangular(chol.T,cla.solve_triangular(chol,b.astype(dtype),lower=True),lower=False))
                r['methods'][name]={'prepare_seconds':r['seconds'][name+'_matrix']+r['seconds'][name+'_factor'],
                    'dtype':str(chol.dtype),'solve_seconds':seconds,'quality':check(weights)}
                del matrix,chol,weights;cp.get_default_memory_pool().free_all_blocks();save()
        else:r['methods']['cholesky']={'status':'not run beyond declared dense benchmark cap; no measured runtime claim',
                                       'single_matrix_bytes':8*len(x)**2}
        spectrum=stage('nufft_spectrum',lambda:FourierCovariance(a.cutoff))
        finite=stage('nufft_plans',lambda:PackedFourierKernel(x,spectrum,.02,rhs_count=3))
        def nufft_solve():
            approximate=lambda rhs:finite.solve(rhs,rtol=1e-5,maxiter=2000)
            initial,info=approximate(b)
            w,refinement=refine_analytic(action,approximate,b,initial,rtol=a.rtol/2,max_corrections=12)
            return w,{'initial':info,'refinement':refinement}
        r['methods']['nufft_refined']={'prepare_seconds':r['seconds']['analytic_geometry_setup']+
            r['seconds']['nufft_spectrum']+r['seconds']['nufft_plans'],'runs':[]}
        for repeat in range(a.repeats):
            (w,info),seconds=timed(nufft_solve)
            row={'repeat':repeat,'solve_seconds':seconds,'solver':info,'quality':check(w)}
            r['methods']['nufft_refined']['runs'].append(row);save();print('nufft',row,flush=True)
        del finite,spectrum,w;gc.collect();cp.get_default_memory_pool().free_all_blocks()
        k=Matern32Kernel(.01,.1);kernel=ProductKernel(k,k,k)
        xj,yj=jnp.asarray(x),jnp.asarray(y)
        pilots=[];best=None
        if a.selected_sdd:
            source=json.loads(a.selected_sdd.read_text())
            if source['n']!=len(x) or source['hashes']!=r['hashes']:raise ValueError('selected SDD source inputs differ')
            selected=source['methods']['sdd']['selected'];batch=selected['batch'];eta=selected['eta']
            solver=StochasticDualDescent(xj,yj,kernel,batch=min(batch,len(x)),lr=eta*min(batch,len(x))/len(x))
            _,comp=timed(solver.compile)
            best=(0.,solver,comp,batch,eta,selected['iterations'])
            r['methods']['sdd']={'selection_source':str(a.selected_sdd),'pilots':[]}
        for batch in ([] if a.selected_sdd else a.batches):
            for eta in a.etas:
                solver=StochasticDualDescent(xj,yj,kernel,batch=min(batch,len(x)),lr=eta*min(batch,len(x))/len(x))
                _,comp=timed(solver.compile);elapsed=0.;validation=0.;history=[]
                while solver.iterations<a.maxiter:
                    jw,seconds=timed(lambda:solver.advance(a.chunk));elapsed+=seconds
                    def residual():
                        w=own_jax(jw).astype('d')
                        return float(cp.linalg.norm(action(w)-b)/target_norm)
                    res,seconds=timed(residual);validation+=seconds
                    history.append({'iterations':solver.iterations,'updates_seconds':elapsed,'full_analytic_relative_residual':res})
                    if res<=a.rtol or not np.isfinite(res) or res>10 or elapsed>=a.budget:break
                row={'batch':min(batch,len(x)),'eta':eta,'compile_seconds':comp,'validation_seconds':validation,'history':history,
                     'status':'pass' if res<=a.rtol else 'budget_or_divergence'}
                pilots.append(row);r['methods']['sdd']={'pilots':pilots};save();print('sdd pilot',row['batch'],eta,history[-1],flush=True)
                if res<=a.rtol and (best is None or elapsed<best[0]):best=(elapsed,solver,comp,batch,eta,solver.iterations)
        if best is not None:
            _,solver,comp,batch,eta,steps=best
            sdd=r['methods']['sdd'];sdd.update(selected={'batch':batch,'eta':eta,'iterations':steps},prepare_seconds=comp,runs=[])
            sdd['confirmation_policy']='full analytic residual stopping' if a.adaptive_confirm else 'frozen pilot iteration count'
            sdd['verification_setup_seconds']=r['seconds']['analytic_geometry_setup'] if a.adaptive_confirm else 0.
            for repeat in range(a.repeats):
                # Independent coordinate sequence checks whether a pilot-picked
                # budget generalizes. Failures stay failures, never retuned here.
                zero=jnp.zeros_like(yj)
                solver.state=(jnp.asarray(0,dtype=jnp.int32),jax.random.key(500+repeat),zero,zero,zero)
                jw,seconds=timed(lambda:solver.advance(steps));verification_seconds=0.;history=[]
                w=own_jax(jw).astype('d')
                if a.adaptive_confirm:
                    while True:
                        res,dt=timed(lambda:float(cp.linalg.norm(action(w)-b)/target_norm));verification_seconds+=dt
                        history.append({'iterations':solver.iterations,'update_seconds':seconds,'full_analytic_relative_residual':res})
                        if res<=a.rtol or solver.iterations>=a.maxiter or seconds>=a.budget:break
                        jw,dt=timed(lambda:solver.advance(a.chunk));seconds+=dt;w=own_jax(jw).astype('d')
                row={'repeat':repeat,'seed':500+repeat,'solve_seconds':seconds,'quality':check(w),
                     'iterations':solver.iterations,'stopping_history':history,
                     'verification_seconds':verification_seconds,'verified_solve_seconds':seconds+verification_seconds}
                sdd['runs'].append(row);save();print('sdd confirmation',row,flush=True)
        r['complete']=True
    except Exception as exc:r['failure']=f'{type(exc).__name__}: {exc}';raise
    finally:r['gpu_audit']=monitor.stop();save()


if __name__=='__main__':main()
