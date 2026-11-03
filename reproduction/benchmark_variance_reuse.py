"""Cold preparation and changing-query scaling for marginal mean/variance.

Synthetic surface-query proxy on the released scorpion. The Fourier cache is
an exact finite-cross-kernel Cholesky hybrid, not a NUFFT conditioning solver.
Pilot tuning is recorded separately; only passing held-out results count.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import cupy as cp
import cupyx.scipy.linalg as cla
import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import ndtr

from geospsr.iterative import StochasticDualDescent
from geospsr.kernels import Matern32Kernel, ProductKernel
from geospsr.spectral import FourierPoisson
from geospsr.cached_spectral import CachedFourierEvaluator
from geospsr.spectral_tail import analytic_covariance_rows64
from reproduction.tabulated_cross import TabulatedCrossKernel
from reproduction.variance_series import prepare_variance_series, FourierSeriesEvaluator
from reproduction.gpu_audit import GpuProcessAudit

ROOT=Path(__file__).resolve().parents[1]


def timed(fn):
    cp.cuda.runtime.deviceSynchronize(); begin=time.perf_counter()
    result=fn()
    jax.block_until_ready(result)
    cp.cuda.runtime.deviceSynchronize()
    return result,time.perf_counter()-begin


def own_jax(v):
    v.block_until_ready()
    result=cp.from_dlpack(v).copy()
    cp.cuda.get_current_stream().synchronize()
    return result


def quality(reference, candidate, prior):
    mu,var=reference; cm,cv=candidate
    mu,var,cm,cv=[cp.asnumpy(a) for a in (mu,var,cm,cv)]
    contrast=var-var.mean()
    contrast_error=np.linalg.norm((cv-cv.mean())-contrast)/max(np.linalg.norm(contrast),1e-300)
    variance_error=np.linalg.norm(cv-var)/max(np.linalg.norm(var),1e-300)
    mean_error=np.linalg.norm(cm-mu)/max(np.linalg.norm(mu),1e-300)
    positive=bool(np.all(var>0)&np.all(cv>0))
    delta=ndtr(-cm/np.sqrt(np.maximum(cv,1e-300)))-ndtr(-mu/np.sqrt(np.maximum(var,1e-300)))
    rms=float(np.sqrt(np.mean(delta**2))); maximum=float(np.max(np.abs(delta)))
    return {'mean_relative_l2':float(mean_error),'variance_relative_l2':float(variance_error),
        'contrast_relative_l2':float(contrast_error),'probability_rms':rms,'probability_max':maximum,
        'positive':positive,'pass':bool(positive and mean_error<=.002 and variance_error<=.0002
            and contrast_error<=.01 and rms<=.0005 and maximum<=.005),
        'correction_relative_l2':float(np.linalg.norm(cv-var)/max(np.linalg.norm(prior-var),1e-300))}


class SDDRunner:
    def __init__(self,x,count,ell,noise,batch,eta):
        k=Matern32Kernel(ell,.1); kernel=ProductKernel(k,k,k)
        self.solver=StochasticDualDescent(jnp.asarray(x),jnp.zeros((len(x),count)),kernel,
            noise_variance=noise,batch=min(batch,len(x)),lr=eta*min(batch,len(x))/len(x))
        self.solver.compile()

    def reset(self,rhs):
        b=jax.dlpack.from_dlpack(cp.ascontiguousarray(rhs,dtype='f'))
        self.solver.y=b
        z=jnp.zeros_like(b)
        self.solver.state=(jnp.asarray(0,dtype=jnp.int32),jax.random.key(0),z,z,z)

    def solve(self,rhs,steps):
        self.reset(rhs)
        return own_jax(self.solver.advance(steps)).astype('d')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--case',choices=['A','B'],default='A')
    p.add_argument('--n',type=int,default=256)
    p.add_argument('--cutoff',type=int,default=12)
    p.add_argument('--ell',type=float,default=.01)
    p.add_argument('--grid',type=int,default=200)
    p.add_argument('--queries',type=int,nargs='+',default=[64,256,1024,4096])
    p.add_argument('--query-batch',type=int,default=64)
    p.add_argument('--dense-batch',type=int,default=1024)
    p.add_argument('--factor-batch',type=int,default=8)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--sdd',action='store_true')
    p.add_argument('--sdd-maxiter',type=int,default=8192)
    p.add_argument('--spatial-grids',type=int,nargs='*',default=[])
    p.add_argument('--query-domain',choices=['surface','mixed'],default='surface')
    p.add_argument('--spatial-domain',choices=['object','periodic'],default='object')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if any(m%a.query_batch for m in a.queries):p.error('query counts must be multiples of query-batch')
    a.output.mkdir(parents=True,exist_ok=False)
    data=np.load(ROOT/f'reproduction/results/matched/{a.case}/input.npz')
    ids=np.random.default_rng(92).permutation(len(data['points']))[:a.n]
    x,y=data['points'][ids],data['normals'][ids]; n=len(x)
    xp=cp.asarray(x); yp=cp.asarray(y,dtype='d')
    r={'complete':False,'args':{**vars(a),'output':str(a.output)},'n':n,'seconds':{},'measurements':[],
       'scope':'synthetic changing surface-query proxy; dense-analytic conditioning, finite scalar/cross kernel; no ContactFusion replay',
       'gate':{'mean_relative_l2':.002,'variance_relative_l2':.0002,'centered_contrast_relative_l2':.01,
               'probability_rms':.0005,'probability_max':.005},
       'device':cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
       'device_memory':cp.cuda.runtime.memGetInfo()[1],
       'versions':{'cupy':cp.__version__,'jax':jax.__version__,'numpy':np.__version__},
       'hashes':{key:hashlib.sha256(value.tobytes()).hexdigest() for key,value in [('points',x),('normals',y)]}}
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
        op=stage('cross_spectrum',lambda:FourierPoisson(a.cutoff,lengthscale=a.ell,dtype='float64',eps=1e-11))
        def matrix():
            mat=analytic_covariance_rows64(x,x,lengthscale=a.ell)
            mat.flat[::n+1]+=.02
            return mat
        mat=stage('analytic_matrix',matrix)
        chol=stage('cholesky',lambda:cp.linalg.cholesky(mat))
        alpha=stage('dense_mean_solve',lambda:cla.solve_triangular(chol.T,cla.solve_triangular(chol,yp,lower=True),lower=False))
        mean_coeff=stage('mean_aggregate',lambda:op.aggregate(x,alpha))
        offset=stage('mean_center',lambda:float(op.evaluate(x,mean_coeff).mean()))
        (variance_coeff,record)=stage('variance_series_prepare',lambda:prepare_variance_series(x,chol,op,a.factor_batch))
        r['variance_series']=record;prior=record['prior_variance']
        variance_eval=FourierSeriesEvaluator(variance_coeff)
        mean_eval=FourierSeriesEvaluator(mean_coeff)
        def interpolation_cache():
            combined=cp.zeros((2,*variance_coeff.shape),dtype='complex128')
            sl=slice(a.cutoff,3*a.cutoff+1)
            combined[0,sl,sl,sl]=mean_coeff[0]
            combined[1]=variance_coeff
            return CachedFourierEvaluator(combined,eps=1e-6,oversampling=2.,batch=1)
        grid_cache=stage('moment_grid_prepare',interpolation_cache)
        r['moment_cache_bytes']=grid_cache.cache_bytes
        table=stage('reference_table',lambda:TabulatedCrossKernel(op,a.grid))
        def reference(q,exact=False):
            ms=[];vs=[]
            for start in range(0,len(q),a.dense_batch):
                block=q[start:start+a.dense_batch]
                c=(op.cross_values((block[:,None].astype('d')-xp[None].astype('d')).reshape(-1,3)).reshape(len(block),n,3)
                   if exact else table.rows(block,xp))
                ms.append((c*alpha[None]).sum((1,2))-offset)
                rhs=cp.ascontiguousarray(c.transpose(1,0,2).reshape(n,-1))
                z=cla.solve_triangular(chol,rhs,lower=True)
                vs.append(prior-(z*z).sum(0).reshape(-1,3).sum(1))
            return cp.concatenate(ms),cp.concatenate(vs)
        def cached(q):return mean_eval.evaluate(q)[0]-offset,variance_eval.evaluate(q)[0]
        def cached_grid(q):
            values=grid_cache.evaluate(q)
            return values[0]-offset,values[1]
        # Independent surface coordinates, no repeated/cached answers. The mesh
        # itself is released; these are constructed query workloads, not sensor data.
        from experiments.utils import uniform_sample_mesh
        qhost,_=uniform_sample_mesh(data['vertices'],data['faces'],max(a.queries),np.random.default_rng(716))
        qhost=qhost.astype('f');qhost+=np.random.default_rng(717).normal(0,.01,qhost.shape).astype('f')
        if a.query_domain=='mixed':
            # Unknown future query regions: half near-surface, half workspace
            # probes. Interleave so every prefix has the same mixture.
            qhost[1::2]=np.random.default_rng(718).uniform(-np.pi,np.pi,qhost[1::2].shape).astype('f')
        r['hashes']['base_queries']=hashlib.sha256(qhost.tobytes()).hexdigest()
        pilot=cp.asarray(qhost[:a.query_batch])
        exact=stage('validation_direct_pilot',lambda:reference(pilot,True))
        r['table_pilot_quality']=quality(exact,reference(pilot),prior)
        r['cache_pilot_quality']=quality(exact,cached(pilot),prior);save()
        print('pilot quality',r['table_pilot_quality'],r['cache_pilot_quality'],flush=True)
        chol32=stage('cholesky_float32',lambda:cp.linalg.cholesky(mat.astype('f')))
        alpha32=stage('mean_solve_float32',lambda:cla.solve_triangular(chol32.T,cla.solve_triangular(chol32,yp.astype('f'),lower=True),lower=False))
        coeff32=stage('mean_aggregate_float32',lambda:op.aggregate(x,alpha32))
        offset32=stage('mean_center_float32',lambda:float(op.evaluate(x,coeff32).mean()))
        def reference32(q):
            ms=[];vs=[]
            for start in range(0,len(q),a.dense_batch):
                c=table.rows(q[start:start+a.dense_batch],xp).astype('f')
                ms.append((c*alpha32[None]).sum((1,2))-offset32)
                rhs=cp.ascontiguousarray(c.transpose(1,0,2).reshape(n,-1))
                z=cla.solve_triangular(chol32,rhs,lower=True)
                vs.append(prior-(z*z).sum(0).reshape(-1,3).sum(1))
            return cp.concatenate(ms),cp.concatenate(vs)
        r['float32_pilot_quality']=quality(exact,reference32(pilot),prior)
        sdd_fn=None
        if a.sdd:
            c=table.rows(pilot,xp);rhs=cp.ascontiguousarray(c.transpose(1,0,2).reshape(n,-1))
            r['sdd_pilots']=[];best=None
            for batch,eta in [(128,2.),(128,5.),(512,2.),(512,5.)]:
                runner,compile_seconds=timed(lambda:SDDRunner(x,rhs.shape[1],a.ell,.02,batch,eta))
                runner.reset(rhs);elapsed=0.;history=[]
                while runner.solver.iterations<a.sdd_maxiter:
                    w,seconds=timed(lambda:own_jax(runner.solver.advance(256)).astype('d'));elapsed+=seconds
                    vv=prior-(rhs*w).sum(0).reshape(-1,3).sum(1)
                    check=quality(exact,(exact[0],vv),prior)
                    history.append({'iterations':runner.solver.iterations,'seconds':elapsed,'quality':check})
                    if check['pass']:break
                row={'batch':min(batch,n),'eta':eta,'compile_seconds':compile_seconds,'history':history}
                r['sdd_pilots'].append(row);save();print('sdd pilot',batch,eta,history[-1],flush=True)
                if check['pass'] and (best is None or elapsed<best[0]):best=(elapsed,runner,runner.solver.iterations,batch,eta,compile_seconds)
            if best is not None:
                _,runner,steps,batch,eta,compile_seconds=best
                mean_runner=stage('sdd_mean_compile',lambda:SDDRunner(x,3,a.ell,.02,batch,eta))
                mean_runner.reset(yp);mean_seconds=0.;mean_history=[]
                while mean_runner.solver.iterations<a.sdd_maxiter:
                    sa,dt=timed(lambda:own_jax(mean_runner.solver.advance(256)).astype('d'))
                    mean_seconds+=dt
                    residual=float(cp.linalg.norm(mat@sa-yp)/cp.linalg.norm(yp))
                    mean_history.append({'iterations':mean_runner.solver.iterations,'update_seconds':mean_seconds,
                                         'analytic_relative_residual':residual})
                    if residual<=.0005:break
                r['seconds']['sdd_mean_solve']=mean_seconds;r['sdd_mean_history']=mean_history
                sc=stage('sdd_mean_aggregate',lambda:op.aggregate(x,sa))
                so=stage('sdd_mean_center',lambda:float(op.evaluate(x,sc).mean()))
                r['sdd_selected']={'batch':min(batch,n),'eta':eta,'iterations':steps,'query_compile_seconds':compile_seconds}
                def sdd_fn(q):
                    ms=[];vs=[]
                    for start in range(0,len(q),a.query_batch):
                        c=table.rows(q[start:start+a.query_batch],xp)
                        rhs=cp.ascontiguousarray(c.transpose(1,0,2).reshape(n,-1))
                        w=runner.solve(rhs,steps)
                        ms.append((c*sa[None]).sum((1,2))-so)
                        vs.append(prior-(rhs*w).sum(0).reshape(-1,3).sum(1))
                    return cp.concatenate(ms),cp.concatenate(vs)
            else:r['sdd_status']='no passing pilot within declared iteration budget'
        methods={'cholesky_direct':reference,'cholesky_nufft_variance_cache':cached,
                 'cholesky_nufft_moment_grid':cached_grid,'cholesky_direct_float32':reference32}
        if sdd_fn is not None:methods['sdd_direct']=sdd_fn
        if a.spatial_grids:
            from reproduction.spatial_moment_grid import SpatialMomentGrid
            if a.spatial_domain=='periodic':
                lower=np.full(3,-np.pi);upper=np.full(3,np.pi)
            else:
                lower=data['vertices'].min(0)-.15;upper=data['vertices'].max(0)+.15
            screen_q=cp.asarray(qhost[:min(1024,len(qhost))])
            screen_ref=reference(screen_q,True);r['spatial_grid_screens']=[]
            for size in a.spatial_grids:
                # Only accuracy is screened using the candidate. Construction
                # timing for the baseline below uses direct Cholesky evaluation.
                grid,seconds=timed(lambda:SpatialMomentGrid.prepare(cached_grid,size,lower,upper,periodic=a.spatial_domain=='periodic'))
                test=quality(screen_ref,grid.evaluate(screen_q),prior)
                r['spatial_grid_screens'].append({'size':size,'quality':test,'screen_seconds':seconds})
                save();print('spatial screen',size,test,flush=True)
                if test['pass']:
                    baseline=reference32 if r['float32_pilot_quality']['pass'] else reference
                    r['spatial_grid_builder']='float32' if baseline is reference32 else 'float64'
                    grid=stage('cholesky_spatial_grid_prepare',lambda:SpatialMomentGrid.prepare(baseline,size,lower,upper,periodic=a.spatial_domain=='periodic'))
                    r['spatial_grid']={'size':size,'order':3,'lower':lower.tolist(),'upper':upper.tolist(),
                                       'cache_bytes':grid.cache_bytes}
                    methods['cholesky_spatial_grid']=grid.evaluate
                    break
                del grid
        # Every repeat changes geometry; method order rotates to expose warmup.
        for m in a.queries:
            for repeat in range(a.repeats):
                q=cp.asarray(qhost[:m]+np.array([.001*repeat,-.0007*repeat,.0003*repeat],dtype='f'))
                if a.query_domain=='mixed':q=(q+np.pi)%(2*np.pi)-np.pi
                order=list(methods);order=order[repeat%len(order):]+order[:repeat%len(order)]
                values={}; row={'queries':m,'repeat':repeat,'seconds':{},'quality':{}}
                for name in order:
                    values[name],row['seconds'][name]=timed(lambda:methods[name](q))
                # Dense table calculation is the timed reference; an independent
                # exact cross-kernel panel audits approximation at new locations.
                ref=values['cholesky_direct']
                for name in order:row['quality'][name]=quality(ref,values[name],prior)
                heldout_ids=np.linspace(0,m-1,min(m,a.query_batch),dtype=int)
                precise=reference(q[heldout_ids],True)
                row['heldout_exact_quality']={name:quality(precise,(v[0][heldout_ids],v[1][heldout_ids]),prior) for name,v in values.items()}
                r['measurements'].append(row);save();print('measurement',m,repeat,row['seconds'],{k:v['pass'] for k,v in row['heldout_exact_quality'].items()},flush=True)
        r['complete']=True
    except Exception as exc:r['failure']=f'{type(exc).__name__}: {exc}';raise
    finally:r['gpu_audit']=monitor.stop();save()


if __name__=='__main__':main()
