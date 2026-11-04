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
    p.add_argument('--moment-method',choices=['squares','gradient'],default='squares')
    p.add_argument('--moment-dtype',choices=['float64','float32'],default='float64')
    p.add_argument('--source-dtype',choices=['float64','float32'],default='float64')
    p.add_argument('--paired-sources',action='store_true')
    p.add_argument('--inverse-batch',type=int,default=0)
    p.add_argument('--probability',action='store_true',help='include the marginal probability CDF in measured query work')
    p.add_argument('--shared-mean',action='store_true',help='use the same accurate prepared mean with all direct variance baselines')
    p.add_argument('--fast-reference',action='store_true',help='use validated float32 direct queries for full-batch comparison; retain exact double held-out panels')
    p.add_argument('--spatial-screen-queries',type=int,default=1024)
    p.add_argument('--spatial-order',type=int,choices=[3,5],default=3)
    p.add_argument('--replay-batches',type=int,default=0)
    p.add_argument('--replay-queries',type=int,default=65536)
    p.add_argument('--replay-budget',type=float,default=180.)
    p.add_argument('--sdd-measure-max',type=int,default=0)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--sdd',action='store_true')
    p.add_argument('--sdd-shared-mean',action='store_true',help='give SDD the same prepared Cholesky mean; isolate variance workload')
    p.add_argument('--sdd-maxiter',type=int,default=8192)
    p.add_argument('--spatial-grids',type=int,nargs='*',default=[])
    p.add_argument('--variance-spatial-grids',type=int,nargs='*',default=[],help='strong hybrid: common prepared NUFFT mean plus a Cholesky-built variance grid')
    p.add_argument('--query-domain',choices=['surface','mixed'],default='surface')
    p.add_argument('--spatial-domain',choices=['object','periodic'],default='object')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if any(m%a.query_batch for m in a.queries):p.error('query counts must be multiples of query-batch')
    if a.replay_batches and a.replay_queries%a.query_batch:p.error('replay queries must be a multiple of query-batch')
    a.output.mkdir(parents=True,exist_ok=False)
    data=np.load(ROOT/f'reproduction/results/matched/{a.case}/input.npz')
    ids=np.random.default_rng(92).permutation(len(data['points']))[:a.n]
    x,y=data['points'][ids],data['normals'][ids]; n=len(x)
    xp=cp.asarray(x); yp=cp.asarray(y,dtype='d')
    r={'complete':False,'args':{**vars(a),'output':str(a.output)},'n':n,'seconds':{},'measurements':[],
       'query_output':'mean, variance, marginal inside probability and mean ambiguity' if a.probability else 'mean and variance',
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
        (variance_coeff,record)=stage('variance_series_prepare',lambda:prepare_variance_series(
            x,chol,op,a.factor_batch,method=a.moment_method,working_dtype=a.moment_dtype,
            source_dtype=a.source_dtype,paired_sources=a.paired_sources,inverse_batch=a.inverse_batch))
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
        query_count=max(max(a.queries),a.replay_queries if a.replay_batches else 0,a.spatial_screen_queries)
        qhost,_=uniform_sample_mesh(data['vertices'],data['faces'],query_count,np.random.default_rng(716))
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
        if a.variance_spatial_grids or a.sdd_shared_mean or a.shared_mean:
            mean_cache32=stage('prepared_mean_float32',lambda:CachedFourierEvaluator(coeff32,eps=1e-6,oversampling=2.,batch=1))
            def common_mean(q):return mean_cache32.evaluate(q)[0]-offset32
        def reference32(q,mean=True):
            ms=[];vs=[]
            for start in range(0,len(q),a.dense_batch):
                c=table.rows(q[start:start+a.dense_batch],xp).astype('f')
                ms.append((c*alpha32[None]).sum((1,2))-offset32 if mean and not a.shared_mean else cp.zeros(len(c),dtype='f'))
                rhs=cp.ascontiguousarray(c.transpose(1,0,2).reshape(n,-1))
                z=cla.solve_triangular(chol32,rhs,lower=True)
                vs.append(prior-(z*z).sum(0).reshape(-1,3).sum(1))
            return (common_mean(q) if mean and a.shared_mean else cp.concatenate(ms)),cp.concatenate(vs)
        r['float32_pilot_quality']=quality(exact,reference32(pilot),prior)
        if a.fast_reference and not r['float32_pilot_quality']['pass']:raise RuntimeError('float32 reference pilot failed')
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
                if not (a.sdd_shared_mean or a.shared_mean):
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
                        ms.append(common_mean(q[start:start+a.query_batch]) if a.sdd_shared_mean or a.shared_mean else (c*sa[None]).sum((1,2))-so)
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
            screen_q=cp.asarray(qhost[:min(a.spatial_screen_queries,len(qhost))])
            screen_ref=reference(screen_q,True);r['spatial_grid_screens']=[]
            for size in a.spatial_grids:
                # Only accuracy is screened using the candidate. Construction
                # timing for the baseline below uses direct Cholesky evaluation.
                grid,seconds=timed(lambda:SpatialMomentGrid.prepare(cached_grid,size,lower,upper,periodic=a.spatial_domain=='periodic',order=a.spatial_order))
                test=quality(screen_ref,grid.evaluate(screen_q),prior)
                r['spatial_grid_screens'].append({'size':size,'quality':test,'screen_seconds':seconds})
                save();print('spatial screen',size,test,flush=True)
                if test['pass']:
                    baseline=reference32 if r['float32_pilot_quality']['pass'] else reference
                    r['spatial_grid_builder']='float32' if baseline is reference32 else 'float64'
                    grid=stage('cholesky_spatial_grid_prepare',lambda:SpatialMomentGrid.prepare(baseline,size,lower,upper,periodic=a.spatial_domain=='periodic',order=a.spatial_order))
                    r['spatial_grid']={'size':size,'order':a.spatial_order,'lower':lower.tolist(),'upper':upper.tolist(),
                                       'cache_bytes':grid.cache_bytes}
                    methods['cholesky_spatial_grid']=grid.evaluate
                    break
                del grid
        if a.variance_spatial_grids:
            from reproduction.spatial_moment_grid import SpatialMomentGrid
            lower=np.full(3,-np.pi) if a.spatial_domain=='periodic' else data['vertices'].min(0)-.15
            upper=np.full(3,np.pi) if a.spatial_domain=='periodic' else data['vertices'].max(0)+.15
            screen_q=cp.asarray(qhost[:min(a.spatial_screen_queries,len(qhost))])
            screen_ref=reference(screen_q,True);screen_mean=common_mean(screen_q)
            r['spatial_variance_grid_screens']=[]
            def variance_builder(q):return reference32(q,mean=False)
            for size in a.variance_spatial_grids:
                screen,dt=timed(lambda:SpatialMomentGrid.prepare(cached_grid,size,lower,upper,periodic=a.spatial_domain=='periodic',order=a.spatial_order))
                check=quality(screen_ref,(screen_mean,screen.evaluate_channel(screen_q,1)),prior)
                r['spatial_variance_grid_screens'].append({'size':size,'quality':check,'seconds':dt})
                save();print('variance grid screen',size,check,flush=True)
                del screen
                if check['pass']:
                    vgrid=stage('cholesky_spatial_variance_prepare',lambda:SpatialMomentGrid.prepare(variance_builder,size,lower,upper,periodic=a.spatial_domain=='periodic',order=a.spatial_order))
                    r['spatial_variance_grid']={'size':size,'order':a.spatial_order,'cache_bytes':vgrid.cache_bytes,'mean_cache_bytes':mean_cache32.cache_bytes,
                        'lower':lower.tolist(),'upper':upper.tolist(),'builder':'direct float32 Cholesky; common prepared NUFFT mean'}
                    def spatial_variance(q):return common_mean(q),vgrid.evaluate_channel(q,1)
                    methods['cholesky_spatial_variance_grid']=spatial_variance
                    break
        if a.fast_reference:methods.pop('cholesky_direct')
        # Every repeat changes geometry; method order rotates to expose warmup.
        def evaluate_method(name,q):
            mu,var=methods[name](q)
            if a.probability:
                from cupyx.scipy.special import ndtr as gpu_ndtr
                probability=gpu_ndtr(mu.astype('d')/cp.sqrt(var.astype('d')))
                return mu,var,probability,cp.minimum(probability,1-probability).mean()
            return mu,var
        for m in a.queries:
            for repeat in range(a.repeats):
                q=cp.asarray(qhost[:m]+np.array([.001*repeat,-.0007*repeat,.0003*repeat],dtype='f'))
                if a.query_domain=='mixed':q=(q+np.pi)%(2*np.pi)-np.pi
                order=[name for name in methods if name!='sdd_direct' or not a.sdd_measure_max or m<=a.sdd_measure_max]
                order=order[repeat%len(order):]+order[:repeat%len(order)]
                values={}; row={'queries':m,'repeat':repeat,'seconds':{},'quality':{}}
                for name in order:
                    values[name],row['seconds'][name]=timed(lambda:evaluate_method(name,q))
                # Dense table calculation is the timed reference; an independent
                # exact cross-kernel panel audits approximation at new locations.
                reference_name='cholesky_direct_float32' if a.fast_reference else 'cholesky_direct'
                ref=values[reference_name][:2]
                for name in order:row['quality'][name]=quality(ref,values[name][:2],prior)
                if a.probability:
                    pref=values[reference_name][2]
                    row['probability']={name:{'rms_error':float(cp.sqrt(cp.mean((v[2]-pref)**2))),
                                              'max_error':float(cp.max(cp.abs(v[2]-pref))),
                                              'expected_occupied_fraction':float(v[2].mean()),
                                              'mean_ambiguity':float(v[3])}
                                        for name,v in values.items()}
                heldout_ids=np.linspace(0,m-1,min(m,a.query_batch),dtype=int)
                precise=reference(q[heldout_ids],True)
                row['heldout_exact_quality']={name:quality(precise,(v[0][heldout_ids],v[1][heldout_ids]),prior) for name,v in values.items()}
                r['measurements'].append(row);save();print('measurement',m,repeat,row['seconds'],{k:v['pass'] for k,v in row['heldout_exact_quality'].items()},flush=True)
        if a.replay_batches:
            selected=[name for name in ('cholesky_direct_float32','cholesky_nufft_moment_grid','cholesky_spatial_grid','cholesky_spatial_variance_grid','sdd_direct') if name in methods]
            r['replay']={'queries_per_batch':a.replay_queries,'requested_batches':a.replay_batches,
                'sdd_query_budget_seconds':a.replay_budget,'budget_checked':'between native SDD query chunks; final chunk can exceed budget',
                'geometry':'candidate translations: [0.06 cos(theta),0.06 sin(theta),0.03 sin(2theta)]; theta=2pi*batch/batches',
                'variance_ablation':[],
                'methods':{name:{'query_seconds':0.,'completed_batches':0,'completed_queries':0,'batches':[]} for name in selected}}
            for repeat in range(a.replay_batches):
                theta=2*np.pi*repeat/a.replay_batches
                q=cp.asarray(qhost[:a.replay_queries]+np.array([.06*np.cos(theta),.06*np.sin(theta),.03*np.sin(2*theta)],dtype='f'))
                if a.query_domain=='mixed':q=(q+np.pi)%(2*np.pi)-np.pi
                active=[name for name in selected if name!='sdd_direct' or r['replay']['methods'][name]['query_seconds']<a.replay_budget]
                if not active:break
                active=active[repeat%len(active):]+active[:repeat%len(active)]
                ids=np.linspace(0,len(q)-1,min(128,len(q)),dtype=int)
                precise=reference(q[ids],True)
                batch_values={}
                for name in active:
                    item=r['replay']['methods'][name]
                    if name=='sdd_direct':
                        parts=[];dt=0.
                        for start in range(0,len(q),a.query_batch):
                            part,elapsed=timed(lambda:evaluate_method(name,q[start:start+a.query_batch]))
                            parts.append(part);dt+=elapsed
                            if item['query_seconds']+dt>=a.replay_budget:break
                        values=tuple(cp.concatenate([part[j] for part in parts]) for j in range(3 if a.probability else 2))
                        if a.probability:values+= (cp.minimum(values[2],1-values[2]).mean(),)
                    else:values,dt=timed(lambda:evaluate_method(name,q))
                    count=len(values[0]);batch_values[name]=values
                    if count==len(q):check=quality(precise,(values[0][ids],values[1][ids]),prior)
                    else:
                        prefix_ids=np.linspace(0,count-1,min(128,count),dtype=int)
                        prefix_ref=reference(q[prefix_ids],True)
                        check=quality(prefix_ref,(values[0][prefix_ids],values[1][prefix_ids]),prior)
                    item['query_seconds']+=dt;item['completed_batches']+=int(count==len(q));item['completed_queries']+=count
                    item['batches'].append({'repeat':repeat,'queries':count,'seconds':dt,'heldout_exact_quality':check,
                        'probability_mean':float(values[2].mean()) if a.probability else None,
                        'mean_ambiguity':float(values[3]) if a.probability else None})
                    save();print('replay',repeat,name,dt,check['pass'],flush=True)
                ref_values=batch_values['cholesky_direct_float32']
                if a.probability:
                    from cupyx.scipy.special import ndtr as gpu_ndtr
                    ablation={'repeat':repeat}
                    for label,value in [('prior',prior),('constant_posterior',float(ref_values[1].mean()))]:
                        p0=gpu_ndtr(ref_values[0].astype('d')/cp.sqrt(cp.asarray(value)))
                        delta=p0-ref_values[2]
                        ablation[label]={'probability_rms_error':float(cp.sqrt(cp.mean(delta*delta))),
                            'probability_max_error':float(cp.max(cp.abs(delta))),
                            'mean_ambiguity':float(cp.minimum(p0,1-p0).mean())}
                    r['replay']['variance_ablation'].append(ablation)
                for name,values in batch_values.items():
                    item=r['replay']['methods'][name]['batches'][-1]
                    count=len(values[0])
                    item['full_quality']=quality((ref_values[0][:count],ref_values[1][:count]),values[:2],prior)
                    if a.probability:
                        delta=values[2]-ref_values[2][:count]
                        item['probability_rms_error']=float(cp.sqrt(cp.mean(delta*delta)))
                        item['probability_max_error']=float(cp.max(cp.abs(delta)))
                save()
            for item in r['replay']['methods'].values():
                item['complete']=item['completed_batches']==a.replay_batches
                item['quality_pass']=bool(item['batches']) and all(v['heldout_exact_quality']['pass'] and v['full_quality']['pass'] for v in item['batches'])
                if a.probability and item['complete']:
                    ref_scores=np.array([v['mean_ambiguity'] for v in r['replay']['methods']['cholesky_direct_float32']['batches']])
                    scores=np.array([v['mean_ambiguity'] for v in item['batches']])
                    selected=int(np.argmax(scores));gap=float(ref_scores.max()-ref_scores[selected])
                    item['ambiguity_ranking']={'selected_batch':selected,'reference_best_batch':int(np.argmax(ref_scores)),
                        'max_score_absolute_error':float(np.max(np.abs(scores-ref_scores))),
                        'reference_score_range':float(np.ptp(ref_scores)),
                        'regret':gap,'range_normalized_regret':gap/max(float(np.ptp(ref_scores)),1e-300)}
        r['complete']=True
    except Exception as exc:r['failure']=f'{type(exc).__name__}: {exc}';raise
    finally:r['gpu_audit']=monitor.stop();save()


if __name__=='__main__':main()
