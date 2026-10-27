"""Larger distinct-input mean fit, independent residuals, and budgeted SDD."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import cupy as cp
import numpy as np
from geospsr.spectral import FourierPoisson
from geospsr.spectral_tail import analytic_covariance_rows64
from reproduction.common_rhs_integrity import array_digest
from reproduction.gpu_audit import GpuProcessAudit
from reproduction.posterior import sync
from reproduction.scale_training_bandwidth import residual_statistics


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--method',choices=['half','sdd'],required=True)
    p.add_argument('--cutoff',type=int,default=200)
    p.add_argument('--time-budget',type=float,default=600.)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.cutoff<50 or a.time_budget<=0:p.error('require cutoff >=50 and positive budget')
    a.output.mkdir(parents=True,exist_ok=True)
    r={'complete':False,'args':{**vars(a),'source':str(a.source),'output':str(a.output)},
       'scope':'mean fit only, not covariance or application quality or upstream infeasibility',
       'seconds':{},'history':[],'analytic_residual_gate':.001,
       'device':cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}
    audit=GpuProcessAudit(cp.cuda.Device().pci_bus_id).start();wall=time.perf_counter()
    def save():
        (a.output/'results.json').write_text(json.dumps(r,indent=2)+'\n')
    def timed(name,fn):
        sync();t=time.perf_counter();v=fn();sync()
        r['seconds'][name]=time.perf_counter()-t;print(name,r['seconds'][name],flush=True);save()
        return v
    try:
        def source():
            manifest=json.loads((a.source/'manifest.json').read_text())
            x=np.load(a.source/'points.npy');y=np.load(a.source/'normals.npy')
            if not manifest['complete'] or manifest['distinct_observed_positions']!=len(x):
                raise ValueError('source distinctness not validated')
            for k,v in [('points',x),('normals',y)]:
                if array_digest(v)!=manifest['arrays'][k]['payload_sha256']:raise ValueError('source hash mismatch')
            return x,y,manifest
        x,y,manifest=timed('load_and_verify',source);r['N']=len(x)
        r['source_manifest']=manifest
        root=Path(__file__).resolve().parents[1]
        q=np.load(root/'reproduction/results/matched/A/input.npz')['query']
        r['query_sha256']=hashlib.sha256(q.tobytes()).hexdigest()
        stop=np.random.default_rng(73).choice(len(x),128,replace=False)
        # Select from a separate seeded permutation and exclude every stop row.
        held=np.random.default_rng(917).choice(len(x),1200,replace=False)
        held=held[~np.isin(held,stop)][:1024]
        if len(held)!=1024:raise RuntimeError('insufficient disjoint validation rows')
        r['stopping_rows']=stop.tolist();r['holdout_rows']=held.tolist()
        begin=time.perf_counter()
        if a.method=='half':
            from reproduction.half_posterior import HalfCubePosterior
            model=timed('fit',lambda:HalfCubePosterior(x,y,training_cutoff=a.cutoff))
            alpha=model.alpha;op=model.op;coeff=model.mean_coeff;offset=model.offset
            r['finite_solve']=model.mean_solve;r['status']='finite_solver_converged'
        else:
            from geospsr.iterative import StochasticDualDescent
            from geospsr.kernels import Matern32Kernel,ProductKernel
            from reproduction.matched_teaser import own_jax_array
            k=Matern32Kernel(.01,.1);kernel=ProductKernel(k,k,k)
            r['sdd_parameters']={'batch':512,'eta':5.,'lr':5*512/len(x),'seed':0,
                'momentum':.9,'polyak':.01,'chunk':128,'iteration_budget':1000000}
            solver=timed('sdd_construct',lambda:StochasticDualDescent(x,y,kernel,batch=512,lr=5*512/len(x)))
            timed('sdd_compile',solver.compile)
            rows=timed('stopping_rows_setup',lambda:analytic_covariance_rows64(x[stop],x))
            target=cp.asarray(y[stop],dtype='d');elapsed=0.;checks=0.;r['status']='iteration_budget'
            while solver.iterations<1000000:
                t=time.perf_counter();value=solver.advance(min(128,1000000-solver.iterations))
                elapsed+=time.perf_counter()-t
                t=time.perf_counter();alpha=own_jax_array(value)
                error=rows@alpha.astype('d')+.02*alpha[stop]-target
                stats=residual_statistics(error,target);sync();checks+=time.perf_counter()-t
                r['history'].append({'iterations':solver.iterations,'update_seconds':elapsed,'analytic_residual':stats})
                print('sdd',solver.iterations,elapsed,stats['relative_rms'],flush=True);save()
                if not np.isfinite(stats['relative_rms']) or stats['relative_rms']>1e4:
                    r['status']='diverged';break
                if stats['bootstrap_95_percent_interval'][1]<.001:
                    r['status']='stopping_gate_pass';break
                if elapsed>=a.time_budget:
                    r['status']='update_time_budget';break
            r['seconds']['sdd_updates']=elapsed;r['seconds']['sdd_stopping_checks']=checks
            r['iterations']=solver.iterations
            del rows,target
            op=timed('scalar_operator',lambda:FourierPoisson(50))
            coeff=timed('aggregation',lambda:op.aggregate(x,alpha))
            offset=timed('centering',lambda:op.evaluate(x,coeff).mean())
        mean=timed('mean_queries',lambda:cp.asnumpy(op.evaluate(q,coeff)[0]-offset))
        r['fit_through_mean_seconds']=time.perf_counter()-begin
        # Independent double analytic rows are streamed; no 1024-by-N cache.
        def validation():
            residual=[];weights=alpha.astype('d')
            for start in range(0,len(held),16):
                ids=held[start:start+16]
                rows=analytic_covariance_rows64(x[ids],x)
                residual.append(rows@weights+.02*weights[ids]-cp.asarray(y[ids],dtype='d'))
            values=cp.concatenate(residual)
            return values,residual_statistics(values,cp.asarray(y[held],dtype='d'))
        residual,stats=timed('independent_analytic_validation',validation)
        r['holdout_residual']=stats
        r['quality_pass']=stats['bootstrap_95_percent_interval'][1]<.001
        r['quality_qualification']='1024 disjoint random analytic rows and empirical interval, not a full-system bound or covariance guarantee'
        np.savez_compressed(a.output/'fields.npz',query=q,mean=mean,offset=float(offset),
                            residual=cp.asnumpy(residual),target=y[held])
        np.save(a.output/'weights.npy',cp.asnumpy(alpha))
        r['weights_sha256']=array_digest(np.load(a.output/'weights.npy',mmap_mode='r'))
        r['complete']=True
    except Exception as exc:
        r['failure']=f'{type(exc).__name__}: {exc}';raise
    finally:
        r['work_wall_seconds']=time.perf_counter()-wall
        r['gpu_process_audit']=audit.stop();save()


if __name__=='__main__':main()
