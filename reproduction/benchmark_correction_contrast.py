"""Fixed contrast-quality study with streamed inverse-factor correction draws."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import cupy as cp
import cupyx.scipy.linalg as cla
import numpy as np
from scipy.special import ndtr
from geospsr.spectral import FourierPoisson
from geospsr.spectral_tail import analytic_covariance_rows64
from reproduction.posterior import Posterior, sync
from reproduction.correction_sampling import DenseCorrectionSampler, CorrectionMoments
from reproduction.gpu_audit import GpuProcessAudit


def deterministic_variance(op, x, chol, q, prior, batch=64):
    """Strong streamed cross-factor baseline: no full query-by-source array."""
    result=[]
    for start in range(0,len(q),batch):
        diff=cp.asarray(q[start:start+batch],dtype=op.dtype)[:,None]-cp.asarray(x,dtype=op.dtype)[None]
        cross=op.cross_values(diff.reshape(-1,3)).reshape(-1,len(x),3)
        transformed=cla.solve_triangular(chol,cross.transpose(1,0,2).reshape(len(x),-1),lower=True)
        reduction=(transformed*transformed).sum(0).reshape(-1,3).sum(1)
        result.append(cp.asnumpy(prior-reduction))
    return np.concatenate(result)


def metrics(value,reference,mean,prior):
    error=value-reference
    probability=ndtr(-mean/np.sqrt(np.maximum(value,np.finfo(float).tiny)))
    truth=ndtr(-mean/np.sqrt(reference))
    return {'variance_relative_l2':float(np.linalg.norm(error)/np.linalg.norm(reference)),
            'contrast_relative_l2':float(np.linalg.norm(error)/np.linalg.norm(reference-reference.mean())),
            'reduction_relative_l2':float(np.linalg.norm(error)/np.linalg.norm(prior-reference)),
            'probability_rms':float(np.sqrt(np.mean((probability-truth)**2))),
            'probability_max':float(np.max(np.abs(probability-truth))),
            'nonpositive_count':int((value<=0).sum())}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--case',choices=['A','B'],required=True)
    parser.add_argument('--output',type=Path,required=True); args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    record={'complete':False,'case':args.case,'scope':'direct analytic-vector/scalar50 marginal variance, not coherent posterior events',
            'seconds':{},'runs':[],'checkpoints':[128,512,2048,8192],'seeds':[401,503,607],
            'gates':{'contrast_relative_l2':.01,'probability_rms':.0005,'probability_max':.005},
            'device':cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
            'packages':{name:importlib.metadata.version(name) for name in ['numpy','cupy-cuda12x','cufinufft','jax']}}
    audit=GpuProcessAudit(cp.cuda.Device().pci_bus_id).start(); wall=time.perf_counter()
    def timed(name,fn):
        sync(); start=time.perf_counter(); value=fn(); sync()
        record['seconds'][name]=time.perf_counter()-start
        print(name,record['seconds'][name],flush=True)
        return value
    try:
        root=Path(__file__).resolve().parents[1]
        data=timed('load',lambda:np.load(root/f'reproduction/results/matched/{args.case}/input.npz'))
        x,y,q=data['points'],data['normals'],data['query']
        record.update(N=len(x),queries=len(q),batch=8)
        record['hashes']={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in [('points',x),('normals',y),('queries',q)]}
        model=timed('fit',lambda:Posterior(x,y,upsampfac=1.25))
        mean=timed('mean_queries',lambda:cp.asnumpy(model.mean(q)))
        reference_op=timed('reference_operator',lambda:FourierPoisson(50,dtype='float64',eps=1e-12))
        norm=(reference_op.ns**2).sum(0)
        prior=float(cp.sum(cp.where(norm>0,reference_op.rho/cp.where(norm>0,norm,1),0)))
        matrix=timed('reference_matrix',lambda:analytic_covariance_rows64(x,x)+.02*cp.eye(len(x),dtype='d'))
        chol=timed('reference_factor',lambda:cp.linalg.cholesky(matrix))
        reference=timed('reference_variance',lambda:deterministic_variance(reference_op,x,chol,q,prior))
        assert np.all(reference>0)
        del matrix,chol,reference_op,norm
        cp.get_default_memory_pool().free_all_blocks()
        baseline=[]
        for i in range(3):
            direct=timed(f'deterministic_{i}',lambda:deterministic_variance(model.op,x,model.L,q,prior))
            baseline.append(record['seconds'][f'deterministic_{i}'])
        record['deterministic_quality']=metrics(direct,reference,mean,prior)
        record['deterministic_median_seconds']=float(np.median(baseline))
        np.savez_compressed(args.output/'reference.npz',query=q,mean=mean,variance=reference,
                            prior_variance=prior,deterministic=direct)
        sampler=DenseCorrectionSampler(x,model.L,model.op)
        query_plan=timed('stochastic_query_plan',lambda:model.op.plan(2,q,8))
        for seed in record['seeds']:
            state=CorrectionMoments(); cumulative={'generation':0.,'queries':0.,'reduction':0.}
            stream=iter(sampler.draw_batches(8192,seed=seed,batch=8)); checkpoints=[]; saved={}
            start_wall=time.perf_counter()
            for index in range(1024):
                sync(); t=time.perf_counter(); block=next(stream); sync()
                cumulative['generation']+=time.perf_counter()-t
                t=time.perf_counter(); values=cp.asnumpy(query_plan.execute(block['coefficients']).real); sync()
                cumulative['queries']+=time.perf_counter()-t
                t=time.perf_counter(); state.update(values); cumulative['reduction']+=time.perf_counter()-t
                if state.count in record['checkpoints']:
                    result=state.summary(prior,looks=4); m=metrics(result['variance'],reference,mean,prior)
                    good=all(m[k]<=v for k,v in record['gates'].items()) and m['nonpositive_count']==0
                    checkpoints.append({'count':state.count,'metrics':m,'quality_pass':good,
                        'cumulative_seconds':dict(cumulative),'elapsed_seconds':time.perf_counter()-start_wall,
                        'queries_covered':int(np.sum((result['lower']<=reference)&(reference<=result['upper']))),
                        'interval_width_l2_over_contrast':float(np.linalg.norm(result['upper']-result['lower'])/
                                                                     np.linalg.norm(reference-reference.mean()))})
                    saved[f'variance_{state.count}']=result['variance']
                    print('checkpoint',seed,state.count,m,flush=True)
            stream.close()
            np.savez_compressed(args.output/f'seed_{seed}.npz',**saved)
            record['runs'].append({'seed':seed,'checkpoints':checkpoints})
        record['complete']=True
    except Exception as exc:
        record['failure']=f'{type(exc).__name__}: {exc}';raise
    finally:
        sync(); record['work_wall_seconds']=time.perf_counter()-wall
        record['gpu_process_audit']=audit.stop()
        (args.output/'results.json').write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':main()
