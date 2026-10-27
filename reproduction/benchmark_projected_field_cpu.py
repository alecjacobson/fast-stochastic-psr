"""CPU overlap control for coherent functions; never a GPU timing claim."""
import argparse,hashlib,json,resource,sys,time
from pathlib import Path
import numpy as np
import finufft
from scipy.linalg import cho_factor,cho_solve
from reproduction.projected_field_cpu import prepare
from reproduction.query_projection import assemble_projected
from reproduction.common_rhs_integrity import array_digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--rank',type=int);p.add_argument('--samples',type=int,default=64)
    args=p.parse_args();root=Path(__file__).resolve().parents[1]
    out=root/'reproduction/results/projected_field'/f'A_r{args.rank if args.rank is not None else "auto"}_cpu'
    out.mkdir(parents=True,exist_ok=False)
    files=['geospsr/projected_field.py','reproduction/projected_field_cpu.py','reproduction/benchmark_projected_field_cpu.py']
    r={'complete':False,'args':vars(args),'scope':'CPU dense/FINUFFT overlap control; shared host with GPU pilot; no GPU timing/speedup claim',
       'quality_gate':1e-7,'numpy_version':np.__version__,'finufft_version':finufft.__version__,'seconds':{},
       'source_sha256':{f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in files}}
    def save():(out/'results.json').write_text(json.dumps(r,indent=2,default=lambda v:v.item())+'\n')
    def timed(name,fn):
        start=time.perf_counter();value=fn();r['seconds'][name]=time.perf_counter()-start;return value
    try:
        data=np.load(root/'reproduction/results/matched/A/input.npz');x,y=data['points'],data['normals']
        old=np.load(root/'reproduction/results/query_posterior/A_applications/controls.npz');q=old['queries'].astype(float)
        ids=np.arange(0,len(q),16);near=(q[ids]+q[ids+1])*.5
        far=np.random.default_rng(5409).uniform(x.min(0),x.max(0),(64,3));all_q=np.vstack([q,near,far])
        r['hashes']={k:array_digest(v) for k,v in [('points',x),('normals',y),('queries',q),('heldout_near',near),('heldout_bbox',far)]}
        a=prepare(x,q,y,rank=args.rank,progress=lambda done,total:print('CPU basis',done,total,flush=True) if done%32==0 or done==total else None)
        op,sampler=a['operator'],a['sampler'];r['preparation']=a['record'];save()
        prior=timed('validation_all_prior',lambda:op.covariance(all_q));prior=(prior+prior.T)*.5
        l=timed('validation_cross_action',lambda:op.evaluate(all_q,a['cross_coefficients']).T)
        b=timed('validation_lift',lambda:op.evaluate(all_q,a['lift_coefficients']).T)
        actual_g=sampler.scalar_gram+sampler.remainder_root@sampler.remainder_root.T
        covariance=prior-l@b.T-b@l.T+b@actual_g@b.T;covariance=(covariance+covariance.T)*.5
        def dense_reference():
            delta=(all_q[:,None]-x[None].astype(float)).reshape(-1,3)
            cross=op.evaluate(delta,op.cross).reshape(3,len(all_q),len(x))
            matrix=op.analytic_training(x,.02);chol=cho_factor(matrix,lower=True)
            target=prior-sum(c@cho_solve(chol,c.T) for c in cross)
            mean=sum(c@cho_solve(chol,y.astype(float))[:,d] for d,c in enumerate(cross))-a['offset']
            return (target+target.T)*.5,mean
        target,mean=timed('validation_dense',dense_reference)
        actual_mean=op.evaluate(all_q,a['mean_coefficients'][None,:])[0]-a['offset']
        r['mean_error_over_prior_std']=float(np.max(np.abs(actual_mean-mean)/np.sqrt(np.diag(prior))))
        r['archived_gpu_covariance_relative_l2']=float(np.linalg.norm(target[:len(q),:len(q)]-old['reference_covariance'])/np.linalg.norm(old['reference_covariance']))
        r['panels']={}
        for name,idx in [('reference',np.arange(len(q))),('near',np.arange(len(q),len(q)+64)),('bbox',np.arange(len(q)+64,len(all_q))),('all_joint',np.arange(len(all_q)))]:
            s=target[np.ix_(idx,idx)];e=covariance[np.ix_(idx,idx)]-s
            metric={'covariance_relative_l2':float(np.linalg.norm(e)/np.linalg.norm(s)),
                'maximum_marginal_relative_error':float(np.max(np.abs(np.diag(e))/np.diag(s)))}
            metric['quality_pass']=max(metric.values())<=r['quality_gate'];r['panels'][name]=metric
        root_lam=np.sqrt(a['eigenvalues']);cross_l=l[:len(q)]*root_lam
        wr=sum((w*root_lam[:,None])@(res*root_lam[:,None]).T for w,res in zip(a['weights'],a['residual']))
        original_upper=root_lam[:,None]*a['solve_upper']*root_lam[None,:]
        _,_,budget=assemble_projected(a['prior'],a['basis'],cross_l,wr,a['tail'],original_upper)
        r['reference_solve_truncation_budget']=budget
        r['lift_relative_error']=float(np.linalg.norm(b[:len(q)]-a['basis']*root_lam)/np.linalg.norm(b[:len(q)]))
        innovation=prior-b@b.T;upper=2*innovation+2*b@a['solve_upper']@b.T
        r['upper_minus_error_min_eigenvalue']=float(np.linalg.eigvalsh((upper+upper.T)*.5-(covariance-target)).min())
        values=[];first=None;draw_seconds=eval_seconds=0.
        for start in range(0,args.samples,4):
            count=min(4,args.samples-start);clock=time.perf_counter();coeff=sampler.draw(count,seed=5410+start)
            draw_seconds+=time.perf_counter()-clock;clock=time.perf_counter()
            values.append(op.evaluate(all_q,coeff+a['mean_coefficients'][None,:])-a['offset'])
            eval_seconds+=time.perf_counter()-clock
            if first is None:first=coeff[:1].copy()
        values=np.concatenate(values);r['seconds'].update(draw_coefficients=draw_seconds,evaluate_fields=eval_seconds)
        reused=op.evaluate(all_q[len(q):],first+a['mean_coefficients'][None,:])[0]-a['offset']
        r['heldout_reuse_max_absolute_difference']=float(np.max(np.abs(reused-values[0,len(q):])))
        subset=slice(0,32);s=covariance[subset,subset];mu=actual_mean[subset]
        se=np.sqrt(np.diag(s)/len(values));cse=np.sqrt((s*s+np.diag(s)[:,None]*np.diag(s)[None,:])/(len(values)-1))
        r['moment_sanity']={'mean_max_standardized_error':float(np.max(np.abs(values[:,subset].mean(0)-mu)/se)),
            'covariance_max_standardized_error':float(np.max(np.abs(np.cov(values[:,subset].T)-s)/cse))}
        r['sampling_qualification']='small draw count checks only; no accurate ray/collision probability or view-decision claim'
        r['reference_quality_pass']=r['panels']['reference']['quality_pass'] and r['archived_gpu_covariance_relative_l2']<=1e-7 and r['mean_error_over_prior_std']<=1e-7
        r['all_queries_quality_pass']=all(v['quality_pass'] for v in r['panels'].values()) and r['reference_quality_pass']
        np.savez_compressed(out/'controls.npz',queries=all_q,prior=prior,covariance=covariance,reference_covariance=target,
            mean=actual_mean,reference_mean=mean,cross_action=l,lift=b,projected_covariance=a['projected_covariance'],
            scalar_gram=sampler.scalar_gram,remainder_root=sampler.remainder_root,solve_upper=a['solve_upper'],
            basis=a['basis'],eigenvalues=a['eigenvalues'],weights=a['weights'],residual=a['residual'],fields=values)
        np.savez_compressed(out/'first_coefficients.npz',coefficients=first,mean_coefficients=a['mean_coefficients'],offset=a['offset'])
        r['complete']=True;print('complete',r['panels'],r['moment_sanity'],flush=True)
    except Exception as exc:r['failure']=f'{type(exc).__name__}: {exc}';raise
    finally:
        r['gpu_modules_imported']=[m for m in ['cupy','cufinufft','jax'] if m in sys.modules]
        r['peak_host_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;save()


if __name__=='__main__':main()
