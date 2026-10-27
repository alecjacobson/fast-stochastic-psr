"""Host-backed residual products on retained control and application panels."""
import argparse,hashlib,json,resource
from pathlib import Path
import cupy as cp
import numpy as np
from geospsr.spectral import FourierPoisson
from geospsr.training_spectrum import FourierCovariance
from reproduction.query_posterior import AnalyticConditioner,FixedQueryPosterior,principal_root
from reproduction.query_error import QueryResidualAudit,HostQueryResidualAudit
from reproduction.analytic_posterior import timed
from reproduction.common_rhs_integrity import array_digest
from reproduction.gpu_audit import GpuProcessAudit


def main():
    p=argparse.ArgumentParser();p.add_argument('--case',choices=['B','million'],required=True)
    p.add_argument('--panel',choices=['moments','applications'],required=True)
    p.add_argument('--storage',choices=['host','gpu'],default='host')
    p.add_argument('--steps',type=int,choices=[0,1,2],default=1)
    p.add_argument('--cutoff',type=int,choices=[50,100],default=100)
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    label=f'{a.case}_{a.panel}_{a.storage}_s{a.steps}'+(f'_c{a.cutoff}' if a.cutoff!=100 else '')
    out=root/'reproduction/results/query_host'/label
    out.mkdir(parents=True,exist_ok=True)
    if (out/'results.json').exists():raise FileExistsError('preserve previous results before rerunning')
    r={'complete':False,'args':vars(a),'seconds':{},'finite_solves':[],
       'gates':{'covariance_relative_l2':1e-7,'mean_error_over_prior_std':1e-7,
                'marginal_variance_relative_max':1e-7,'paired_event_disagreement_rate':1e-4},
       'scope':'storage and conditioning-effort pilot on fixed panels; not an upstream speedup or geometry-convergence result',
       'source_sha256':{f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in
            ['reproduction/query_error.py','reproduction/query_posterior.py','reproduction/benchmark_query_host.py']}}
    def save():(out/'results.json').write_text(json.dumps(r,indent=2,default=lambda v:v.item())+'\n')
    gpu=GpuProcessAudit(cp.cuda.Device().pci_bus_id).start()
    try:
        if a.case=='million' and cp.cuda.runtime.memGetInfo()[0]<40*2**30:raise MemoryError('need40GiB initially free')
        reference=np.load(root/f'reproduction/results/query_posterior/B_{a.panel}/controls.npz');q=reference['queries']
        if a.case=='B':
            data=np.load(root/'reproduction/results/matched/B/input.npz');x,y=data['points'],data['normals']
        else:
            x=np.load(root/'reproduction/results/range_compact/scale1m/control.npz')['points']
            y=np.load(root/'reproduction/results/analytic_refinement/million/controls.npz')['normals']
            if a.panel!='moments' or a.steps!=0 or a.cutoff!=100:raise ValueError('million overlap uses moments/steps0/cutoff100; full application validation requires a separate control')
        r.update(n=len(x),m=len(q),hashes={k:array_digest(v) for k,v in [('points',x),('normals',y),('queries',q)]})
        condition,r['seconds']['condition_setup']=timed(lambda:AnalyticConditioner(x,
            solver='analytic_cg' if a.case=='B' else 'defect',fit_cutoff=200,rtol=1e-10,
            max_cache_bytes=(1 if a.case=='B' else 32)*2**30,max_entries=200_000_000))
        alpha,r['seconds']['mean_solve']=timed(lambda:condition.solve(cp.asarray(y,dtype='d')))
        r['mean_solve']=condition.records[-1]
        op,r['seconds']['scalar_setup']=timed(lambda:FourierPoisson(50,dtype='float64',eps=1e-12))
        coeff,r['seconds']['mean_aggregate']=timed(lambda:op.aggregate(x,alpha))
        offset,r['seconds']['mean_center']=timed(lambda:op.evaluate(x,coeff).mean())
        condition.kernel=condition.spectrum=None;condition.rhs_count=None;cp.get_default_memory_pool().free_all_blocks()
        condition.spectrum,r['seconds']['finite_setup']=timed(lambda:FourierCovariance(a.cutoff,upsampfac=1.25))
        def approximate(b):
            w,info=condition.finite_solve(b);record={'initial':info,'corrections':[]}
            for _ in range(a.steps):
                residual=b-condition.action(w)
                delta,info=condition.finite_solve(residual);w+=delta;record['corrections'].append(info)
            r['finite_solves'].append(record);return w
        def make_audit():
            if a.storage=='host':return HostQueryResidualAudit(condition.action,approximate,.02,len(x),len(q),row_chunk=4096)
            return QueryResidualAudit(condition.action,approximate,.02)
        audit,r['seconds']['audit_setup']=timed(make_audit)
        batch=4 if a.case=='B' else 1
        model,r['seconds']['query_posterior']=timed(lambda:FixedQueryPosterior(x,q,op,audit,coeff,offset,batch=batch,residual_audit=audit,
            progress=lambda done,total:print(a.case,a.panel,a.storage,done,total,flush=True) if done%32==0 or done==total else None))
        r['seconds']['prepare_total']=sum(r['seconds'].values());r['query_record']=model.record;save()
        condition.kernel=condition.spectrum=None;cp.get_default_memory_pool().free_all_blocks()
        product_controls={}
        if a.storage=='host':
            # Validation only: compare chunked products to unchunked GPU
            # products of the EXACT SAME retained vectors, avoiding differences
            # caused by rerunning float32 NUFFT conditioning in another process.
            def product_reference():
                if 2*len(x)*len(q)*8>2*2**30:raise MemoryError('independent full-component product check capped at2GiB')
                cross=cp.zeros((len(q),len(q)),dtype='d');gram=cp.zeros_like(cross)
                for d in range(3):
                    wd=cp.asarray(audit.host_w[d]);rd=cp.asarray(audit.host_r[d])
                    cross+=wd@rd.T;gram+=rd@rd.T
                    del wd,rd
                return (cross+cross.T)*.5,(gram+gram.T)*(.5/.02)
            (shift_ref,upper_ref),r['seconds']['validation_same_vector_products']=timed(product_reference)
            r['same_vector_products']={k:float(cp.linalg.norm(v-ref)/cp.maximum(cp.linalg.norm(ref),1e-300)) for k,v,ref in
                [('shift_relative_l2',audit.shift,shift_ref),('upper_relative_l2',audit.upper,upper_ref)]}
            if max(r['same_vector_products'].values())>1e-11:raise ValueError('host chunked products disagree with same-vector GPU products')
            product_controls={'same_vector_reference_shift':shift_ref.get(),'same_vector_reference_upper':upper_ref.get()}
        audit.clear_vectors()
        if a.case=='B':target=reference['reference_covariance'];mu=reference['reference_mean']
        else:
            old=np.load(root/'reproduction/results/query_error/million_c100/controls.npz')
            target,mu=old['covariance'],old['mean']
            r['reference_scope']='previous GPU-resident approximate covariance at identical bandwidth/steps, not an independent exact million-point posterior'
            r['upper_relative_l2_difference']=float(cp.linalg.norm(audit.upper-cp.asarray(old['upper']))/cp.linalg.norm(cp.asarray(old['upper'])))
            r['shift_relative_l2_difference']=float(cp.linalg.norm(audit.shift-cp.asarray(old['shift']))/cp.linalg.norm(cp.asarray(old['shift'])))
        cov=cp.asarray(target);mean=cp.asarray(mu)
        r['covariance_relative_l2']=float(cp.linalg.norm(model.covariance-cov)/cp.linalg.norm(cov))
        r['mean_error_over_prior_std']=float(cp.max(cp.abs(model.mean-mean)/cp.sqrt(cp.diag(model.prior))))
        r['marginal_variance_relative_max']=float(cp.max(cp.abs(cp.diag(model.covariance-cov))/cp.diag(cov)))
        r['quality_pass']=all(r[k]<=r['gates'][k] for k in ['covariance_relative_l2','mean_error_over_prior_std','marginal_variance_relative_max'])
        prior_eig=cp.linalg.eigvalsh((model.prior+model.prior.T)*.5).get()[::-1]
        r['prior_spectral_tail']={str(k):float(prior_eig[k]) for k in [16,32,64,128,256,512,768] if k<len(q)}
        r['prior_psd_roundoff_budget']=float(100*np.finfo(float).eps*len(q)*np.abs(prior_eig).max())
        np.savez_compressed(out/'controls.npz',queries=q,mean=model.mean.get(),reference_mean=mu,covariance=model.covariance.get(),
            reference_covariance=target,prior=model.prior.get(),prior_eigenvalues=prior_eig,upper=audit.upper.get(),shift=audit.shift.get(),
            cross_norms=audit.cross_norms,residual_norms=audit.residual_norms,**product_controls)
        if a.panel=='applications':
            (_,factor,psd),r['seconds']['validation_reference_root']=timed(lambda:principal_root(cov));r['reference_psd']=psd
            count=32768
            white,r['seconds']['draw_rng']=timed(lambda:cp.random.RandomState(4804).standard_normal((count,len(q)),dtype='d'))
            def summarize(values):
                return {'fields':values[:,:32].get(),
                    'rays':cp.cumprod(values[:,:768].reshape(-1,3,256)>0,axis=2).astype('uint8').get(),
                    'collision':(values[:,768:]<=0).any(1).get()}
            values,r['seconds']['draw_fields']=timed(lambda:model.draw(count,white=white))
            output,r['seconds']['draw_events']=timed(lambda:summarize(values))
            refvalues=white@factor.T+mean;refoutput=summarize(refvalues)
            r['paired_field_relative_l2_all_queries']=float(cp.linalg.norm(values-refvalues)/cp.linalg.norm(refvalues))
            r['paired_field_relative_l2_first32']=float(np.linalg.norm(output['fields']-refoutput['fields'])/np.linalg.norm(refoutput['fields']))
            r['samples']=count;r['events']={}
            for k in ['rays','collision']:
                count_diff=int(np.count_nonzero(output[k]!=refoutput[k]));rate=count_diff/output[k].size
                r['events'][k]={'disagreements':count_diff,'rate':rate}
                r['quality_pass']=r['quality_pass'] and rate<r['gates']['paired_event_disagreement_rate']
            np.savez_compressed(out/'outputs.npz',**{'query_'+k:v for k,v in output.items()},**{'reference_'+k:v for k,v in refoutput.items()})
        r['complete']=True;print('complete',r['quality_pass'],r['seconds'],r['covariance_relative_l2'],flush=True)
    except Exception as exc:r['failure']=f'{type(exc).__name__}: {exc}';raise
    finally:
        r['host_process_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        r['gpu_process_audit']=gpu.stop();save()


if __name__=='__main__':main()
