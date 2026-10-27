"""Reduced query-basis posterior on the full retained ray/sphere panel."""
import argparse,hashlib,json,resource
from pathlib import Path
import cupy as cp
import numpy as np
from geospsr.spectral import FourierPoisson
from geospsr.training_spectrum import FourierCovariance
from reproduction.query_posterior import AnalyticConditioner,principal_root
from reproduction.query_projection import ProjectedQueryPosterior
from reproduction.analytic_posterior import timed
from reproduction.common_rhs_integrity import array_digest
from reproduction.gpu_audit import GpuProcessAudit


def main():
    p=argparse.ArgumentParser();p.add_argument('--case',choices=['B','million'],default='B')
    p.add_argument('--rank',type=int);p.add_argument('--cutoff',type=int,choices=[50,100,200],default=50)
    p.add_argument('--steps',type=int,choices=range(7),default=2);p.add_argument('--batch',type=int,default=4)
    p.add_argument('--save-vectors',action='store_true')
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    label=f'{a.case}_r{a.rank if a.rank is not None else "auto"}_c{a.cutoff}_s{a.steps}_b{a.batch}'
    out=root/'reproduction/results/query_projection'/label;out.mkdir(parents=True,exist_ok=True)
    if (out/'results.json').exists():raise FileExistsError('preserve previous results before regenerating')
    r={'complete':False,'args':vars(a),'seconds':{},'finite_solves':[],
       'gates':{'covariance_relative_l2':1e-7,'maximum_marginal_relative_error':1e-7,'mean_error_over_prior_std':1e-7,'event_disagreement_rate':1e-4},
       'scope':'fixed1024-query full analytic training/finite scalar50 panel; no upstream speedup or continuous-geometry guarantee',
       'source_sha256':{f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in ['reproduction/query_projection.py','reproduction/query_error.py','reproduction/query_posterior.py','reproduction/benchmark_query_projection.py']}}
    def save():(out/'results.json').write_text(json.dumps(r,indent=2,default=lambda v:v.item())+'\n')
    gpu=GpuProcessAudit(cp.cuda.Device().pci_bus_id).start()
    try:
        if a.case=='million' and cp.cuda.runtime.memGetInfo()[0]<40*2**30:raise MemoryError('need40GiB initially free')
        ref=np.load(root/'reproduction/results/query_posterior/B_applications/controls.npz');q=ref['queries']
        if a.case=='B':
            source=np.load(root/'reproduction/results/matched/B/input.npz');x,y=source['points'],source['normals']
        else:
            x=np.load(root/'reproduction/results/range_compact/scale1m/control.npz')['points']
            y=np.load(root/'reproduction/results/analytic_refinement/million/controls.npz')['normals']
        r.update(n=len(x),m=len(q),hashes={k:array_digest(v) for k,v in [('points',x),('normals',y),('queries',q)]})
        condition,r['seconds']['condition_setup']=timed(lambda:AnalyticConditioner(x,solver='analytic_cg' if a.case=='B' else 'defect',
            fit_cutoff=200,rtol=1e-10,max_cache_bytes=(1 if a.case=='B' else 32)*2**30,max_entries=200_000_000))
        alpha,r['seconds']['mean_solve']=timed(lambda:condition.solve(cp.asarray(y,dtype='d')));r['mean_solve']=condition.records[-1]
        op,r['seconds']['scalar_setup']=timed(lambda:FourierPoisson(50,dtype='float64',eps=1e-12))
        coeff,r['seconds']['mean_aggregate']=timed(lambda:op.aggregate(x,alpha))
        offset,r['seconds']['mean_center']=timed(lambda:op.evaluate(x,coeff).mean())
        condition.kernel=condition.spectrum=None;condition.rhs_count=None;cp.get_default_memory_pool().free_all_blocks()
        condition.spectrum,r['seconds']['finite_setup']=timed(lambda:FourierCovariance(a.cutoff,upsampfac=1.25))
        def approximate(b):
            w,info=condition.finite_solve(b);record={'initial':info,'corrections':[]}
            for _ in range(a.steps):
                delta,info=condition.finite_solve(b-condition.action(w));w+=delta;record['corrections'].append(info)
            r['finite_solves'].append(record);return w
        print('mean ready',r['seconds'],flush=True)
        model,r['seconds']['query_posterior']=timed(lambda:ProjectedQueryPosterior(x,q,op,approximate,condition.action,coeff,offset,
            rank=a.rank,batch=a.batch,progress=lambda done,total:print('basis',done,total,flush=True) if done%16==0 or done==total else None))
        r['query_record']=model.record;r['seconds']['prepare_total']=sum(r['seconds'].values());r['budget_pass']=model.record['budget_pass'];save()
        if a.save_vectors and hasattr(model,'audit'):
            def archive_vectors():
                archive=Path('/opt/geospsr-runtime/query-projection-vectors')/label;archive.mkdir(parents=True,exist_ok=True)
                record={}
                for name,value in [('weights',model.audit.host_w),('residuals',model.audit.host_r)]:
                    path=archive/(name+'.npy');np.save(path,value)
                    record[name]={'path':str(path),'shape':list(value.shape),'dtype':str(value.dtype),
                                  'array_sha256':array_digest(value.reshape(-1))}
                record['qualification']='local scratch checkpoint, not a published or durable full-vector archive'
                return record
            r['vector_archive'],r['seconds']['validation_vector_archive']=timed(archive_vectors);save()
        condition.kernel=condition.spectrum=None;cp.get_default_memory_pool().free_all_blocks()
        arrays={k:getattr(model,k).get() for k in ['queries','mean','prior','basis','tail','cross_action','weight_residual','solve_upper','covariance','upper']}
        if a.case=='B':
            target=cp.asarray(ref['reference_covariance']);mu=cp.asarray(ref['reference_mean'])
            error=model.covariance-target
            r['covariance_relative_l2']=float(cp.linalg.norm(error)/cp.linalg.norm(target))
            r['maximum_marginal_relative_error']=float(cp.max(cp.abs(cp.diag(error))/cp.diag(target)))
            r['mean_error_over_prior_std']=float(cp.max(cp.abs(model.mean-mu)/cp.sqrt(cp.diag(model.prior))))
            r['upper_minus_error_min_eigenvalue']=float(cp.linalg.eigvalsh(model.upper-error).min())
            r['error_min_eigenvalue']=float(cp.linalg.eigvalsh(error).min())
            arrays.update(reference_covariance=target.get(),reference_mean=mu.get())
            r['quality_pass']=r['budget_pass'] and all(r[k]<=r['gates'][k] for k in ['covariance_relative_l2','maximum_marginal_relative_error','mean_error_over_prior_std'])
            (_,factor,psd),r['seconds']['validation_reference_root']=timed(lambda:principal_root(target));r['reference_psd']=psd
        else:
            # All-row R norms/Gram support the bound. Independent sampled direct
            # rows audit the numerical actions, not the full residual norm.
            from geospsr.spectral_tail import analytic_covariance_rows64
            rows=np.load(root/'reproduction/results/range_compact/scale1m/control.npz')['rows']
            def direct_audit():
                pieces=[];captured=[];rhsp=[];basis=model.basis
                for start in range(0,len(rows),32):
                    rr=rows[start:start+32];mat=analytic_covariance_rows64(x[rr],x)
                    delta=q[:,None].astype('d')-x[rr][None].astype('d')
                    cross=op.cross_values(delta.reshape(-1,3)).reshape(len(q),len(rr),3)
                    rhs=cp.einsum('mnd,mk->dnk',cross,basis)
                    direct=cp.empty_like(rhs)
                    for d in range(3):
                        for begin in range(0,basis.shape[1],16):
                            end=min(begin+16,basis.shape[1]);w=cp.asarray(np.ascontiguousarray(model.audit.host_w[d,begin:end]))
                            direct[d,:,begin:end]=rhs[d,:,begin:end]-mat@w.T-.02*w[:,rr].T
                    observed=np.stack([model.audit.host_r[d][:,rr].T for d in range(3)])
                    pieces.append(direct.get());captured.append(observed);rhsp.append(rhs.get())
                    print('direct rows',start+len(rr),len(rows),flush=True)
                direct=np.concatenate(pieces,axis=1);observed=np.concatenate(captured,axis=1);rhs=np.concatenate(rhsp,axis=1)
                metric=np.linalg.norm(direct-observed,axis=1)/np.maximum(np.linalg.norm(rhs,axis=1),1e-300)
                arrays.update(sampled_direct_residual=direct,sampled_captured_residual=observed,sampled_rhs=rhs,rows=rows)
                return {'maximum_discrepancy_over_rhs':float(metric.max()),'per_component_basis_column':metric.tolist(),
                    'qualification':'sampled direct rows, all sources/all reduced RHS; numerical diagnostic, not an interval bound'}
            r['direct_validation'],r['seconds']['validation_direct']=timed(direct_audit)
            manifest=json.loads((root/'reproduction/results/analytic_refinement/confirmation/release_manifest.json').read_text())
            host=np.load('/opt/geospsr-runtime/analytic-refinement/confirmation/plain_weights.npy')
            assert array_digest(host)==manifest['assets']['plain']['array_sha256']
            r['mean_weights_relative_l2']=float(cp.linalg.norm(alpha-cp.asarray(host))/cp.linalg.norm(cp.asarray(host)))
            r['quality_pass']=r['budget_pass'] and r['direct_validation']['maximum_discrepancy_over_rhs']<=1e-7 and r['mean_weights_relative_l2']<=1e-7
        model.clear_vectors()
        np.savez_compressed(out/'controls.npz',**arrays)
        count=32768
        white,r['seconds']['draw_rng']=timed(lambda:cp.random.RandomState(4804).standard_normal((count,len(q)),dtype='d'))
        def summarize(values):return {'fields':values[:,:32].get(),
            'rays':cp.cumprod(values[:,:768].reshape(-1,3,256)>0,axis=2).astype('uint8').get(),'collision':(values[:,768:]<=0).any(1).get()}
        values,r['seconds']['draw_fields']=timed(lambda:model.draw(count,white=white))
        output,r['seconds']['draw_events']=timed(lambda:summarize(values));outputs={'query_'+k:v for k,v in output.items()};r['samples']=count
        if a.case=='B':
            reference=white@factor.T+mu;other=summarize(reference);outputs.update({'reference_'+k:v for k,v in other.items()})
            r['paired_field_relative_l2']=float(cp.linalg.norm(values-reference)/cp.linalg.norm(reference));r['events']={}
            for k in ['rays','collision']:
                diff=int(np.count_nonzero(output[k]!=other[k]));rate=diff/output[k].size
                r['events'][k]={'disagreements':diff,'rate':rate};r['quality_pass']=r['quality_pass'] and rate<r['gates']['event_disagreement_rate']
        np.savez_compressed(out/'outputs.npz',**outputs)
        r['complete']=True;print('complete',r['quality_pass'],r['seconds'],model.record['budget'],flush=True)
    except Exception as exc:r['failure']=f'{type(exc).__name__}: {exc}';raise
    finally:
        r['host_process_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        r['gpu_process_audit']=gpu.stop();save()


if __name__=='__main__':main()
