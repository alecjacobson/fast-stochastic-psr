"""Fixed-panel coherent covariance versus strong dense query/source samplers."""
import argparse,hashlib,json,time
from pathlib import Path
import cupy as cp
import cupyx.scipy.linalg as cla
import numpy as np
from geospsr.spectral import FourierPoisson
from geospsr.spectral_tail import analytic_covariance_rows64
from geospsr.training_spectrum import FourierCovariance,finite_covariance_rows
from reproduction.query_posterior import AnalyticConditioner,FixedQueryPosterior,principal_root
from reproduction.analytic_posterior import timed
from reproduction.common_rhs_integrity import array_digest
from reproduction.gpu_audit import GpuProcessAudit


def main():
    p=argparse.ArgumentParser();p.add_argument('--case',choices=['A','B'],required=True)
    p.add_argument('--panel',choices=['moments','applications'],required=True)
    p.add_argument('--samples',type=int,default=8192);p.add_argument('--source-samples',type=int,default=64)
    a=p.parse_args()
    if min(a.samples,a.source_samples)<2:p.error('sample counts must exceed one')
    root=Path(__file__).resolve().parents[1];out=root/f'reproduction/results/query_posterior/{a.case}_{a.panel}';out.mkdir(parents=True,exist_ok=True)
    r={'complete':False,'case':a.case,'panel':a.panel,'samples':a.samples,'source_samples':a.source_samples,
       'seconds':{},'source_batches':[],'scope':'fixed query panel, full analytic training and finite scalar50; no arbitrary-query or massive claim',
       'source_sha256':{f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in ['reproduction/query_posterior.py','reproduction/benchmark_query_posterior.py','reproduction/deflated_variance.py']}}
    def save():(out/'results.json').write_text(json.dumps(r,indent=2)+'\n')
    audit=GpuProcessAudit(cp.cuda.Device().pci_bus_id).start()
    try:
        data=np.load(root/f'reproduction/results/matched/{a.case}/input.npz');x,y=data['points'],data['normals']
        old=root/f'reproduction/results/analytic_posterior/{a.case}'
        geo=np.load(old/'geometry.npz')
        if a.panel=='moments':q=geo['query'][np.load(old/'moment_target.npz')['query_indices']]
        else:q=np.concatenate([geo['rays'][[0,8,17]].reshape(-1,3),geo['objects'][8]])
        r.update(n=len(x),m=len(q),hashes={k:array_digest(v) for k,v in [('points',x),('normals',y),('queries',q)]})
        op,r['seconds']['scalar_setup']=timed(lambda:FourierPoisson(50,dtype='float64',eps=1e-12))
        condition,r['seconds']['analytic_setup']=timed(lambda:AnalyticConditioner(x))
        alpha,r['seconds']['analytic_mean_solve']=timed(lambda:condition.solve(cp.asarray(y,dtype='d')))
        coeff,r['seconds']['analytic_mean_aggregate']=timed(lambda:op.aggregate(x,alpha))
        offset,r['seconds']['analytic_mean_center']=timed(lambda:op.evaluate(x,coeff).mean())
        model,r['seconds']['query_posterior']=timed(lambda:FixedQueryPosterior(x,q,op,condition.solve,coeff,offset,batch=4,
            progress=lambda done,total:print(a.case,a.panel,'covariance',done,total,flush=True) if done%64==0 or done==total else None))
        r['query_record']=model.record;r['solve_records']=condition.records;save()
        def dense_setup():
            matrix=analytic_covariance_rows64(x,x);matrix.flat[::len(x)+1]+=.02
            return matrix,cp.linalg.cholesky(matrix)
        (matrix,chol),r['seconds']['dense_setup']=timed(dense_setup)
        def solve(rhs):return cla.solve_triangular(chol.T,cla.solve_triangular(chol,rhs,lower=True),lower=False)
        reference_alpha,r['seconds']['dense_mean_solve']=timed(lambda:solve(cp.asarray(y,dtype='d')))
        reference_coeff,r['seconds']['dense_mean_aggregate']=timed(lambda:op.aggregate(x,reference_alpha))
        reference_offset,r['seconds']['dense_mean_center']=timed(lambda:op.evaluate(x,reference_coeff).mean())
        def dense_query():
            cross=cp.empty((len(q),len(x),3),dtype='d')
            for start in range(0,len(q),16):
                delta=q[start:start+16,None].astype('d')-x[None].astype('d')
                cross[start:start+16]=op.cross_values(delta.reshape(-1,3)).reshape(-1,len(x),3)
            covariance=model.prior.copy()
            for d in range(3):covariance-=cross[:,:,d]@solve(cross[:,:,d].T)
            mean=cp.einsum('qnd,nd->q',cross,reference_alpha)-reference_offset
            covariance,factor,psd=principal_root(covariance)
            return mean,covariance,factor,psd
        (mean,covariance,factor,psd),r['seconds']['dense_query']=timed(dense_query)
        r['dense_psd']=psd
        r['covariance_relative_l2']=float(cp.linalg.norm(model.covariance-covariance)/cp.linalg.norm(covariance))
        r['mean_relative_l2']=float(cp.linalg.norm(model.mean-mean)/cp.linalg.norm(mean))
        r['mean_error_over_prior_std']=float(cp.max(cp.abs(model.mean-mean)/cp.sqrt(cp.diag(model.prior))))
        r['marginal_variance_relative_max']=float(cp.max(cp.abs(cp.diag(model.covariance-covariance))/cp.diag(covariance)))
        def summarize(values):
            fields=values[:,:min(32,len(q))].get()
            if a.panel=='applications':
                rays=cp.cumprod(values[:,:768].reshape(-1,3,256)>0,axis=2).astype('uint8').get()
                collision=(values[:,768:]<=0).any(1).get()
            else:
                rays=np.empty((len(values),0),dtype='uint8');collision=(values<=0).any(1).get()
            return {'fields':fields,'rays':rays,'collision':collision}
        outputs={}
        # All methods at equal counts; larger counts are separate reuse tests.
        for count in sorted({64,a.samples,a.source_samples}):
            white=cp.random.RandomState(4804).standard_normal((count,len(q)),dtype='d')
            production,ptime=timed(lambda:summarize(model.draw(count,white=white)))
            reference,dtime=timed(lambda:summarize(white@factor.T+mean))
            # Normal generation is shared, so time and charge it to both paths.
            _,rngtime=timed(lambda:cp.random.RandomState(4804).standard_normal((count,len(q)),dtype='d'))
            r.setdefault('query_draws',[]).append({'count':count,'production_seconds':ptime+rngtime,'dense_seconds':dtime+rngtime,
                'paired_field_relative_l2':float(np.linalg.norm(production['fields']-reference['fields'])/np.linalg.norm(reference['fields'])),
                'collision_disagreements':int(np.count_nonzero(production['collision']!=reference['collision'])),
                'ray_disagreements':int(np.count_nonzero(production['rays']!=reference['rays']))})
            if count==a.samples:outputs['query']=production;outputs['reference']=reference
        # Independent source-space pathwise samples at exactly this panel.
        def tail_setup():
            spectrum=FourierCovariance(50,dtype='float64')
            low=finite_covariance_rows(x,x,spectrum,dtype='float64')
            return cp.linalg.cholesky(matrix-(low+low.T)*.5)
        tail_chol,r['seconds']['source_tail_setup']=timed(tail_setup)
        def plans():return op.plan(2,x,24),op.plan(1,x,24),op.plan(2,q,8)
        (back,forward,query),r['seconds']['source_plans']=timed(plans)
        rng=cp.random.RandomState(4805);parts={k:[] for k in ['fields','rays','collision']}
        def source_stream():
            for start in range(0,a.source_samples,8):
                count=min(8,a.source_samples-start)
                # Draw complete groups of8, retaining only the requested prefix.
                xi=rng.standard_normal((8,3,*op.shape,2),dtype='d')
                vector,scalar=op.prior_coefficients(xi)
                v=back.execute(cp.ascontiguousarray(vector.reshape(24,*op.shape))).real.T
                white=rng.standard_normal((len(x),24),dtype='d')
                rhs=cp.tile(cp.asarray(y,dtype='d'),(1,8))-v-tail_chol@white
                weights=solve(rhs)
                transformed=forward.execute(cp.ascontiguousarray(weights.T,dtype='complex128')).reshape(8,3,*op.shape)
                c=cp.ascontiguousarray(-1j*(transformed*op.multiplier[None]).sum(1)+scalar)
                values=query.execute(c).real-reference_offset
                result=summarize(values[:count])
                for k,v in result.items():parts[k].append(v)
                if (start+count)%512==0 or start+count==a.source_samples:print('source draws',start+count,flush=True)
            return {k:np.concatenate(v) for k,v in parts.items()}
        outputs['source'],r['seconds']['source_draws']=timed(source_stream)
        r['source_moment_sanity']={}
        ids=slice(0,min(32,len(q)));target=covariance[ids,ids].get();mu=mean[ids].get()
        for side,output in outputs.items():
            values=output['fields'];se=np.sqrt(np.diag(target)/len(values));cse=np.sqrt((target**2+np.diag(target)[:,None]*np.diag(target)[None,:])/(len(values)-1))
            r['source_moment_sanity'][side]={'max_mean_standardized_error':float(np.max(np.abs(values.mean(0)-mu)/se)),
                'max_covariance_standardized_error':float(np.max(np.abs(np.cov(values.T)-target)/cse)),
                'joint_inside_probability':float(output['collision'].mean()),
                'joint_inside_standard_error':float(output['collision'].std(ddof=1)/np.sqrt(len(values)))}
        common=[r['seconds'][k] for k in ['scalar_setup','dense_setup','dense_mean_solve','dense_mean_aggregate','dense_mean_center']]
        r['seconds']['source_prepare_total']=sum(common)+r['seconds']['source_tail_setup']+r['seconds']['source_plans']
        # The dense query reference shares the independently checked Fourier
        # prior, so charge its measured prior assembly to this baseline too.
        r['seconds']['dense_query_prepare_total']=sum(common)+r['seconds']['dense_query']+model.record['seconds']['prior']
        r['seconds']['query_prepare_total']=sum(r['seconds'][k] for k in ['scalar_setup','analytic_setup','analytic_mean_solve','analytic_mean_aggregate','analytic_mean_center','query_posterior'])
        r['quality_pass']=r['covariance_relative_l2']<1e-7 and r['mean_error_over_prior_std']<1e-7 and r['marginal_variance_relative_max']<1e-7
        np.savez_compressed(out/'controls.npz',queries=q,mean=model.mean.get(),reference_mean=mean.get(),covariance=model.covariance.get(),reference_covariance=covariance.get())
        np.savez_compressed(out/'outputs.npz',**{side+'_'+k:value for side,output in outputs.items() for k,value in output.items()})
        r['complete']=True
    except Exception as exc:r['failure']=f'{type(exc).__name__}: {exc}';raise
    finally:r['gpu_process_audit']=audit.stop();save()


if __name__=='__main__':main()
