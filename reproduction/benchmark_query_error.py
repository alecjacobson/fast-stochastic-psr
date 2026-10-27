"""Quadratic residual correction on the retained 32-query scorpion panel.

The million-point run uses full analytic residual norms, with independent direct
row checks. Those sampled checks are diagnostics, not substitutes for full norms.
The fixed relative covariance budget covers solve error only in exact arithmetic.
"""
import argparse, hashlib, json
from pathlib import Path
import cupy as cp
import cupyx.scipy.linalg as cla
import numpy as np
from geospsr.spectral import FourierPoisson
from geospsr.spectral_tail import analytic_covariance_rows64
from geospsr.training_spectrum import FourierCovariance
from reproduction.query_posterior import AnalyticConditioner, FixedQueryPosterior
from reproduction.query_error import QueryResidualAudit
from reproduction.analytic_posterior import timed
from reproduction.common_rhs_integrity import array_digest
from reproduction.gpu_audit import GpuProcessAudit


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--case',choices=['B','million'],required=True)
    p.add_argument('--cutoff',type=int,choices=[50,100,200],required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    out=root/f'reproduction/results/query_error/{a.case}_c{a.cutoff}'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'results.json').exists():raise FileExistsError('retain previous runs; choose a fresh output directory')
    r={'complete':False,'case':a.case,'cutoff':a.cutoff,'seconds':{},'finite_solves':[],
       'gates':{'relative_covariance_solve_budget':1e-4,'dense_identity_relative_l2':1e-10,
                'direct_rhs_relative_l2':1e-9,'direct_action_error_over_rhs':1e-9},
       'scope':'32 fixed control queries; finite scalar50/full analytic training; solve-only covariance budget; no application/SDD speedup',
       'source_sha256':{f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in
          ['reproduction/query_error.py','reproduction/query_posterior.py','reproduction/benchmark_query_error.py']}}
    def save():(out/'results.json').write_text(json.dumps(r,indent=2,default=lambda v:v.item())+'\n')
    gpu=GpuProcessAudit(cp.cuda.Device().pci_bus_id).start()
    try:
        free,total=cp.cuda.runtime.memGetInfo();r['initial_gpu_free_bytes']=free
        r['environment']={'gpu':cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
                          'cupy':cp.__version__,'numpy':np.__version__,'gpu_total_bytes':total}
        if a.case=='million' and free<40*2**30:raise MemoryError('need40GiB initially free')
        ref=np.load(root/'reproduction/results/query_posterior/B_moments/controls.npz');q=ref['queries']
        if a.case=='B':
            data=np.load(root/'reproduction/results/matched/B/input.npz');x,y=data['points'],data['normals']
            rows=np.arange(len(x))
        else:
            data=np.load(root/'reproduction/results/range_compact/scale1m/control.npz');x,rows=data['points'],data['rows']
            y=np.load(root/'reproduction/results/analytic_refinement/million/controls.npz')['normals']
        r.update(n=len(x),m=len(q),hashes={k:array_digest(v) for k,v in [('points',x),('normals',y),('queries',q),('rows',rows)]})
        condition,r['seconds']['condition_setup']=timed(lambda:AnalyticConditioner(x,
            solver='defect' if a.case=='million' else 'analytic_cg',fit_cutoff=200,rtol=1e-10,
            max_cache_bytes=(32 if a.case=='million' else 1)*2**30,max_entries=200_000_000))
        alpha,r['seconds']['mean_solve']=timed(lambda:condition.solve(cp.asarray(y,dtype='d')))
        r['mean_solve']=condition.records[-1];print('mean done',r['seconds'],flush=True)
        op,r['seconds']['scalar_setup']=timed(lambda:FourierPoisson(50,dtype='float64',eps=1e-12))
        coeff,r['seconds']['mean_aggregate']=timed(lambda:op.aggregate(x,alpha))
        offset,r['seconds']['mean_center']=timed(lambda:op.evaluate(x,coeff).mean())
        # Release the mean solver's finite plan before selecting query bandwidth.
        condition.kernel=condition.spectrum=None;condition.rhs_count=None
        cp.get_default_memory_pool().free_all_blocks()
        condition.spectrum,r['seconds']['finite_spectrum_setup']=timed(lambda:FourierCovariance(a.cutoff,upsampfac=1.25))
        sampled_rhs=[]
        def approximate(b):
            if a.case=='million':sampled_rhs.append(b[rows].get())
            w,info=condition.finite_solve(b);r['finite_solves'].append(info);return w
        audit=QueryResidualAudit(condition.action,approximate,.02)
        model,r['seconds']['query_posterior']=timed(lambda:FixedQueryPosterior(x,q,op,audit,coeff,offset,
            batch=1,residual_audit=audit,progress=lambda done,total:print('query',done,total,flush=True) if done%4==0 else None))
        r['query_record']=model.record
        r['seconds']['prepare_total']=sum(r['seconds'].values())
        r['budget_pass']=(audit.record['relative_budget']['informative'] and
            audit.record['relative_budget']['upper_relative_to_exact_covariance']<=r['gates']['relative_covariance_solve_budget'])
        arrays={'queries':q,'rows':rows,'mean':model.mean.get(),'covariance':model.covariance.get(),
                'prior':model.prior.get(),'raw_covariance':(model.covariance+audit.shift).get(),
                'shift':audit.shift.get(),'upper':audit.upper.get(),
                'cross_norms':audit.cross_norms,'residual_norms':audit.residual_norms}
        save()
        # Validation time is separate; no dense matrix is used by production.
        condition.kernel=condition.spectrum=None;cp.get_default_memory_pool().free_all_blocks()
        if a.case=='B':
            def dense_audit():
                mat=analytic_covariance_rows64(x,x);mat.flat[::len(x)+1]+=.02
                chol=cp.linalg.cholesky(mat)
                delta=q[:,None].astype('d')-x[None].astype('d')
                rhs=op.cross_values(delta.reshape(-1,3)).reshape(len(q),len(x),3).transpose(1,0,2).reshape(len(x),-1)
                rd=rhs-mat@audit.w
                invr=cla.solve_triangular(chol.T,cla.solve_triangular(chol,rd,lower=True),lower=False)
                exact_error=sum(rd[:,d::3].T@invr[:,d::3] for d in range(3))
                target=cp.asarray(ref['reference_covariance']);norm=cp.linalg.norm(target)
                error=model.covariance-target
                result={'raw_covariance_relative_l2':float(cp.linalg.norm(model.covariance+audit.shift-target)/norm),
                    'corrected_covariance_relative_l2':float(cp.linalg.norm(error)/norm),
                    'identity_discrepancy_relative_l2':float(cp.linalg.norm(error-exact_error)/norm),
                    'upper_minus_exact_error_min_eigenvalue':float(cp.linalg.eigvalsh(audit.upper-exact_error).min()),
                    'max_marginal_relative_error':float(cp.max(cp.abs(cp.diag(error))/cp.diag(target))),
                    'mean_error_over_prior_std':float(cp.max(cp.abs(model.mean-cp.asarray(ref['reference_mean']))/cp.sqrt(cp.diag(model.prior)))),
                    'direct_residual_discrepancy_over_rhs':float(cp.linalg.norm(rd-audit.r)/cp.linalg.norm(rhs))}
                arrays.update(reference_covariance=target.get(),reference_mean=ref['reference_mean'],exact_remaining_error=exact_error.get())
                return result
            r['dense_validation'],r['seconds']['validation_dense']=timed(dense_audit)
            d=r['dense_validation'];r['validation_pass']=(d['identity_discrepancy_relative_l2']<=1e-10 and d['direct_residual_discrepancy_over_rhs']<=1e-9 and
                d['upper_minus_exact_error_min_eigenvalue']>=-1e-10*np.linalg.norm(arrays['reference_covariance']))
        else:
            def direct_audit():
                direct=[];assembled=[];rhs_parts=[]
                for begin in range(0,len(rows),64):
                    rr=rows[begin:begin+64]
                    delta=q[:,None].astype('d')-x[rr][None].astype('d')
                    rhs=op.cross_values(delta.reshape(-1,3)).reshape(len(q),len(rr),3).transpose(1,0,2).reshape(len(rr),-1)
                    mat=analytic_covariance_rows64(x[rr],x)
                    direct.append((rhs-mat@audit.w-.02*audit.w[rr]).get())
                    assembled.append(audit.r[rr].get());rhs_parts.append(rhs.get())
                    print('direct rows',begin+len(rr),len(rows),flush=True)
                rd=np.concatenate(direct);ra=np.concatenate(assembled);rhs=np.concatenate(rhs_parts)
                source_rhs=np.concatenate(sampled_rhs,axis=1)
                denom=np.maximum(np.linalg.norm(rhs,axis=0),1e-300)
                arrays.update(sampled_direct_residual=rd,sampled_range_residual=ra,sampled_rhs=rhs,sampled_assembled_rhs=source_rhs)
                return {'rows':len(rows),'sources':len(x),'all96columns':True,
                    'residual_discrepancy_over_rhs_per_column':(np.linalg.norm(rd-ra,axis=0)/denom).tolist(),
                    'cross_discrepancy_over_rhs_per_column':(np.linalg.norm(rhs-source_rhs,axis=0)/denom).tolist(),
                    'action_discrepancy_over_rhs_per_column':(np.linalg.norm(rd-ra-(rhs-source_rhs),axis=0)/denom).tolist(),
                    'direct_relative_residual_per_column':(np.linalg.norm(rd,axis=0)/np.maximum(np.linalg.norm(rhs,axis=0),1e-300)).tolist(),
                    'qualification':'sampled direct rows validate numerical action consistency, not the full residual norm or an interval bound'}
            r['direct_validation'],r['seconds']['validation_direct']=timed(direct_audit)
            manifest=json.loads((root/'reproduction/results/analytic_refinement/confirmation/release_manifest.json').read_text())
            path=Path('/opt/geospsr-runtime/analytic-refinement/confirmation/plain_weights.npy')
            host=np.load(path);assert array_digest(host)==manifest['assets']['plain']['array_sha256']
            reference=cp.asarray(host);r['mean_weights_relative_l2']=float(cp.linalg.norm(alpha-reference)/cp.linalg.norm(reference))
            r['mean_reference']=manifest['assets']['plain']
            r['validation_pass']=all(max(r['direct_validation'][k])<=1e-9 for k in
                ['residual_discrepancy_over_rhs_per_column','cross_discrepancy_over_rhs_per_column','action_discrepancy_over_rhs_per_column']) and r['mean_weights_relative_l2']<=1e-7
        audit.clear_vectors()
        np.savez_compressed(out/'controls.npz',**arrays)
        r['quality_pass']=r['budget_pass'] and r['validation_pass'];r['complete']=True
        print('complete',r['quality_pass'],r['seconds'],audit.record['relative_budget'],flush=True)
    except Exception as exc:r['failure']=f'{type(exc).__name__}: {exc}';raise
    finally:r['gpu_process_audit']=gpu.stop();save()


if __name__=='__main__':main()
