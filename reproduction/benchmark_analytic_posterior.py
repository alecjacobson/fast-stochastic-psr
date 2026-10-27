"""Complete coherent analytic conditioning and paired application controls."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import cupy as cp
import cupyx.scipy.linalg as cla
import numpy as np
from reproduction.analytic_posterior import AnalyticPosterior,timed
from reproduction.common_rhs_integrity import array_digest
from reproduction.gpu_audit import GpuProcessAudit
from reproduction.paired_applications import application_geometry,uncertain_objects,events,field_quality,quality
from geospsr.spectral_tail import analytic_covariance_rows64
from geospsr.training_spectrum import finite_covariance_rows,FourierCovariance


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--case',choices=['A','B'],required=True)
    parser.add_argument('--samples',type=int,default=64)
    args=parser.parse_args()
    if args.samples<2:parser.error('need at least two samples')
    root=Path(__file__).resolve().parents[1]
    out=root/'reproduction/results/analytic_posterior'/args.case;out.mkdir(parents=True,exist_ok=True)
    record={'complete':False,'case':args.case,'samples':args.samples,'seed':4694,'seconds':{},'draws':[],
            'model':{'scalar_cutoff':50,'fit_cutoff':200,'lengthscale':.01,'noise_variance':.02,'variance_per_axis':.1,'root_rtol':1e-7,'solve_rtol':1e-8,'normalization':'analytic'},
            'scope':'direct finite-scalar/full-analytic-vector numerical comparison; no released-interpolant, geometry-convergence or absolute MC guarantee',
            'device':cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
            'source_sha256':{f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in ['reproduction/analytic_posterior.py','reproduction/benchmark_analytic_posterior.py','geospsr/analytic_refinement.py','geospsr/analytic_tail.py','geospsr/square_root.py','geospsr/spectral.py']}}
    def save():(out/'results.json').write_text(json.dumps(record,indent=2)+'\n')
    audit=GpuProcessAudit(cp.cuda.Device().pci_bus_id).start();begin=time.perf_counter()
    try:
        data=np.load(root/f'reproduction/results/matched/{args.case}/input.npz');x,y=data['points'],data['normals']
        record['n']=len(x);record['input_hashes']={k:array_digest(v) for k,v in [('points',x),('normals',y)]}
        model,record['seconds']['production_setup_fit']=timed(lambda:AnalyticPosterior(x,y))
        record['production_setup_stages']=model.setup;record['root_bound']=model.bound;record['root_polynomial']=model.polynomial.record;record['mean_solve']=model.mean_solve
        def matrix():
            a=analytic_covariance_rows64(x,x);a.flat[::len(x)+1]+=.02;return a
        a,record['seconds']['dense_matrix']=timed(matrix)
        factor,record['seconds']['dense_factor']=timed(lambda:cp.linalg.cholesky(a))
        def solve(rhs):return cla.solve_triangular(factor.T,cla.solve_triangular(factor,rhs,lower=True),lower=False)
        alpha,record['seconds']['dense_mean_solve']=timed(lambda:solve(cp.asarray(y,dtype='d')))
        (dense_source_plan,dense_prior_plan),record['seconds']['dense_shared_plans']=timed(lambda:(model.op.plan(1,x,3),model.op.plan(2,x,3)))
        def dense_aggregate(weights):
            modes=dense_source_plan.execute(cp.ascontiguousarray(weights.T,dtype='complex128'))
            return cp.ascontiguousarray(-1j*(modes.reshape(1,3,*model.op.shape)*model.op.multiplier[None]).sum(1))
        coeff,record['seconds']['dense_mean_aggregate']=timed(lambda:model.op.aggregate(x,alpha))
        offset,record['seconds']['dense_mean_centering']=timed(lambda:model.op.evaluate(x,coeff).mean())
        spectrum,record['seconds']['dense_tail_spectrum']=timed(lambda:FourierCovariance(50,dtype='float64'))
        def residual():
            low=finite_covariance_rows(x,x,spectrum,dtype='float64');return a-(low+low.T)*.5
        residual,record['seconds']['dense_tail_matrix']=timed(residual)
        tail_chol,record['seconds']['dense_tail_cholesky']=timed(lambda:cp.linalg.cholesky(residual))
        if args.case=='A':
            def root_matrix():
                eig,vec=cp.linalg.eigh(residual);return (vec*cp.sqrt(eig))@vec.T
            dense_root,record['seconds']['paired_reference_root_setup']=timed(root_matrix)
            reference_tail=lambda white:dense_root@white
            record['paired_reference_root']='dense principal root (eigendecomposition)'
        else:
            reference_tail=lambda white:model.polynomial.apply(lambda v:residual@v,white,xp=cp)
            record['paired_reference_root']='same bounded fixed polynomial, independent dense covariance actions; not exact principal root'
        def geometry():
            rng=np.random.default_rng(4695)
            q=np.concatenate([rng.uniform(-1,1,(256,3)),x[rng.choice(len(x),256,replace=False)]+rng.normal(0,.01,(256,3))]).astype('f')
            rays,objects,distance=application_geometry(x)
            norm2=(model.op.ns**2).sum(0)
            prior_std=float(cp.sqrt(cp.sum(cp.where(norm2>0,model.op.rho/cp.where(norm2>0,norm2,1),0))))
            objects=uncertain_objects(lambda pos:model.op.evaluate(pos,coeff)[0]-offset,objects,prior_std)
            return q,rays,objects,distance
        (q,rays,objects,distance),record['seconds']['geometry']=timed(geometry)
        positions=np.concatenate([q,rays.reshape(-1,3),objects.reshape(-1,3)])
        record['geometry_hashes']={k:array_digest(v) for k,v in [('query',q),('rays',rays),('objects',objects),('distance',distance)]}
        np.savez_compressed(out/'geometry.npz',query=q,rays=rays,objects=objects,distance=distance)
        def query(c,off):
            values=model.op.evaluate(positions,c)-off
            end=len(q)+rays.shape[0]*rays.shape[1]
            return {'fields':values[:,:len(q)].get(),
                    'rays':events(values[:,len(q):end],rays.shape[:2],'rays').get(),
                    'collision':events(values[:,end:],objects.shape[:2],'collision').get()}
        outputs={side:{k:[] for k in ['fields','rays','collision']} for side in ['production','paired_reference','dense_cholesky']}
        control={}
        for block in model.draw_batches(args.samples,seed=4694,batch=1,retain_components=True):
            entry={'index':block['start'],'production':block['record']}
            tail,entry['paired_root_seconds']=timed(lambda:reference_tail(block['tail_white']))
            rhs=cp.asarray(y,dtype='d')-block['vector_values']-tail
            weights,entry['paired_solve_seconds']=timed(lambda:solve(rhs))
            c,entry['paired_aggregate_seconds']=timed(lambda:dense_aggregate(weights)+block['prior'])
            entry['tail_relative_l2']=float(cp.linalg.norm(tail-block['tail'])/cp.linalg.norm(tail))
            entry['weights_relative_l2']=float(cp.linalg.norm(weights-block['weights'])/cp.linalg.norm(weights))
            entry['independent_solve_residual_per_rhs']=(cp.linalg.norm(a@block['weights']-block['rhs'],axis=0)/cp.linalg.norm(block['rhs'],axis=0)).get().tolist()
            baseline_tail,entry['dense_root_seconds']=timed(lambda:tail_chol@block['tail_white'])
            baseline_rhs=cp.asarray(y,dtype='d')-block['vector_values']-baseline_tail
            baseline_weights,entry['dense_solve_seconds']=timed(lambda:solve(baseline_rhs))
            baseline_c,entry['dense_aggregate_seconds']=timed(lambda:dense_aggregate(baseline_weights)+block['prior'])
            for side,cvalue,off in [('production',block['coefficients'],model.offset),('paired_reference',c,offset),('dense_cholesky',baseline_c,offset)]:
                value,entry[side+'_query_seconds']=timed(lambda:query(cvalue,off))
                for k,v in value.items():outputs[side][k].append(v)
            if block['start']==0:
                control={k:v.get() for k,v in [('production_weights',block['weights']),('production_rhs',block['rhs']),('paired_weights',weights),('paired_rhs',rhs),('white_noise',block['tail_white']),('production_tail',block['tail']),('paired_tail',tail)]}
            record['draws'].append(entry);save()
            print(args.case,'draw',block['stop'],'root',block['record']['tail_root_seconds'],'solve',block['record']['solve_seconds'],'residual',max(entry['independent_solve_residual_per_rhs']),flush=True)
        outputs={s:{k:np.concatenate(v) for k,v in d.items()} for s,d in outputs.items()}
        moment_ids=np.r_[0:16,256:272]
        def moment_target():
            selected=q[moment_ids].astype('d')
            cross=model.op.cross_values((selected[:,None]-x.astype('d')[None]).reshape(-1,3)).reshape(len(selected),len(x),3)
            solved=solve(cross.transpose(1,0,2).reshape(len(x),-1)).reshape(len(x),len(selected),3)
            norm2=(model.op.ns**2).sum(0)
            rho=cp.where(norm2>0,model.op.rho/cp.where(norm2>0,norm2,1),0)
            prior=model.op.evaluate((selected[:,None]-selected[None]).reshape(-1,3),rho).reshape(len(selected),len(selected))
            cov=prior-cp.einsum('qnd,nkd->qk',cross,solved)
            mean=cp.einsum('qnd,nd->q',cross,alpha)-offset
            return mean.get(),cov.get()
        (target_mean,target_cov),record['seconds']['moment_validation']=timed(moment_target)
        assert np.linalg.eigvalsh(target_cov).min()>0
        moment_controls={'query_indices':moment_ids,'mean':target_mean,'covariance':target_cov}
        record['moment_sanity']={}
        for side,values in outputs.items():
            draws=values['fields'][:,moment_ids]
            mean_se=np.sqrt(np.diag(target_cov)/args.samples)
            cov_se=np.sqrt((target_cov**2+np.diag(target_cov)[:,None]*np.diag(target_cov)[None,:])/(args.samples-1))
            mean_z=float(np.max(np.abs(draws.mean(0)-target_mean)/mean_se))
            cov_z=float(np.max(np.abs(np.cov(draws.T)-target_cov)/cov_se))
            record['moment_sanity'][side]={'max_mean_standardized_error':mean_z,'max_covariance_standardized_error':cov_z,'six_standard_error_sanity_pass':mean_z<6 and cov_z<6}
        np.savez_compressed(out/'moment_target.npz',**moment_controls)
        np.savez_compressed(out/'outputs.npz',**{s+'_'+k:v for s,d in outputs.items() for k,v in d.items()})
        np.savez_compressed(out/'first_draw.npz',**control,mean_weights=model.alpha.get(),dense_mean_weights=alpha.get())
        record['field_quality']=field_quality(cp.asarray(outputs['paired_reference']['fields']),cp.asarray(outputs['production']['fields']))
        record['event_quality']=quality(outputs['paired_reference'],outputs['production'],distance)
        record['maximum_probability_mc_standard_error']={k:float(np.sqrt((outputs['production'][k].mean(0)*(1-outputs['production'][k].mean(0)))/args.samples).max()) for k in ['rays','collision']}
        record['quality_pass']=bool(record['field_quality']['pass'] and record['event_quality']['compatibility_pass'] and record['event_quality']['uncertainty_coverage_pass'] and max(max(d['independent_solve_residual_per_rhs']) for d in record['draws'])<=1e-8 and max(d['tail_relative_l2'] for d in record['draws'])<=1e-7)
        production=sum(sum(d['production'][k] for k in ['plan_seconds','finite_prior_seconds','tail_root_seconds','solve_seconds','aggregation_seconds'])+d['production_query_seconds'] for d in record['draws'])
        # Charge shared finite-prior generation to both methods. The dense
        # baseline executes its own root, solve, aggregation and queries.
        dense=sum(d['production']['finite_prior_seconds']+sum(d[k] for k in ['dense_root_seconds','dense_solve_seconds','dense_aggregate_seconds','dense_cholesky_query_seconds']) for d in record['draws'])
        dense_setup=sum(record['seconds'][k] for k in ['dense_matrix','dense_factor','dense_mean_solve','dense_mean_aggregate','dense_mean_centering','dense_tail_spectrum','dense_tail_matrix','dense_tail_cholesky','dense_shared_plans'])+model.setup['scalar_spectrum_seconds']
        record['seconds'].update(production_draws_and_queries=production,dense_draws_and_queries=dense,
                                 dense_setup_fit=dense_setup,production_total=record['seconds']['production_setup_fit']+production,
                                 dense_total=dense_setup+dense)
        record['complete']=True
    except Exception as exc:record['failure']=f'{type(exc).__name__}: {exc}';raise
    finally:
        record['seconds']['instrumented_wall']=time.perf_counter()-begin;record['gpu_process_audit']=audit.stop();save()


if __name__=='__main__':main()
