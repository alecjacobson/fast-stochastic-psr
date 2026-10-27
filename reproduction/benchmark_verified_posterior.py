"""Paired coherent applications after independently verified conditioning."""
import argparse,json,time
from pathlib import Path
from contextlib import closing
import cupy as cp
import numpy as np
from geospsr.spectral import FourierPoisson,spectral_axis
from geospsr.refinement import refine_solution
from reproduction.half_posterior import HalfCubePosterior
from reproduction.posterior import sync
from reproduction.common_rhs_integrity import array_digest
from reproduction.gpu_audit import GpuProcessAudit
from reproduction.paired_applications import application_geometry,uncertain_objects,events,field_quality,quality


def dense_matrix(x):
    modes,axis=spectral_axis(200)
    result=np.empty((len(x),len(x)),dtype='d')
    for start in range(0,len(x),32):
        rows=np.ones((min(32,len(x)-start),len(x)))
        for d in range(3):
            rows*=np.cos((x[start:start+32,None,d].astype('d')-x[None,:,d].astype('d'))[...,None]*modes)@axis
        result[start:start+32]=rows
    return cp.asarray(result+.02*np.eye(len(x)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--case',choices=['A','three_million'],required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--archive',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);a.archive.mkdir(parents=True,exist_ok=True)
    r={'complete':False,'case':a.case,'samples':8,'seed':701,'batch':1,'seconds':{},'draws':[],
       'scope':'paired same-prior numerical conditioning and application diagnostic; no speedup or absolute MC claim',
       'model':{'training_cutoff':200,'scalar_cutoff':50,'lengthscale':.01,'variance':.1,'noise_variance':.02,
                'normalization':'analytic','production_rtol':1e-5,'actual_residual_threshold':2e-5,
                'reference_residual_threshold':2e-7},'device':cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}
    def save():(a.output/'results.json').write_text(json.dumps(r,indent=2)+'\n')
    def timed(fn):
        sync();start=time.perf_counter();v=fn();sync();return v,time.perf_counter()-start
    def archive(name,value):
        value=cp.asnumpy(value) if isinstance(value,cp.ndarray) else value
        np.save(a.archive/(name+'.npy'),value);return array_digest(value)
    audit=GpuProcessAudit(cp.cuda.Device().pci_bus_id).start();begin=time.perf_counter()
    try:
        if a.case=='A':
            data=np.load('reproduction/results/matched/A/input.npz');x,y=data['points'],data['normals']
        else:
            source=Path('/opt/geospsr-runtime/large-cloud-3000000')
            x,y=[np.load(source/(k+'.npy')) for k in ['points','normals']]
        # Fit the exact stored float32 positions in both precisions.
        x,y=x.astype('f'),y.astype('f');r['N']=len(x)
        r['source_hashes']={'points':array_digest(x),'normals':array_digest(y)}
        model,r['seconds']['production_fit']=timed(lambda:HalfCubePosterior(x,y,refine_solves=True))
        r['mean_solve']=model.mean_solve
        double,r['seconds']['reference_scalar_setup']=timed(lambda:FourierPoisson(50,dtype='float64',eps=1e-10))
        L=None
        if a.case=='A':L,r['seconds']['dense_setup']=timed(lambda:cp.linalg.cholesky(dense_matrix(x)))
        def reference(rhs,initial):
            if L is not None:
                from cupyx.scipy.linalg import solve_triangular
                weights=solve_triangular(L.T,solve_triangular(L,rhs.astype('d'),lower=True),lower=False)
                residual=rhs.astype('d')-model._verification_action(weights,200)
                rel=cp.linalg.norm(residual,axis=0)/cp.linalg.norm(rhs.astype('d'),axis=0)
                return weights,{'converged':bool(cp.all(rel<=2e-7)),'relative_residual':rel.tolist(),'kind':'dense_same_finite_cube'}
            def correction(defect):
                weights,info=model.K.solve(defect.astype('f'),rtol=.1,maxiter=1000)
                return weights.astype('d'),info
            return refine_solution(correction,lambda w:model._verification_action(w,200),rhs,initial.astype('d'),
                                   target_rtol=2e-7,max_corrections=4,xp=cp)
        (alpha,info),r['seconds']['reference_mean_refinement']=timed(lambda:reference(model.normals,model.alpha))
        r['reference_mean_solve']=info
        if not info['converged']:raise RuntimeError('reference mean verification failed')
        coeff,r['seconds']['reference_mean_aggregation']=timed(lambda:double.aggregate(model.x.astype('d'),alpha))
        offset,r['seconds']['reference_mean_centering']=timed(lambda:double.evaluate(model.x.astype('d'),coeff).mean())
        r['offsets']={'production':float(model.offset),'reference':float(offset)}
        r['mean_hashes']={k:archive(k,v) for k,v in [('production_mean_coeff',model.mean_coeff),('reference_mean_coeff',coeff)]}
        def geometry():
            rng=np.random.default_rng(702)
            q=np.concatenate([rng.uniform(-1,1,(256,3)),x[rng.choice(len(x),256,replace=False)]+rng.normal(0,.01,(256,3))]).astype('f')
            rays,objects,distance=application_geometry(x)
            # Scalar prior variance = sum(rho/|k|^2), excluding k=0.
            norm2=(double.ns**2).sum(0)
            prior_std=float(cp.sqrt(cp.sum(cp.where(norm2>0,double.rho/cp.where(norm2>0,norm2,1),0))))
            objects=uncertain_objects(lambda pos:double.evaluate(pos,coeff)-offset,objects,prior_std)
            return q,rays,objects,distance
        (q,rays,objects,distance),r['seconds']['geometry']=timed(geometry)
        positions=np.concatenate([q,rays.reshape(-1,3),objects.reshape(-1,3)])
        np.savez_compressed(a.output/'geometry.npz',query=q,rays=rays,objects=objects,distance=distance)
        r['geometry_hashes']={k:array_digest(v) for k,v in [('query',q),('rays',rays),('objects',objects),('distance',distance)]}
        def query(op,coefs,off):
            values=op.evaluate(positions,coefs)-off
            fields=cp.asnumpy(values[:,:len(q)])
            ray_end=len(q)+rays.shape[0]*rays.shape[1]
            ray_events=cp.asnumpy(events(values[:,len(q):ray_end],rays.shape[:2],'rays'))
            coll_events=cp.asnumpy(events(values[:,ray_end:],objects.shape[:2],'collision'))
            return {'fields':fields,'rays':ray_events,'collision':coll_events}
        outputs={side:{k:[] for k in ['fields','rays','collision']} for side in ['production','reference']}
        with closing(model.draw_batches(8,seed=701,batch=1,retain_components=True,retain_rhs=True)) as stream:
            for block in stream:
                index=block['start'];rhs=block['rhs'][0];weights=block['weights'][0]
                entry={'index':index,'production':block['record']}
                (strong,info),entry['reference_solve_seconds']=timed(lambda:reference(rhs,weights))
                entry['reference_solve']=info
                if not info['converged']:raise RuntimeError('reference draw verification failed')
                strong_coeff,entry['reference_aggregation_seconds']=timed(lambda:double.aggregate(model.x.astype('d'),strong)+block['prior'].astype('complex128'))
                for side,op,c,off in [('production',model.op,block['coefficients'],model.offset),('reference',double,strong_coeff,offset)]:
                    value,entry[side+'_query_seconds']=timed(lambda:query(op,c,off))
                    for k,v in value.items():outputs[side][k].append(v)
                entry['hashes']={name:archive(f'{index}_{name}',v) for name,v in [('rhs',rhs),('prior',block['prior']),
                    ('production_weights',weights),('reference_weights',strong),('production_coeff',block['coefficients']),('reference_coeff',strong_coeff)]}
                r['draws'].append(entry);save();print('draw',index,'reference',info['converged'],flush=True)
        outputs={s:{k:np.concatenate(v) for k,v in d.items()} for s,d in outputs.items()}
        np.savez_compressed(a.output/'fields.npz',**{s+'_'+k:v for s,d in outputs.items() for k,v in d.items()})
        r['field_quality']=field_quality(cp.asarray(outputs['reference']['fields']),cp.asarray(outputs['production']['fields']))
        r['event_quality']=quality(outputs['reference'],outputs['production'],distance)
        r['numerical_gate_pass']=bool(r['field_quality']['pass'] and r['event_quality']['compatibility_pass'] and r['event_quality']['uncertainty_coverage_pass'])
        r['seconds']['production_draw_components']=sum(d['production'][k] for d in r['draws'] for k in ['kernel_setup_seconds','prior_seconds','solve_seconds','aggregation_seconds'])
        r['seconds']['production_queries']=sum(d['production_query_seconds'] for d in r['draws'])
        r['complete']=True
    except Exception as exc:r['failure']=f'{type(exc).__name__}: {exc}';raise
    finally:
        r['seconds']['instrumented_work_wall']=time.perf_counter()-begin;r['gpu_process_audit']=audit.stop();save()


if __name__=='__main__':main()
