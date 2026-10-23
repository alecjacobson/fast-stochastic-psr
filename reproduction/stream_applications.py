"""Stream coherent finite-cube samples into application and convergence checks.

The complete instrumented execution includes fitting, sampling and queries.
No upstream timing ratio is inferred. Absolute Monte Carlo bounds are separate
from numerical agreement and the two-resolution geometry diagnostic.
"""
import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
import time
import cupy as cp
import numpy as np
from geospsr.cached_spectral import CachedFourierEvaluator
from geospsr.training_spectrum import finite_covariance_rows
from reproduction.half_posterior import HalfCubePosterior
from reproduction.gpu_audit import GpuProcessAudit
from reproduction.paired_applications import events,field_quality,quality
from reproduction.streaming_statistics import OnlineMoments,OnlineEvents,view_scores

ROOT=Path(__file__).resolve().parents[1]


def refined_geometry(rays,objects):
    """Nested ray midpoints and a larger common sphere-surface point set."""
    fine=np.empty((len(rays),2*rays.shape[1]-1,3),dtype='f')
    fine[:,::2]=rays;fine[:,1::2]=(rays[:,:-1]+rays[:,1:])*.5
    n=objects.shape[1]
    unit=np.random.default_rng(1).normal(size=(2*n,3))
    unit/=np.linalg.norm(unit,axis=1)[:,None]
    centers=objects[:,0].astype('d')-.12*unit[0]
    larger=(centers[:,None]+.12*unit[None]).astype('f')
    # Original geometry is preserved exactly, including float32 coordinates.
    if not np.allclose(larger[:,:n],objects,atol=2e-6,rtol=0):
        raise ValueError('saved objects do not match the declared sphere sampling')
    larger[:,:n]=objects
    return fine,larger


def serializable(value):
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,dict): return {k:serializable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [serializable(v) for v in value]
    return value


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--scratch',type=Path,default=Path('/opt/geospsr-runtime/common-rhs-128'))
    p.add_argument('--samples',type=int,default=128)
    p.add_argument('--batch',type=int,default=2)
    p.add_argument('--seed',type=int,default=51)
    p.add_argument('--checkpoints',type=int,nargs='+',default=[8,32,128])
    p.add_argument('--validation-rows',type=int,default=32)
    p.add_argument('--retained-control',action='store_true')
    p.add_argument('--coefficient-dir',type=Path,help='optional CPU/disk blocks for later moving-query replay; not a resume protocol')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if min(a.samples,a.batch,a.validation_rows)<1 or a.samples<2:
        p.error('require at least two samples and positive batch/row counts')
    checkpoints=sorted(set(a.checkpoints))
    if not checkpoints or checkpoints[-1]!=a.samples or any(n<2 or n>a.samples or (n%a.batch and n!=a.samples) for n in checkpoints):
        p.error('checkpoints must end at samples and align with batch boundaries')
    a.output.mkdir(parents=True,exist_ok=True)
    if a.coefficient_dir: a.coefficient_dir.mkdir(parents=True,exist_ok=True)
    record={'args':{**vars(a),'scratch':str(a.scratch),'output':str(a.output),
                    'coefficient_dir':None if a.coefficient_dir is None else str(a.coefficient_dir)},
            'complete':False,'seconds':{},'checkpoints':{},'batches':[],
            'scope':'instrumented finite-cube fit/prior/conditioning/query execution; numerical and two-resolution diagnostics, not a matched-quality application speedup',
            'device':cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
            'model':{'training_cutoff':200,'scalar_cutoff':50,'lengthscale':.01,'variance':.1,'noise_variance':.02,'rtol':1e-5,'eps':1e-6,'normalization':'analytic','pack_rhs':True,'centered_hybrid':False}}
    def save():
        (a.output/'results.json').write_text(json.dumps(serializable(record),indent=2)+'\n')
    def timed(key,fn):
        cp.cuda.runtime.deviceSynchronize();start=time.perf_counter()
        value=fn();cp.cuda.runtime.deviceSynchronize()
        record['seconds'][key]=record['seconds'].get(key,0.)+time.perf_counter()-start
        return value
    audit=GpuProcessAudit(cp.cuda.Device().pci_bus_id).start();begin=time.perf_counter()
    try:
        x=timed('input_load',lambda:np.load(a.scratch/'points.npy'))
        y=timed('input_load',lambda:np.load(a.scratch/'normals.npy'))
        geo=np.load(a.scratch/'geometry.npz')
        q,rays,objects,distance=[geo[k] for k in ['query','rays','objects','distance']]
        fine_rays,fine_objects=refined_geometry(rays,objects)
        record['hashes']={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in [('points',x),('normals',y),('query',q),('rays',rays),('objects',objects)]}
        record['N']=len(x)
        model=timed('fit',lambda:HalfCubePosterior(x,y))
        record['offset']=float(model.offset);record['saved_geometry_offset']=float(geo['offset'])
        ids=np.sort(np.random.default_rng(3187).choice(len(x),a.validation_rows,replace=False))
        rows=timed('validation_rows_setup',lambda:finite_covariance_rows(x[ids],x,model.training,dtype='d'))
        record['validation_ids']=ids
        mean=cp.asnumpy(model.mean(q))
        reference=None
        if a.retained_control:
            timed('retained_control_draw',lambda:model.draw(a.samples,seed=a.seed,batch=a.batch,pack_rhs=True))
            reference={'fields':timed('retained_control_queries',lambda:cp.asnumpy(model.samples(q)))}
            for key,pos in [('rays',rays),('collision',objects)]:
                reference[key]=timed('retained_control_queries',lambda:cp.asnumpy(events(model.samples(pos.reshape(-1,3)),pos.shape[:2],key)))
        moments=OnlineMoments();covariance=OnlineMoments(covariance=True)
        counts={k:OnlineEvents() for k in ['rays','collision','fine_rays','fine_collision','ray_refinement','collision_refinement']}
        dimensions={'rays':rays.shape[:2],'collision':objects.shape[:1],
                    'fine_rays':fine_rays.shape[:2],'fine_collision':objects.shape[:1],
                    'ray_refinement':rays.shape[:2],'collision_refinement':objects.shape[:1]}
        family=sum(int(np.prod(v)) for v in dimensions.values())
        maximum_residual=0.;maximum_cached_error=0.
        paired={k:[] for k in ['fields','rays','collision']}
        stream_begin=time.perf_counter()
        with closing(model.draw_batches(a.samples,seed=a.seed,batch=a.batch,pack_rhs=True,
                                       retain_components=True,retain_rhs=True)) as batches:
            for block in batches:
                n=block['stop'];size=n-block['start']
                weights=block['weights'].transpose(1,0,2).reshape(len(x),-1)
                rhs=block['rhs'][:,ids].transpose(1,0,2).reshape(len(ids),-1)
                def residual_check():
                    residual=rows@weights.astype('d')+.02*weights[ids]-rhs
                    return float(cp.max(cp.linalg.norm(residual,axis=0)/cp.linalg.norm(rhs.astype('d'),axis=0)))
                maximum_residual=max(maximum_residual,timed('finite_residual_checks',residual_check))
                del weights,rhs
                cache=timed('query_cache_setup',lambda:CachedFourierEvaluator(block['coefficients'],batch=1))
                fields=timed('field_queries',lambda:cp.asnumpy(cache.evaluate(q)-model.offset))
                direct=timed('direct_query_validation',lambda:cp.asnumpy(model.op.evaluate(q,block['coefficients'])-model.offset))
                scale=np.linalg.norm(direct-mean)
                error=np.linalg.norm(fields-direct)/max(scale,1e-30)
                maximum_cached_error=max(maximum_cached_error,float(error))
                values={}
                for key,pos,kind in [('rays',rays,'rays'),('collision',objects,'collision'),
                                     ('fine_rays',fine_rays,'rays'),('fine_collision',fine_objects,'collision')]:
                    query_values=timed(key+'_queries',lambda:cache.evaluate(pos.reshape(-1,3))-model.offset)
                    values[key]=timed('event_reductions_transfer',lambda:cp.asnumpy(events(query_values,pos.shape[:2],kind)))
                    del query_values
                values['ray_refinement']=values['rays']-values['fine_rays'][:,:,::2]
                values['collision_refinement']=values['fine_collision']-values['collision']
                def accumulate():
                    moments.update(fields);covariance.update(fields[:,:32])
                    for key,value in values.items(): counts[key].update(value)
                timed('online_statistics',accumulate)
                if reference is not None:
                    paired['fields'].append(fields)
                    for key in ['rays','collision']: paired[key].append(values[key])
                if a.coefficient_dir:
                    target=a.coefficient_dir/f'{block["start"]:08d}-{n:08d}.npy'
                    timed('coefficient_export',lambda:np.save(target,cp.asnumpy(block['coefficients'])))
                del cache
                record['batches'].append({**block['record'],'coefficient_bytes':block['coefficients'].nbytes,
                                          'cupy_live_bytes_at_yield':cp.get_default_memory_pool().used_bytes(),
                                          'cupy_pool_bytes_at_yield':cp.get_default_memory_pool().total_bytes()})
                if n in checkpoints:
                    stats={k:v.summary(looks=len(checkpoints),family_size=family) for k,v in counts.items()}
                    scores=view_scores(stats['rays'],distance)
                    record['checkpoints'][str(n)]={'simultaneous_probability_radius':stats['rays']['simultaneous_radius'],
                        'probability_radius_2pp_pass':stats['rays']['simultaneous_radius']<=.02,
                        'ray_refinement_max_change':float(stats['ray_refinement']['probability'].max()),
                        'collision_refinement_max_change':float(stats['collision_refinement']['probability'].max()),
                        'next_view':scores,'uncertain_collision_count':int(np.count_nonzero((stats['collision']['probability']>0)&(stats['collision']['probability']<1))),
                        'elapsed_stream_seconds':time.perf_counter()-stream_begin}
                    arrays={'field_mean':moments.summary()['mean'],'field_variance':moments.summary()['variance'],
                            'field_covariance32':covariance.summary()['covariance'],'mean':mean,'query':q,'distance':distance}
                    for key,stat in stats.items():
                        for item in ['successes','probability','lower','upper']: arrays[key+'_'+item]=stat[item]
                    np.savez_compressed(a.output/f'checkpoint_{n}.npz',**arrays)
                    save();print('checkpoint',n,'radius',stats['rays']['simultaneous_radius'],flush=True)
                del block
        record['seconds']['stream_wall']=time.perf_counter()-stream_begin
        record['maximum_independent_finite_residual']=maximum_residual
        record['maximum_cached_error_over_fluctuations']=maximum_cached_error
        record['finite_residual_gate_pass']=maximum_residual<=2e-5
        record['cache_gate_pass']=maximum_cached_error<5e-4
        record['retained_ensemble_coefficients']=model.sample_coeff is not None
        record['sample_count']=moments.count
        if reference is not None:
            candidate={k:np.concatenate(v) for k,v in paired.items()}
            record['retained_control_quality']={'fields':field_quality(cp.asarray(reference['fields']),cp.asarray(candidate['fields'])),
                'events':quality(reference,candidate,distance)}
            if not record['retained_control_quality']['fields']['pass'] or not record['retained_control_quality']['events']['compatibility_pass']:
                raise RuntimeError('retained-control quality gate failed')
        if not record['finite_residual_gate_pass'] or not record['cache_gate_pass'] or moments.count!=a.samples:
            raise RuntimeError('streaming numerical gate failed')
        record['complete']=True
    except Exception as exc:
        record['failure']=f'{type(exc).__name__}: {exc}';raise
    finally:
        record['seconds']['instrumented_work_wall']=time.perf_counter()-begin
        record['gpu_process_audit']=audit.stop();save()


if __name__=='__main__':
    main()
