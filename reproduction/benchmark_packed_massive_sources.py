"""All-3M-point source aggregation of eight archived coherent posterior fields."""
import json,time
from pathlib import Path
import cupy as cp
import numpy as np
from geospsr.spectral import FourierPoisson
from geospsr.packed_sources import PackedRealSourcePlan
from reproduction.posterior import sync
from reproduction.paired_applications import events,field_quality,quality
from reproduction.common_rhs_integrity import array_digest
from reproduction.gpu_audit import GpuProcessAudit


def main():
    root=Path(__file__).resolve().parents[1]
    out=root/'reproduction/results/packed_factor_sources/three_million';out.mkdir(exist_ok=True)
    archive=Path('/opt/geospsr-runtime/packed-factor-sources/three_million');archive.mkdir(exist_ok=True)
    src=root/'reproduction/results/verified_preconditioning/strict/fused16'
    old=json.loads((src/'results.json').read_text());truth=np.load(src/'fields.npz')
    srcarchive=Path('/opt/geospsr-runtime/verified-preconditioning/strict/fused16')
    priorarchive=Path('/opt/geospsr-runtime/verified-posterior/three_million')
    if not old['complete'] or not old['quality_pass']:raise ValueError('invalid source')
    r={'complete':False,'N':old['N'],'draws':8,'batch':1,'repetitions':3,'methods':{},'hashes':{},
       'model':old['model'],'scalar_query_upsampfac':2.,
       'source_result_sha256':array_digest(np.frombuffer((src/'results.json').read_bytes(),dtype='u1')),
       'scope':'prepared coherent weights, source aggregation and queries only; no fitting/prior/solve cost'}
    def timed(fn):
        sync();start=time.perf_counter();v=fn();sync();return v,time.perf_counter()-start
    audit=GpuProcessAudit(cp.cuda.Device().pci_bus_id).start()
    try:
        x=np.load('/opt/geospsr-runtime/large-cloud-3000000/points.npy')
        if array_digest(x)!=old['source_hashes']['points'] or len(x)!=3000000:raise ValueError('point mismatch')
        r['hashes']['points']=array_digest(x)
        geo=np.load(root/'reproduction/results/verified_posterior/three_million/geometry.npz')
        if any(array_digest(geo[k])!=v for k,v in old['source_geometry_hashes'].items()):raise ValueError('geometry mismatch')
        r['geometry_hashes']=old['source_geometry_hashes']
        q,rays,objects,distance=[geo[k] for k in ['query','rays','objects','distance']]
        positions=np.concatenate([q,rays.reshape(-1,3),objects.reshape(-1,3)]);end=len(q)+np.prod(rays.shape[:2])
        weights=[];priors=[]
        for d in old['draws']:
            i=d['index'];w=np.load(srcarchive/f'{i}_weights.npy');p=np.load(priorarchive/f'{i}_prior.npy')
            if array_digest(w)!=d['archive_hashes']['weights'] or array_digest(p)!=d['prior_sha256']:raise ValueError('draw mismatch')
            r['hashes'][str(i)]={'weights':array_digest(w),'prior':array_digest(p)};weights.append(w);priors.append(p)
        (points,weights,priors),r['transfer_seconds']=timed(lambda:(cp.asarray(x),[cp.asarray(w) for w in weights],[cp.asarray(p) for p in priors]))
        op=FourierPoisson(50);offset=cp.asarray(old['offset'],dtype=op.dtype)
        query_plan,r['query_setup_seconds']=timed(lambda:op.plan(2,positions,1))
        for name in ['generic','packed']:
            plan,setup=timed(lambda:op.plan(1,points,3) if name=='generic' else PackedRealSourcePlan(op,points,3))
            method={'source_setup_seconds':setup,'passes':[],'coefficient_hashes':[]};r['methods'][name]=method
            for rep in range(3):
                output={k:[] for k in ['fields','rays','collision']};agg=queries=0.
                for i,(w,prior) in enumerate(zip(weights,priors)):
                    def aggregate():
                        modes=plan.execute(cp.ascontiguousarray(w.T,dtype=op.cdtype)).reshape(1,3,*op.shape)
                        return cp.ascontiguousarray(-1j*(modes*op.multiplier[None]).sum(1))+prior
                    coefficients,seconds=timed(aggregate);agg+=seconds
                    def evaluate():
                        v=query_plan.execute(coefficients).real-offset
                        return {'fields':cp.asnumpy(v[:,:len(q)]),'rays':cp.asnumpy(events(v[:,len(q):end],rays.shape[:2],'rays')),
                                'collision':cp.asnumpy(events(v[:,end:],objects.shape[:2],'collision'))}
                    values,seconds=timed(evaluate);queries+=seconds
                    for k,v in values.items():output[k].append(v)
                    if rep==2:
                        a=cp.asnumpy(coefficients);np.save(archive/f'{name}_{i}_coefficients.npy',a);method['coefficient_hashes'].append(array_digest(a))
                output={k:np.concatenate(v) for k,v in output.items()}
                fq=field_quality(truth['fields'],output['fields']);eq=quality(truth,output,distance)
                exact=all(np.array_equal(truth[k],output[k]) for k in ['rays','collision'])
                method['passes'].append({'aggregation_seconds':agg,'query_seconds':queries,'field_quality':fq,'event_quality':eq,
                                         'events_exact':exact,'quality_pass':bool(fq['pass'] and exact and eq['compatibility_pass'] and eq['uncertainty_coverage_pass'])})
                print(name,rep,agg,queries,fq['pass'],exact,flush=True)
            np.savez_compressed(out/f'{name}_fields.npz',**output)
            del plan
        r['complete']=True
    except Exception as exc:r['failure']=str(exc);raise
    finally:r['gpu_process_audit']=audit.stop();(out/'results.json').write_text(json.dumps(r,indent=2)+'\n')


if __name__=='__main__':main()
