"""Fixed archived coherent draws; independently verified conditioning costs."""
import argparse,json,time
from pathlib import Path
import cupy as cp
import numpy as np
from reproduction.half_posterior import HalfCubePosterior
from reproduction.common_rhs_integrity import array_digest
from reproduction.posterior import sync
from reproduction.gpu_audit import GpuProcessAudit
from reproduction.paired_applications import events,field_quality,quality


def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['plain','float32','fused16'],required=True)
    p.add_argument('--source',type=Path,default=Path('reproduction/results/verified_posterior/three_million'))
    p.add_argument('--source-archive',type=Path,default=Path('/opt/geospsr-runtime/verified-posterior/three_million'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--archive',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);a.archive.mkdir(parents=True,exist_ok=True)
    source=json.loads((a.source/'results.json').read_text())
    if not source['complete'] or not source['numerical_gate_pass']:raise ValueError('invalid source run')
    r={'complete':False,'method':a.method,'N':source['N'],'samples':8,'seed':701,'batch':1,
       'model':source['model'],'seconds':{},'draws':[],
       'scope':'same archived coherent RHS/prior; verified conditional solves, aggregation and queries; excludes prior generation',
       'reference':'archived tighter same-finite-cube numerical control from verified_posterior/three_million',
       'source_hashes':source['source_hashes'],'source_record_sha256':array_digest(np.frombuffer((a.source/'results.json').read_bytes(),dtype='u1')),
       'source_geometry_hashes':source['geometry_hashes'],
       'device':cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
       'compilation_cache':'existing CuPy disk cache; no untimed solve warmup; first use included'}
    def save():(a.output/'results.json').write_text(json.dumps(r,indent=2)+'\n')
    def timed(label,fn):
        sync();begin=time.perf_counter();value=fn();sync();seconds=time.perf_counter()-begin
        print(label,seconds,flush=True);return value,seconds
    audit=GpuProcessAudit(cp.cuda.Device().pci_bus_id).start();begin=time.perf_counter()
    try:
        cloud=Path('/opt/geospsr-runtime/large-cloud-3000000')
        x,y=[np.load(cloud/(k+'.npy')) for k in ['points','normals']]
        if array_digest(x)!=r['source_hashes']['points'] or array_digest(y)!=r['source_hashes']['normals']:
            raise ValueError('input hashes differ')
        geo=np.load(a.source/'geometry.npz');q,rays,objects,distance=[geo[k] for k in ['query','rays','objects','distance']]
        if any(array_digest(geo[k])!=v for k,v in r['source_geometry_hashes'].items()):raise ValueError('geometry hashes differ')
        positions=np.concatenate([q,rays.reshape(-1,3),objects.reshape(-1,3)])
        reference=np.load(a.source/'fields.npz')
        model,r['seconds']['fit']=timed('fit',lambda:HalfCubePosterior(x,y,refine_solves=True))
        r['mean_solve']=model.mean_solve;r['offset']=float(model.offset)
        np.save(a.archive/'mean_coeff.npy',cp.asnumpy(model.mean_coeff));r['mean_coeff_sha256']=array_digest(cp.asnumpy(model.mean_coeff))
        r['seconds']['coarse_setup']=0.
        if a.method!='plain':
            coarse,r['seconds']['coarse_setup']=timed('coarse_setup',lambda:model.prepare_conditioning_preconditioner(
                rank=1024,seed=41,row_batch=8192,basis_dtype='float16' if a.method=='fused16' else 'float32',
                recompute_features=True,fused_apply=a.method=='fused16'))
            r['preconditioner']=coarse.record
        save()
        output={k:[] for k in ['fields','rays','collision']}
        with model._draw_context(pack_rhs=True,preconditioned=a.method!='plain'):
            for original in source['draws']:
                i=original['index'];entry={'index':i}
                rhs_cpu=np.load(a.source_archive/f'{i}_rhs.npy');prior_cpu=np.load(a.source_archive/f'{i}_prior.npy')
                entry['rhs_sha256']=array_digest(rhs_cpu);entry['prior_sha256']=array_digest(prior_cpu)
                if any(entry[k+'_sha256']!=original['hashes'][k] for k in ['rhs','prior']):raise ValueError('archived draw mismatch')
                (rhs,prior),entry['transfer_seconds']=timed('transfer',lambda:(cp.asarray(rhs_cpu),cp.asarray(prior_cpu)))
                (weights,info),entry['solve_seconds']=timed('solve',lambda:model._solve_draw(model.K,rhs,False))
                entry['solve']=info;entry['spectrum']=model._spectrum_record()
                if not info['converged']:raise RuntimeError('verified solve failed')
                coefficients,entry['aggregation_seconds']=timed('aggregate',lambda:model.op.aggregate(model.x,weights)+prior)
                def query():
                    values=model.op.evaluate(positions,coefficients)-model.offset
                    end=len(q)+np.prod(rays.shape[:2])
                    return {'fields':cp.asnumpy(values[:,:len(q)]),
                            'rays':cp.asnumpy(events(values[:,len(q):end],rays.shape[:2],'rays')),
                            'collision':cp.asnumpy(events(values[:,end:],objects.shape[:2],'collision'))}
                values,entry['query_seconds']=timed('query',query)
                for k,v in values.items():output[k].append(v)
                entry['archive_hashes']={}
                for name,value in [('weights',weights),('coefficients',coefficients)]:
                    cpu=cp.asnumpy(value);np.save(a.archive/f'{i}_{name}.npy',cpu)
                    entry['archive_hashes'][name]=array_digest(cpu)
                r['draws'].append(entry);save();print('draw',i,'done',flush=True)
                del rhs,prior,weights,coefficients
        output={k:np.concatenate(v) for k,v in output.items()}
        np.savez_compressed(a.output/'fields.npz',**output)
        r['field_quality']=field_quality(reference['reference_fields'],output['fields'])
        r['event_quality']=quality({k:reference['reference_'+k] for k in ['rays','collision']},output,distance)
        r['quality_pass']=bool(r['field_quality']['pass'] and r['event_quality']['compatibility_pass'] and r['event_quality']['uncertainty_coverage_pass'])
        r['complete']=True
    except Exception as exc:r['failure']=f'{type(exc).__name__}: {exc}';raise
    finally:
        r['seconds']['instrumented_work_wall']=time.perf_counter()-begin;r['gpu_process_audit']=audit.stop();save()


if __name__=='__main__':main()
