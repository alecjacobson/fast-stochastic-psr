"""Nested geometry and strong cached/direct queries on a checked draw archive."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import cupy as cp
import numpy as np
from geospsr.cached_spectral import CachedFourierEvaluator
from geospsr.packed_queries import PackedDirectEvaluator
from reproduction.coefficient_archive import CoefficientArchive,digest
from reproduction.gpu_audit import GpuProcessAudit
from reproduction.nested_queries import nested_rays,nested_spheres,discrepancy_bound
from reproduction.stream_applications import serializable
from reproduction.streaming_statistics import view_scores


def level_events(values,ray_grids,sphere_grids):
    b=len(values);ray_count,steps=ray_grids[-1].shape[:2]
    split=ray_count*steps
    ray=values[:,:split].reshape(b,ray_count,steps)
    sphere=values[:,split:].reshape(b,*sphere_grids[-1].shape[:2])
    result={}
    levels=len(ray_grids)-1
    for level in range(levels+1):
        stride=2**(levels-level)
        result['rays',level]=np.logical_and.accumulate(ray[:,:,::stride]>0,axis=2)
        result['sphere',level]=(sphere[:,:,:sphere_grids[level].shape[1]]<=0).any(2)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--source-results',type=Path,required=True,help='completed generation result directory, for replay event-count checks')
    p.add_argument('--levels',type=int,default=4)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if not 1<=a.levels<=8: p.error('levels must be 1..8')
    a.output.mkdir(parents=True,exist_ok=True)
    record={'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
            'complete':False,'seconds':{},'scope':'prepared Fourier draw replay; no conditioning or source-data fitting in query timers',
            'device':cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}
    def save(): (a.output/'results.json').write_text(json.dumps(serializable(record),indent=2)+'\n')
    def timed(key,fn):
        cp.cuda.runtime.deviceSynchronize();begin=time.perf_counter()
        result=fn();cp.cuda.runtime.deviceSynchronize()
        record['seconds'][key]=record['seconds'].get(key,0.)+time.perf_counter()-begin
        return result
    audit=GpuProcessAudit(cp.cuda.Device().pci_bus_id).start();begin=time.perf_counter()
    try:
        archive=timed('archive_metadata',lambda:CoefficientArchive(a.archive))
        record['archive_manifest_sha256']=digest(a.archive/'manifest.json')
        record['source_metadata']=archive.record['metadata']
        source=json.loads((a.source_results/'results.json').read_text())
        if not source['complete'] or source['sample_count']!=archive.record['count']:
            raise ValueError('source generation is incomplete or has a different sample count')
        if source['hashes']!=archive.record['metadata']['source_hashes'] or source['offset']!=archive.record['metadata']['offset']:
            raise ValueError('source generation provenance differs from archive')
        if (source['model']!=archive.record['metadata']['model'] or
                any(source['args'][key]!=archive.record['metadata'][key] for key in ['seed','batch'])):
            raise ValueError('source model or random schedule differs from archive')
        if not all(archive.record['validation'].get(k) for k in ['finite_residual_gate_pass','cache_gate_pass']):
            raise ValueError('archive lacks producer numerical validation')
        geometry=timed('archive_metadata',archive.geometry)
        for key in ['query','rays','objects']:
            if hashlib.sha256(geometry[key].tobytes()).hexdigest()!=source['hashes'][key]:
                raise ValueError('archived geometry differs from producer hashes')
        mean_coeff=timed('archive_metadata',archive.mean)
        offset=float(archive.record['metadata']['offset'])
        if not np.isfinite(offset): raise ValueError('offset must be finite')
        ray_grids,distances=nested_rays(geometry['rays'],geometry['distance'],a.levels)
        sphere_grids=nested_spheres(geometry['objects'],a.levels)
        points=np.concatenate([ray_grids[-1].reshape(-1,3),sphere_grids[-1].reshape(-1,3)])
        record['ray_steps']=[g.shape[1] for g in ray_grids]
        record['sphere_points']=[g.shape[1] for g in sphere_grids]
        direct=timed('direct_plan_setup',lambda:PackedDirectEvaluator(points,archive.record['shape']))
        for n in sorted({1,*[b['stop']-b['start'] for b in archive.record['blocks']]}):
            timed('direct_plan_setup',lambda:direct.prepare(n))
        mean=timed('mean_query_validation',lambda:direct.evaluate(cp.asarray(mean_coeff))[0]-offset)
        totals={backend:{(kind,level):np.zeros((len(ray_grids[level]),ray_grids[level].shape[1]) if kind=='rays'
                        else (len(sphere_grids[level]),),dtype=np.int64)
                        for kind in ['rays','sphere'] for level in range(a.levels+1)} for backend in ['cached','direct']}
        delta_totals={(kind,level):np.zeros_like(totals['cached'][kind,level]) for kind in ['rays','sphere'] for level in range(a.levels)}
        disagreement=[];backend_disagreement=[];max_error=0.;count=0
        blocks=archive.blocks()
        while True:
            try: block=timed('archive_block_io_hash',lambda:next(blocks))
            except StopIteration: break
            coefficients=timed('coefficient_transfer',lambda:cp.asarray(block['coefficients']))
            cache=timed('cached_grid_setup',lambda:CachedFourierEvaluator(coefficients,batch=1))
            cached_values=timed('cached_query',lambda:cache.evaluate(points)-offset)
            direct_values=timed('direct_query_with_packing',lambda:direct.evaluate(coefficients)-offset)
            def field_error():
                return float(cp.linalg.norm((cached_values-direct_values).astype('d'))/
                             cp.maximum(cp.linalg.norm((direct_values-mean[None]).astype('d')),1e-30))
            error=timed('field_validation',field_error)
            if not np.isfinite(error): raise RuntimeError('nonfinite query error')
            max_error=max(max_error,error)
            pairs={}
            for backend,v in [('cached',cached_values),('direct',direct_values)]:
                raw=timed('field_transfer',lambda:cp.asnumpy(v))
                pairs[backend]=timed('event_reductions',lambda:level_events(raw,ray_grids,sphere_grids))
                for key,value in pairs[backend].items(): totals[backend][key]+=value.sum(0,dtype=np.int64)
            any_difference=[]
            for kind in ['rays','sphere']:
                for level in range(a.levels):
                    low=pairs['cached'][kind,level]
                    high=pairs['cached'][kind,level+1]
                    if kind=='rays':
                        high=high[:,:,::2];delta=low.astype(np.int8)-high.astype(np.int8)
                    else: delta=high.astype(np.int8)-low.astype(np.int8)
                    if not np.all((delta==0)|(delta==1)): raise RuntimeError('nested events violate inclusion')
                    delta_totals[kind,level]+=delta.sum(0,dtype=np.int64)
                    any_difference.append(delta.reshape(len(delta),-1).any(1))
            disagreement.append(np.stack(any_difference,axis=1))
            backend_disagreement.append(np.any(np.stack([
                (pairs['cached'][key]!=pairs['direct'][key]).reshape(len(coefficients),-1).any(1)
                for key in pairs['cached']],axis=1),axis=1))
            count+=len(coefficients)
            del block,coefficients,cache,cached_values,direct_values,pairs,raw
        if count!=archive.record['count']: raise RuntimeError('incomplete replay')
        disagreement=np.concatenate(disagreement);backend_disagreement=np.concatenate(backend_disagreement)
        record['samples']=count;record['maximum_cached_error_over_fluctuations']=max_error
        record['field_gate_pass']=max_error<5e-4
        max_probability_error=max(float(np.abs(totals['cached'][key]-totals['direct'][key]).max()/count) for key in totals['cached'])
        record['maximum_backend_probability_difference']=max_probability_error
        record['backend_event_gate_pass']=max_probability_error<=.02
        record['backend_any_disagreeing_draws']=int(backend_disagreement.sum())
        record['refinement']=[];index=0
        arrays={'per_draw_refinement_disagreement':disagreement,'backend_any_disagreement':backend_disagreement}
        for kind in ['rays','sphere']:
            for level in range(a.levels):
                bound=discrepancy_bound(int(disagreement[:,index].sum()),count,comparisons=2*a.levels)
                record['refinement'].append({'kind':kind,'level':level,
                    'coarse_size':record['ray_steps' if kind=='rays' else 'sphere_points'][level],
                    'fine_size':record['ray_steps' if kind=='rays' else 'sphere_points'][level+1],
                    'maximum_empirical_probability_change':float(delta_totals[kind,level].max()/count),**bound})
                arrays[f'{kind}_{level}_refinement_counts']=delta_totals[kind,level];index+=1
        record['next_view']=[]
        for level,distance in enumerate(distances):
            probability=totals['cached']['rays',level]/count
            score=view_scores({'probability':probability,'lower':probability,'upper':probability},distance)
            record['next_view'].append({'level':level,'steps':len(distance),'selected':score['selected'],'scores':score['scores']})
            arrays[f'distance_{level}']=distance
        final=record['next_view'][-1]['scores']
        for entry in record['next_view']:
            entry['empirical_regret_vs_finest']=float(max(final)-final[entry['selected']])
        for backend,values in totals.items():
            for (kind,level),value in values.items(): arrays[f'{backend}_{kind}_{level}_counts']=value
        original=np.load(a.source_results/f'checkpoint_{count}.npz')
        if not np.array_equal(original['distance'],geometry['distance']):
            raise ValueError('distance grid differs from producer checkpoint')
        matches={}
        for kind,name in [('rays','rays'),('sphere','collision')]:
            for level,prefix in [(0,''),(1,'fine_')]:
                matches[f'{kind}_{level}']=bool(np.array_equal(totals['cached'][kind,level],original[prefix+name+'_successes']))
        record['source_event_replay_exact']=matches
        np.savez_compressed(a.output/'counts.npz',**arrays)
        if not all(matches.values()) or not record['field_gate_pass'] or not record['backend_event_gate_pass']:
            raise RuntimeError('replay/source or numerical query gate failed')
        record['complete']=True
    except Exception as exc:
        record['failure']=f'{type(exc).__name__}: {exc}';raise
    finally:
        record['seconds']['work_wall']=time.perf_counter()-begin
        record['gpu_process_audit']=audit.stop();save()
    print(json.dumps(serializable({k:record[k] for k in ['samples','refinement','next_view','maximum_backend_probability_difference']})),flush=True)


if __name__=='__main__': main()
