"""Replay published query fields/events and check the local coefficient archive."""
import argparse,json
from pathlib import Path
import cupy as cp
import numpy as np
from geospsr.spectral import FourierPoisson
from reproduction.common_rhs_integrity import array_digest
from reproduction.paired_applications import events,field_quality,quality


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--archive',type=Path,required=True)
    a=p.parse_args();r=json.loads((a.output/'results.json').read_text())
    if not r['complete']:raise ValueError('incomplete run')
    geo=np.load(a.output/'geometry.npz');saved=np.load(a.output/'fields.npz')
    for k,digest in r['geometry_hashes'].items():
        if array_digest(geo[k])!=digest:raise ValueError('geometry hash mismatch')
    for name,digest in r['mean_hashes'].items():
        if array_digest(np.load(a.archive/(name+'.npy'),mmap_mode='r'))!=digest:raise ValueError('mean hash mismatch')
    q,rays,objects,distance=[geo[k] for k in ['query','rays','objects','distance']]
    positions=np.concatenate([q,rays.reshape(-1,3),objects.reshape(-1,3)])
    ray_end=len(q)+np.prod(rays.shape[:2])
    ops={'production':FourierPoisson(50),'reference':FourierPoisson(50,dtype='float64',eps=1e-10)}
    errors=[];hashes=0
    for draw in r['draws']:
        i=draw['index']
        for name,digest in draw['hashes'].items():
            if array_digest(np.load(a.archive/f'{i}_{name}.npy',mmap_mode='r'))!=digest:raise ValueError('archive hash mismatch')
            hashes+=1
        for side,op in ops.items():
            coeff=cp.asarray(np.load(a.archive/f'{i}_{side}_coeff.npy'))
            # Recreate original offset precision, including float32 subtraction.
            offset=cp.asarray(r['offsets'][side],dtype=op.dtype)
            values=op.evaluate(positions,coeff)-offset
            fields=cp.asnumpy(values[:,:len(q)])
            original=saved[side+'_fields'][i:i+1]
            relative=float(np.linalg.norm(fields-original)/max(np.linalg.norm(original),1e-30))
            if relative>1e-5:raise ValueError('field replay mismatch')
            for name,pos,sl in [('rays',rays,slice(len(q),ray_end)),('collision',objects,slice(ray_end,None))]:
                actual=cp.asnumpy(events(values[:,sl],pos.shape[:2],name))
                if not np.array_equal(actual,saved[side+'_'+name][i:i+1]):raise ValueError('event replay mismatch')
            errors.append(relative)
    fq=field_quality(saved['reference_fields'],saved['production_fields'])
    eq=quality({k:saved['reference_'+k] for k in ['rays','collision']},
               {k:saved['production_'+k] for k in ['rays','collision']},distance)
    if fq!=r['field_quality'] or eq!=r['event_quality']:raise ValueError('quality summary mismatch')
    result={'complete':True,'archive_arrays_checked':hashes+len(r['mean_hashes']),
            'field_replays':len(errors),'maximum_relative_replay_error':max(errors),
            'event_replays_exact':True,'quality_summary_replayed':True,
            'scope':'hashes and coefficient-to-field/event replay; conditioning residuals and prior law not re-executed'}
    (a.output/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(result)


if __name__=='__main__':main()
