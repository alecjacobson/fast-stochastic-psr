"""Check archived payloads and replay coefficient fields/events."""
import argparse,json
from pathlib import Path
import cupy as cp
import numpy as np
from geospsr.spectral import FourierPoisson
from reproduction.paired_applications import events
from reproduction.common_rhs_integrity import array_digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--source',type=Path,default=Path('reproduction/results/verified_posterior/three_million'))
    a=p.parse_args();r=json.loads((a.output/'results.json').read_text())
    if not r['complete']:raise ValueError('incomplete run')
    geo=np.load(a.source/'geometry.npz');fields=np.load(a.output/'fields.npz')
    if any(array_digest(geo[k])!=v for k,v in r['source_geometry_hashes'].items()):raise ValueError('geometry hash mismatch')
    if array_digest(np.load(a.archive/'mean_coeff.npy'))!=r['mean_coeff_sha256']:raise ValueError('mean hash mismatch')
    if array_digest(np.frombuffer((a.source/'results.json').read_bytes(),dtype='u1'))!=r['source_record_sha256']:raise ValueError('source record mismatch')
    q,rays,objects=[geo[k] for k in ['query','rays','objects']]
    positions=np.concatenate([q,rays.reshape(-1,3),objects.reshape(-1,3)]);end=len(q)+np.prod(rays.shape[:2])
    op=FourierPoisson(50);offset=cp.asarray(r['offset'],dtype='f');errors=[];count=1
    for d in r['draws']:
        i=d['index']
        for k,v in d['archive_hashes'].items():
            if array_digest(np.load(a.archive/f'{i}_{k}.npy',mmap_mode='r'))!=v:raise ValueError('archive hash mismatch')
            count+=1
        coeff=cp.asarray(np.load(a.archive/f'{i}_coefficients.npy'));values=op.evaluate(positions,coeff)-offset
        current=cp.asnumpy(values[:,:len(q)]);saved=fields['fields'][i:i+1]
        error=float(np.linalg.norm(current-saved)/max(np.linalg.norm(saved),1e-30));errors.append(error)
        if error>1e-5:raise ValueError('field replay mismatch')
        for kind,pos,sl in [('rays',rays,slice(len(q),end)),('collision',objects,slice(end,None))]:
            if not np.array_equal(cp.asnumpy(events(values[:,sl],pos.shape[:2],kind)),fields[kind][i:i+1]):raise ValueError('event replay mismatch')
    result={'complete':True,'archive_arrays_checked':count,'fields_replayed':len(errors),'maximum_relative_field_error':max(errors),
            'events_exact':True,'scope':'archive hashes and coefficient replay; does not re-execute residual checks or prior generation'}
    (a.output/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(result)


if __name__=='__main__':main()
