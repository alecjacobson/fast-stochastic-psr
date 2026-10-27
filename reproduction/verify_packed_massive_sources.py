"""Replay stored massive-source coefficients and events, without reconditioning."""
import json
from pathlib import Path
import cupy as cp
import numpy as np
from geospsr.spectral import FourierPoisson
from reproduction.common_rhs_integrity import array_digest
from reproduction.paired_applications import events


def main():
    root=Path(__file__).resolve().parents[1];out=root/'reproduction/results/packed_factor_sources/three_million'
    r=json.loads((out/'results.json').read_text());archive=Path('/opt/geospsr-runtime/packed-factor-sources/three_million')
    if not r['complete']:raise ValueError('incomplete run')
    source=root/'reproduction/results/verified_preconditioning/strict/fused16/results.json';old=json.loads(source.read_text())
    if array_digest(np.frombuffer(source.read_bytes(),dtype='u1'))!=r['source_result_sha256']:raise ValueError('source record mismatch')
    if array_digest(np.load('/opt/geospsr-runtime/large-cloud-3000000/points.npy',mmap_mode='r'))!=r['hashes']['points']:raise ValueError('point mismatch')
    hashes=1
    for i,entry in r['hashes'].items():
        if i=='points':continue
        for key,path in [('weights',f'/opt/geospsr-runtime/verified-preconditioning/strict/fused16/{i}_weights.npy'),
                         ('prior',f'/opt/geospsr-runtime/verified-posterior/three_million/{i}_prior.npy')]:
            if array_digest(np.load(path,mmap_mode='r'))!=entry[key]:raise ValueError('draw payload mismatch')
            hashes+=1
    geo=np.load(root/'reproduction/results/verified_posterior/three_million/geometry.npz')
    if any(array_digest(geo[k])!=v for k,v in r['geometry_hashes'].items()):raise ValueError('geometry mismatch')
    q,rays,objects=[geo[k] for k in ['query','rays','objects']]
    positions=np.concatenate([q,rays.reshape(-1,3),objects.reshape(-1,3)]);end=len(q)+np.prod(rays.shape[:2])
    op=FourierPoisson(50);plan=op.plan(2,positions,1);offset=cp.asarray(old['offset'],dtype=op.dtype);errors=[]
    for method,m in r['methods'].items():
        saved=np.load(out/f'{method}_fields.npz')
        for i,digest in enumerate(m['coefficient_hashes']):
            coeff=np.load(archive/f'{method}_{i}_coefficients.npy')
            if array_digest(coeff)!=digest:raise ValueError('coefficient hash mismatch')
            hashes+=1;v=plan.execute(cp.asarray(coeff)).real-offset
            field=cp.asnumpy(v[:,:len(q)]);target=saved['fields'][i:i+1]
            error=float(np.linalg.norm(field-target)/max(np.linalg.norm(target),1e-30));errors.append(error)
            if error>1e-5:raise ValueError('field replay mismatch')
            for kind,pos,sl in [('rays',rays,slice(len(q),end)),('collision',objects,slice(end,None))]:
                if not np.array_equal(cp.asnumpy(events(v[:,sl],pos.shape[:2],kind)),saved[kind][i:i+1]):raise ValueError('event replay mismatch')
    result={'complete':True,'arrays_hashed':hashes,'fields_replayed':len(errors),'max_relative_field_error':max(errors),'events_exact':True,
            'scope':'input/coefficient hashes and field/event replay; no solve or prior regeneration'}
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(result)


if __name__=='__main__':main()
