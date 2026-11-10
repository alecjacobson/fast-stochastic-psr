"""Uniform-grid display with exact all-pairs means at every grid node."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.spatial import cKDTree

from geospsr.nonperiodic import FreeSpacePoissonMatern
from reproduction.observed_fidelity import OUT, OLD, write
from reproduction.partial_occupancy import load
from reproduction.stanford_partial_data import CACHE, digest


def run(args):
    import cupy as cp
    import igl
    from skimage.measure import marching_cubes
    from reproduction.fused_mean import FusedMean
    from reproduction.gpu_audit import GpuProcessAudit
    source=Path(args.source);r=json.loads(source.read_text());field=np.load(source.with_suffix('.npz'))
    assert r['status']=='completed' and r['gpu_audit']['sampled_isolation_pass']
    assert r['field_sha256']==digest(source.with_suffix('.npz'))
    config=r.get('configuration',r['arguments']);group=r['arguments']['group'];n=r['arguments']['n']
    x,_,_,_=load(group,n)
    panel=dict(np.load((OUT if 'configuration' in r else OLD)/(group+'-panel.npz')))
    contacts=panel['contacts'][:config['contacts']];model=FreeSpacePoissonMatern(config['ell'],1/3)
    path=OUT/f'display-{args.tag}-{args.resolution}.json'
    if path.exists():raise FileExistsError(path)
    source_hashes={p:digest(p) for p in [__file__,'reproduction/fused_mean.py']}
    for p,h in source_hashes.items():(OUT/'sources'/(h+'.py')).write_bytes(Path(p).read_bytes())
    evaluator=FusedMean(model,x,contacts,field['coefficients'],field['contact_weights'])
    difference=float(np.max(abs(evaluator(panel['queries'])-field['mean'])))
    if difference>1e-10:raise ValueError(f'fused evaluator differs from saved independent moments: {difference}')
    ref=np.load(CACHE/'reference.npz');v,f=ref['vertices'],ref['faces'];dims=[args.resolution,round(1.2*args.resolution),args.resolution]
    axes=[np.linspace(v[:,d].min()-.1,v[:,d].max()+.1,size) for d,size in enumerate(dims)]
    volume=np.empty(dims,dtype=np.float32)
    yz=np.stack(np.meshgrid(axes[1],axes[2],indexing='ij'),axis=-1).reshape(-1,2)
    audit=GpuProcessAudit(cp.cuda.Device().pci_bus_id,interval=.5,cuda_device=cp.cuda.Device().id).start();start=perf_counter()
    for i,a in enumerate(axes[0]):
        q=np.column_stack((np.full(len(yz),a),yz));volume[i]=evaluator(q).reshape(dims[1:])
        if (i+1)%20==0:print('DISPLAY',args.tag,i+1,dims[0],perf_counter()-start,flush=True)
    elapsed=perf_counter()-start;measured=audit.stop()
    vertices,faces,_,_=marching_cubes(volume,level=0,gradient_direction='ascent',spacing=[a[1]-a[0] for a in axes])
    vertices+=np.array([a[0] for a in axes])
    edges=np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1)
    _,multiplicity=np.unique(edges,axis=0,return_counts=True)
    distance=np.sqrt(igl.point_mesh_squared_distance(v,vertices,faces)[0]);reverse=np.sqrt(igl.point_mesh_squared_distance(vertices,v,f)[0])
    input_distance=np.sqrt(igl.point_mesh_squared_distance(x,vertices,faces)[0]);coverage=cKDTree(x).query(v)[0]
    stats={}
    for label,values in [('observed',distance[coverage<.025]),('occluded',distance[coverage>.075]),
                         ('mean_to_reference',reverse),('input_to_mean',input_distance)]:
        stats[label]=dict(count=len(values),median=float(np.median(values)),p95=float(np.quantile(values,.95)),mae=float(np.mean(values)))
    # Retain meshes and quantitative fields; the uniform volume is regenerable.
    np.savez_compressed(path.with_suffix('.npz'),vertices=vertices,faces=faces,reference_distance=distance,
                        reverse_distance=reverse,input_distance=input_distance)
    write(path,dict(source_record=str(source),source_record_sha256=digest(source),field_sha256=digest(path.with_suffix('.npz')),
        source_sha256=source_hashes,grid=dims,queries=int(np.prod(dims)),mean_seconds=elapsed,gpu_audit=measured,
        saved_mean_agreement=difference,vertices=len(vertices),faces=len(faces),boundary_edges=int((multiplicity==1).sum()),
        statistics=stats,scope='Every uniform grid node evaluated by float64 all-pairs sum; float32 extraction storage. No narrow band, truncation, repair, or capping. Display costs are not task timings.'))
    print('DISPLAY DONE',args.tag,args.resolution,elapsed,stats,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--tag',required=True)
    p.add_argument('--resolution',type=int,choices=[160,240,320],required=True);run(p.parse_args())
