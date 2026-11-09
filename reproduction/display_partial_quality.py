"""Matched display-grid convergence using saved full-posterior mean coefficients."""
import argparse
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from scipy.spatial import cKDTree

from geospsr.nonperiodic import FreeSpacePoissonMatern
from reproduction.partial_quality import OUT
from reproduction.partial_occupancy import load
from reproduction.stanford_partial_data import CACHE, digest


def run(args):
    import cupy as cp
    import igl
    from skimage.measure import marching_cubes
    from reproduction.partial_direct_gpu import cross_rhs
    from reproduction.compute_partial_figures import mean_field
    from reproduction.gpu_audit import GpuProcessAudit
    source=Path(args.source);record=json.loads(source.read_text());config=record['arguments']
    assert record['status']=='completed' and record['gpu_audit']['sampled_isolation_pass']
    assert digest(source.with_suffix('.npz'))==record['field_sha256']
    fields=np.load(source.with_suffix('.npz'))
    path=OUT/f'display-{args.tag}-{args.resolution}.json'
    if path.exists():raise FileExistsError(path)
    x,_,_,_=load(config['group'],config['n'])
    contacts=np.load(OUT/(config['group']+'-panel.npz'))['contacts'][:config['contacts']]
    model=FreeSpacePoissonMatern(config['ell'],1/3)
    posterior=SimpleNamespace(model=model,x=cp.asarray(x),contacts=contacts)
    coefficients=cp.asarray(fields['coefficients']),fields['contact_weights']
    ref=np.load(CACHE/'reference.npz');v=ref['vertices'];f=ref['faces']
    dims=[args.resolution,round(args.resolution*1.2),args.resolution]
    axes=[np.linspace(v[:,d].min()-.1,v[:,d].max()+.1,r) for d,r in enumerate(dims)]
    q=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,3)
    cross_rhs(model,contacts[:2],x[:2]);cp.cuda.Device().synchronize()
    audit=GpuProcessAudit(cp.cuda.Device().pci_bus_id,interval=.5,cuda_device=cp.cuda.Device().id).start();start=perf_counter()
    mean=mean_field(posterior,q,coefficients).reshape(dims)
    measured=audit.stop();seconds=perf_counter()-start
    vertices,faces,_,_=marching_cubes(mean,level=0.,gradient_direction='ascent',spacing=[a[1]-a[0] for a in axes])
    vertices+=np.array([a[0] for a in axes])
    edges=np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1)
    _,counts=np.unique(edges,axis=0,return_counts=True)
    reference_distance=np.sqrt(igl.point_mesh_squared_distance(v,vertices,faces)[0])
    reverse_distance=np.sqrt(igl.point_mesh_squared_distance(vertices,v,f)[0])
    coverage=cKDTree(x).query(v)[0]
    stats={}
    for label,values in [('observed',reference_distance[coverage<.025]),('occluded',reference_distance[coverage>.075]),
        ('all',reference_distance),('mean_to_reference',reverse_distance)]:
        stats[label]=dict(count=len(values),mae=float(values.mean()),median=float(np.median(values)),p95=float(np.quantile(values,.95)))
    np.savez_compressed(path.with_suffix('.npz'),mean=mean,vertices=vertices,faces=faces,axis_x=axes[0],axis_y=axes[1],axis_z=axes[2],
        reference_distance=reference_distance,reverse_distance=reverse_distance)
    result=dict(source_record=str(source),source_record_sha256=digest(source),source_sha256=digest(__file__),
        source_fields_sha256=digest(source.with_suffix('.npz')),grid=dims,queries=len(q),mean_only_seconds=seconds,
        gpu_audit=measured,vertices=len(vertices),faces=len(faces),boundary_edges=int((counts==1).sum()),statistics=stats,
        field_sha256=digest(path.with_suffix('.npz')),scope='Mean-only display work excluded from benchmark; vertex-to-triangle distances to same-acquisition reference, no repair/capping')
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('DISPLAY',args.tag,dims,seconds,stats,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--tag',required=True)
    p.add_argument('--resolution',type=int,choices=[80,120],required=True);run(p.parse_args())
