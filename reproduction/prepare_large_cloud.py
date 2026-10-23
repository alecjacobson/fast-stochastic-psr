"""Prepare a distinct larger scorpion scan on CPU; no reconstruction claim."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import numpy as np
from reproduction.million_teaser import visible_cloud,ROOT
from reproduction.common_rhs_integrity import array_digest


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--points',type=int,required=True)
    p.add_argument('--seed',type=int,default=2718)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.points<1:p.error('point count must be positive')
    # A generated input is immutable; never overwrite an existing experiment.
    a.output.mkdir(parents=True,exist_ok=False)
    record={'complete':False,'point_count':a.points,'seed':a.seed,
            'scope':'synthetic area-uniform visible scorpion observations; input preparation only',
            'mesh_sha256':hashlib.sha256((ROOT/'experiments/meshes/scorpion.obj').read_bytes()).hexdigest(),
            'generator_sha256':hashlib.sha256((ROOT/'reproduction/million_teaser.py').read_bytes()).hexdigest(),
            'camera_positions':(np.array([[0,-2,0],[1,1,1]])*np.pi).tolist(),
            'coordinate_noise_std':.02,'normal_noise_std':.02,
            'versions':{name:importlib.metadata.version(name) for name in ['numpy','gpytoolbox','libigl']}}
    begin=time.perf_counter()
    try:
        x,y,clean,_,_=visible_cloud(a.points,a.seed)
        assert x.shape==y.shape==clean.shape==(a.points,3)
        assert all(np.isfinite(v).all() and v.dtype==np.dtype('float32') for v in [x,y,clean])
        unique=len(np.unique(x,axis=0));record['distinct_observed_positions']=unique
        if unique!=len(x):raise RuntimeError('observed positions are not all distinct')
        record['arrays']={}
        for name,values in [('points',x),('normals',y),('clean_points',clean)]:
            np.save(a.output/(name+'.npy'),values)
            loaded=np.load(a.output/(name+'.npy'),mmap_mode='r',allow_pickle=False)
            expected=array_digest(values)
            if array_digest(loaded)!=expected:raise RuntimeError('saved input hash differs')
            record['arrays'][name]={'shape':list(values.shape),'dtype':values.dtype.str,
                                   'payload_sha256':expected,'bytes':values.nbytes}
        record['complete']=True
    except Exception as exc:
        record['failure']=f'{type(exc).__name__}: {exc}';raise
    finally:
        record['preparation_wall_seconds']=time.perf_counter()-begin
        (a.output/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)


if __name__=='__main__':main()
