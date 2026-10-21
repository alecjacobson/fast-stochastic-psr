"""Three-step next-view acquisition versus a fixed-seed random policy.

Single illustrative trial on the public scorpion, not the unreleased Figure 8.
Geometry error is symmetric mean nearest-neighbor distance to mesh samples.
"""
import sys,json,time
from pathlib import Path
import numpy as np
import cupy as cp
import igl
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes
import matplotlib.pyplot as plt
from posterior import Posterior,sync
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'reproduction/results'
sys.path.insert(0,str(ROOT/'experiments'))
from utils import normalize_points,simulate_scan,uniform_sample_mesh
read_obj=getattr(igl,'read_obj',None) or igl.readOBJ
v,_,_,f,_,_=read_obj(str(ROOT/'experiments/meshes/scorpion.obj'));v=normalize_points(v)*np.pi
rng=np.random.default_rng(42);truth,_=uniform_sample_mesh(v,f,20000,rng)
data=np.load(out/'upstream_baseline.npz')
center=v.mean(0);angles=np.arange(24)*2*np.pi/24
origins=center+2.5*np.stack([np.cos(angles),np.sin(angles),np.full(24,.2)],axis=1)
directions=center-origins;directions/=np.linalg.norm(directions,axis=1)[:,None]
t=np.linspace(0,5,128);rays=origins[:,None]+directions[:,None]*t[None,:,None]
g=np.linspace(-1.7,1.7,48);grid=np.stack(np.meshgrid(g,g,g,indexing='ij'),axis=-1).reshape(-1,3)
result={'trial_seed':42,'steps':3,'score_samples':64,'metric':'symmetric mean nearest-neighbor distance between reconstructed mesh vertices and 20000 area-uniform ground-truth surface samples; radians coordinates; mesh resolution 48^3','limitation':'one trial; mesh vertices are not area-uniform; no statistical superiority claim','policies':{}}
for policy in ['uncertainty','random']:
    x=data['points'].copy();y=data['normals'].copy();records=[];used=[]
    rand=np.random.default_rng(42)
    for step in range(4):
        start=time.perf_counter();model=Posterior(x,y)
        field=cp.asnumpy(model.mean(grid)).reshape(48,48,48)
        verts,faces,_,_=marching_cubes(field,0,spacing=(g[1]-g[0],)*3);verts+=g[0]
        error=.5*(cKDTree(truth).query(verts)[0].mean()+cKDTree(verts).query(truth)[0].mean())
        record={'step':step,'N':len(x),'geometry_error':float(error)}
        if step<3:
            if policy=='uncertainty':
                model.draw(64,seed=42+step,exact=False)
                s=model.samples(rays.reshape(-1,3)).reshape(64,24,128)
                trans=cp.asnumpy(cp.cumprod(s>0,axis=2).mean(0))
                score=np.trapezoid((trans>=.05)&(trans<=.95),t,axis=1);score[used]=-np.inf
                choice=int(score.argmax());record['score']=float(score[choice])
            else:choice=int(rand.choice([i for i in range(24) if i not in used]))
            used.append(choice);record['camera_index']=choice
            pts,ns=simulate_scan(v,f,origins[choice:choice+1],directions[choice:choice+1],np.pi/3,200)
            pts+=rand.normal(size=pts.shape)*.02;ns+=rand.normal(size=ns.shape)*.02
            x=np.concatenate([x,pts]);y=np.concatenate([y,ns]);record['new_points']=len(pts)
        sync();record['wall_seconds']=time.perf_counter()-start;records.append(record)
        print(policy,record,flush=True)
        del model
        cp.get_default_memory_pool().free_all_blocks()
    result['policies'][policy]=records
    (out/'sequential_scan.json').write_text(json.dumps(result,indent=2))
fig,ax=plt.subplots(figsize=(6,4))
for name,rows in result['policies'].items():ax.plot([r['step'] for r in rows],[r['geometry_error'] for r in rows],'o-',label=name)
ax.set(xlabel='Additional views',ylabel='Geometry error (coordinate units)',title='Sequential scanning: one fixed-seed trial');ax.legend();fig.tight_layout();fig.savefig(out/'sequential_scan.png',dpi=160)
