"""Reconstruct the paper's application types on the published scorpion scan.

Scenes are recreated here; these are not the unreleased dragon/cat/falcon scenes.
All collision and visibility events use coherent joint posterior samples.
"""
import argparse
import json
from pathlib import Path
import time
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import marching_cubes
from posterior import Posterior, sync

p=argparse.ArgumentParser(); p.add_argument('--samples',type=int,default=128); p.add_argument('--grid',type=int,default=48)
a=p.parse_args(); out=Path(__file__).resolve().parent/'results'
data=np.load(out/'upstream_baseline.npz'); x=data['points']; normals=data['normals']
result={'scene':'published scorpion scan; reconstructed application scenarios','N':len(x),'samples':a.samples,'grid':a.grid,'seconds':{},'inside_convention':'negative with outward normals','noise_variance':.02,'cross_modes':101**3,'prior_modes':101**3,'uncertainty':'Monte Carlo; finite-f/full-analytic-v posterior, with exact residual-vector covariance and observation noise'}
def timed(name,fn):
    sync(); t=time.perf_counter(); z=fn(); sync(); result['seconds'][name]=time.perf_counter()-t
    print(name,result['seconds'][name],flush=True); return z
model=timed('fit_first',lambda:Posterior(x,normals))
timed('draw_coefficients_first',lambda:model.draw(a.samples,seed=42))
# One set of random functions is reused in every application, preserving joint events.
q=data['query']
sm=timed('slice_mean',lambda:model.mean(q))
s=timed('slice_samples',lambda:model.samples(q))
np.savez_compressed(out/'applications_slice.npz',mean=cp.asnumpy(sm),samples=cp.asnumpy(s))
# Sign sanity check from normal probes; increasing f along outward normals.
probe=cp.asnumpy(model.mean(x+.02*normals)-model.mean(x-.02*normals))
result['outward_derivative_positive_fraction']=float(np.mean(probe>0))
# Candidate cameras on a horizontal ring; rays aim at the centroid of the scan.
center=x.mean(0); angles=np.arange(24)*2*np.pi/24
origins=center+2.5*np.stack([np.cos(angles),np.sin(angles),np.full(24,.2)],axis=1)
directions=center-origins; directions/=np.linalg.norm(directions,axis=1)[:,None]
t=np.linspace(0,5,256)
rays=origins[:,None,:]+directions[:,None,:]*t[None,:,None]
rs=timed('ray_samples',lambda:model.samples(rays.reshape(-1,3))).reshape(a.samples,24,256)
trans=cp.asnumpy(cp.cumprod(rs>0,axis=2).mean(0))
scores=np.trapezoid((trans>=.05)&(trans<=.95),t,axis=1)
best=int(np.argmax(scores))
result['next_view']={'best_index':best,'origin':origins[best].tolist(),'scores':scores.tolist(),'criterion':'length of ray with transmittance between .05 and .95'}
result['transmittance_monotone']=bool(np.all(np.diff(trans,axis=1)<=0))
# Collision proxies: spheres near the core, at an uncertain flank, and far away.
rng=np.random.default_rng(1); unit=rng.normal(size=(512,3)); unit/=np.linalg.norm(unit,axis=1)[:,None]
centers=np.array([center, center+np.array([.5,0,0]), center+np.array([1.8,0,0])])
objects=centers[:,None,:]+.12*unit[None]
cs=timed('collision_samples',lambda:model.samples(objects.reshape(-1,3))).reshape(a.samples,3,512)
collision=cp.asnumpy((cs<=0).any(axis=2).mean(axis=0))
result['collision']={'centers':centers.tolist(),'radius':.12,'surface_points':512,'probabilities':collision.tolist(),'event':'at least one sampled proxy surface point inside reconstructed solid; discrete overlap proxy','max_monte_carlo_standard_error':.5/np.sqrt(a.samples)}
# Conservative hitbox surfaces from Monte Carlo marginal moments.
g=np.linspace(-1.9,1.9,a.grid); grid=np.stack(np.meshgrid(g,g,g,indexing='ij'),axis=-1).reshape(-1,3)
gm=timed('volume_mean',lambda:model.mean(grid))
gs=timed('volume_samples',lambda:model.samples(grid))
std=gs.std(axis=0,ddof=1)
meshes=[]
for eta in [0.,1.,2.]:
    field=cp.asnumpy(gm-eta*std).reshape((a.grid,)*3)
    if field.min()<0<field.max():
        v,f,_,_=marching_cubes(field,0,spacing=(g[1]-g[0],)*3); v+=g[0]
        import gpytoolbox
        name='hitbox_eta%d.obj'%eta; gpytoolbox.write_mesh(str(out/name),v,f)
        meshes.append((eta,v,f)); result.setdefault('hitboxes',[]).append({'eta':eta,'vertices':len(v),'triangles':len(f),'file':name,'touches_domain_boundary':bool(np.any(field[[0,-1]]<0) or np.any(field[:,[0,-1]]<0) or np.any(field[:,:,[0,-1]]<0))})
    else: result.setdefault('hitboxes',[]).append({'eta':eta,'status':'no zero crossing in volume'})
fig,axes=plt.subplots(2,3,figsize=(14,8))
ng=int(np.sqrt(len(q))); q1=q[:,0].reshape(ng,ng); q2=q[:,2].reshape(ng,ng)
for ax,z,title in zip(axes[0],[cp.asnumpy(sm),cp.asnumpy(s.std(0)),cp.asnumpy(s[0])],['Mean (direct Fourier)','Monte Carlo standard deviation','Coherent posterior sample']):
    im=ax.pcolormesh(q1,q2,z.reshape(ng,ng),shading='auto',cmap='viridis'); fig.colorbar(im,ax=ax); ax.set_title(title); ax.set_aspect('equal')
    if title!='Monte Carlo standard deviation':ax.contour(q1,q2,z.reshape(ng,ng),[0],colors='w',linewidths=.7)
for i in [best,(best+6)%24,(best+12)%24]:axes[1,0].plot(t,trans[i],label='view %d'%i)
axes[1,0].set(xlabel='Distance',ylabel='P(no hit so far)',title='Ray transmittance'); axes[1,0].legend()
axes[1,1].bar(np.arange(24),scores); axes[1,1].set(xlabel='Camera index',ylabel='Uncertain distance',title='Next-view scores')
axes[1,2].bar(['Core','Flank','Far'],collision); axes[1,2].set(ylim=(0,1),ylabel='P(overlap)',title='Joint collision queries')
fig.tight_layout();fig.savefig(out/'applications.png',dpi=160)
fig=plt.figure(figsize=(12,4))
for i,(eta,v,f) in enumerate(meshes):
    ax=fig.add_subplot(1,len(meshes),i+1,projection='3d'); ax.plot_trisurf(v[:,0],v[:,1],v[:,2],triangles=f,color='steelblue',linewidth=0);ax.set_title('Hitbox eta=%g'%eta);ax.set_box_aspect((1,1,1));ax.set(xlim=(-1.9,1.9),ylim=(-1.9,1.9),zlim=(-1.9,1.9));ax.view_init(30,-60)
fig.tight_layout();fig.savefig(out/'hitboxes.png',dpi=150)
np.savez_compressed(out/'applications.npz',transmittance=trans,scores=scores,collision=collision,volume_mean=cp.asnumpy(gm),volume_std=cp.asnumpy(std),grid=g)
(out/'applications.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2),flush=True)
