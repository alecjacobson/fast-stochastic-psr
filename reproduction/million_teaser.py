"""Million-observation scorpion fields for a Figure-1-style scientific teaser.

All observations enter NUFFT-CG. Posterior variance is estimated from coherent
pathwise draws of the same finite Fourier GP used in training, not a small subset.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cupy as cp
import gpytoolbox as gpy
import igl
import numpy as np
from scipy.special import ndtr

from geospsr.spectral import FourierKernel, FourierPoisson

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
from utils import normalize_points,uniform_sample_mesh


def sync(): cp.cuda.runtime.deviceSynchronize()

def timed(fn):
    sync();t=time.perf_counter();v=fn();sync()
    return v,time.perf_counter()-t


def visible_cloud(n,seed=2718):
    """Area-uniform distinct mesh samples visible from the upstream two cameras."""
    read=getattr(igl,'read_obj',None) or igl.readOBJ
    v,_,_,f,_,_=read(str(ROOT/'experiments/meshes/scorpion.obj'))
    v=normalize_points(v)*np.pi
    tree=gpy.ray_mesh_intersect_precompute(v,f)
    cameras=np.array([[0,-2,0],[1,1,1]])*np.pi
    rng=np.random.default_rng(seed);points=[];normals=[];total=0
    while total<n:
        p,nr=uniform_sample_mesh(v,f,min(300000,max(10000,n-total)*2),rng)
        visible=np.zeros(len(p),dtype=bool)
        for cam in cameras:
            d=p-cam;dist=np.linalg.norm(d,axis=1);d/=dist[:,None]
            hits,_,_=gpy.ray_mesh_intersect(np.broadcast_to(cam,p.shape).copy(),d,v,f,intersector=tree)
            visible|=np.abs(hits-dist)<1e-5
        points.append(p[visible]);normals.append(nr[visible]);total+=int(visible.sum())
        print('visible surface samples',total,flush=True)
    clean=np.concatenate(points)[:n].astype('f');normals=np.concatenate(normals)[:n].astype('f')
    # Match the upstream example's coordinate and normal perturbations.
    x=clean+rng.normal(size=clean.shape).astype('f')*.02
    normals+=rng.normal(size=normals.shape).astype('f')*.02
    return x,normals,clean,v,f


class FinitePosterior:
    """Matrix-free posterior for one explicitly finite-spectrum GP model."""
    def __init__(self,x,normals,cutoff=50,lengthscale=.01,noise_variance=.02,eps=1e-6):
        self.x=cp.asarray(x,dtype='f');self.y=cp.asarray(normals,dtype='f')
        self.op=FourierPoisson(cutoff,lengthscale,.1,eps=eps)
        self.noise=noise_variance
        self.K=FourierKernel(self.x,self.op,noise_variance,rhs_count=3)
        self.alpha,self.mean_solve=self.K.solve(self.y,rtol=2e-5,maxiter=1000)
        if not self.mean_solve['converged']:
            raise RuntimeError(f'mean solve failed: {self.mean_solve}')
        self.coeff=self.op.aggregate(self.x,self.alpha)
        self.offset=self.op.evaluate(self.x,self.coeff).mean()
        sync()

    def mean(self,q):return self.op.evaluate(q,self.coeff)[0]-self.offset

    def variance(self,q,count=256,batch=8,seed=31415,progress=None,rtol=1e-4):
        rng=np.random.default_rng(seed);n=len(self.x)
        total=cp.zeros(len(q),dtype=cp.float64);total_mean=cp.zeros(len(q),dtype=cp.float64)
        records=[];K=None;query=None;last_p=None
        for start in range(0,count,batch):
            p=min(batch,count-start);sync();begin=time.perf_counter()
            if p!=last_p:
                K=FourierKernel(self.x,self.op,self.noise,rhs_count=3*p)
                query=self.op.plan(2,q,p)
                last_p=p
            xi=cp.asarray(rng.standard_normal((p,3,*self.op.shape,2),dtype=np.float32))
            vc,fc=self.op.prior_coefficients(xi)
            vp=K.backward.execute(cp.ascontiguousarray(vc.reshape(3*p,*self.op.shape))).real.T
            del vc,xi
            noise=cp.asarray(rng.standard_normal((n,3*p),dtype=np.float32))*np.sqrt(self.noise)
            alpha,info=K.solve(vp+noise,rtol=rtol,maxiter=1000)
            del vp,noise
            if not info['converged']:
                raise RuntimeError(f'sample solve failed: {info}')
            # Reuse the transform plans; each sample's three vector components
            # become one scalar Poisson-corrected coefficient grid.
            ac=K.forward.execute(cp.ascontiguousarray(alpha.T,dtype=self.op.cdtype)).reshape(p,3,*self.op.shape)
            correction=-1j*(ac*self.op.multiplier[None]).sum(1)
            samples=query.execute(cp.ascontiguousarray(fc-correction)).real
            total+=cp.sum(samples.astype(cp.float64)**2,axis=0)
            total_mean+=samples.sum(0)
            sync();info['seconds']=time.perf_counter()-begin;info['samples_done']=start+p
            records.append(info)
            if progress:progress(records)
            print('posterior draws',start+p,'/',count,'iterations',info['iterations'],'seconds',info['seconds'],flush=True)
            del ac,alpha,correction,fc,samples
        return (total/count).astype('f'),(total_mean/count).astype('f'),records


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--points',type=int,default=1000000)
    p.add_argument('--samples',type=int,default=256)
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--resolution',type=int,default=256)
    p.add_argument('--cutoff',type=int,default=50)
    a=p.parse_args()
    out=ROOT/'reproduction/results/teaser';out.mkdir(exist_ok=True,parents=True)
    result={'args':vars(a),'seed':2718,'device':cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
            'model':'finite product-Matern32 GP, ell=.01, per-axis variance=.1, noise variance=.02',
            'sampling':'area-uniform surface points visible from either original camera; independent .02 coordinate/normal noise',
            'timing_policy':'synchronized GPU wall time; first calls include initialization; sample stages include CPU RNG and transfers',
            'variance_estimator':'mean squared zero-mean conditional draws; no small-N substitution',
            'seconds':{}}
    def save(): (out/'million.json').write_text(json.dumps(result,indent=2))
    cloudfile=out/f'cloud_{a.points}.npz'
    if cloudfile.exists():
        d=np.load(cloudfile);x,y,clean,v,f=(d[k] for k in ['points','normals','clean','vertices','faces'])
    else:
        begin=time.perf_counter();x,y,clean,v,f=visible_cloud(a.points)
        result['seconds']['generate_observations']=time.perf_counter()-begin
        np.savez_compressed(cloudfile,points=x,normals=y,clean=clean,vertices=v,faces=f)
    result['points_sha256']=hashlib.sha256(x.tobytes()).hexdigest();result['N']=len(x)
    print('training N=',len(x),flush=True)
    model,secs=timed(lambda:FinitePosterior(x,y,cutoff=a.cutoff))
    result['seconds']['fit_and_source_aggregation_first']=secs;result['mean_solve']=model.mean_solve;save()
    print('fit',secs,model.mean_solve,flush=True)
    axis=np.linspace(-1.7,1.7,a.resolution,dtype='f')
    xg,zg=np.meshgrid(axis,axis,indexing='xy')
    heights=np.array([.4,-.75],dtype='f')
    q=np.concatenate([np.stack([xg.ravel(),np.full(xg.size,h),zg.ravel()],axis=1) for h in heights])
    mu,secs=timed(lambda:model.mean(q));result['seconds']['slices_mean_first']=secs
    # Benchmark mean action on the exact figure queries, with the same solved weights.
    query_times=[];aggregate_times=[]
    for _ in range(3):
        _,secs=timed(lambda:model.mean(q));query_times.append(secs)
        _,secs=timed(lambda:model.op.aggregate(model.x,model.alpha));aggregate_times.append(secs)
    result['mean_query_warm_seconds']=query_times;result['mean_aggregate_warm_seconds']=aggregate_times
    result['mean_aggregate_plus_query_medians']=float(np.median(query_times)+np.median(aggregate_times))
    result['M']=len(q);save()
    def progress(records):
        result['sampling_batches']=records;save()
    (var,zero_mean,records),secs=timed(lambda:model.variance(q,count=a.samples,batch=a.batch,progress=progress))
    result['seconds']['posterior_variance_all_draws']=secs
    mu=cp.asnumpy(mu);var=cp.asnumpy(var)
    prob=ndtr(-mu/np.sqrt(np.maximum(var,1e-30)))
    result['variance_relative_mc_standard_error']=float(np.sqrt(2/a.samples))
    result['conditional_draw_mean_standardized_rms']=float(cp.sqrt(cp.mean((zero_mean/cp.sqrt(cp.asarray(var)))**2)))
    result['field_ranges']={k:[float(z.min()),float(z.max())] for k,z in [('mean',mu),('variance',var),('probability_inside',prob)]}
    np.savez_compressed(out/'fields.npz',axis=axis,heights=heights,mean=mu.reshape(2,a.resolution,a.resolution),variance=var.reshape(2,a.resolution,a.resolution),probability=prob.reshape(2,a.resolution,a.resolution),offset=float(model.offset))
    # Save the fitted weights for separate, repeatable upstream-kernel comparisons.
    np.savez_compressed(out/'weights.npz',alpha=cp.asnumpy(model.alpha),offset=float(model.offset))
    save()
    print('FIELDS COMPLETE',json.dumps(result['seconds']),flush=True)

if __name__=='__main__':main()
