"""Held-out quality controls for issue124; screening uses full float64 kernels."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import ndtr

from geospsr.nonperiodic import FreeSpacePoissonMatern
from geospsr.scalar_contacts import ScalarContactUpdate
from reproduction.partial_occupancy import ROOT, load, farthest_contacts
from reproduction.stanford_partial_data import CACHE, digest
from reproduction.evaluate_partial_controls import contact_only_variance

OUT=Path('reproduction/results/quality_controls')
CONTACTS=[32,128,512]


def snapshot():
    folder=OUT/'sources';folder.mkdir(parents=True,exist_ok=True)
    paths=[__file__,'geospsr/nonperiodic.py','geospsr/nonperiodic_dense_gpu.py','geospsr/scalar_contacts.py',
           'reproduction/partial_direct_gpu.py','reproduction/partial_occupancy.py','reproduction/evaluate_partial_controls.py']
    result={}
    for path in paths:
        data=Path(path).read_bytes();h=hashlib.sha256(data).hexdigest();(folder/(h+'.py')).write_bytes(data);result[str(path)]=h
    return result


def prepare():
    OUT.mkdir(exist_ok=True)
    mesh=np.load(CACHE/'reference.npz');v,f=mesh['vertices'],mesh['faces']
    area=np.linalg.norm(np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]),axis=1)
    for i,group in enumerate(['validation','test_side','test_back','test_other_side']):
        path=OUT/(group+'-panel.npz')
        if path.exists():raise FileExistsError(path)
        x,y,c,panel=load(group,768)
        rows=np.load(CACHE/(group+'.npy'));contacts=farthest_contacts(rows[:,:3],512)
        np.testing.assert_array_equal(contacts[:32],c)
        ids=np.linspace(0,len(panel['queries'])-1,512,dtype=int)
        rng=np.random.default_rng(93124+i);fi=rng.choice(len(f),512,p=area/area.sum());bary=rng.dirichlet([1,1,1],512)
        surface=np.sum(v[f[fi]]*bary[:,:,None],axis=1)
        normal=np.cross(v[f[fi,1]]-v[f[fi,0]],v[f[fi,2]]-v[f[fi,0]]);normal/=np.linalg.norm(normal,axis=1)[:,None]
        offsets=np.linspace(-.15,.15,33);probes=surface[:,None,:]+normal[:,None,:]*offsets[None,:,None]
        distance=cKDTree(rows[:,:3]).query(surface)[0]
        np.savez_compressed(path,contacts=contacts,query_indices=ids,queries=panel['queries'][ids],inside=panel['inside'][ids],
            observed=panel['observed'][ids],occluded=panel['occluded'][ids],near_surface=panel['near_surface'][ids],
            surface=surface,normal=normal,offsets=offsets,probes=probes,surface_observed=distance<.025,surface_occluded=distance>.075)
    (OUT/'inputs.json').write_text(json.dumps(dict(dataset_sha256=digest(ROOT/'dataset.json'),reference_sha256=digest(CACHE/'reference.npz'),
        protocol_sha256=digest(OUT/'PROTOCOL.md'),panels={g:digest(OUT/(g+'-panel.npz')) for g in ['validation','test_side','test_back','test_other_side']}),indent=2)+'\n')


def occupancy(mean,variance,panel):
    if not np.isfinite(mean).all() or not np.isfinite(variance).all() or np.any(variance<=0):raise ValueError('invalid posterior moments')
    p=ndtr(-mean/np.sqrt(variance));inside=panel['inside'];result={}
    for label,mask in [('all',np.ones(len(p),bool)),('observed',panel['observed']),('occluded',panel['occluded']),('near_surface',panel['near_surface'])]:
        result[label]=dict(count=int(mask.sum()),brier=float(np.mean((p[mask]-inside[mask])**2)),
            error_rate=float(np.mean((p[mask]>=.5)!=inside[mask])),
            confident_error_rate=float(np.mean(((p[mask]<.05)&inside[mask])|((p[mask]>.95)&~inside[mask]))))
    return result


def roots(values,panel):
    t=panel['offsets'];v=np.asarray(values).reshape(len(panel['surface']),len(t))
    if not np.isfinite(v).all():raise ValueError('nonfinite surface-probe mean')
    cross=(v[:,:-1]*v[:,1:]<=0)
    denominator=v[:,1:]-v[:,:-1]
    fraction=np.divide(-v[:,:-1],denominator,out=np.zeros_like(denominator),where=denominator!=0)
    rr=t[:-1]+fraction*np.diff(t)
    rr=np.where(cross,abs(rr),np.inf);error=np.min(rr,axis=1);missing=~np.isfinite(error);error=np.where(missing,.15,error)
    result={}
    for label,mask in [('all',np.ones(len(error),bool)),('observed',panel['surface_observed']),('occluded',panel['surface_occluded'])]:
        result[label]=dict(count=int(mask.sum()),mae=float(np.mean(error[mask])),median=float(np.median(error[mask])),
            p90=float(np.quantile(error[mask],.9)),missing_fraction=float(np.mean(missing[mask])))
    return result,error,missing


class DenseContactFamily:
    """Reuse a full float64 factor and nested contact solves across controls."""
    def __init__(self,model,x,y,contacts,noise,counts=CONTACTS):
        import cupy as cp
        from geospsr.nonperiodic_dense_gpu import GPUDenseMaternSystem
        from reproduction.partial_direct_gpu import cross_rhs
        self.counts=tuple(counts)
        if not self.counts or min(self.counts)<1 or max(self.counts)>len(contacts):raise ValueError("invalid contact counts")
        self.model,self.x,self.contacts=model,cp.asarray(x),contacts
        self.system=GPUDenseMaternSystem(model,x,noise,dtype='float64')
        self.alpha=cp.asarray(self.system.solve(y));b=cross_rhs(model,contacts,x)
        self.zc=cp.asarray(self.system.solve(cp.asnumpy(b)));width=len(contacts)
        mean=sum(b[:,d*width:(d+1)*width].T@self.alpha[:,d] for d in range(3))
        reduction=sum(b[:,d*width:(d+1)*width].T@self.zc[:,d*width:(d+1)*width] for d in range(3))
        covariance=model.scalar_covariance(contacts,contacts)-cp.asnumpy((reduction+reduction.T)*.5)
        self.updates={c:ScalarContactUpdate(cp.asnumpy(mean[:c]),covariance[:c,:c],np.zeros(c),.0001) for c in self.counts}
        self.coefficients=[]
        for c in self.counts:
            w=cp.asarray(self.updates[c].weights)
            self.coefficients.append(self.alpha-cp.column_stack([self.zc[:,d*width:d*width+c]@w for d in range(3)]))
        self.weights=np.column_stack([np.pad(self.updates[c].weights,(0,width-c)) for c in self.counts])

    def predict(self,q,batch=96):
        import cupy as cp
        from cupyx.scipy.linalg import solve_triangular
        from reproduction.partial_direct_gpu import cross_rhs
        nc=len(self.contacts);means=np.empty((len(q),len(self.counts)));variances=np.empty_like(means)
        for first in range(0,len(q),batch):
            qq=q[first:first+batch];m=len(qq);b=cross_rhs(self.model,qq,self.x)
            bm=cp.asnumpy(sum(b[:,d*m:(d+1)*m].T@self.alpha[:,d] for d in range(3)))
            w=solve_triangular(self.system.factor,b,lower=True,check_finite=False)
            bv=self.model.prior_variance-cp.asnumpy(cp.sum(w*w,axis=0).reshape(3,m).sum(axis=0))
            cross=self.model.scalar_covariance(qq,self.contacts)-cp.asnumpy(sum(b[:,d*m:(d+1)*m].T@self.zc[:,d*nc:(d+1)*nc] for d in range(3)))
            for k,c in enumerate(self.counts):means[first:first+m,k],variances[first:first+m,k]=self.updates[c].apply(bm,bv,cross[:,:c])
        return means,variances

    def mean(self,q,batch=192):
        import cupy as cp
        from reproduction.partial_direct_gpu import cross_rhs
        result=np.empty((len(q),len(self.counts)))
        coefficients=[cp.column_stack([a[:,d] for a in self.coefficients]) for d in range(3)]
        for first in range(0,len(q),batch):
            qq=q[first:first+batch];m=len(qq);b=cross_rhs(self.model,qq,self.x)
            result[first:first+m]=cp.asnumpy(sum(b[:,d*m:(d+1)*m].T@coefficients[d] for d in range(3)))
            result[first:first+m]+=self.model.scalar_covariance(qq,self.contacts)@self.weights
        return result


def screen(n,ell,sigma):
    import cupy as cp
    from reproduction.gpu_audit import GpuProcessAudit
    output=OUT/f'screen-n{n}-ell{ell}-sigma{sigma}.json'
    if output.exists():raise FileExistsError(output)
    sources=snapshot();panel=dict(np.load(OUT/'validation-panel.npz'));x,y,_,_=load('validation',n)
    model=FreeSpacePoissonMatern(ell,1/3);audit=GpuProcessAudit(cp.cuda.Device().pci_bus_id,interval=.25,cuda_device=cp.cuda.Device().id).start()
    start=perf_counter()
    record=dict(n=n,ell=ell,sigma=sigma,group='validation',backend='full float64 Cholesky',
        source_sha256=sources,panel_sha256=digest(OUT/'validation-panel.npz'))
    try:
        family=DenseContactFamily(model,x,y,panel['contacts'],sigma**2)
        m,v=family.predict(panel['queries']);folded=family.mean(panel['queries'])
        if np.max(abs(m-folded))>1e-10:raise RuntimeError('folded mean disagrees with direct conditioning')
        probe_mean=family.mean(panel['probes'].reshape(-1,3));records=[];root_errors=[];missing_roots=[]
        for k,c in enumerate(CONTACTS):
            geometry,error,missing=roots(probe_mean[:,k],panel);root_errors.append(error);missing_roots.append(missing)
            cv=contact_only_variance(model,panel['contacts'][:c],panel['queries'])
            records.append(dict(contacts=c,occupancy=occupancy(m[:,k],v[:,k],panel),geometry=geometry,
                contact_only=occupancy(m[:,k],cv,panel),prior_variance=model.prior_variance))
        cp.cuda.Device().synchronize()
        np.savez_compressed(output.with_suffix('.npz'),mean=m,variance=v,probe_mean=probe_mean,root_error=np.array(root_errors).T,missing=np.array(missing_roots).T)
        record.update(status='completed',records=records,field_sha256=digest(output.with_suffix('.npz')))
    except Exception as exc:
        record.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        record.update(seconds=perf_counter()-start,gpu_audit=audit.stop())
        output.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print('DONE',n,ell,sigma,record['seconds'],[(r['contacts'],r['occupancy']['all']['brier'],r['geometry']['observed']['mae'],r['geometry']['occluded']['mae']) for r in records],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','screen']);p.add_argument('--n',type=int,default=16384)
    p.add_argument('--ell',type=float,default=.48);p.add_argument('--sigma',type=float,default=.2);a=p.parse_args()
    if a.action=='prepare':prepare()
    else:screen(a.n,a.ell,a.sigma)
