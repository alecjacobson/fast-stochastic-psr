"""Stream a full-kernel reference for the observed-fidelity candidate."""
import argparse
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from geospsr.nonperiodic import FreeSpacePoissonMatern
from geospsr.scalar_contacts import ScalarContactUpdate
from reproduction.observed_fidelity import OUT, OLD, write, sources, fidelity
from reproduction.partial_occupancy import load
from reproduction.partial_quality import occupancy, roots
from reproduction.stanford_partial_data import digest


class StreamingReference:
    """Keep contact inverse actions on host while the dense factor is resident."""
    def __init__(self,model,x,y,contacts,noise,scalar_variance,block=96):
        import cupy as cp
        from reproduction.partial_quality_large import RefinedDenseSystem
        from reproduction.partial_direct_gpu import cross_rhs
        self.model,self.x,self.contacts,self.block=model,cp.asarray(x),contacts,block
        self.system=RefinedDenseSystem(model,x,noise)
        self.alpha=cp.asarray(self.system.solve(y));nc=len(contacts)
        self.zc=np.empty((len(x),3*nc))
        for first in range(0,nc,block):
            qq=contacts[first:first+block];b=cross_rhs(model,qq,self.x);m=len(qq)
            z=self.system.solve(cp.asnumpy(b))
            for d in range(3):self.zc[:,d*nc+first:d*nc+first+m]=z[:,d*m:(d+1)*m]
            del b,z;cp.get_default_memory_pool().free_all_blocks()
            print('CONTACT INVERSE',first+m,nc,flush=True)
        m,cross=self.cross_moments(contacts)
        self.update=ScalarContactUpdate(m,(cross+cross.T)*.5,np.zeros(nc),scalar_variance)
        coefficients=self.alpha.copy();w=cp.asarray(self.update.weights)
        for d in range(3):
            z=cp.asarray(self.zc[:,d*nc:(d+1)*nc]);coefficients[:,d]-=z@w
            del z;cp.get_default_memory_pool().free_all_blocks()
        self.coefficients=coefficients

    def cross_moments(self,q):
        import cupy as cp
        from reproduction.partial_direct_gpu import cross_rhs
        nc=len(self.contacts);m=np.zeros(len(q));cross=self.model.scalar_covariance(q,self.contacts)
        for d in range(3):
            z=cp.asarray(self.zc[:,d*nc:(d+1)*nc])
            for first in range(0,len(q),self.block):
                qq=q[first:first+self.block];width=len(qq);raw=cross_rhs(self.model,qq,self.x);b=raw[:,d*width:(d+1)*width]
                m[first:first+width]+=cp.asnumpy(b.T@self.alpha[:,d])
                cross[first:first+width]-=cp.asnumpy(b.T@z)
                del b,raw
            del z;cp.get_default_memory_pool().free_all_blocks()
        return m,cross

    def predict(self,q):
        import cupy as cp
        from reproduction.partial_direct_gpu import cross_rhs
        m,cross=self.cross_moments(q);v=np.empty(len(q))
        for first in range(0,len(q),self.block):
            qq=q[first:first+self.block];width=len(qq);b=cross_rhs(self.model,qq,self.x)
            z,_,residual=self.system.variance_solve(b,0)
            v[first:first+width]=self.model.prior_variance-cp.asnumpy(cp.sum(b*z+z*residual,axis=0).reshape(3,width).sum(axis=0))
            del b,z,residual;cp.get_default_memory_pool().free_all_blocks()
        return self.update.apply(m,v,cross)

    def release_factor(self):
        import cupy as cp
        self.histories=self.system.histories
        del self.system,self.zc
        cp.get_default_memory_pool().free_all_blocks()


def evaluate_mean(model,x,contacts,coefficients,weights,panel):
    import cupy as cp
    from reproduction.compute_partial_figures import mean_field
    posterior=SimpleNamespace(model=model,x=cp.asarray(x),contacts=contacts)
    coeff=cp.asarray(coefficients),weights
    p=panel['fit_points'];t=panel['fit_offsets'];normal=panel['fit_normals']
    probe_mean=mean_field(posterior,(p[:,None,:]+t[None,:,None]*normal[:,None,:]).reshape(-1,3),coeff)
    delta=float(panel['spacing'])/8;offsets=np.concatenate((np.eye(3),-np.eye(3)))*delta
    g=mean_field(posterior,(p[:,None,:]+offsets[None,:,:]).reshape(-1,3),coeff).reshape(len(p),6)
    gradient=(g[:,:3]-g[:,3:])/(2*delta)
    fit,error,angle=fidelity(probe_mean,gradient,panel)
    reference_mean=mean_field(posterior,panel['probes'].reshape(-1,3),coeff)
    return dict(fidelity=fit,geometry=roots(reference_mean,panel)[0]),dict(fit_mean=probe_mean,gradient=gradient,
                root_error=error,normal_angle=angle,reference_mean=reference_mean)


def run(args):
    import cupy as cp
    from reproduction.gpu_audit import GpuProcessAudit
    path=OUT/(args.tag+'.json')
    if path.exists():raise FileExistsError(path)
    chosen=json.loads((OUT/'selection.json').read_text())['promotion_order'][args.candidate-1]
    config={k:chosen[k] for k in ['ell','sigma','contacts','scalar_variance']}
    if args.group!='validation':
        frozen=json.loads((OUT/'selected.json').read_text())
        assert frozen['configuration']==config
    panel=dict(np.load(OUT/(args.group+'-panel.npz')));x,y,_,_=load(args.group,args.n)
    contacts=panel['contacts'][:config['contacts']];model=FreeSpacePoissonMatern(config['ell'],1/3)
    source=sources();h=digest(__file__);(OUT/'sources'/(h+'.py')).write_bytes(Path(__file__).read_bytes());source[__file__]=h
    record=dict(arguments=vars(args),configuration=config,source_sha256=source,
        panel_sha256=digest(OUT/(args.group+'-panel.npz')),selection_sha256=digest(OUT/'selection.json'))
    audit=GpuProcessAudit(cp.cuda.Device().pci_bus_id,interval=.5,cuda_device=cp.cuda.Device().id).start();start=perf_counter()
    try:
        posterior=StreamingReference(model,x,y,contacts,config['sigma']**2,config['scalar_variance'])
        m,v=posterior.predict(panel['queries']);posterior.release_factor()
        coefficients=cp.asnumpy(posterior.coefficients);weights=posterior.update.weights
        scores,fields=evaluate_mean(model,x,contacts,coefficients,weights,panel)
        record.update(scores,occupancy=occupancy(m,v,panel),refinement_history=posterior.histories)
        cheap=ScalarContactUpdate(np.zeros(len(contacts)),model.scalar_covariance(contacts,contacts),np.zeros(len(contacts)),config['scalar_variance'])
        cv=cheap.apply(np.zeros(len(m)),np.full(len(m),model.prior_variance),model.scalar_covariance(panel['queries'],contacts))[1]
        record['contact_only']=occupancy(m,cv,panel)
        np.savez_compressed(path.with_suffix('.npz'),mean=m,variance=v,coefficients=coefficients,contact_weights=weights,**fields)
        record.update(status='completed',field_sha256=digest(path.with_suffix('.npz')))
    except Exception as exc:
        record.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        record.update(seconds=perf_counter()-start,gpu_audit=audit.stop());write(path,record)
    print('LARGE',args.tag,record['seconds'],record['fidelity'],record['occupancy']['all'],flush=True)


def freeze(args):
    from datetime import datetime, timezone
    path=OUT/(args.tag+'.json');r=json.loads(path.read_text())
    assert r['status']=='completed' and r['gpu_audit']['sampled_isolation_pass']
    assert r['arguments']['group']=='validation' and r['arguments']['n']==70652 and r['fidelity']['passes']
    assert r['field_sha256']==digest(path.with_suffix('.npz'))
    target=OUT/'selected.json'
    if target.exists():raise FileExistsError(target)
    write(target,dict(configuration=r['configuration'],candidate=r['arguments']['candidate'],validation_record=str(path),
        validation_sha256=digest(path),protocol_sha256=digest(OUT/'PROTOCOL.md'),frozen_at_utc=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['run','freeze'])
    p.add_argument('--tag',required=True);p.add_argument('--group',default='validation');p.add_argument('--n',type=int,default=70652)
    p.add_argument('--candidate',type=int,default=1);a=p.parse_args()
    if a.action=='run':run(a)
    else:freeze(a)
