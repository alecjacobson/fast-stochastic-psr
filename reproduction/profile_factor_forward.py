"""Instrument the existing forward factor; diagnostic timings, not speed ratios."""
import json,time
from pathlib import Path
import cupy as cp
import cupyx.scipy.linalg as cla
import numpy as np
from reproduction.posterior import Posterior,sync
from reproduction.factor_deflation import FourierCorrectionFactor
from reproduction.common_rhs_integrity import array_digest
from reproduction.gpu_audit import GpuProcessAudit


def main():
    root=Path(__file__).resolve().parents[1];out=root/'reproduction/results/planar_queries/profile.json'
    d=np.load(root/'reproduction/results/matched/B/input.npz');x,y,q=d['points'],d['normals'],d['query']
    old=json.loads((root/'reproduction/results/factor_deflation/batch16/B/results.json').read_text())
    basis=np.load('/opt/geospsr-runtime/factor-deflation/batch16/B/709_512_basis.npy')
    if array_digest(basis)!=old['runs'][0]['settings']['512']['hashes']['basis']:raise ValueError('basis hash mismatch')
    if {k:array_digest(v) for k,v in [('points',x),('normals',y),('queries',q)]}!=old['hashes']:raise ValueError('input mismatch')
    r={'complete':False,'case':'B','rank':512,'batch':16,'basis_sha256':array_digest(basis),
       'scope':'one instrumented forward action with synchronization per stage; not a speedup baseline','seconds':{}}
    audit=GpuProcessAudit(cp.cuda.Device().pci_bus_id).start()
    def timed(name,fn):
        sync();t=time.perf_counter();v=fn();sync();r['seconds'][name]=r['seconds'].get(name,0.)+time.perf_counter()-t;return v
    try:
        model=timed('fit',lambda:Posterior(x,y,upsampfac=1.25))
        factor=timed('factor_setup',lambda:FourierCorrectionFactor(x,q,model.op,model.L,batch=16))
        factor.prepare(16);op=model.op;n=len(x)
        reference=timed('production_forward',lambda:factor.forward(cp.asarray(basis)))
        values=timed('cast',lambda:cp.asarray(basis,dtype=op.dtype));output=cp.empty_like(reference)
        for start in range(0,512,16):
            v=values[:,start:start+16];p=v.shape[1];_,_,xf,qb=factor.prepare(p)
            rhs=timed('layout',lambda:cp.ascontiguousarray(v.reshape(n,3,p).transpose(0,2,1).reshape(n,3*p)))
            weights=timed('triangular_solve',lambda:cla.solve_triangular(model.L.T,rhs,lower=False))
            source=timed('source_layout',lambda:cp.ascontiguousarray(weights.T,dtype=op.cdtype))
            modes=timed('source_transform',lambda:xf.execute(source).reshape(p,3,*op.shape))
            coeff=timed('component_contraction',lambda:cp.ascontiguousarray((modes*factor.coeff[None]).sum(1)))
            field=timed('query_transform',lambda:qb.execute(coeff).real.T)
            timed('output_copy',lambda:output.__setitem__((slice(None),slice(start,start+p)),field))
        r['relative_field_error']=float(cp.linalg.norm(output-reference)/cp.linalg.norm(reference))
        if r['relative_field_error']>1e-5:raise ValueError('instrumentation changed output')
        r['complete']=True
    finally:r['gpu_process_audit']=audit.stop();out.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r['seconds'],indent=2))


if __name__=='__main__':main()
