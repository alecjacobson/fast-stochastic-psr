"""Both planar query directions on the original B slices; query-only timings."""
import json,time
from pathlib import Path
import cupy as cp
import numpy as np
from reproduction.posterior import Posterior,sync
from reproduction.planar_fourier import PlanarFourierPlan
from reproduction.correction_sampling import DenseCorrectionSampler
from reproduction.common_rhs_integrity import array_digest
from reproduction.gpu_audit import GpuProcessAudit
from geospsr.spectral import FourierPoisson


def main():
    root=Path(__file__).resolve().parents[1];output=root/'reproduction/results/planar_queries/microbenchmark.json'
    r={'complete':False,'case':'B','batch':16,'repeats':3,'methods':{},
       'scope':'query-plan setup and execution only, same original planes and full cube50; no application ratio'}
    audit=GpuProcessAudit(cp.cuda.Device().pci_bus_id).start()
    def timed(fn):
        sync();start=time.perf_counter();v=fn();sync();return v,time.perf_counter()-start
    try:
        d=np.load(root/'reproduction/results/matched/B/input.npz');x,y,q=d['points'],d['normals'],d['query']
        r['query_hash']=array_digest(q);r['heights']=np.unique(q[:,1]).tolist()
        model=Posterior(x,y,upsampfac=1.25)
        sampler=DenseCorrectionSampler(x,model.L,model.op)
        coefficients=next(sampler.draw_batches(16,seed=43,batch=16))['coefficients']
        strengths=cp.random.RandomState(43).standard_normal((16,len(q)),dtype=model.op.dtype).astype(model.op.cdtype)
        reference={}
        double=FourierPoisson(50,dtype='float64',eps=1e-12,upsampfac=2.)
        for kind,values in [(1,strengths),(2,coefficients)]:
            reference[kind]=double.plan(kind,q,16).execute(values.astype(double.cdtype))
        for name,make in [('generic',lambda kind:model.op.plan(kind,q,16)),('planar',lambda kind:PlanarFourierPlan(kind,model.op,q,16))]:
            record={};r['methods'][name]=record
            for kind,values in [(1,strengths),(2,coefficients)]:
                plan,setup=timed(lambda:make(kind));times=[]
                for repeat in range(3):out,seconds=timed(lambda:plan.execute(values));times.append(seconds)
                item={'setup_seconds':setup,'execute_seconds':times,'median_seconds':float(np.median(times))}
                record[str(kind)]=item
                item['relative_complex_l2']=float(cp.linalg.norm(out-reference[kind])/cp.linalg.norm(reference[kind]))
                item['precision_pass']=bool(item['relative_complex_l2']<=1e-5)
                if name=='planar' and not item['precision_pass']:raise ValueError('planar mismatch to double reference')
                del plan
        r['complete']=True
    finally:r['gpu_process_audit']=audit.stop();output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r['methods'],indent=2))


if __name__=='__main__':main()
