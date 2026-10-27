"""CUDA event stage timing; no hardware-counter profiler required."""
import json
from pathlib import Path
import cupy as cp
import numpy as np
from geospsr.periodic_cached import CachedPeriodicRangeKernel,_scan
from geospsr.periodic_range import _direct
from reproduction.gpu_audit import GpuProcessAudit


def main():
    root=Path(__file__).resolve().parents[1];r={'scope':'CUDA event stage timings, not hardware counters','cases':{}}
    audit=GpuProcessAudit(cp.cuda.Device().pci_bus_id).start()
    def event(fn):
        start,end=cp.cuda.Event(),cp.cuda.Event();start.record();fn();end.record();end.synchronize()
        return cp.cuda.get_elapsed_time(start,end)/1000
    try:
        for case in 'AB':
            x=np.load(root/f'reproduction/results/matched/{case}/input.npz')['points'];op=CachedPeriodicRangeKernel(x,scan=True)
            w=cp.asarray(np.random.default_rng(4501).normal(size=(len(x),3)));out=cp.zeros_like(w)
            fs=lambda:_scan((op.record['far_groups']*6,),(512,),(op.x,*op.far,op.anchors,op.coeff,op.decays,w,out,3,op.a,op.om,op.r,op.c0,op.c1,op.scale))
            ds=lambda:_direct((op.record['direct_groups']*3,),(32,),(op.x,*op.direct,w,out,3,op.a,op.c0,op.c1,op.norm,op.variance))
            v={'plan':op.record,'first':{'far':event(fs),'direct':event(ds)},'next':[]};r['cases'][case]=v
            for _ in range(3):v['next'].append({'far':event(fs),'direct':event(ds)})
            print(case,v,flush=True)
    finally:
        r['gpu_process_audit']=audit.stop();(root/'reproduction/results/range_compact/profile.json').write_text(json.dumps(r,indent=2)+'\n')


if __name__=='__main__':main()
