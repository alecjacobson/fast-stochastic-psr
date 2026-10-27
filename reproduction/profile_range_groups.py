"""Size-bucket GPU event profile of the million-point compact range action."""
import json
from pathlib import Path
import cupy as cp
import numpy as np
from geospsr.native_range import NativeCompactRangeKernel
from geospsr.periodic_compact import _SOURCE,_scan
from geospsr.periodic_range import _direct
from reproduction.gpu_audit import GpuProcessAudit

source=_SOURCE.replace('void compact_scan_ranges(', 'void selected_scan(')
source=source.replace('const double* weights, double* out, int nrhs,','const int* selected, int totalgroups, const double* weights, double* out, int nrhs,')
source=source.replace('g=blockIdx.x/(2*nrhs);','g=selected[blockIdx.x/(2*nrhs)];')
source=source.replace('offsets[gridDim.x/(2*nrhs)]','offsets[totalgroups]')
selected_scan=cp.RawKernel(source,'selected_scan')


def main():
    root=Path(__file__).resolve().parents[1];r={'scope':'CUDA event size-bucket stages, not hardware counters','cases':{}}
    audit=GpuProcessAudit(cp.cuda.Device().pci_bus_id).start()
    def event(fn):
        s,e=cp.cuda.Event(),cp.cuda.Event();s.record();fn();e.record();e.synchronize();return cp.cuda.get_elapsed_time(s,e)/1000
    try:
        for case in ['B','million']:
            if case=='B':x=np.load(root/'reproduction/results/matched/B/input.npz')['points']
            else:x=np.load(root/'reproduction/results/range_compact/scale1m/control.npz')['points']
            op=NativeCompactRangeKernel(x,max_entries=200_000_000,max_cache_bytes=32*2**30)
            w=cp.asarray(np.random.default_rng(4501).normal(size=(len(x),3)));out=cp.zeros_like(w)
            full=lambda:_scan((op.record['far_groups']*6,),(512,),(op.x,*op.far,op.anchors,op.coeff,op.decays,w,out,3,op.a,op.om,op.r,op.c0,op.c1,op.scale))
            direct=lambda:_direct((op.record['direct_groups']*3,),(32,),(op.x,*op.direct,w,out,3,op.a,op.c0,op.c1,op.norm,op.variance))
            v={'plan':op.record,'first':{'far':event(full),'direct':event(direct)},'next':[],'buckets':{}};r['cases'][case]=v
            for _ in range(3):v['next'].append({'far':event(full),'direct':event(direct)})
            out.fill(0);full();truth=out.copy();out.fill(0)
            lengths=np.diff(op.far[2].get());methods=[]
            for low,high in [(1,32),(33,128),(129,512),(513,8192),(8193,2**31-1)]:
                ids=np.flatnonzero((lengths>=low)&(lengths<=high)).astype('i');pick=cp.asarray(ids)
                if not len(ids):continue
                def selected(pick=pick):selected_scan((len(pick)*6,),(512,),(op.x,*op.far,op.anchors,op.coeff,op.decays,pick,op.record['far_groups'],w,out,3,op.a,op.om,op.r,op.c0,op.c1,op.scale))
                m={'groups':len(ids),'entries':int(lengths[ids].sum()),'first_seconds':event(selected),'next_seconds':[]}
                for _ in range(3):m['next_seconds'].append(event(selected))
                v['buckets'][f'{low}-{high}']=m;methods.append(selected)
            out.fill(0)
            for f in methods:f()
            error=float(cp.linalg.norm(out-truth)/cp.linalg.norm(truth));v['partition_relative_error']=error
            if error>1e-12:raise ArithmeticError('partitioned kernel changed action')
            print(case,json.dumps(v),flush=True)
            del op,full,direct,selected,methods,out,truth,w;cp.get_default_memory_pool().free_all_blocks()
    finally:
        r['gpu_process_audit']=audit.stop();(root/'reproduction/results/range_hybrid/profile.json').write_text(json.dumps(r,indent=2)+'\n')


if __name__=='__main__':main()
