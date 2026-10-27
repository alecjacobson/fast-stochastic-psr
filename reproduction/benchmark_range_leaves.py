"""Hybrid/cached-leaf action comparison against archived independent dense actions."""
import hashlib,json,time
from pathlib import Path
import cupy as cp
import numpy as np
from geospsr.periodic_range import PeriodicRangeKernel
from geospsr.periodic_compact import CompactPeriodicRangeKernel
from geospsr.native_range import NativeCompactRangeKernel
from geospsr.periodic_hybrid import HybridPeriodicRangeKernel
from geospsr.periodic_leaf_cache import LeafCachedRangeKernel
from geospsr.periodic_cached import CachedPeriodicRangeKernel
from reproduction.common_rhs_integrity import array_digest
from reproduction.gpu_audit import GpuProcessAudit


def main():
    root=Path(__file__).resolve().parents[1];out=root/'reproduction/results/range_hybrid/leaves';out.mkdir(exist_ok=True);out.mkdir(exist_ok=True);out.mkdir(exist_ok=True)
    r={'complete':False,'scope':'matrix actions only','cases':{},'relative_l2_gate':1e-7,
       'shared_scratch_bytes_per_scan_block':4096,'additional_global_scratch_bytes':0,
       'source_sha256':{f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in ['geospsr/periodic_scan.py','geospsr/periodic_range.py','geospsr/periodic_cached.py','geospsr/periodic_compact.py','geospsr/native_range.py','geospsr/native/range_plan.cpp','geospsr/periodic_hybrid.py','geospsr/periodic_leaf_cache.py','reproduction/benchmark_range_leaves.py']}}
    def timed(fn):
        cp.cuda.get_current_stream().synchronize();start=time.perf_counter();v=fn();cp.cuda.get_current_stream().synchronize()
        return v,time.perf_counter()-start
    audit=GpuProcessAudit(cp.cuda.Device().pci_bus_id).start()
    try:
        for case in 'AB':
            old=root/f'reproduction/results/periodic_range/{case}';meta=json.loads((old/'results.json').read_text());ref=np.load(old/'actions.npz')
            x=np.load(root/f'reproduction/results/matched/{case}/input.npz')['points'];w=ref['weights']
            assert array_digest(x)==meta['hashes']['points'] and array_digest(w)==meta['hashes']['weights']
            wd=cp.asarray(w);reference=ref['reference'];cov=reference-.02*w
            v={'n':len(x),'hashes':meta['hashes'],'reference_payload_sha256':hashlib.sha256((old/'actions.npz').read_bytes()).hexdigest(),'methods':{}}
            r['cases'][case]=v;outputs={}
            for name,cls in [('hybrid',HybridPeriodicRangeKernel),('leaf32',LeafCachedRangeKernel),('leaf128',lambda x:LeafCachedRangeKernel(x,leaf_size=128))]:
                op,setup=timed(lambda:cls(x));m={'setup_seconds':setup,'plan':op.record,'actions':[]};v['methods'][name]=m
                m['plan_hashes']=[array_digest(cp.asnumpy(a)) for a in (op.x,*op.far,*op.direct,op.anchors)]
                for repeat in range(4):
                    y,seconds=timed(lambda:op(wd));y=y.get();outputs[name]=y
                    e=[float(np.linalg.norm(y[:,j]-reference[:,j])/np.linalg.norm(reference[:,j])) for j in range(3)]
                    ke=[float(np.linalg.norm(y[:,j]-.02*w[:,j]-cov[:,j])/np.linalg.norm(cov[:,j])) for j in range(3)]
                    m['actions'].append({'seconds':seconds,'action_relative_l2_per_rhs':e,'covariance_relative_l2_per_rhs':ke,'quality_pass':max(e+ke)<=r['relative_l2_gate']})
                print(case,name,m,flush=True)
                del op
            assert v['methods']['hybrid']['plan_hashes']==v['methods']['leaf32']['plan_hashes']
            v['hybrid_leaf32_plans_exact']=True
            np.savez_compressed(out/f'{case}_actions.npz',**outputs)
        r['complete']=True
    except Exception as exc:r['failure']=str(exc);raise
    finally:
        r['gpu_process_audit']=audit.stop();(out/'results.json').write_text(json.dumps(r,indent=2)+'\n')


if __name__=='__main__':main()
