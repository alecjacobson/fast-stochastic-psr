"""Fresh compact32 versus leaf128 million-point action comparison."""
import hashlib,json,resource,time
from pathlib import Path
import cupy as cp
import numpy as np
from geospsr.native_range import NativeCompactRangeKernel
from geospsr.periodic_leaf_cache import LeafCachedRangeKernel
from reproduction.common_rhs_integrity import array_digest
from reproduction.gpu_audit import GpuProcessAudit


def main():
    root=Path(__file__).resolve().parents[1];out=root/'reproduction/results/range_hybrid/scale1m';out.mkdir(exist_ok=True)
    old=root/'reproduction/results/range_compact/scale1m/control.npz'
    r={'complete':False,'n':1000000,'seed':4501,'query_rows':1024,'rhs_count':3,'relative_l2_gate':1e-7,
       'scope':'sampled-row full analytic action only; no posterior/application speedup',
       'reference_payload_sha256':hashlib.sha256(old.read_bytes()).hexdigest(),'methods':{},
       'source_sha256':{f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in ['geospsr/periodic_hybrid.py','geospsr/periodic_leaf_cache.py','geospsr/periodic_compact.py','geospsr/native_range.py','geospsr/native/range_plan.cpp','reproduction/benchmark_range_leaf_scale.py']}}
    def timed(fn):
        cp.cuda.get_current_stream().synchronize();start=time.perf_counter();v=fn();cp.cuda.get_current_stream().synchronize();return v,time.perf_counter()-start
    audit=GpuProcessAudit(cp.cuda.Device().pci_bus_id).start()
    try:
        free,total=cp.cuda.runtime.memGetInfo();r['initial_gpu_free_bytes']=free
        if free<40*2**30:raise MemoryError('need40GiB initially free GPU memory')
        data=np.load(old);x=data['points'];rows=data['rows'];ref=data['reference'];w=np.random.default_rng(r['seed']).normal(size=(len(x),3))
        r['hashes']={k:array_digest(v) for k,v in [('points',x),('weights',w),('rows',rows)]}
        wd=cp.asarray(w);cov=ref-.02*w[rows];outputs={}
        for name,cls,leaf in [('compact32',NativeCompactRangeKernel,32),('leaf128',LeafCachedRangeKernel,128)]:
            op,setup=timed(lambda:cls(x,leaf_size=leaf,max_entries=200_000_000,max_cache_bytes=32*2**30))
            m={'setup_seconds':setup,'plan':op.record,'process_rss_high_water_after_plan_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'actions':[]};r['methods'][name]=m
            print(name,'setup',setup,'retained',op.record['cached_total_device_bytes'],flush=True)
            for repeat in range(4):
                actual,seconds=timed(lambda:op(wd));v=actual[rows].get();outputs[name]=v
                e=[float(np.linalg.norm(v[:,j]-ref[:,j])/np.linalg.norm(ref[:,j])) for j in range(3)]
                ke=[float(np.linalg.norm(v[:,j]-.02*w[rows,j]-cov[:,j])/np.linalg.norm(cov[:,j])) for j in range(3)]
                m['actions'].append({'seconds':seconds,'action_relative_l2_per_rhs':e,'covariance_relative_l2_per_rhs':ke,'quality_pass':max(e+ke)<=r['relative_l2_gate']})
                print(name,repeat,seconds,max(ke),flush=True)
            del op,actual;cp.get_default_memory_pool().free_all_blocks()
        np.savez_compressed(out/'actions.npz',**outputs);r['complete']=True
    except Exception as exc:r['failure']=str(exc);raise
    finally:
        r['gpu_process_audit']=audit.stop();(out/'results.json').write_text(json.dumps(r,indent=2)+'\n')


if __name__=='__main__':main()
