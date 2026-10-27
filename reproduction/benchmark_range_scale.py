"""100k full analytic action pilot with 1024 independently checked rows."""
import hashlib,json,resource,time
from pathlib import Path
import cupy as cp
import numpy as np
from geospsr.native_range import NativeCompactRangeKernel
from geospsr.spectral_tail import analytic_covariance_rows64
from reproduction.common_rhs_integrity import array_digest
from reproduction.gpu_audit import GpuProcessAudit


def main():
    root=Path(__file__).resolve().parents[1];out=root/'reproduction/results/range_compact/scale100k';out.mkdir(exist_ok=True)
    r={'complete':False,'n':100000,'seed':4501,'query_seed':4601,'query_rows':1024,'rhs_count':3,
       'relative_l2_gate':1e-7,'scope':'sampled rows of full analytic matrix action; no fit/posterior/application speedup',
       'source_sha256':{f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in ['geospsr/periodic_compact.py','geospsr/native_range.py','geospsr/native/range_plan.cpp','reproduction/benchmark_range_scale.py']},'actions':[]}
    def timed(fn):
        cp.cuda.get_current_stream().synchronize();start=time.perf_counter();v=fn();cp.cuda.get_current_stream().synchronize()
        return v,time.perf_counter()-start
    audit=GpuProcessAudit(cp.cuda.Device().pci_bus_id).start()
    try:
        x=np.array(np.load('/opt/geospsr-runtime/large-cloud-3000000/points.npy',mmap_mode='r')[:r['n']])
        w=np.random.default_rng(r['seed']).normal(size=(len(x),3));rows=np.sort(np.random.default_rng(r['query_seed']).choice(len(x),r['query_rows'],replace=False))
        r['hashes']={k:array_digest(v) for k,v in [('points',x),('weights',w),('rows',rows)]}
        wd=cp.asarray(w);r['process_rss_high_water_before_plan_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        op,r['setup_seconds']=timed(lambda:NativeCompactRangeKernel(x,max_entries=20_000_000,max_cache_bytes=4*2**30));r['plan']=op.record
        r['process_rss_high_water_after_plan_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        print('100k setup',r['setup_seconds'],op.record,flush=True)
        # Independent direct summation at sampled rows, all sources included.
        def reference():
            return cp.concatenate([analytic_covariance_rows64(x[rows[b:b+128]],x)@wd+.02*wd[rows[b:b+128]] for b in range(0,len(rows),128)]).get()
        ref,r['reference_seconds']=timed(reference);cov=ref-.02*w[rows]
        for repeat in range(4):
            actual,seconds=timed(lambda:op(wd));values=actual[rows].get()
            e=[float(np.linalg.norm(values[:,j]-ref[:,j])/np.linalg.norm(ref[:,j])) for j in range(3)]
            ke=[float(np.linalg.norm(values[:,j]-.02*w[rows,j]-cov[:,j])/np.linalg.norm(cov[:,j])) for j in range(3)]
            r['actions'].append({'seconds':seconds,'action_relative_l2_per_rhs':e,'covariance_relative_l2_per_rhs':ke,'quality_pass':max(e+ke)<=r['relative_l2_gate']})
            print('100k action',repeat,seconds,max(ke),flush=True)
        np.savez_compressed(out/'control.npz',points=x,rows=rows,reference=ref,actual=values)
        r['complete']=True
    except Exception as exc:r['failure']=str(exc);raise
    finally:
        r['process_rss_high_water_final_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        r['gpu_process_audit']=audit.stop();(out/'results.json').write_text(json.dumps(r,indent=2)+'\n')


if __name__=='__main__':main()
