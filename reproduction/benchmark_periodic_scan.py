"""Fresh serial/warp-scan comparison against archived independent dense actions."""
import hashlib,json,time
from pathlib import Path
import cupy as cp
import numpy as np
from geospsr.periodic_range import PeriodicRangeKernel
from geospsr.periodic_scan import PeriodicScanKernel
from reproduction.common_rhs_integrity import array_digest
from reproduction.gpu_audit import GpuProcessAudit


def main():
    root=Path(__file__).resolve().parents[1];out=root/'reproduction/results/range_scan'
    r={'complete':False,'scope':'matrix actions only','cases':{},'relative_l2_gate':1e-7,
       'shared_scratch_bytes_per_scan_block':4096,'additional_global_scratch_bytes':0,
       'source_sha256':{f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in ['geospsr/periodic_scan.py','geospsr/periodic_range.py','reproduction/benchmark_periodic_scan.py']}}
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
            for name,cls in [('serial',PeriodicRangeKernel),('scan',PeriodicScanKernel)]:
                op,setup=timed(lambda:cls(x));m={'setup_seconds':setup,'plan':op.record,'actions':[]};v['methods'][name]=m
                for repeat in range(4):
                    y,seconds=timed(lambda:op(wd));y=y.get();outputs[name]=y
                    e=[float(np.linalg.norm(y[:,j]-reference[:,j])/np.linalg.norm(reference[:,j])) for j in range(3)]
                    ke=[float(np.linalg.norm(y[:,j]-.02*w[:,j]-cov[:,j])/np.linalg.norm(cov[:,j])) for j in range(3)]
                    m['actions'].append({'seconds':seconds,'action_relative_l2_per_rhs':e,'covariance_relative_l2_per_rhs':ke,'quality_pass':max(e+ke)<=r['relative_l2_gate']})
                print(case,name,m,flush=True)
                del op
            np.savez_compressed(out/f'{case}_actions.npz',**outputs)
        r['complete']=True
    except Exception as exc:r['failure']=str(exc);raise
    finally:
        r['gpu_process_audit']=audit.stop();(out/'results.json').write_text(json.dumps(r,indent=2)+'\n')


if __name__=='__main__':main()
