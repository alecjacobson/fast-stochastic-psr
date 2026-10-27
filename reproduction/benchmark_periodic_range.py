"""Full analytic range action versus dense analytic and packed cube200 NUFFT."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import cupy as cp
import numpy as np
from geospsr.periodic_range import PeriodicRangeKernel
from geospsr.spectral import FourierPoisson
from geospsr.packed_spectral import PackedFourierKernel
from geospsr.spectral_tail import analytic_covariance_rows64
from reproduction.common_rhs_integrity import array_digest
from reproduction.gpu_audit import GpuProcessAudit


def main():
    p=argparse.ArgumentParser();p.add_argument('--case',choices=['A','B'],required=True)
    args=p.parse_args();root=Path(__file__).resolve().parents[1]
    out=root/'reproduction/results/periodic_range'/args.case;out.mkdir(parents=True,exist_ok=True)
    r={'complete':False,'case':args.case,'model':{'lengthscale':.01,'variance_per_axis':.1,'noise_variance':.02},
       'seed':4501,'rhs_count':3,'repetitions_after_first':3,'relative_l2_gate':1e-7,
       'scope':'matrix action only; no CG, posterior, or application speedup',
       'device':cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),'methods':{}}
    r['source_sha256']={f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in ['geospsr/periodic_range.py','reproduction/benchmark_periodic_range.py']}
    def timed(fn):
        cp.cuda.get_current_stream().synchronize();start=time.perf_counter();v=fn()
        cp.cuda.get_current_stream().synchronize();return v,time.perf_counter()-start
    audit=GpuProcessAudit(cp.cuda.Device().pci_bus_id).start()
    try:
        x=np.load(root/f'reproduction/results/matched/{args.case}/input.npz')['points']
        if x.dtype!=np.float32:raise ValueError('reference expects float32 coordinates')
        w=np.random.default_rng(r['seed']).normal(size=(len(x),3))
        r['n']=len(x);r['hashes']={'points':array_digest(x),'weights':array_digest(w)}
        (xd,wd),r['shared_input_transfer_seconds']=timed(lambda:(cp.asarray(x),cp.asarray(w)))
        k,seconds=timed(lambda:analytic_covariance_rows64(xd,xd))
        truth,first=timed(lambda:k@wd+.02*wd)
        d={'setup_seconds':seconds,'first_action_seconds':first,'retained_matrix_bytes':k.nbytes,'subsequent_seconds':[]}
        r['methods']['dense']=d
        for _ in range(3):_,seconds=timed(lambda:k@wd+.02*wd);d['subsequent_seconds'].append(seconds)
        reference=truth.get();covreference=reference-.02*w
        outputs={'reference':reference,'weights':w}
        del k;cp.get_default_memory_pool().free_all_blocks()
        for name in ['range','packed_nufft200']:
            if name=='range':
                op,setup=timed(lambda:PeriodicRangeKernel(x,leaf_size=32))
            else:
                spectral,spectral_setup=timed(lambda:FourierPoisson(200,upsampfac=1.25))
                op,setup=timed(lambda:PackedFourierKernel(xd,spectral,.02,rhs_count=3));setup+=spectral_setup
            value,first=timed(lambda:op(wd));m={'setup_seconds':setup,'first_action_seconds':first,'subsequent_seconds':[],'quality':[]}
            if name=='range':m['plan']=op.record
            else:m.update(dtype='float32',cutoff=200,upsampfac=1.25,eps=1e-6)
            r['methods'][name]=m
            def measure(v):
                y=v.get();outputs[name]=y
                errors=[float(np.linalg.norm(y[:,j]-reference[:,j])/np.linalg.norm(reference[:,j])) for j in range(3)]
                kernel_errors=[float(np.linalg.norm(y[:,j]-.02*w[:,j]-covreference[:,j])/np.linalg.norm(covreference[:,j])) for j in range(3)]
                return {'action_relative_l2_per_rhs':errors,'covariance_relative_l2_per_rhs':kernel_errors,
                        'quality_pass':max(errors+kernel_errors)<=r['relative_l2_gate']}
            m['quality'].append(measure(value))
            for repeat in range(3):
                value,seconds=timed(lambda:op(wd));m['subsequent_seconds'].append(seconds);m['quality'].append(measure(value))
            print(args.case,name,json.dumps(m),flush=True)
            del op,value
            if name!='range':del spectral
            cp.get_default_memory_pool().free_all_blocks()
        np.savez_compressed(out/'actions.npz',**outputs)
        r['complete']=True
    except Exception as exc:r['failure']=str(exc);raise
    finally:
        r['gpu_process_audit']=audit.stop()
        (out/'results.json').write_text(json.dumps(r,indent=2)+'\n')


if __name__=='__main__':main()
