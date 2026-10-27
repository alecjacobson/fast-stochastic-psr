"""Use stored common RHS to test explicit mixed-precision defect correction."""
import argparse,json,time
from pathlib import Path
import numpy as np
import cupy as cp
from geospsr.half_spectral import HalfCubeKernel,HalfCubeSpectrum
from geospsr.refinement import refine_solution
from reproduction.posterior import sync
from reproduction.common_rhs_integrity import array_digest
from reproduction.gpu_audit import GpuProcessAudit


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    old=json.loads((a.source/'results.json').read_text())
    r={'complete':False,'N':old['N'],'draws':[],'seconds':{},'correction_rtol':.1,'max_corrections':2,
       'scope':'common-RHS same-model mixed-precision solve validation; no application-speedup claim'}
    audit=GpuProcessAudit(cp.cuda.Device().pci_bus_id).start();wall=time.perf_counter()
    def timed(label,fn):
        sync();start=time.perf_counter();v=fn();sync();seconds=time.perf_counter()-start
        print(label,seconds,flush=True);return v,seconds
    def save():(a.output/'results.json').write_text(json.dumps(r,indent=2)+'\n')
    try:
        x=np.load('/opt/geospsr-runtime/large-cloud-3000000/points.npy',mmap_mode='r')[:old['N']]
        r['source_sha256']=array_digest(x)
        if r['source_sha256']!=old['source_sha256']:raise ValueError('source mismatch')
        def setup():
            return (HalfCubeKernel(x,HalfCubeSpectrum(200),.02),
                    HalfCubeKernel(x,HalfCubeSpectrum(200,dtype='float64',eps=1e-10,upsampfac=2.),.02,rhs_count=1))
        (native,double),r['seconds']['operators']=timed('operators',setup)
        for d in old['draws']:
            draw=d['draw'];rhs=np.load(a.source/f'rhs_{draw}.npy')
            if array_digest(rhs)!=d['rhs_sha256']:raise ValueError('RHS mismatch')
            rhs=cp.asarray(rhs)
            entry={'draw':draw,'rhs_sha256':d['rhs_sha256'],'correction_seconds':[],'verification_seconds':[]}
            (initial,info),entry['initial_solve_seconds']=timed('initial_solve',lambda:native.solve(rhs,rtol=1e-5,maxiter=1000))
            entry['initial_native_solve']=info
            def verify(value):
                def action():return cp.concatenate([double(value[:,c:c+1]) for c in range(3)],axis=1)
                result,seconds=timed('verify',action);entry['verification_seconds'].append(seconds);return result
            def correction(residual):
                result,seconds=timed('correction',lambda:native.solve(residual,rtol=.1,maxiter=1000))
                entry['correction_seconds'].append(seconds);return result
            (weights,refined),entry['refinement_wall_seconds']=timed('refine',lambda:refine_solution(correction,verify,rhs,initial,xp=cp))
            entry['refinement']=refined;entry['quality_pass']=refined['converged']
            entry['solve_and_verify_seconds']=entry['initial_solve_seconds']+entry['refinement_wall_seconds']
            np.save(a.output/f'weights_{draw}.npy',cp.asnumpy(weights));entry['weights_sha256']=array_digest(cp.asnumpy(weights))
            r['draws'].append(entry);save()
        r['complete']=True
    except Exception as exc:
        r['failure']=f'{type(exc).__name__}: {exc}';raise
    finally:
        r['seconds']['work_wall']=time.perf_counter()-wall;r['gpu_process_audit']=audit.stop();save()


if __name__=='__main__':main()
