"""Distinguish native residual floors from same-model double residuals."""
import argparse,json,time
from pathlib import Path
import numpy as np
import cupy as cp
from geospsr.half_spectral import HalfCubeKernel,HalfCubeSpectrum
from geospsr.spectral import FourierPoisson
from reproduction.posterior import sync
from reproduction.common_rhs_integrity import array_digest
from reproduction.gpu_audit import GpuProcessAudit


def main():
    p=argparse.ArgumentParser();p.add_argument('--count',type=int,choices=[1000000,3000000],required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    r={'complete':False,'N':a.count,'draws':[],'seconds':{},'double_rhs_batch':1,
       'scope':'same finite cube residual diagnostic; not application or covariance quality'}
    audit=GpuProcessAudit(cp.cuda.Device().pci_bus_id).start();wall=time.perf_counter()
    def save():(a.output/'results.json').write_text(json.dumps(r,indent=2)+'\n')
    def timed(label,fn):
        sync();start=time.perf_counter();v=fn();sync();elapsed=time.perf_counter()-start
        print(label,elapsed,flush=True);return v,elapsed
    try:
        x=np.load('/opt/geospsr-runtime/large-cloud-3000000/points.npy',mmap_mode='r')[:a.count]
        r['source_sha256']=array_digest(x)
        def setup():
            native=HalfCubeKernel(x,HalfCubeSpectrum(200),.02)
            double=HalfCubeKernel(x,HalfCubeSpectrum(200,dtype='float64',eps=1e-10,upsampfac=2.),.02,rhs_count=1)
            return native,double,FourierPoisson(50)
        (native,double,op),r['seconds']['operators']=timed('operators',setup)
        rng=cp.random.RandomState(509)
        for draw in range(2):
            entry={'draw':draw,'attempts':[]}
            def rhs_fn():
                v,_=native.joint_prior(rng,op)
                return v+np.float32(np.sqrt(.02))*rng.standard_normal(v.shape,dtype='f')
            rhs,entry['rhs_seconds']=timed('rhs',rhs_fn)
            rhs64=rhs.astype('d');denom=cp.linalg.norm(rhs64,axis=0)
            np.save(a.output/f'rhs_{draw}.npy',cp.asnumpy(rhs))
            entry['rhs_sha256']=array_digest(np.load(a.output/f'rhs_{draw}.npy',mmap_mode='r'))
            weights=None
            for attempt in range(3):
                (weights,info),seconds=timed(f'draw{draw}_attempt{attempt}',lambda:native.solve(rhs,rtol=1e-5,maxiter=1000,x0=weights))
                def validation():
                    errors=[]
                    for column in range(3):
                        residual=double(weights[:,column:column+1].astype('d'))[:,0]-rhs64[:,column]
                        errors.append(cp.linalg.norm(residual)/denom[column])
                    return cp.asnumpy(cp.stack(errors))
                errors,validation_seconds=timed('double_residual',validation)
                passed=bool(np.max(errors)<=2e-5)
                entry['attempts'].append({'attempt':attempt,'native_solve':info,'solve_seconds':seconds,
                    'double_relative_residual':errors.tolist(),'double_validation_seconds':validation_seconds,'double_pass':passed})
                print('residuals',info['relative_residual'],errors.tolist(),flush=True)
                if passed:break
            entry['quality_pass']=passed
            entry['weights_sha256']=array_digest(cp.asnumpy(weights))
            np.save(a.output/f'weights_{draw}.npy',cp.asnumpy(weights))
            r['draws'].append(entry);save()
        r['complete']=True
    except Exception as exc:
        r['failure']=f'{type(exc).__name__}: {exc}';raise
    finally:
        r['seconds']['work_wall']=time.perf_counter()-wall;r['gpu_process_audit']=audit.stop();save()


if __name__=='__main__':main()
