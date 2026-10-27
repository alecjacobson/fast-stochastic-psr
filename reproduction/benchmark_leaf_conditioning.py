"""Full analytic leaf128-range CG versus independent dense Cholesky mean fits."""
import hashlib,json,time
from pathlib import Path
from types import SimpleNamespace
import cupy as cp
import cupyx.scipy.linalg as cla
import numpy as np
from geospsr.periodic_leaf_cache import LeafCachedRangeKernel
from geospsr.spectral import FourierKernel,FourierPoisson
from geospsr.spectral_tail import analytic_covariance_rows64
from reproduction.common_rhs_integrity import array_digest
from reproduction.gpu_audit import GpuProcessAudit


class SolveAdapter:
    def __init__(self,kernel):self.kernel=kernel;self.op=SimpleNamespace(dtype=np.dtype('float64'))
    def __call__(self,w):return self.kernel(w)
    solve=FourierKernel.solve


def main():
    root=Path(__file__).resolve().parents[1];out=root/'reproduction/results/range_hybrid/conditioning';out.mkdir(exist_ok=True)
    r={'complete':False,'scope':'mean fitting pilot, full analytic training and finite scalar/cross cutoff50; no SDD or application comparison',
       'gates':{'residual':1e-8,'weights_relative_l2':1e-7,'mean_relative_l2':1e-7},'cases':{},
       'source_sha256':{f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in ['geospsr/periodic_range.py','geospsr/periodic_scan.py','geospsr/periodic_cached.py','geospsr/spectral.py','geospsr/periodic_compact.py','geospsr/native_range.py','geospsr/native/range_plan.cpp','geospsr/periodic_hybrid.py','geospsr/periodic_leaf_cache.py','reproduction/benchmark_leaf_conditioning.py']}}
    def timed(fn):
        cp.cuda.get_current_stream().synchronize();start=time.perf_counter();v=fn();cp.cuda.get_current_stream().synchronize()
        return v,time.perf_counter()-start
    audit=GpuProcessAudit(cp.cuda.Device().pci_bus_id).start()
    try:
        for case,ell,noise,label in [('A',.01,.02,'A'),('B',.01,.02,'B'),('B',.005,.0004,'B_short_low_noise')]:
            data=np.load(root/f'reproduction/results/matched/{case}/input.npz');x,y,q=[data[k] for k in ['points','normals','query']]
            v={'n':len(x),'lengthscale':ell,'noise_variance':noise,'variance_per_axis':.1,'methods':{},
               'hashes':{k:array_digest(a) for k,a in [('points',x),('normals',y),('query',q)]}}
            r['cases'][label]=v;xd=cp.asarray(x);yd=cp.asarray(y,dtype='d')
            def matrix():
                k=analytic_covariance_rows64(xd,xd,ell,.1);k.flat[::len(x)+1]+=noise;return k
            k,v['dense_matrix_seconds']=timed(matrix)
            chol,v['dense_factor_seconds']=timed(lambda:cp.linalg.cholesky(k))
            def dense_solve():return cla.solve_triangular(chol.T,cla.solve_triangular(chol,yd,lower=True),lower=False)
            reference,first=timed(dense_solve);v['methods']['dense']={'first_solve_seconds':first,'subsequent_solve_seconds':[]}
            for _ in range(3):_,seconds=timed(dense_solve);v['methods']['dense']['subsequent_solve_seconds'].append(seconds)
            v['dense_residual_per_rhs']=(cp.linalg.norm(k@reference-yd,axis=0)/cp.linalg.norm(yd,axis=0)).get().tolist()
            print(label,'dense factor',v['dense_factor_seconds'],flush=True)
            kernel,v['range_setup_seconds']=timed(lambda:LeafCachedRangeKernel(x,ell,.1,noise,leaf_size=128))
            v['range_plan']=kernel.record;op=SolveAdapter(kernel);m={'solves':[]};v['methods']['range']=m
            for repeat in range(4):
                (alpha,info),seconds=timed(lambda:op.solve(yd,rtol=5e-9,maxiter=1000))
                residual=(cp.linalg.norm(k@alpha-yd,axis=0)/cp.linalg.norm(yd,axis=0)).get().tolist()
                err=(cp.linalg.norm(alpha-reference,axis=0)/cp.linalg.norm(reference,axis=0)).get().tolist()
                m['solves'].append({'seconds':seconds,'info':info,'independent_residual_per_rhs':residual,
                                    'weights_relative_l2_per_rhs':err,'quality_pass':bool(max(residual)<=1e-8 and max(err)<=1e-7 and info['converged'])})
                print(label,'range solve',repeat,seconds,info['iterations'],max(residual),flush=True)
            scalar,v['shared_scalar_operator_seconds']=timed(lambda:FourierPoisson(50,ell,.1,dtype='float64',eps=1e-12))
            def mean(weights):
                coefficients=scalar.aggregate(xd,weights);offset=scalar.evaluate(xd,coefficients).mean()
                return scalar.evaluate(q,coefficients)[0]-offset
            meanref,v['dense_mean_seconds']=timed(lambda:mean(reference))
            meanactual,v['range_mean_seconds']=timed(lambda:mean(alpha))
            v['mean_relative_l2']=float(cp.linalg.norm(meanactual-meanref)/cp.linalg.norm(meanref))
            v['quality_pass']=bool(max(v['dense_residual_per_rhs'])<=1e-8 and all(s['quality_pass'] for s in m['solves']) and v['mean_relative_l2']<=1e-7)
            np.savez_compressed(out/f'{label}_fields.npz',reference_weights=reference.get(),range_weights=alpha.get(),reference_mean=meanref.get(),range_mean=meanactual.get())
            del kernel,op,scalar,k,chol,reference,alpha,meanref,meanactual
            cp.get_default_memory_pool().free_all_blocks()
        r['complete']=True
    except Exception as exc:r['failure']=str(exc);raise
    finally:
        r['gpu_process_audit']=audit.stop();(out/'results.json').write_text(json.dumps(r,indent=2)+'\n')


if __name__=='__main__':main()
