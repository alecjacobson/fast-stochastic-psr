"""NUFFT initial fits, analytic CG warm starts, and verified defect correction."""
import hashlib,json,time
from pathlib import Path
import cupy as cp
import cupyx.scipy.linalg as cla
import numpy as np
from geospsr.periodic_leaf_cache import LeafCachedRangeKernel
from geospsr.spectral import FourierPoisson
from geospsr.packed_spectral import PackedFourierKernel
from geospsr.analytic_refinement import refine_analytic
from geospsr.spectral_tail import analytic_covariance_rows64
from reproduction.benchmark_leaf_conditioning import SolveAdapter
from reproduction.common_rhs_integrity import array_digest
from reproduction.gpu_audit import GpuProcessAudit


def main():
    root=Path(__file__).resolve().parents[1];out=root/'reproduction/results/analytic_refinement'
    r={'complete':False,'scope':'full analytic mean fits; finite scalar/cross50; no SDD/posterior/application claim',
       'gates':{'residual':1e-8,'weights_relative_l2':1e-7,'mean_relative_l2':1e-7},'cases':{},
       'source_sha256':{f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in ['geospsr/analytic_refinement.py','geospsr/periodic_leaf_cache.py','geospsr/spectral.py','geospsr/packed_spectral.py','reproduction/benchmark_analytic_refinement.py']}}
    def timed(fn):
        cp.cuda.get_current_stream().synchronize();start=time.perf_counter();v=fn();cp.cuda.get_current_stream().synchronize();return v,time.perf_counter()-start
    audit=GpuProcessAudit(cp.cuda.Device().pci_bus_id).start()
    try:
        for case,ell,noise,label in [('A',.01,.02,'A'),('B',.01,.02,'B'),('B',.005,.0004,'B_short_low_noise')]:
            data=np.load(root/f'reproduction/results/matched/{case}/input.npz');x,y,q=[data[k] for k in ['points','normals','query']]
            v={'n':len(x),'lengthscale':ell,'noise_variance':noise,'methods':{k:{'runs':[]} for k in ['plain','nufft','warm_cg','defect']},
               'hashes':{k:array_digest(a) for k,a in [('points',x),('normals',y),('query',q)]}}
            r['cases'][label]=v;yd=cp.asarray(y,dtype='d')
            def matrix():
                k=analytic_covariance_rows64(x,x,ell,.1);k.flat[::len(x)+1]+=noise;return k
            k,v['dense_matrix_seconds']=timed(matrix);chol,v['dense_factor_seconds']=timed(lambda:cp.linalg.cholesky(k))
            def dense_solve():return cla.solve_triangular(chol.T,cla.solve_triangular(chol,yd,lower=True),lower=False)
            ref,v['dense_first_solve_seconds']=timed(dense_solve);v['dense_reused_seconds']=[]
            for _ in range(3):_,sec=timed(dense_solve);v['dense_reused_seconds'].append(sec)
            kernel,v['analytic_setup_seconds']=timed(lambda:LeafCachedRangeKernel(x,ell,.1,noise,leaf_size=128))
            v['analytic_plan']=kernel.record;analytic=SolveAdapter(kernel)
            spectral,v['finite_spectrum_setup_seconds']=timed(lambda:FourierPoisson(200,ell,.1,upsampfac=1.25))
            finite,v['finite_plan_setup_seconds']=timed(lambda:PackedFourierKernel(x,spectral,noise,rhs_count=3))
            def approx(rhs):return finite.solve(rhs,rtol=1e-5,maxiter=500)
            last={}
            def assess(name,weights,info,sec,repeat,initial=0.):
                last[name]=weights
                residual=(cp.linalg.norm(k@weights.astype('d')-yd,axis=0)/cp.linalg.norm(yd,axis=0)).get().tolist()
                error=(cp.linalg.norm(weights.astype('d')-ref,axis=0)/cp.linalg.norm(ref,axis=0)).get().tolist()
                value={'seconds':sec,'initial_nufft_seconds':initial,'solve_total_seconds':sec+initial,'info':info,
                       'independent_residual_per_rhs':residual,'weights_relative_l2_per_rhs':error,
                       'fit_quality_pass':max(residual)<=1e-8 and max(error)<=1e-7}
                v['methods'][name]['runs'].append(value)
                print(label,name,repeat,sec+initial,info.get('iterations',info.get('corrections')),max(residual),flush=True)
            for repeat in range(4):
                (a,info),sec=timed(lambda:analytic.solve(yd,rtol=5e-9,maxiter=1000));assess('plain',a,info,sec,repeat)
                (initial,iinfo),isec=timed(lambda:approx(yd));assess('nufft',initial,iinfo,isec,repeat)
                (a,info),sec=timed(lambda:analytic.solve(yd,rtol=5e-9,maxiter=1000,x0=initial));assess('warm_cg',a,info,sec,repeat,isec)
                (a,info),sec=timed(lambda:refine_analytic(kernel,approx,yd,initial,rtol=1e-8,max_corrections=8));assess('defect',a,info,sec,repeat,isec)
            scalar,v['shared_scalar_setup_seconds']=timed(lambda:FourierPoisson(50,ell,.1,dtype='float64',eps=1e-12))
            def mean(weights):
                c=scalar.aggregate(x,weights);return scalar.evaluate(q,c)[0]-scalar.evaluate(x,c).mean()
            meanref,v['dense_mean_seconds']=timed(lambda:mean(ref));arrays={'reference_weights':ref.get(),'reference_mean':meanref.get()}
            for name,weights in last.items():
                values,sec=timed(lambda:mean(weights));m=v['methods'][name]
                m['mean_seconds']=sec;m['mean_relative_l2']=float(cp.linalg.norm(values-meanref)/cp.linalg.norm(meanref))
                m['quality_pass']=all(a['fit_quality_pass'] for a in m['runs']) and m['mean_relative_l2']<=1e-7
                arrays[name+'_weights']=weights.get();arrays[name+'_mean']=values.get()
            np.savez_compressed(out/f'{label}_fields.npz',**arrays)
            del kernel,analytic,finite,spectral,scalar,k,chol,ref,last,initial,a,weights,values,meanref,yd
            cp.get_default_memory_pool().free_all_blocks()
        r['complete']=True
    except Exception as exc:r['failure']=str(exc);raise
    finally:
        r['gpu_process_audit']=audit.stop();(out/'results.json').write_text(json.dumps(r,indent=2)+'\n')


if __name__=='__main__':main()
