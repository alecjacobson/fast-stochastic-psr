"""Bounded matrix-free tail roots: released A overlap and scale feasibility.

This measures a residual-prior component, not posterior generation or an app.
The large-N section computes bounds/degrees only, never large-N root timings.
"""
import hashlib
import json
import time
from pathlib import Path
import cupy as cp
import numpy as np
from geospsr.analytic_tail import AnalyticTailAction
from geospsr.periodic_leaf_cache import LeafCachedRangeKernel
from geospsr.spectral import FourierKernel
from geospsr.training_spectrum import FourierCovariance, finite_covariance_rows
from geospsr.spectral_tail import analytic_covariance_rows64
from geospsr.square_root import FixedChebyshevRoot, analytic_tail_spectral_bound
from reproduction.common_rhs_integrity import array_digest
from reproduction.gpu_audit import GpuProcessAudit


def main():
    root = Path(__file__).resolve().parents[1]
    out = root/'reproduction/results/analytic_tail'; out.mkdir(exist_ok=True)
    files = ['geospsr/square_root.py','geospsr/inverse_root.py','geospsr/analytic_tail.py',
             'geospsr/training_spectrum.py','geospsr/spectral.py','geospsr/periodic_leaf_cache.py',
             'reproduction/benchmark_analytic_tail.py']
    record = {'complete':False, 'scope':'tail-plus-noise root component; no posterior/application speedup',
              'root_relative_tolerance':1e-7, 'paired_draw_count':3, 'seed':4624,
              'gates':{'action_relative_l2':1e-8,'paired_root_relative_l2':1e-7,
                       'dense_polynomial_whitened_covariance_operator_error':2.1e-7},
              'source_sha256':{f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in files},
              'cases':{}, 'scale_bounds':{}}
    def timed(fn):
        cp.cuda.get_current_stream().synchronize(); start=time.perf_counter()
        value=fn(); cp.cuda.get_current_stream().synchronize()
        return value,time.perf_counter()-start
    audit = GpuProcessAudit(cp.cuda.Device().pci_bus_id).start()
    try:
        x = np.load(root/'reproduction/results/matched/A/input.npz')['points']
        xi = cp.asarray(np.random.default_rng(4624).normal(size=(len(x),3)))
        for label,ell,noise in [('A',.01,.02),('A_short_low_noise',.005,.0004),('A_long',1.,.00001)]:
            v = {'n':len(x),'lengthscale':ell,'noise_variance':noise,'cutoff':50,'variance_per_axis':.1,
                 'input_sha256':array_digest(x),'runs':[]}; record['cases'][label]=v
            bound,v['bound_seconds'] = timed(lambda:analytic_tail_spectral_bound(x,50,ell,noise_variance=noise,grid=128))
            poly,v['polynomial_setup_seconds'] = timed(lambda:FixedChebyshevRoot(bound['lower'],bound['upper'],1e-7))
            v['bound'],v['polynomial'] = bound,poly.record
            spectrum,v['finite_spectrum_seconds'] = timed(lambda:FourierCovariance(50,ell,dtype='float64',eps=1e-12,upsampfac=2.))
            finite,v['finite_plan_seconds'] = timed(lambda:FourierKernel(x,spectrum,0,rhs_count=3))
            analytic,v['analytic_plan_seconds'] = timed(lambda:LeafCachedRangeKernel(x,ell,.1,noise,leaf_size=128))
            action = AnalyticTailAction(analytic,finite)
            def matrix():
                a = analytic_covariance_rows64(x,x,ell)
                a.flat[::len(x)+1] += noise
                low = finite_covariance_rows(x,x,spectrum,dtype='float64')
                return a-(low+low.T)*.5
            target,v['dense_covariance_seconds'] = timed(matrix)
            chol,v['dense_cholesky_seconds'] = timed(lambda:cp.linalg.cholesky(target))
            (eig,vec),v['reference_eigh_seconds'] = timed(lambda:cp.linalg.eigh(target))
            assert float(eig.min()) >= noise*(1-1e-8) and float(eig.max()) <= bound['upper']*(1+1e-10)
            v['eigenvalue_range']=[float(eig.min()),float(eig.max())]
            reference = (vec*cp.sqrt(eig))@(vec.T@xi)
            v['action_relative_l2'] = float(cp.linalg.norm(action(xi)-target@xi)/cp.linalg.norm(target@xi))
            dense_root = poly.apply(lambda w:target@w,cp.eye(len(x),dtype='d'),xp=cp)
            whiten = (vec/cp.sqrt(eig))@vec.T
            covariance_error = whiten@(dense_root@dense_root.T-target)@whiten
            v['dense_polynomial_whitened_covariance_operator_error'] = float(cp.max(cp.abs(cp.linalg.eigvalsh((covariance_error+covariance_error.T)*.5))))
            for repeat in range(4):
                actual,sec = timed(lambda:poly.apply(action,xi,xp=cp))
                _,dense_sec = timed(lambda:chol@xi)
                error = float(cp.linalg.norm(actual-reference)/cp.linalg.norm(reference))
                v['runs'].append({'root_seconds':sec,'dense_cholesky_draw_seconds':dense_sec,
                                  'paired_root_relative_l2':error})
                print(label,repeat,'degree',poly.degree,'root',sec,'paired error',error,flush=True)
            # Dense Cholesky and symmetric root have the same intended law,
            # but are not equal pathwise for the same white-noise vector.
            v['dense_setup_first_draw_seconds'] = v['finite_spectrum_seconds']+v['dense_covariance_seconds']+v['dense_cholesky_seconds']+v['runs'][0]['dense_cholesky_draw_seconds']
            v['matrix_free_setup_first_draw_seconds'] = sum(v[k] for k in ['bound_seconds','polynomial_setup_seconds','finite_spectrum_seconds','finite_plan_seconds','analytic_plan_seconds'])+v['runs'][0]['root_seconds']
            v['quality_pass'] = bool(v['action_relative_l2']<=1e-8 and max(a['paired_root_relative_l2'] for a in v['runs'])<=1e-7 and v['dense_polynomial_whitened_covariance_operator_error']<=2.1e-7)
            np.savez_compressed(out/f'{label}_controls.npz',white_noise=xi.get(),reference=reference.get(),actual=actual.get(),eigenvalues=eig.get(),coefficients=poly.coefficients)
            del spectrum,finite,analytic,action,target,chol,eig,vec,dense_root,whiten,covariance_error,actual,reference
            cp.get_default_memory_pool().free_all_blocks()
        for label,path in [('B','reproduction/results/matched/B/input.npz'),('million','reproduction/results/range_compact/scale1m/control.npz')]:
            points=np.load(root/path)['points']; row={'n':len(points),'input_sha256':array_digest(points),'settings':[]}
            record['scale_bounds'][label]=row
            for cutoff in [50,200]:
                for ell,noise in [(.01,.02),(.005,.0004)]:
                    bound,seconds=timed(lambda:analytic_tail_spectral_bound(points,cutoff,ell,noise_variance=noise,grid=128))
                    poly=FixedChebyshevRoot(bound['lower'],bound['upper'],1e-7)
                    row['settings'].append({'cutoff':cutoff,'lengthscale':ell,'noise_variance':noise,'bound_seconds':seconds,'bound':bound,'polynomial':poly.record,'root_measured':False})
                    print('bound only',label,cutoff,ell,'degree',poly.degree,flush=True)
        record['complete']=True
    except Exception as exc:
        record['failure']=str(exc);raise
    finally:
        record['gpu_process_audit']=audit.stop()
        (out/'results.json').write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':main()
