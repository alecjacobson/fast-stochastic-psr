"""Posterior covariance bracket for all omitted positive-block frequencies.

Dense A/B controls validate corners and outside-cube contributions separately.
Massive inputs use block NUFFT solves and streamed frequency-feature estimates.
Empirical errors/intervals are diagnostics, not simultaneous certificates.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import cupy as cp
import numpy as np
from geospsr.block_spectral import BlockSpectrum,BlockKernel
from geospsr.block_tail import sample_block_tail_frequencies
from geospsr.spectral import FourierPoisson
from geospsr.spectral_blocks import axis_weights
from geospsr.spectral_tail import analytic_covariance_rows64,tail_quadratic_samples
from geospsr.training_spectrum import finite_covariance_rows
from reproduction.audit_centered_hybrid import chol_solve
from reproduction.block_reference import block_covariance_rows
from reproduction.gpu_audit import GpuProcessAudit

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset',choices=['A','B','1000000'],default='A')
    p.add_argument('--core',type=int,default=100)
    p.add_argument('--extent',type=int,default=200)
    p.add_argument('--lengthscale',type=float,default=.01)
    p.add_argument('--noise',type=float,default=.02)
    p.add_argument('--frequencies',type=int,default=16384,help='per nonempty stratum and seed')
    p.add_argument('--seeds',type=int,nargs='+',default=[71,72,73])
    p.add_argument('--precision',choices=['float32','float64'],default='float32')
    p.add_argument('--rtol',type=float,default=1e-5)
    p.add_argument('--eps',type=float,default=1e-6)
    p.add_argument('--probe-indices',type=int,nargs='+')
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    if not 50<=args.core<=args.extent or args.frequencies<2:
        p.error('need 50 <= core <= extent and at least two frequencies')
    if min(args.lengthscale,args.noise,args.rtol,args.eps)<=0:
        p.error('length scale, noise and tolerances must be positive')
    dense = args.dataset in ('A','B')
    if dense:
        data = np.load(ROOT/f'reproduction/results/matched/{args.dataset}/input.npz')
        x = data['points'].astype('f')
    else:
        data = np.load(ROOT/'reproduction/results/teaser/cloud_1000000.npz')
        x = data['points'][np.random.default_rng(19).permutation(len(data['points']))].astype('f')
    panel = np.load(ROOT/f'reproduction/results/spectral_tail/{args.dataset}/fields.npz')
    ids = np.arange(len(panel['query'])) if args.probe_indices is None else np.asarray(args.probe_indices)
    if len(np.unique(ids))!=len(ids) or min(ids)<0 or max(ids)>=len(panel['query']):
        p.error('probe indices must be distinct members of the saved panel')
    q = panel['query'][ids].astype('f')
    n,count = len(x),len(q)
    args.output.mkdir(parents=True,exist_ok=True)
    record = {'args':{**vars(args),'output':str(args.output)},'N':n,'complete':False,
              'normalization':'analytic','dense_reference':dense,'seconds':{},'seeds':[],
              'scope':'fixed-panel covariance diagnostic for finite scalar/cross50 and full analytic vector model; not application convergence',
              'hashes':{k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in [('points',x),('queries',q)]},
              'device':cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}
    arrays = {'query':q,'query_ids':panel['query_ids'][ids],'probe_indices':ids}
    def save():
        (args.output/'results.json').write_text(json.dumps(record,indent=2)+'\n')
    def timed(name,fn):
        cp.cuda.runtime.deviceSynchronize();begin=time.perf_counter()
        value=fn();cp.cuda.runtime.deviceSynchronize()
        record['seconds'][name]=time.perf_counter()-begin
        save();print(name,record['seconds'][name],flush=True)
        return value
    audit = GpuProcessAudit(cp.cuda.Device().pci_bus_id).start()
    try:
        dtype = 'float64' if dense else args.precision
        scalar = FourierPoisson(50,args.lengthscale,dtype=dtype,eps=1e-11 if dense else args.eps,upsampfac=2.)
        xg = cp.asarray(x,dtype=dtype)
        cross = timed('cross',lambda:cp.stack([scalar.cross_values(qi[None]-xg) for qi in cp.asarray(q,dtype=dtype)],axis=1))
        norm2 = (scalar.ns**2).sum(0)
        rho = cp.where(norm2>0,scalar.rho/cp.maximum(norm2,1),0)
        prior = timed('scalar_prior',lambda:scalar.evaluate((q.astype(dtype)[:,None]-q.astype(dtype)[None]).reshape(-1,3),rho)[0].reshape(count,count).astype('d'))
        exact = {}
        if dense:
            block = timed('block_matrix',lambda:block_covariance_rows(x,x,args.core,args.extent,args.lengthscale))
            full = timed('analytic_matrix',lambda:analytic_covariance_rows64(x,x,args.lengthscale))
            # The separable reference requires only 1D coefficients. Never
            # allocate an extent^3 Fourier cube merely to validate its rows.
            axis = SimpleNamespace(cutoff=args.extent,axis64=cp.asarray(axis_weights(np.arange(-args.extent,args.extent+1),args.lengthscale)))
            cube = timed('cube_matrix',lambda:finite_covariance_rows(x,x,axis,dtype='d'))
            block_diagonal,full_diagonal = block.diagonal().copy(),full.diagonal().copy()
            block.flat[::n+1] += args.noise
            full.flat[::n+1] += args.noise
            lb = timed('block_cholesky',lambda:cp.linalg.cholesky(block))
            a = timed('block_cross_solve',lambda:chol_solve(lb,cross.reshape(n,-1)).reshape(n,count,3))
            lf = timed('analytic_cholesky',lambda:cp.linalg.cholesky(full))
            afull = timed('analytic_cross_solve',lambda:chol_solve(lf,cross.reshape(n,-1)).reshape(n,count,3))
            full_cov = prior-cp.einsum('nqd,nrd->qr',cross,afull)
            delta = cube-block
            # Avoid subtracting the observation noise and adding it back to
            # recover a tiny spectral diagonal, especially at long scales.
            delta.flat[::n+1] = cube.diagonal()-block_diagonal
            exact['corners'] = cp.asnumpy(cp.einsum('nqd,nrd->qr',a,(delta@a.reshape(n,-1)).reshape(n,count,3)))
            delta = full-cube
            delta.flat[::n+1] = full_diagonal-cube.diagonal()
            exact['outside'] = cp.asnumpy(cp.einsum('nqd,nrd->qr',a,(delta@a.reshape(n,-1)).reshape(n,count,3)))
            del block,full,cube,lb,lf,afull,delta,axis,block_diagonal,full_diagonal
        else:
            spectrum = timed('block_spectrum',lambda:BlockSpectrum(args.core,args.extent,args.lengthscale,dtype=dtype,eps=args.eps))
            kernel = timed('block_plans',lambda:BlockKernel(xg,spectrum,args.noise))
            weights,records = [],[]
            def solve_all():
                for j in range(count):
                    value,info = kernel.solve(cross[:,j],rtol=args.rtol,maxiter=1000)
                    if not info['converged']:
                        raise RuntimeError(info)
                    weights.append(value);records.append(info)
                    print('probe solve',j+1,info['iterations'],flush=True)
                return cp.stack(weights,axis=1)
            a = timed('block_cross_solves',solve_all)
            record['solves'] = records
            del weights,kernel,spectrum
        finite_cov = prior-cp.einsum('nqd,nrd->qr',cross.astype('d'),a.astype('d'))
        finite_cov = cp.asnumpy((finite_cov+finite_cov.T)/2)
        variance,scale = np.diag(finite_cov),np.linalg.norm(finite_cov)
        if variance.min()<=0:
            raise ArithmeticError('nonpositive finite posterior variance')
        arrays['finite_covariance'] = finite_cov
        if dense:
            bound = sum(exact.values());bound=(bound+bound.T)/2
            gap = cp.asnumpy(full_cov)-finite_cov;gap=(gap+gap.T)/2
            record['exact'] = {'covariance_gap_relative_frobenius':float(np.linalg.norm(gap)/scale),
                               'bound_relative_frobenius':float(np.linalg.norm(bound)/scale),
                               'trace_bound_over_covariance_frobenius':float(np.trace(bound)/scale),
                               'trace_by_region_over_covariance_frobenius':{k:float(np.trace(v)/scale) for k,v in exact.items()},
                               'minimum_gap_eigenvalue':float(np.linalg.eigvalsh(gap).min()),
                               'minimum_bound_minus_gap_eigenvalue':float(np.linalg.eigvalsh(bound-gap).min()),
                               'gap_roundoff_indicator':64*np.finfo(float).eps,
                               'covariance_gap_resolved_above_roundoff':bool(np.linalg.norm(gap)/scale>64*np.finfo(float).eps)}
            if min(record['exact']['minimum_gap_eigenvalue'],record['exact']['minimum_bound_minus_gap_eigenvalue']) < -1e-10*scale:
                raise ArithmeticError('PSD covariance bracket failed')
            record['exact']['bracket_within_numerical_tolerance'] = True
            arrays.update(analytic_covariance=cp.asnumpy(full_cov),exact_bound=bound,
                          exact_corner_bound=exact['corners'],exact_outside_bound=exact['outside'])
            print('exact',json.dumps(record['exact']),flush=True)
        del scalar,cross,prior
        gc.collect();cp.get_default_memory_pool().free_all_blocks()
        for seed in args.seeds:
            strata = sample_block_tail_frequencies(args.frequencies,args.core,args.extent,args.lengthscale,seed=seed)
            total = np.zeros((count,count));diag_var=np.zeros(count);trace_var=0.
            row = {'seed':seed,'strata':{}}
            for name,region in strata.items():
                modes,mass = region['frequencies'],region['mass']
                if mass==0:
                    row['strata'][name]={'mass':0.,'count':0,'trace_estimate_over_covariance_frobenius':0.}
                    continue
                draws = timed(f'{seed}_{name}_features',lambda:tail_quadratic_samples(xg,a,modes,mass))
                h = draws.mean(0)
                diag = np.diagonal(draws,axis1=1,axis2=2)
                traces = np.trace(draws,axis1=1,axis2=2)
                dse = diag.std(0,ddof=1)/np.sqrt(len(draws))
                tse = traces.std(ddof=1)/np.sqrt(len(draws))
                total+=h;diag_var+=dse**2;trace_var+=tse**2
                detail = {'mass':mass,'count':len(modes),'maximum_absolute_frequency':int(np.abs(modes).max()),
                          'trace_estimate_over_covariance_frobenius':float(traces.mean()/scale),
                          'trace_standard_error_over_covariance_frobenius':float(tse/scale)}
                if dense:
                    detail['trace_error_standard_errors'] = float((traces.mean()-np.trace(exact[name]))/tse)
                    detail['maximum_diagonal_error_standard_errors'] = float(np.max(np.abs(np.diag(h-exact[name]))/dse))
                row['strata'][name]=detail
                arrays[f'{seed}_{name}_estimate'],arrays[f'{seed}_{name}_frequencies']=h,modes
                arrays[f'{seed}_{name}_diagonal_samples'],arrays[f'{seed}_{name}_trace_samples']=diag,traces
            row.update(trace_estimate_over_covariance_frobenius=float(np.trace(total)/scale),
                       trace_standard_error_over_covariance_frobenius=float(np.sqrt(trace_var)/scale),
                       maximum_variance_bound_estimate_fraction=float(np.max(np.diag(total)/variance)),
                       maximum_variance_marginal_normal_95_upper_fraction=float(np.max((np.diag(total)+1.96*np.sqrt(diag_var))/variance)),
                       qualification='stratified estimates and empirical standard errors; normal intervals are marginal, not rigorous simultaneous certificates; solver and transform errors separate')
            if dense:
                row['trace_error_standard_errors']=float((np.trace(total)-np.trace(bound))/np.sqrt(trace_var))
                row['maximum_diagonal_error_standard_errors']=float(np.max(np.abs(np.diag(total-bound))/np.sqrt(diag_var)))
            record['seeds'].append(row)
            arrays[f'{seed}_bound_estimate'],arrays[f'{seed}_diagonal_standard_error']=total,np.sqrt(diag_var)
            save();np.savez_compressed(args.output/'fields.npz',**arrays)
            print('stratified estimate',json.dumps(row),flush=True)
        record['complete']=True
    except Exception as exc:
        record['failure']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        record['gpu_process_audit']=audit.stop();save()


if __name__=='__main__':
    main()
