"""Dense analytic overlap for a positive core-plus-slabs finite posterior.

Reports exact posterior moments on the previously fixed 16-query panel, plus
production float32 NUFFT actions/fits. Timings are diagnostics, not competitive
benchmarks: setup and sampling are not included in the fit timer.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import cupy as cp
import numpy as np
from scipy.special import ndtr
from geospsr.block_spectral import BlockSpectrum,BlockKernel
from geospsr.spectral import FourierPoisson
from geospsr.spectral_blocks import block_audit
from geospsr.spectral_tail import analytic_covariance_rows64
from reproduction.audit_centered_hybrid import chol_solve
from reproduction.block_reference import block_covariance_rows
from reproduction.gpu_audit import GpuProcessAudit

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--case',choices=['A','B'],default='A')
    p.add_argument('--lengthscale',type=float,default=.01)
    p.add_argument('--noise',type=float,default=.02)
    p.add_argument('--blocks',nargs='+',default=['100:200','100:400','150:400'])
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    blocks = [tuple(map(int,s.split(':'))) for s in args.blocks]
    if any(len(b)!=2 or not 50<=b[0]<=b[1] for b in blocks):
        p.error('blocks must be core:extent with 50 <= core <= extent')
    args.output.mkdir(parents=True,exist_ok=True)
    data = np.load(ROOT/f'reproduction/results/matched/{args.case}/input.npz')
    panel = np.load(ROOT/f'reproduction/results/spectral_tail/{args.case}/fields.npz')
    x,y,q = data['points'].astype('f'),data['normals'].astype('f'),panel['query'].astype('f')
    n,count = len(x),len(q)
    record = {'args':{**vars(args),'output':str(args.output)},'N':n,'complete':False,
              'normalization':'analytic','seconds':{},'rows':[],
              'scope':'fixed 16-query dense posterior moments and production mean; not full application quality or sampling timings',
              'hashes':{k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in [('points',x),('normals',y),('queries',q)]},
              'quality_gates':{'mean_relative_l2':.002,'variance_relative_l2':.0002,'covariance_relative_l2':.0002,
                               'probability_rms':.001,'probability_max':.02,'analytic_residual':.001},
              'device':cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}
    arrays = {'query':q,'query_ids':panel['query_ids']}
    def save():
        (args.output/'results.json').write_text(json.dumps(record,indent=2)+'\n')
    def timed(name,fn):
        cp.cuda.runtime.deviceSynchronize();start=time.perf_counter()
        value=fn();cp.cuda.runtime.deviceSynchronize()
        record['seconds'][name]=time.perf_counter()-start
        save();print(name,record['seconds'][name],flush=True)
        return value
    relative = lambda a,b:float(cp.linalg.norm(a)/cp.maximum(cp.linalg.norm(b),1e-30))
    audit = GpuProcessAudit(cp.cuda.Device().pci_bus_id).start()
    try:
        scalar = FourierPoisson(50,args.lengthscale,eps=1e-11,dtype='float64',upsampfac=2.)
        xg,yg = cp.asarray(x,dtype='d'),cp.asarray(y,dtype='d')
        cross = timed('cross',lambda:cp.stack([scalar.cross_values(qi[None]-xg) for qi in cp.asarray(q,dtype='d')],axis=1))
        c = cross.reshape(n,-1)
        norm2 = (scalar.ns**2).sum(0)
        rho = cp.where(norm2>0,scalar.rho/cp.maximum(norm2,1),0)
        prior = timed('scalar_prior',lambda:scalar.evaluate((q.astype('d')[:,None]-q.astype('d')[None]).reshape(-1,3),rho)[0].reshape(count,count))
        def centered_mean(alpha):
            coeff = scalar.aggregate(xg,alpha)
            return scalar.evaluate(q,coeff)[0]-scalar.evaluate(xg,coeff).mean()
        full = timed('analytic_matrix',lambda:analytic_covariance_rows64(x,x,args.lengthscale))
        full.flat[::n+1] += args.noise
        lower = timed('analytic_cholesky',lambda:cp.linalg.cholesky(full))
        afull = chol_solve(lower,yg)
        zfull = chol_solve(lower,c).reshape(n,count,3)
        covfull = prior-cp.einsum('nqd,nrd->qr',cross,zfull)
        meanfull = timed('analytic_mean',lambda:centered_mean(afull))
        varfull = cp.diag(covfull)
        probfull = ndtr(-cp.asnumpy(meanfull)/np.sqrt(cp.asnumpy(varfull)))
        arrays.update({'analytic_mean':cp.asnumpy(meanfull),'analytic_covariance':cp.asnumpy(covfull),'analytic_probability':probfull})
        del lower,afull,zfull
        for core,extent in blocks:
            key = f'{core}_{extent}'
            matrix = timed(key+'_matrix',lambda:block_covariance_rows(x,x,core,extent,args.lengthscale))
            matrix.flat[::n+1] += args.noise
            lower = timed(key+'_cholesky',lambda:cp.linalg.cholesky(matrix))
            alpha = chol_solve(lower,yg)
            z = chol_solve(lower,c).reshape(n,count,3)
            covariance = prior-cp.einsum('nqd,nrd->qr',cross,z)
            mean = timed(key+'_dense_mean',lambda:centered_mean(alpha))
            variance = cp.diag(covariance)
            if bool(cp.any(variance<=0)):
                raise RuntimeError('nonpositive variance')
            probability = ndtr(-cp.asnumpy(mean)/np.sqrt(cp.asnumpy(variance)))
            errors = {'mean_relative_l2':relative(mean-meanfull,meanfull),
                      'variance_relative_l2':relative(variance-varfull,varfull),
                      'covariance_relative_l2':relative(covariance-covfull,covfull),
                      'probability_rms':float(np.sqrt(np.mean((probability-probfull)**2))),
                      'probability_max':float(np.max(np.abs(probability-probfull))),
                      'analytic_residual':relative(full@alpha-yg,yg)}
            spectrum = timed(key+'_spectrum',lambda:BlockSpectrum(core,extent,args.lengthscale))
            kernel = timed(key+'_plans',lambda:BlockKernel(x,spectrum,args.noise))
            action = timed(key+'_action',lambda:kernel(y))
            fitted,info = timed(key+'_fit',lambda:kernel.solve(y,rtol=1e-5,maxiter=1000))
            if not info['converged']:
                raise RuntimeError(info)
            production_mean = centered_mean(fitted.astype('d'))
            row = {'core':core,'extent':extent,'prior_mass':block_audit(core,extent,args.lengthscale),
                   'stored_modes':spectrum.stored_modes,'retained_modes':spectrum.retained_modes,
                   'errors_vs_analytic':errors,
                   'panel_gates_pass':all(errors[k]<=v for k,v in record['quality_gates'].items()),
                   'production_solve':info,
                   'production_action_relative_error':relative(action-matrix@yg,matrix@yg),
                   'production_mean_relative_error_vs_dense_block':relative(production_mean-mean,mean),
                   'production_analytic_residual':relative(full@fitted.astype('d')-yg,yg)}
            record['rows'].append(row);save();print(json.dumps(row),flush=True)
            arrays[key+'_mean'],arrays[key+'_covariance'] = cp.asnumpy(mean),cp.asnumpy(covariance)
            arrays[key+'_probability'] = probability
            del kernel,spectrum,matrix,lower,alpha,z,action,fitted
            gc.collect();cp.get_default_memory_pool().free_all_blocks()
        np.savez_compressed(args.output/'fields.npz',**arrays)
        record['complete'] = True
    except Exception as exc:
        record['failure'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        record['gpu_process_audit'] = audit.stop();save()


if __name__=='__main__':
    main()
