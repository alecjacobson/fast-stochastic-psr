"""Dense analytic overlap for a wider prior and centered narrower inverse.

This measures posterior moments directly, without Monte Carlo sampling error.
Scalar/cross cutoff is 50; the held mean and coherent vector prior use cutoff200.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import cupy as cp
import cupyx.scipy.linalg as cla
import numpy as np
from scipy.special import ndtr
from geospsr.spectral import FourierPoisson
from geospsr.training_spectrum import FourierCovariance,finite_covariance_rows
from geospsr.spectral_tail import analytic_covariance_rows64
from reproduction.gpu_audit import GpuProcessAudit

ROOT = Path(__file__).resolve().parents[1]


def chol_solve(lower,rhs):
    intermediate=cla.solve_triangular(lower,rhs,lower=True)
    return cla.solve_triangular(lower,intermediate,lower=True,trans='T')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--case',choices=['A','B'],default='A')
    p.add_argument('--lengthscale',type=float,default=.01)
    p.add_argument('--noise',type=float,default=.02)
    p.add_argument('--cutoffs',type=int,nargs='+',default=[100,150,200])
    p.add_argument('--block-core',type=int,help='optional positive-block inverse core; cutoffs specify extents')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if not all(50 <= c <= 200 for c in args.cutoffs):
        p.error('inverse cutoffs must cover scalar50 and not exceed prior200')
    if args.block_core is not None and not 50 <= args.block_core <= min(args.cutoffs):
        p.error('block core must cover scalar50 and not exceed any extent')
    args.output.mkdir(parents=True,exist_ok=True)
    data=np.load(ROOT/f'reproduction/results/matched/{args.case}/input.npz')
    panel=np.load(ROOT/f'reproduction/results/spectral_tail/{args.case}/fields.npz')
    x,y=data['points'].astype('f'),data['normals'].astype('f')
    q=panel['query'].astype('f')
    n=len(x);count=len(q)
    record={'args':{**vars(args),'output':str(args.output)},'N':n,'complete':False,'seconds':{},'rows':[],
            'scope':'float64 dense analytic/finite/hybrid moments on fixed 16-query panel; not full application coverage',
            'hashes':{k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in [('points',x),('normals',y),('queries',q)]},
            'quality_gates':{'mean_relative_l2':.002,'variance_relative_l2':.0002,'covariance_relative_l2':.0002,
                             'probability_rms':.001,'probability_max':.02},
            'device':cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}
    arrays={'query':q,'query_ids':panel['query_ids']}
    def save():
        (args.output/'results.json').write_text(json.dumps(record,indent=2)+'\n')
    def timed(name,fn):
        cp.cuda.runtime.deviceSynchronize();start=time.perf_counter();value=fn();cp.cuda.runtime.deviceSynchronize()
        record['seconds'][name]=time.perf_counter()-start;save();print(name,record['seconds'][name],flush=True)
        return value
    relative=lambda e,b:float(cp.linalg.norm(e)/cp.maximum(cp.linalg.norm(b),1e-30))
    audit=GpuProcessAudit(cp.cuda.Device().pci_bus_id).start()
    try:
        op=FourierPoisson(50,args.lengthscale,eps=1e-11,dtype='float64',upsampfac=2.)
        xg,yg=cp.asarray(x,dtype='d'),cp.asarray(y,dtype='d')
        cross=timed('cross',lambda:cp.stack([op.cross_values(qi[None]-xg) for qi in cp.asarray(q,dtype='d')],axis=1))
        c=cross.reshape(n,-1)
        norm2=(op.ns**2).sum(0);rho=cp.where(norm2>0,op.rho/cp.maximum(norm2,1),0)
        prior=timed('scalar_prior',lambda:op.evaluate((q.astype('d')[:,None]-q.astype('d')[None]).reshape(-1,3),rho)[0].reshape(count,count))
        full=timed('analytic_matrix',lambda:analytic_covariance_rows64(x,x,args.lengthscale))
        full.flat[::n+1]+=args.noise
        lf=timed('analytic_cholesky',lambda:cp.linalg.cholesky(full))
        full_alpha=timed('analytic_mean_weights',lambda:chol_solve(lf,yg))
        full_z=timed('analytic_cross_weights',lambda:chol_solve(lf,c)).reshape(n,count,3)
        vfull=prior-cp.einsum('nqd,nrd->qr',cross,full_z)
        def centered_mean(alpha):
            coeff=op.aggregate(xg,alpha)
            offset=op.evaluate(xg,coeff).mean()
            return op.evaluate(q,coeff)[0]-offset,offset
        full_mean,full_offset=timed('analytic_mean',lambda:centered_mean(full_alpha))
        del lf,full_z,full_alpha
        spectrum=FourierCovariance(200,args.lengthscale,dtype='float64')
        wide=timed('prior200_matrix',lambda:finite_covariance_rows(x,x,spectrum,dtype='d'))
        del spectrum
        wide.flat[::n+1]+=args.noise
        lw=timed('prior200_cholesky',lambda:cp.linalg.cholesky(wide))
        mean_alpha=timed('prior200_mean_weights',lambda:chol_solve(lw,yg))
        z=timed('prior200_cross_weights',lambda:chol_solve(lw,c)).reshape(n,count,3)
        vwide=prior-cp.einsum('nqd,nrd->qr',cross,z)
        mean,offset=timed('held_mean',lambda:centered_mean(mean_alpha))
        record['held_mean_analytic_residual']=relative(full@mean_alpha-yg,yg)
        del full,lw
        full_var=cp.diag(vfull)
        if bool(cp.any(full_var <= 0)): raise RuntimeError('nonpositive analytic variance')
        full_probability=ndtr(-cp.asnumpy(full_mean)/np.sqrt(cp.asnumpy(full_var)))
        arrays.update({'analytic_mean':cp.asnumpy(full_mean),'held_mean':cp.asnumpy(mean),
                       'analytic_covariance':cp.asnumpy(vfull),'prior200_covariance':cp.asnumpy(vwide),
                       'analytic_probability':full_probability})
        record['analytic_offset'],record['held_offset']=float(full_offset),float(offset)
        for cutoff in args.cutoffs:
            if cutoff==200 and args.block_core is None:
                a=z
            else:
                if args.block_core is None:
                    spectrum=FourierCovariance(cutoff,args.lengthscale,dtype='float64')
                    narrow=timed(f'{cutoff}_matrix',lambda:finite_covariance_rows(x,x,spectrum,dtype='d'))
                    del spectrum
                else:
                    from reproduction.block_reference import block_covariance_rows
                    narrow=timed(f'{cutoff}_matrix',lambda:block_covariance_rows(x,x,args.block_core,cutoff,args.lengthscale))
                narrow.flat[::n+1]+=args.noise
                ln=timed(f'{cutoff}_cholesky',lambda:cp.linalg.cholesky(narrow))
                a=timed(f'{cutoff}_cross_weights',lambda:chol_solve(ln,c)).reshape(n,count,3)
                del narrow,ln
            difference=a-z
            # Stable PSD expression for the second-order excess over V_prior.
            delta=timed(f'{cutoff}_covariance_excess',lambda:cp.einsum(
                'nqd,nrd->qr',difference,(wide@difference.reshape(n,-1)).reshape(n,count,3)))
            hybrid=vwide+delta
            var=cp.diag(hybrid)
            if bool(cp.any(var <= 0)): raise RuntimeError('nonpositive hybrid variance')
            probability=ndtr(-cp.asnumpy(mean)/np.sqrt(cp.asnumpy(var)))
            errors={'mean_relative_l2':relative(mean-full_mean,full_mean),
                    'variance_relative_l2':relative(var-full_var,full_var),
                    'covariance_relative_l2':relative(hybrid-vfull,vfull),
                    'probability_rms':float(np.sqrt(np.mean((probability-full_probability)**2))),
                    'probability_max':float(np.max(np.abs(probability-full_probability)))}
            row={'solve_cutoff':cutoff,'prior_cutoff':200,'mean_cutoff':200,'errors_vs_analytic':errors,
                 'block_core':args.block_core,
                 'covariance_excess_over_prior200_relative_l2':relative(delta,vwide),
                 'largest_variance_excess_fraction':float(cp.max(cp.diag(delta)/cp.diag(vwide))),
                 'minimum_excess_eigenvalue':float(cp.linalg.eigvalsh((delta+delta.T)/2).min()),
                 'panel_gates_pass':all(errors[k] <= v for k,v in record['quality_gates'].items())}
            record['rows'].append(row);save();print(json.dumps(row),flush=True)
            arrays[f'{cutoff}_covariance']=cp.asnumpy(hybrid);arrays[f'{cutoff}_probability']=probability
            del a,difference,delta,hybrid
            gc.collect();cp.get_default_memory_pool().free_all_blocks()
        np.savez_compressed(args.output/'fields.npz',**arrays)
        record['complete']=True
    except Exception as exc:
        record['failure']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        record['gpu_process_audit']=audit.stop();save()


if __name__=='__main__':
    main()
