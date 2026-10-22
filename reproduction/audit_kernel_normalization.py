"""Pinned-release comparison and full scorpion-slice sensitivity to normalization.

Validation/sensitivity measurements only; this is not a timing benchmark.
The analytic vector kernel and observation noise stay fixed. Scalar/cross
spectra use cutoff 50, independently of the normalization correction.
"""
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import cupy as cp
import cupyx.scipy.linalg as cla
import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import ndtr

from geospsr.kernels import Matern32Kernel
from geospsr.spectral import FourierPoisson, spectral_axis
from geospsr.spectral_blocks import axis_weights
from geospsr.spectral_tail import analytic_covariance_rows64

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reproduction/results/normalization'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    upstream = Path(os.environ.get('GEOSPSR_RUNTIME','/opt/geospsr-runtime'))/'upstream'
    source = upstream/'geospsr/kernels.py'
    spec = importlib.util.spec_from_file_location('released_kernel_normalization',source)
    release = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(release)
    record = {'scope':'normalization correction and full released-A slice sensitivity; not a speed comparison',
              'complete':False,'normalization':'analytic','compatibility':'released',
              'upstream_commit':subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip(),
              'upstream_kernel_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'axis_checks':[],'slices':[],'scalar_cross_cutoff':50,'noise_variance':.02}
    def save():
        (OUT/'results.json').write_text(json.dumps(record,indent=2)+'\n')
    with jax.experimental.enable_x64():
        for ell in [.005,.01,.04,.1,.5,1.,10.]:
            ns,corrected = spectral_axis(200,ell)
            _,legacy = spectral_axis(200,ell,normalization='released')
            expected = axis_weights(ns,ell)
            old = np.asarray(release.Matern32Kernel(ell,.1).spectral_density(jnp.asarray(ns)))
            local_old = np.asarray(Matern32Kernel(ell,.1,normalization='released').spectral_density(jnp.asarray(ns)))
            local_new = np.asarray(Matern32Kernel(ell,.1).spectral_density(jnp.asarray(ns)))
            errors = {'numpy_vs_independent_max_relative':float(np.max(np.abs(corrected/expected-1))),
                      'jax_vs_independent_max_relative':float(np.max(np.abs(local_new/expected-1))),
                      'compatibility_jax_vs_pinned_max_relative':float(np.max(np.abs(local_old/old-1))),
                      'compatibility_numpy_vs_pinned_max_relative':float(np.max(np.abs(legacy/old-1)))}
            if max(errors.values()) > 3e-12:
                raise AssertionError((ell,errors))
            record['axis_checks'].append({'lengthscale':ell,**errors,
                'numpy_corrected_and_released_bitwise_equal':bool(np.array_equal(corrected,legacy)),
                'infinite_released_product_diagonal_deficit':float(1-(1+np.exp(-np.sqrt(3)/ell))**-3)})
            save()
    data = np.load(ROOT/'reproduction/results/matched/A/input.npz')
    x,y,q = [data[key].astype('d') for key in ['points','normals','query']]
    record.update({'N':len(x),'query_count':len(q),
                   'input_sha256':hashlib.sha256((ROOT/'reproduction/results/matched/A/input.npz').read_bytes()).hexdigest(),
                   'slices':[]})
    arrays = {'query':q,'axis':data['axis'],'heights':data['heights']}
    for ell in [.01,.1,.5,1.,10.]:
        print('lengthscale',ell,flush=True)
        k = analytic_covariance_rows64(x,x,ell)
        k.flat[::len(x)+1] += .02
        lower = cp.linalg.cholesky(k)
        def solve(rhs):
            return cla.solve_triangular(lower,cla.solve_triangular(lower,rhs,lower=True),lower=True,trans='T')
        alpha = solve(cp.asarray(y))
        op = FourierPoisson(50,ell,dtype='float64',eps=1e-11,upsampfac=2.)
        coeff = op.aggregate(x,alpha)
        offset = op.evaluate(x,coeff).mean()
        mean = cp.asnumpy(op.evaluate(q,coeff)[0]-offset)
        norm2 = (op.ns**2).sum(0)
        prior = float(cp.where(norm2>0,op.rho/cp.maximum(norm2,1),0).sum())
        correction = np.empty(len(q))
        for start in range(0,len(q),256):
            block = cp.asarray(q[start:start+256])
            cross = op.cross_values((block[:,None]-cp.asarray(x)[None]).reshape(-1,3)).reshape(len(block),len(x),3)
            c = cross.transpose(1,0,2).reshape(len(x),-1)
            solved = solve(c).reshape(len(x),len(block),3).transpose(1,0,2)
            correction[start:start+len(block)] = cp.asnumpy((cross*solved).sum(axis=(1,2)))
            if start % 5120 == 0:
                print('variance queries',start+len(block),flush=True)
        factor = (1+np.exp(-np.sqrt(3)/ell))**3
        # Shared analytic inverse: both cross terms scale, while the scalar
        # prior scales once. Validate the compatibility evaluation separately.
        legacy_mean = mean/factor
        variance = prior-correction
        legacy_variance = prior/factor-correction/factor**2
        if np.any(variance <= 0) or np.any(legacy_variance <= 0):
            raise ArithmeticError('nonpositive variance')
        probability = ndtr(-mean/np.sqrt(variance))
        legacy_probability = ndtr(-legacy_mean/np.sqrt(legacy_variance))
        old_op = FourierPoisson(50,ell,dtype='float64',eps=1e-11,upsampfac=2.,normalization='released')
        old_coeff = old_op.aggregate(x,alpha)
        old_offset = old_op.evaluate(x,old_coeff).mean()
        check_ids = np.random.default_rng(905).choice(len(q),17,replace=False)
        direct_old_mean = cp.asnumpy(old_op.evaluate(q[check_ids],old_coeff)[0]-old_offset)
        c_old = old_op.cross_values((cp.asarray(q[check_ids])[:,None]-cp.asarray(x)[None]).reshape(-1,3)).reshape(17,len(x),3)
        old_solution = solve(c_old.transpose(1,0,2).reshape(len(x),-1)).reshape(len(x),17,3).transpose(1,0,2)
        old_prior = float(cp.where(norm2>0,old_op.rho/cp.maximum(norm2,1),0).sum())
        direct_old_var = old_prior-cp.asnumpy((c_old*old_solution).sum(axis=(1,2)))
        np.testing.assert_allclose(direct_old_mean,legacy_mean[check_ids],atol=1e-13,rtol=2e-8)
        np.testing.assert_allclose(direct_old_var,legacy_variance[check_ids],atol=1e-15,rtol=2e-8)
        rel = lambda delta,base:float(np.linalg.norm(delta)/np.linalg.norm(base))
        record['slices'].append({'lengthscale':ell,'mean_relative_difference':rel(legacy_mean-mean,mean),
            'variance_relative_difference':rel(legacy_variance-variance,variance),
            'probability_rms_difference':float(np.sqrt(np.mean((legacy_probability-probability)**2))),
            'probability_max_difference':float(np.max(np.abs(legacy_probability-probability))),
            'minimum_corrected_variance':float(variance.min()),'compatibility_17_query_check_pass':True})
        for label,value in [('mean',mean),('variance',variance),('probability',probability),
                            ('released_mean',legacy_mean),('released_variance',legacy_variance),
                            ('released_probability',legacy_probability)]:
            arrays[f'{ell}_{label}'] = value
        np.savez_compressed(OUT/'fields.npz',**arrays)
        save()
        print(record['slices'][-1],flush=True)
        del op,old_op,k,lower,alpha,coeff,old_coeff,cross,c,solved,c_old,old_solution
        gc.collect();cp.get_default_memory_pool().free_all_blocks()
    record['complete'] = True
    save()


if __name__ == '__main__':
    main()
