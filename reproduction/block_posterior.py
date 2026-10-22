"""Experimental finite-block posterior with coherent cached scalar draws."""
import time
import cupy as cp
import numpy as np
from geospsr.block_spectral import BlockSpectrum,BlockKernel
from geospsr.spectral import FourierPoisson
from reproduction.posterior import Posterior,sync


class BlockPosterior(Posterior):
    def __init__(self,x,normals,core=100,extent=200,cutoff=50,lengthscale=.01,
                 variance=.1,noise_variance=.02,rtol=1e-5,eps=1e-6,dtype='float32',upsampfac=1.25):
        if cutoff > core:
            raise ValueError('core must cover the scalar/cross band')
        self.x = cp.asarray(x,dtype=dtype)
        self.normals = cp.asarray(normals,dtype=dtype)
        self.noise_variance,self.rtol = noise_variance,rtol
        self.lengthscale,self.variance,self.cutoff = lengthscale,variance,cutoff
        self.normalization = 'analytic'
        self.op = FourierPoisson(cutoff,lengthscale,variance,eps=eps,dtype=dtype,upsampfac=2.)
        self.training = BlockSpectrum(core,extent,lengthscale,variance,eps,dtype,upsampfac)
        self.K = BlockKernel(self.x,self.training,noise_variance)
        self.alpha,self.mean_solve = self.K.solve(self.normals,rtol=rtol,maxiter=1000)
        if not self.mean_solve['converged']:
            raise RuntimeError(self.mean_solve)
        self.mean_coeff = self.op.aggregate(self.x,self.alpha)
        self.offset = self.op.evaluate(self.x,self.mean_coeff).mean()
        self.mean_cache = self.sample_cache = None
        self.sample_coeff = self.sample_weights = self.sample_prior_coeff = None
        self.cache_batch = 16
        sync()

    def solve(self,rhs):
        if rhs.shape != self.normals.shape:
            raise ValueError('this solve interface expects three vector components')
        result,info = self.K.solve(rhs,rtol=self.rtol,maxiter=1000)
        if not info['converged']:
            raise RuntimeError(info)
        return result

    def draw(self,count=64,seed=0,batch=1,retain_components=False,retain_rhs=False,initial_mean=False,progress=None):
        if count < 1 or batch < 1:
            raise ValueError('counts must be positive')
        self.sample_cache = None
        rng = cp.random.RandomState(seed)
        coefficients,weights,priors,rhs_records,records = [],[],[],[],[]
        kernel,last_size = self.K,1
        for start in range(0,count,batch):
            p = min(batch,count-start)
            if p != last_size:
                if kernel is not self.K:
                    del kernel
                    cp.get_default_memory_pool().free_all_blocks()
                kernel = BlockKernel(self.x,self.training,self.noise_variance,3*p)
                last_size = p
            sync();begin = time.perf_counter()
            vector,scalar = kernel.joint_prior(rng,self.op)
            noise = rng.standard_normal(vector.shape,dtype=self.training.dtype)*np.sqrt(self.noise_variance)
            rhs = cp.tile(self.normals,(1,p))-vector-noise
            del vector,noise
            initial = cp.tile(self.alpha,(1,p)) if initial_mean else None
            alpha,info = kernel.solve(rhs,rtol=self.rtol,maxiter=1000,x0=initial)
            if not info['converged']:
                raise RuntimeError(info)
            alpha = alpha.reshape(len(self.x),p,3).transpose(1,0,2)
            coefficients.append(self.op.aggregate(self.x,alpha)+scalar)
            if retain_components:
                weights.append(alpha.copy());priors.append(scalar.copy())
            if retain_rhs:
                rhs_records.append(rhs.reshape(len(self.x),p,3).transpose(1,0,2).copy())
            sync()
            records.append({**info,'samples_done':start+p,'seconds':time.perf_counter()-begin,
                            'core':self.training.core,'extent':self.training.extent,
                            'normalization':'analytic','initial_mean':initial_mean})
            if progress:
                progress(start+p,count,records[-1])
        self.sample_coeff = cp.concatenate(coefficients)
        self.sample_weights = cp.concatenate(weights) if retain_components else None
        self.sample_prior_coeff = cp.concatenate(priors) if retain_components else None
        self.sample_rhs = cp.concatenate(rhs_records) if retain_rhs else None
        self.sample_solve_records = records
        return self.sample_coeff
