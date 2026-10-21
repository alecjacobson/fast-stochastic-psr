"""Reusable GPU posterior with one factorization and cached Fourier coefficients."""
import time
import cupy as cp
import cupyx.scipy.linalg as cla
import jax
import jax.numpy as jnp
import numpy as np
from geospsr.kernels import Matern32Kernel, ProductKernel
from geospsr.spectral import FourierPoisson


def sync():
    cp.cuda.runtime.deviceSynchronize()


def kernel_matrix(x, lengthscale=.01, variance=.1):
    k=Matern32Kernel(lengthscale,variance)
    kv=ProductKernel(k,k,k)
    return jax.jit(jax.vmap(jax.vmap(kv,(None,0)),(0,None)))(jnp.asarray(x),jnp.asarray(x))


class Posterior:
    def __init__(self,x,normals,lengthscale=.01,variance=.1,noise_variance=.02,cutoff=50,prior_cutoff=20):
        self.x=cp.asarray(x,dtype='f'); self.normals=cp.asarray(normals,dtype='f')
        self.noise_variance=noise_variance
        self.op=FourierPoisson(cutoff,lengthscale,variance)
        self.prior=FourierPoisson(prior_cutoff,lengthscale,variance)
        K=kernel_matrix(x,lengthscale,variance)
        K.block_until_ready()
        self.K=cp.asarray(np.asarray(K))+noise_variance*cp.eye(len(x),dtype='f')
        self.L=cp.linalg.cholesky(self.K)
        self.alpha=self.solve(self.normals)
        self.mean_coeff=self.op.aggregate(self.x,self.alpha)
        self.offset=self.op.evaluate(self.x,self.mean_coeff).mean()
        self.sample_coeff=None
        sync()

    def solve(self,b):
        return cla.solve_triangular(self.L.T,cla.solve_triangular(self.L,b,lower=True),lower=False)

    def mean(self,q):
        return self.op.evaluate(q,self.mean_coeff)[0]-self.offset

    def draw(self,count=64,seed=0,batch=4,exact=True):
        """Draw coherent functions.

        exact=True uses the cross-covariance's full finite spectrum, and samples
        the missing analytic vector-field spectrum at the observed points. This
        matches the finite-f/full-v posterior, up to NUFFT/floating-point error.
        exact=False uses the paper's smaller prior truncation approximation.
        """
        rng=np.random.default_rng(seed)
        prior=self.op if exact else self.prior
        residual_chol=None
        if exact:
            rows=[]
            for start in range(0,len(self.x),128):
                diff=(self.x[start:start+128,None]-self.x[None]).reshape(-1,3)
                rows.append(prior.evaluate(diff,prior.rho)[0].reshape(-1,len(self.x)))
            low=cp.concatenate(rows)
            # K already contains observation noise, ensuring a strictly positive
            # residual covariance even when the Fourier tail is tiny.
            residual=self.K-(low+low.T)*.5
            residual_chol=cp.linalg.cholesky(residual)
        coeff=[]
        for start in range(0,count,batch):
            p=min(batch,count-start)
            xi=cp.asarray(rng.standard_normal((p,3,*prior.shape,2),dtype=np.float32))
            v,f=prior.prior_coefficients(xi)
            vp=prior.evaluate(self.x,v.reshape(p*3,*prior.shape)).reshape(p,3,-1).transpose(0,2,1)
            eps=cp.asarray(rng.standard_normal(vp.shape,dtype=np.float32))
            if exact:
                eps=(residual_chol@eps.transpose(1,0,2).reshape(len(self.x),-1)).reshape(len(self.x),p,3).transpose(1,0,2)
            else:
                eps*=np.sqrt(self.noise_variance)
            residual=self.normals[None]-vp-eps
            alpha=self.solve(residual.transpose(1,0,2).reshape(len(self.x),-1)).reshape(len(self.x),p,3).transpose(1,0,2)
            c=self.op.aggregate(self.x,alpha)
            pad=(self.op.shape[0]-prior.shape[0])//2
            if pad < 0: raise ValueError('prior cutoff must not exceed cross cutoff')
            c+=cp.pad(f,((0,0),(pad,pad),(pad,pad),(pad,pad)))
            coeff.append(c)
        self.sample_coeff=cp.concatenate(coeff)
        sync()

    def samples(self,q,batch=4):
        if self.sample_coeff is None: raise RuntimeError('call draw first')
        return cp.concatenate([self.op.evaluate(q,self.sample_coeff[i:i+batch]) for i in range(0,len(self.sample_coeff),batch)])-self.offset
