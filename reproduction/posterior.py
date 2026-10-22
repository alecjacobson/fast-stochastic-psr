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
    def __init__(self,x,normals,lengthscale=.01,variance=.1,noise_variance=.02,cutoff=50,prior_cutoff=20,upsampfac=2.0):
        self.x=cp.asarray(x,dtype='f'); self.normals=cp.asarray(normals,dtype='f')
        self.noise_variance=noise_variance
        self.lengthscale, self.variance, self.cutoff = lengthscale, variance, cutoff
        self.op=FourierPoisson(cutoff,lengthscale,variance,upsampfac=upsampfac)
        self.prior=FourierPoisson(prior_cutoff,lengthscale,variance,upsampfac=upsampfac)
        K=kernel_matrix(x,lengthscale,variance)
        K.block_until_ready()
        self.K=cp.asarray(np.asarray(K))+noise_variance*cp.eye(len(x),dtype='f')
        self.L=cp.linalg.cholesky(self.K)
        self.alpha=self.solve(self.normals)
        self.mean_coeff=self.op.aggregate(self.x,self.alpha)
        self.offset=self.op.evaluate(self.x,self.mean_coeff).mean()
        self.sample_coeff=None
        self.mean_cache=None
        self.sample_cache=None
        self.cache_batch=16
        sync()

    def solve(self,b):
        return cla.solve_triangular(self.L.T,cla.solve_triangular(self.L,b,lower=True),lower=False)

    def mean(self,q):
        if self.mean_cache is not None:
            return self.mean_cache.evaluate(q)[0]-self.offset
        return self.op.evaluate(q,self.mean_coeff)[0]-self.offset

    def prepare_query_cache(self,batch=16):
        """Cache fixed functions; changing query locations remains supported.

        Extra memory/preparation amortize only over repeated queries. `batch`
        counts packed pairs of real sample functions during interpolation.
        Calling draw again invalidates the sample cache, preserving the mean.
        """
        from geospsr.cached_spectral import CachedFourierEvaluator
        if batch < 1:
            raise ValueError('batch must be positive')
        self.cache_batch=batch
        self.mean_cache=CachedFourierEvaluator(self.mean_coeff,batch=1)
        self.sample_cache=(CachedFourierEvaluator(self.sample_coeff,batch=1)
                           if self.sample_coeff is not None else None)
        sync()
        return {'mean_cache_bytes':self.mean_cache.cache_bytes,
                'sample_cache_bytes':self.sample_cache.cache_bytes if self.sample_cache is not None else 0}

    def draw(self,count=64,seed=0,batch=4,exact=True,retain_components=False,
             retain_rhs=False,covariance_backend='separable',rng_backend='cpu',profile=False,
             joint_solve=False):
        """Draw coherent functions.

        exact=True uses the cross-covariance's full finite spectrum, and samples
        the missing analytic vector-field spectrum at the observed points. This
        matches the finite-f/full-v posterior, up to NUFFT/floating-point error.
        exact=False uses the paper's smaller prior truncation approximation.
        retain_components keeps the common pathwise weights and prior spectrum
        for paired backend comparisons; it adds memory and is off by default.
        covariance_backend selects the separable direct finite sum or the
        previous 'nufft' construction of the same vector covariance.
        GPU randomness changes seeded realizations, not their Gaussian law.
        CPU randomness remains the default for historical reproducibility.
        joint_solve shares triangular factors across all draw right-hand sides;
        it retains those RHS and scalar priors until the solve completes.
        """
        self.sample_cache=None
        if rng_backend not in ('cpu','gpu'):
            raise ValueError('rng_backend must be cpu or gpu')
        rng=np.random.default_rng(seed) if rng_backend == 'cpu' else cp.random.RandomState(seed)
        self.draw_profile={}
        if profile:
            sync()
        last=time.perf_counter()
        def mark(stage):
            nonlocal last
            if profile:
                sync()
                now=time.perf_counter()
                self.draw_profile[stage]=self.draw_profile.get(stage,0.)+now-last
                last=now
        prior=self.op if exact else self.prior
        residual_chol=None
        if exact:
            if covariance_backend == 'separable':
                from geospsr.training_spectrum import FourierCovariance, finite_covariance_rows
                spectrum=FourierCovariance(self.cutoff,self.lengthscale,self.variance)
                low=finite_covariance_rows(self.x,self.x,spectrum)
            elif covariance_backend == 'nufft':
                rows=[]
                for start in range(0,len(self.x),128):
                    diff=(self.x[start:start+128,None]-self.x[None]).reshape(-1,3)
                    rows.append(prior.evaluate(diff,prior.rho)[0].reshape(-1,len(self.x)))
                low=cp.concatenate(rows)
            else:
                raise ValueError('covariance_backend must be separable or nufft')
            # K already contains observation noise, ensuring a strictly positive
            # residual covariance even when the Fourier tail is tiny.
            residual=self.K-(low+low.T)*.5
            mark('finite_covariance_and_residual')
            residual_chol=cp.linalg.cholesky(residual)
            mark('residual_cholesky')
        coeff=[]
        weights=[]
        priors=[]
        rhs=[]
        for start in range(0,count,batch):
            p=min(batch,count-start)
            xi=cp.asarray(rng.standard_normal((p,3,*prior.shape,2),dtype=np.float32))
            mark('vector_randomness')
            v,f=prior.prior_coefficients(xi)
            vp=prior.evaluate(self.x,v.reshape(p*3,*prior.shape)).reshape(p,3,-1).transpose(0,2,1)
            mark('vector_prior')
            eps=cp.asarray(rng.standard_normal(vp.shape,dtype=np.float32))
            mark('noise_randomness')
            if exact:
                eps=(residual_chol@eps.transpose(1,0,2).reshape(len(self.x),-1)).reshape(len(self.x),p,3).transpose(1,0,2)
            else:
                eps*=np.sqrt(self.noise_variance)
            residual=self.normals[None]-vp-eps
            if retain_rhs or joint_solve:
                rhs.append(residual)
            mark('conditional_rhs')
            if joint_solve:
                priors.append(f)
                continue
            alpha=self.solve(residual.transpose(1,0,2).reshape(len(self.x),-1)).reshape(len(self.x),p,3).transpose(1,0,2)
            mark('solve')
            c=self.op.aggregate(self.x,alpha)
            pad=(self.op.shape[0]-prior.shape[0])//2
            if pad < 0: raise ValueError('prior cutoff must not exceed cross cutoff')
            c+=cp.pad(f,((0,0),(pad,pad),(pad,pad),(pad,pad)))
            coeff.append(c)
            if retain_components:
                weights.append(alpha)
                priors.append(f)
            mark('scalar_aggregation')
        if joint_solve:
            combined=cp.concatenate(rhs)
            alpha=self.solve(combined.transpose(1,0,2).reshape(len(self.x),-1))
            alpha=alpha.reshape(len(self.x),count,3).transpose(1,0,2)
            mark('solve')
            for i,start in enumerate(range(0,count,batch)):
                p=min(batch,count-start)
                c=self.op.aggregate(self.x,alpha[start:start+p])
                pad=(self.op.shape[0]-prior.shape[0])//2
                if pad < 0: raise ValueError('prior cutoff must not exceed cross cutoff')
                coeff.append(c+cp.pad(priors[i],((0,0),(pad,pad),(pad,pad),(pad,pad))))
            if retain_components:
                weights.append(alpha)
            mark('scalar_aggregation')
        self.sample_coeff=cp.concatenate(coeff)
        self.sample_weights=cp.concatenate(weights) if retain_components else None
        self.sample_prior_coeff=cp.concatenate(priors) if retain_components else None
        self.sample_rhs=cp.concatenate(rhs) if retain_rhs else None
        mark('finalize')
        sync()

    def samples(self,q,batch=None):
        if self.sample_coeff is None: raise RuntimeError('call draw first')
        if self.sample_cache is not None:
            return self.sample_cache.evaluate(q,self.cache_batch if batch is None else batch)-self.offset
        return self.op.evaluate_batched(q,self.sample_coeff,4 if batch is None else batch)-self.offset
