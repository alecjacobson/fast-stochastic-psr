"""Experimental finite-block posterior with coherent cached scalar draws."""
import time
from contextlib import contextmanager,closing
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

    def _make_kernel(self,rhs_count):
        return BlockKernel(self.x,self.training,self.noise_variance,rhs_count)

    def _spectrum_record(self):
        return {'core':self.training.core,'extent':self.training.extent}

    def _solve_draw(self,kernel,rhs,initial_mean):
        p=rhs.shape[1]//3
        initial=cp.tile(self.alpha,(1,p)) if initial_mean else None
        return kernel.solve(rhs,rtol=self.rtol,maxiter=1000,x0=initial)

    @contextmanager
    def _draw_context(self,initial_mean=False):
        yield

    def draw_batches(self,count=64,seed=0,batch=1,retain_components=False,retain_rhs=False,
                     initial_mean=False,progress=None,**options):
        """Yield coherent coefficient batches without retaining the ensemble.

        Use ``with contextlib.closing(model.draw_batches(...)) as batches``
        so an early break or consumer exception releases solver state. Do not
        retain yielded arrays if bounded memory is required. Batching/seed/RNG
        order matches draw(); changing batch size can change realizations.
        Streaming invalidates any previous retained sample ensemble/cache.
        Only one active stream per model is allowed. Mean queries remain valid.
        This iterator has no checkpoint/resume contract yet.
        """
        if count!=int(count) or batch!=int(batch) or count<1 or batch<1:
            raise ValueError('counts must be positive integers')
        if getattr(self,'_drawing',False):
            raise RuntimeError('a draw stream is already active on this model')
        self._drawing=True
        try:
            with self._draw_context(initial_mean=initial_mean,**options):
                self.sample_cache=self.sample_coeff=self.sample_weights=self.sample_prior_coeff=None
                self.sample_rhs=None
                self.sample_solve_records=[]
                rng=cp.random.RandomState(seed)
                kernel,last_size=self.K,1
                for start in range(0,int(count),int(batch)):
                    p=min(int(batch),int(count)-start)
                    sync();setup_begin=time.perf_counter()
                    if p!=last_size:
                        if kernel is not self.K:
                            del kernel
                            cp.get_default_memory_pool().free_all_blocks()
                        kernel=self._make_kernel(3*p)
                        last_size=p
                    sync();begin=time.perf_counter()
                    vector,scalar=kernel.joint_prior(rng,self.op)
                    noise=rng.standard_normal(vector.shape,dtype=self.training.dtype)*np.sqrt(self.noise_variance)
                    rhs=cp.tile(self.normals,(1,p))-vector-noise
                    del vector,noise
                    sync();prior_end=time.perf_counter()
                    alpha,info=self._solve_draw(kernel,rhs,initial_mean)
                    if not info['converged']:
                        raise RuntimeError(info)
                    sync();solve_end=time.perf_counter()
                    alpha=alpha.reshape(len(self.x),p,3).transpose(1,0,2)
                    result={'start':start,'stop':start+p,'coefficients':self.op.aggregate(self.x,alpha)+scalar}
                    if retain_components:
                        result.update(weights=alpha.copy(),prior=scalar.copy())
                    if retain_rhs:
                        result['rhs']=rhs.reshape(len(self.x),p,3).transpose(1,0,2).copy()
                    del alpha,rhs,scalar
                    sync();end=time.perf_counter()
                    record={**info,'samples_done':start+p,'seconds':end-begin,**self._spectrum_record(),
                            'normalization':self.normalization,'initial_mean':initial_mean,
                            'kernel_setup_seconds':begin-setup_begin,'prior_seconds':prior_end-begin,
                            'solve_seconds':solve_end-prior_end,'aggregation_seconds':end-solve_end}
                    result['record']=record
                    self.sample_solve_records.append(record)
                    if progress:
                        progress(start+p,count,record)
                    yield result
                    del result
        finally:
            self._drawing=False

    def draw(self,count=64,seed=0,batch=1,retain_components=False,retain_rhs=False,
             initial_mean=False,progress=None,**options):
        """Retain the ensemble; use draw_batches() for bounded sample storage."""
        coefficients,weights,priors,rhs_records=[],[],[],[]
        with closing(self.draw_batches(count,seed,batch,retain_components,retain_rhs,initial_mean,progress,**options)) as batches:
            for result in batches:
                coefficients.append(result['coefficients'])
                if retain_components:
                    weights.append(result['weights']);priors.append(result['prior'])
                if retain_rhs:
                    rhs_records.append(result['rhs'])
        self.sample_coeff = cp.concatenate(coefficients)
        self.sample_weights = cp.concatenate(weights) if retain_components else None
        self.sample_prior_coeff = cp.concatenate(priors) if retain_components else None
        self.sample_rhs = cp.concatenate(rhs_records) if retain_rhs else None
        return self.sample_coeff
