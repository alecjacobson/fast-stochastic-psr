"""Matrix-free posterior with separate vector-training and scalar bandwidths.

This is a controlled spectral approximation to the analytic vector covariance,
not an exact analytic-kernel solver. Validate training bandwidth independently
of CG/NUFFT tolerances. Scalar and vector priors share their low-frequency modes.
"""
import time
import cupy as cp
import numpy as np

from geospsr.spectral import FourierPoisson, FourierKernel
from geospsr.training_spectrum import FourierCovariance
from reproduction.posterior import Posterior, sync


def solve_centered_fluctuations(kernel, rhs, normals, mean_alpha, rtol=1e-5, maxiter=1000):
    """Apply an experimental inverse only to zero-mean prior/noise fluctuations.

    rhs contains observed normals minus the joint vector prior and noise.
    The supplied wider-band mean is added independently of this inverse.
    The solve tolerance is relative to the centered RHS; the returned total
    weights need not solve the original wider or analytic training system.
    """
    if rhs.ndim != 2 or rhs.shape[1] < 3 or rhs.shape[1] % 3 or normals.shape != (rhs.shape[0],3) or mean_alpha.shape != normals.shape:
        raise ValueError('expected RHS (N,3P), normals (N,3), and mean weights (N,3)')
    count = rhs.shape[1] // 3
    centered = rhs-cp.tile(normals,(1,count))
    fluctuation,info = kernel.solve(centered,rtol=rtol,maxiter=maxiter)
    info = {**info,'centered_fluctuations':True,'residual_normalization':'centered prior-plus-noise RHS'}
    return fluctuation+cp.tile(mean_alpha,(1,count)),info


class SpectralPosterior(Posterior):
    def __init__(self, x, normals, training_cutoff=100, cutoff=50, lengthscale=.01,
                 variance=.1, noise_variance=.02, rtol=1e-5, eps=1e-6):
        if training_cutoff < cutoff:
            raise ValueError('vector bandwidth must cover every scalar mode')
        self.x = cp.asarray(x, dtype='f')
        self.normals = cp.asarray(normals, dtype='f')
        self.noise_variance = noise_variance
        self.lengthscale,self.variance = lengthscale,variance
        self.rtol = rtol
        self.op = FourierPoisson(cutoff, lengthscale, variance, eps=eps)
        self.training = FourierCovariance(training_cutoff, lengthscale, variance, eps=eps)
        self.K = FourierKernel(self.x, self.training, noise_variance)
        self.alpha, self.mean_solve = self.K.solve(self.normals, rtol=rtol, maxiter=1000)
        if not self.mean_solve['converged']:
            raise RuntimeError(('mean solve did not converge', self.mean_solve))
        self.mean_coeff = self.op.aggregate(self.x, self.alpha)
        self.offset = self.op.evaluate(self.x, self.mean_coeff).mean()
        self.sample_coeff = self.sample_weights = self.sample_prior_coeff = None
        self.mean_cache = self.sample_cache = None
        self.cache_batch = 16
        self.sample_solve_records = []
        sync()

    def solve(self, rhs):
        """Solve three RHS using the existing plans; no dense fallback."""
        if rhs.shape != self.normals.shape:
            raise ValueError('this solve interface expects three vector components')
        result, info = self.K.solve(rhs, rtol=self.rtol, maxiter=1000)
        if not info['converged']:
            raise RuntimeError(('conditional solve did not converge', info))
        return result

    def draw(self, count=64, seed=0, batch=1, retain_components=False, progress=None,
             validation_rows=None, retain_rhs=False, initial_mean=False, solve_cutoff=None):
        """Draw the joint finite-scalar / larger-finite-vector posterior.

        GPU Gaussian vector coefficients cover the complete training band.
        The scalar prior uses the corresponding low-frequency coefficients,
        preserving their cross covariance. Independent observation noise is
        sampled at every source. No dense omitted-spectrum factor is needed.
        Random realizations depend on bandwidth and batching; validate their
        law, not equality to another sampler's seed alone.
        initial_mean reuses the already fitted mean as the CG starting point;
        the stopping tolerance remains relative to each original sample RHS.
        Explicit solve_cutoff enables an experimental hybrid: keep this model's
        prior and mean, but use that bandwidth's inverse on centered fluctuations.
        It must cover the scalar band and not exceed the prior bandwidth.
        When the bands differ, this is not an exact posterior at either bandwidth. Its numerical
        solve residual is for the centered, narrower system only.
        """
        if count < 1 or batch < 1:
            raise ValueError('count and batch must be positive')
        if solve_cutoff is not None and (solve_cutoff != int(solve_cutoff) or
                not self.op.shape[0]//2 <= solve_cutoff <= self.training.cutoff):
            raise ValueError('solve cutoff must lie between scalar and prior cutoffs')
        if solve_cutoff is not None and initial_mean:
            raise ValueError('centered fluctuations use a zero start; mean is added separately')
        solve_spectrum = None if solve_cutoff is None else FourierCovariance(
            int(solve_cutoff),self.lengthscale,self.variance,eps=self.training.eps)
        ids = None if validation_rows is None else cp.asarray(validation_rows, dtype=cp.int64)
        if ids is not None and (ids.ndim != 1 or bool(cp.any((ids < 0) | (ids >= len(self.x))))):
            raise ValueError('validation_rows must be a vector of source indices')
        self.sample_cache = None
        self.sample_validation_rhs = None
        rng = cp.random.RandomState(seed)
        root_rho = cp.sqrt(self.training.rho)
        norm2 = (self.op.ns ** 2).sum(0)
        denominator = cp.where(norm2 > 0, norm2, 1)
        pad = (self.training.shape[0] - self.op.shape[0]) // 2
        sl = slice(pad, pad + self.op.shape[0])
        coefficients, retained_weights, retained_prior = [], [], []
        records, validation_rhs, retained_rhs = [], [], []
        kernel, last_size = self.K, 1
        fluctuation_kernel,fluctuation_size = None,None
        for start in range(0, count, batch):
            sync()
            begin = time.perf_counter()
            p = min(batch, count - start)
            if p != last_size:
                if kernel is not self.K:
                    del kernel
                    cp.get_default_memory_pool().free_all_blocks()
                kernel = FourierKernel(self.x, self.training, self.noise_variance, rhs_count=3 * p)
                last_size = p
            shape = (3 * p, *self.training.shape)
            vector = rng.standard_normal(shape, dtype='f').astype('F')
            vector.imag = -rng.standard_normal(shape, dtype='f')
            vector *= root_rho[None]
            low = vector.reshape(p, 3, *self.training.shape)[:, :, sl, sl, sl]
            scalar = -1j * (low * self.op.ns[None]).sum(1) / denominator
            values = kernel.backward.execute(vector).real.T.reshape(len(self.x), p, 3)
            noise = rng.standard_normal(values.shape, dtype='f') * np.sqrt(self.noise_variance)
            rhs = (self.normals[:, None] - values - noise).reshape(len(self.x), 3 * p)
            if retain_rhs:
                retained_rhs.append(rhs.reshape(len(self.x), p, 3).transpose(1, 0, 2).copy())
            if ids is not None:
                validation_rhs.append(rhs[ids].reshape(len(ids), p, 3).transpose(1, 0, 2).copy())
            del low, vector, values, noise
            sync()
            prior_end = time.perf_counter()
            if solve_spectrum is not None:
                if fluctuation_size != p:
                    del fluctuation_kernel
                    cp.get_default_memory_pool().free_all_blocks()
                    fluctuation_kernel = FourierKernel(self.x,solve_spectrum,self.noise_variance,rhs_count=3*p)
                    fluctuation_size = p
                weights,info = solve_centered_fluctuations(
                    fluctuation_kernel,rhs,self.normals,self.alpha,rtol=self.rtol)
            else:
                initial = cp.tile(self.alpha, (1,p)) if initial_mean else None
                weights, info = kernel.solve(rhs, rtol=self.rtol, maxiter=1000, x0=initial)
                del initial
            sync()
            solve_end = time.perf_counter()
            if not info['converged']:
                raise RuntimeError(('sample solve did not converge', start, info))
            weights = weights.reshape(len(self.x), p, 3).transpose(1, 0, 2)
            correction = self.op.aggregate(self.x, weights)
            coefficients.append(correction + scalar)
            if retain_components:
                retained_weights.append(weights)
                retained_prior.append(scalar)
            sync()
            end = time.perf_counter()
            records.append({**info, 'samples_done': start + p, 'seconds': end - begin,
                            'initial_mean': bool(initial_mean),
                            'solve_cutoff':self.training.cutoff if solve_cutoff is None else int(solve_cutoff),
                            'prior_cutoff':self.training.cutoff,
                            'prior_seconds': prior_end - begin, 'solve_seconds': solve_end - prior_end,
                            'aggregation_seconds': end - solve_end})
            if progress:
                progress(records)
        self.sample_coeff = cp.concatenate(coefficients)
        self.sample_weights = cp.concatenate(retained_weights) if retain_components else None
        self.sample_prior_coeff = cp.concatenate(retained_prior) if retain_components else None
        self.sample_solve_records = records
        self.sample_validation_rhs = cp.concatenate(validation_rhs) if ids is not None else None
        self.sample_rhs = cp.concatenate(retained_rhs) if retain_rhs else None
        sync()
        return records
