"""Exact finite-cross-kernel variance series prepared from dense Cholesky.

This is a solver/evaluator hybrid: it removes observation count from later
marginal queries, but retains the dense preparation cost. No sampling or
rank truncation is involved. Other correction-factor producers can use the
same FourierSquareAccumulator independently.
"""
import cupy as cp
import cupyx.scipy.linalg as cla
import time
from geospsr.fourier_moments import FourierSquareAccumulator


def prepare_variance_series(points, cholesky, operator, batch=8, method='squares', working_dtype='float64',
                            source_dtype='float64', paired_sources=False, inverse_batch=0):
    n = len(points)
    if operator.dtype != cp.dtype('float64') or cholesky.dtype != cp.float64:
        raise ValueError('variance preparation requires double precision')
    if cholesky.shape != (n, n) or n < 1 or batch != int(batch) or batch < 1 or inverse_batch!=int(inverse_batch) or inverse_batch<0:
        raise ValueError('invalid factor shape or batch')
    if method not in ('squares','gradient'):raise ValueError('unknown moment construction')
    if method=='gradient':
        from geospsr.fourier_gradient_moments import FourierGradientSquareAccumulator
        acc=FourierGradientSquareAccumulator(operator.shape[0]//2,xp=cp,working_dtype=working_dtype)
    else:
        if working_dtype!='float64':raise ValueError('squares method requires float64')
        acc = FourierSquareAccumulator(operator.shape[0]//2, xp=cp)
    norm2 = (operator.ns**2).sum(0)
    potential=cp.where(norm2>0,operator.rho/cp.where(norm2>0,norm2,1),0)
    profile={'inverse_factor':0.,'source_transforms':0.,'product_accumulation':0.}
    def clock():
        cp.cuda.get_current_stream().synchronize()
        return time.perf_counter()
    # Preserve the no-copy solve for both conventional C-order factors and
    # the in-place Fortran-order factors used by the large dense baseline.
    factor,lower,transpose=(cholesky,True,'T') if cholesky.flags.f_contiguous else (cholesky.T,False,'N')
    full_weights=None
    if inverse_batch:
        begin=clock();full_weights=cp.empty((n,n),dtype='d')
        for start in range(0,n,inverse_batch):
            count=min(inverse_batch,n-start);rhs=cp.zeros((n,count),dtype='d')
            rhs[start+cp.arange(count),cp.arange(count)]=1
            full_weights[:,start:start+count]=cla.solve_triangular(factor,rhs,lower=lower,trans=transpose)
        profile['inverse_factor']+=clock()-begin
    from types import SimpleNamespace
    import numpy as np
    if source_dtype not in ('float32','float64'):raise ValueError('invalid source dtype')
    source_op=SimpleNamespace(shape=operator.shape,dtype=np.dtype(source_dtype),
        cdtype=np.dtype('complex64' if source_dtype=='float32' else 'complex128'),
        eps=max(operator.eps,1e-6) if source_dtype=='float32' else operator.eps,upsampfac=operator.upsampfac)
    from geospsr.spectral import FourierPoisson
    plans = {}
    for start in range(0, n, batch):
        count = min(batch, n-start)
        begin=clock()
        if full_weights is None:
            rhs = cp.zeros((n, count), dtype='d')
            rhs[start+cp.arange(count), cp.arange(count)] = 1.
            weights = cla.solve_triangular(factor, rhs, lower=lower, trans=transpose)
        else:weights=full_weights[:,start:start+count]
        profile['inverse_factor']+=clock()-begin
        begin=clock()
        transforms=(count+1)//2 if paired_sources else count
        if transforms not in plans:plans[transforms]=FourierPoisson.plan(source_op,1,points,transforms)
        if paired_sources:
            scales=cp.max(cp.abs(weights),axis=0);scales=cp.where(scales>0,scales,1)
            normalized=weights/scales[None]
            packed=cp.zeros((transforms,n),dtype=source_op.cdtype)
            packed.real=normalized[:,::2].T
            packed[:count//2].imag=normalized[:,1::2].T
            transformed=plans[transforms].execute(packed).astype('complex128')
            reverse=transformed[:,::-1,::-1,::-1].conj()
            sources=cp.empty((count,*operator.shape),dtype='complex128')
            sources[::2]=.5*(transformed+reverse)*scales[::2,None,None,None]
            sources[1::2]=(-.5j*(transformed-reverse))[:count//2]*scales[1::2,None,None,None]
        else:sources=plans[transforms].execute(cp.ascontiguousarray(weights.T,dtype=source_op.cdtype)).astype('complex128')
        profile['source_transforms']+=clock()-begin
        begin=clock()
        # Each scalar whitened source yields one function per vector component.
        if method=='gradient':acc.add(sources*potential[None])
        else:
            for d in range(3):
                acc.add(-1j*sources*operator.multiplier[d][None])
        profile['product_accumulation']+=clock()-begin
    prior = float(cp.sum(cp.where(norm2 > 0, operator.rho/cp.where(norm2 > 0, norm2, 1), 0)))
    return acc.coefficients(prior, -1), {'functions':3*n, 'potential_functions':n,
        'construction':method,'working_dtype':working_dtype,'prior_variance':prior,
        'source_dtype':source_dtype,'paired_sources':paired_sources,'inverse_batch':inverse_batch,
        'profile_seconds':profile,
        'product_fft_width':acc.fft_width,
        'variance_cutoff':2*(operator.shape[0]//2),
        'scope':'full analytic observation Cholesky and finite scalar/cross kernel; no Monte Carlo'}


class FourierSeriesEvaluator:
    """Evaluate arbitrary centered real Fourier series without kernel setup."""
    def __init__(self, coefficients, eps=1e-11, upsampfac=2.):
        self.coefficients = cp.ascontiguousarray(coefficients, dtype='complex128')
        if self.coefficients.ndim == 3:
            self.coefficients = self.coefficients[None]
        self.shape = self.coefficients.shape[1:]
        self.dtype, self.cdtype = cp.dtype('float64'), cp.dtype('complex128')
        self.eps, self.upsampfac = eps, upsampfac

    def evaluate(self, points):
        from geospsr.spectral import FourierPoisson
        plan = FourierPoisson.plan(self, 2, points, len(self.coefficients))
        return plan.execute(self.coefficients).real
