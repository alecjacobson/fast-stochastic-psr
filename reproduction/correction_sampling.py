"""Bounded Gaussian correction draws for a dense, fixed GP factorization.

If A=L L.T and xi has independent standard normal entries, alpha=L^-T xi
has covariance A^-1. Cross-kernel evaluation therefore produces correction
covariance C A^-1 C.T. These are NOT posterior function draws and must never
be used as ray/collision samples. A dense factor is still required.
"""
import cupy as cp
import cupyx.scipy.linalg as cla
import numpy as np
from scipy.stats import chi2


class DenseCorrectionSampler:
    def __init__(self, points, cholesky, operator):
        self.points = cp.asarray(points, dtype=operator.dtype)
        self.cholesky = cp.asarray(cholesky)
        self.operator = operator
        n = len(self.points)
        if self.points.shape != (n,3) or n == 0:
            raise ValueError('expected nonempty (N,3) points')
        if self.cholesky.shape != (n,n) or self.cholesky.dtype != operator.dtype:
            raise ValueError('factor must be (N,N) at operator precision')

    def draw_batches(self, count, seed=0, batch=8, retain_weights=False, retain_standard_normal=False):
        """Seeded draws reproducible for a fixed batch schedule; bounded storage.

        The caller owns each yielded block; retaining blocks defeats the memory
        bound. The factor and operator must remain fixed throughout consumption.
        """
        if count != int(count) or count < 1 or batch != int(batch) or batch < 1:
            raise ValueError('count and batch must be positive integers')
        rng = cp.random.RandomState(seed)
        n = len(self.points)
        plans = {}
        for start in range(0,count,batch):
            p = min(batch,count-start)
            xi = rng.standard_normal((n,3*p),dtype=self.operator.dtype)
            weights = cla.solve_triangular(self.cholesky.T,xi,lower=False)
            weights = weights.reshape(n,p,3).transpose(1,0,2)
            if p not in plans: plans[p] = self.operator.plan(1,self.points,3*p)
            modes = plans[p].execute(cp.ascontiguousarray(weights.transpose(0,2,1).reshape(3*p,n),
                                                          dtype=self.operator.cdtype))
            modes = modes.reshape(p,3,*self.operator.shape)
            coefficients = cp.ascontiguousarray(-1j*(modes*self.operator.multiplier[None]).sum(1))
            result = {'start':start,'stop':start+p,'coefficients':coefficients}
            if retain_weights: result['weights'] = weights
            if retain_standard_normal:
                # Observation-major (observation,component,draw) layout.
                result['standard_normal'] = xi.reshape(n,p,3).transpose(0,2,1).reshape(3*n,p)
            yield result


class PolynomialCorrectionSampler:
    """Matrix-free correction draws from a fixed inverse-root polynomial.

    covariance_factory(rhs_count) must return the SAME symmetric covariance
    action at each RHS count. The polynomial and its interval remain fixed
    throughout the stream. Approximation errors are not included in the exact-
    Gaussian target intervals from CorrectionMoments. These draws are not
    posterior functions and must not be used for correlated posterior events.
    """
    def __init__(self, points, operator, polynomial, covariance_factory):
        self.points=cp.asarray(points,dtype=operator.dtype)
        if self.points.ndim!=2 or self.points.shape[1]!=3 or not len(self.points):
            raise ValueError('expected nonempty (N,3) points')
        self.operator,self.polynomial,self.covariance_factory=operator,polynomial,covariance_factory
        self.kernels,self.plans={},{}

    def prepare(self,batch):
        if batch!=int(batch) or batch<1:
            raise ValueError('batch must be a positive integer')
        if batch not in self.kernels:
            self.kernels[batch]=self.covariance_factory(3*batch)
            self.plans[batch]=self.operator.plan(1,self.points,3*batch)

    def aggregate(self,weights):
        if weights.ndim!=2 or weights.shape[0]!=len(self.points) or weights.shape[1]%3:
            raise ValueError('expected weights (N,3*batch)')
        p=weights.shape[1]//3;self.prepare(p)
        modes=self.plans[p].execute(cp.ascontiguousarray(weights.T,dtype=self.operator.cdtype))
        modes=modes.reshape(p,3,*self.operator.shape)
        return cp.ascontiguousarray(-1j*(modes*self.operator.multiplier[None]).sum(1))

    def draw_batches(self,count,seed=0,batch=1,retain_weights=False,progress=None):
        if count!=int(count) or count<1 or batch!=int(batch) or batch<1:
            raise ValueError('count and batch must be positive integers')
        rng=cp.random.RandomState(seed);n=len(self.points)
        for start in range(0,count,batch):
            p=min(batch,count-start);self.prepare(p)
            xi=rng.standard_normal((n,3*p),dtype=self.operator.dtype)
            weights=self.polynomial.apply(self.kernels[p],xi,xp=cp,progress=progress)
            coefficients=self.aggregate(weights)
            result={'start':start,'stop':start+p,'coefficients':coefficients}
            if retain_weights:
                result.update(weights=weights.reshape(n,p,3).transpose(1,0,2),standard_normal=xi)
            yield result


class CorrectionMoments:
    """O(query count) zero-mean second moments; no retained ensemble."""
    def __init__(self):
        self.count = 0
        self.sum_squares = None

    def update(self, values):
        values = np.asarray(values,dtype='d')
        if values.ndim != 2 or not values.size or not np.isfinite(values).all():
            raise ValueError('expected finite nonempty (draw,query) values')
        squares = np.sum(values*values,axis=0)
        if not np.isfinite(squares).all(): raise ValueError('second moment overflow')
        if self.sum_squares is None: self.sum_squares = squares
        else:
            if squares.shape != self.sum_squares.shape: raise ValueError('query shape changed')
            self.sum_squares += squares
        self.count += len(values)

    def summary(self, prior_variance, delta=.05, looks=1):
        if not self.count or not 0 < delta < 1 or looks != int(looks) or looks < 1:
            raise ValueError('require draws, valid delta and positive integer looks')
        v0 = np.broadcast_to(np.asarray(prior_variance,dtype='d'),self.sum_squares.shape)
        if not np.isfinite(v0).all() or np.any(v0<=0): raise ValueError('invalid prior variance')
        tail = delta/(2*len(v0)*looks)
        low,high = chi2.ppf([tail,1-tail],self.count)
        return {'count':self.count,'variance':v0-self.sum_squares/self.count,
                'lower':v0-self.sum_squares/low,'upper':v0-self.sum_squares/high,
                'delta':delta,'looks':looks}
