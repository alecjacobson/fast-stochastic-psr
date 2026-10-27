"""Coherent Gaussian posterior on a fixed query panel.

This assembles K_ff(Q,Q)-C A^-1 C.T in query space. Its samples are correlated
on the whole supplied panel; they do not define cached functions at new points.
Observation/query cross matrices are streamed through existing NUFFT actions.
"""
import cupy as cp
import numpy as np
from geospsr.analytic_refinement import refine_analytic
from geospsr.packed_spectral import PackedFourierKernel
from geospsr.periodic_leaf_cache import LeafCachedRangeKernel
from geospsr.training_spectrum import FourierCovariance
from reproduction.analytic_posterior import _AnalyticCG, timed
from reproduction.deflated_variance import FourierCorrectionCovariance


class AnalyticConditioner:
    """Shared full-analytic solver, with optional finite-NUFFT defect correction."""
    def __init__(self, points, lengthscale=.01, variance=.1, noise_variance=.02,
                 solver='analytic_cg', fit_cutoff=200, rtol=1e-10,
                 max_cache_bytes=2**30,max_entries=100_000_000):
        if solver not in ('analytic_cg','defect') or not 0<rtol<1:
            raise ValueError('invalid solver or tolerance')
        self.action=LeafCachedRangeKernel(points,lengthscale,variance,noise_variance,
                                         leaf_size=128,max_cache_bytes=max_cache_bytes,max_entries=max_entries)
        self.points=points;self.solver=solver;self.rtol=rtol;self.noise=noise_variance
        self.spectrum=(FourierCovariance(fit_cutoff,lengthscale,variance,upsampfac=1.25)
                       if solver=='defect' else None)
        self.kernel=_AnalyticCG(self.action) if solver=='analytic_cg' else None
        self.rhs_count=None;self.records=[]

    def finite_solve(self, rhs):
        """Approximate finite-system solve only; no analytic convergence claim."""
        if self.spectrum is None:raise ValueError('finite_solve requires solver=defect')
        if rhs.shape[1]!=self.rhs_count:
            self.kernel=None
            self.kernel=PackedFourierKernel(self.points,self.spectrum,self.noise,rhs_count=rhs.shape[1])
            self.rhs_count=rhs.shape[1]
        b=cp.asarray(rhs,dtype='d');scale=cp.max(cp.abs(b),axis=0)
        scale=cp.where(scale>0,scale,1)
        value,info=self.kernel.solve(b/scale,rtol=1e-5,maxiter=1000)
        return value.astype('d')*scale,{**info,'rhs_normalized':True}

    def solve(self, rhs):
        if rhs.ndim!=2 or rhs.shape[0]!=self.action.n or rhs.shape[1]<1:
            raise ValueError('expected nonempty (observations,RHS)')
        if self.solver=='analytic_cg':
            value,info=self.kernel.solve(rhs,rtol=self.rtol/2,maxiter=1000)
        else:
            approximate=self.finite_solve
            initial,initial_info=approximate(rhs)
            value,info=refine_analytic(self.action,approximate,rhs,initial,rtol=self.rtol,max_corrections=8)
            info['initial_finite_solve']=initial_info
        self.records.append(info)
        if not info['converged']:raise RuntimeError(('analytic solve did not converge',info))
        return value


def principal_root(covariance):
    """Reject indefiniteness beyond a documented floating-point roundoff budget.

    Small negative eigenvalues within 100*eps*M*||S||_2 may be projected to zero;
    both the threshold and actual projection are returned. This budget does NOT
    cover solve or NUFFT approximation errors and is never enlarged to fit them.
    """
    s=cp.asarray(covariance)
    if s.ndim!=2 or not len(s) or s.shape[0]!=s.shape[1] or s.dtype!=cp.float64 or not bool(cp.isfinite(s).all()):
        raise ValueError('expected finite double square covariance')
    scale=float(cp.linalg.norm(s));asym=float(cp.linalg.norm(s-s.T))/max(scale,1e-300)
    if asym>1e-7:raise ValueError('posterior covariance is materially asymmetric')
    s=(s+s.T)*.5;eig,vec=cp.linalg.eigh(s)
    spectral_scale=float(cp.max(cp.abs(eig)))
    threshold=100*np.finfo(float).eps*len(s)*spectral_scale
    minimum=float(eig.min())
    if minimum < -threshold:
        raise ValueError(f'indefinite posterior covariance: eigenvalue {minimum}, roundoff budget {threshold}')
    clipped=cp.maximum(eig,0)
    factor=(vec*cp.sqrt(clipped))@vec.T
    return s,factor,{'minimum_eigenvalue':minimum,'maximum_eigenvalue':float(eig.max()),
                     'negative_eigenvalues':int((eig<0).sum()),'maximum_projection':max(0.,-minimum),
                     'roundoff_projection_budget':threshold,'relative_asymmetry':asym,
                     'root_covariance_relative_l2':float(cp.linalg.norm(factor@factor.T-s))/max(scale,1e-300)}


class FixedQueryPosterior:
    def __init__(self, points, queries, operator, solve, mean_coefficients, offset,
                 batch=1, progress=None, residual_audit=None):
        if operator.dtype!=np.dtype('float64') or operator.normalization!='analytic':
            raise ValueError('requires double analytic-normalized Fourier operator')
        self.queries=cp.array(queries,dtype='d',copy=True)
        if self.queries.ndim!=2 or self.queries.shape[1]!=3 or not len(self.queries) or not bool(cp.isfinite(self.queries).all()):
            raise ValueError('expected finite nonempty query panel')
        self.record={'queries':len(self.queries),'batch':batch,'seconds':{}}
        action=FourierCorrectionCovariance(points,self.queries,operator,solve,batch=batch)
        m=len(self.queries)
        self.mean,self.record['seconds']['mean']=timed(lambda:operator.evaluate(self.queries,mean_coefficients)[0]-offset)
        def correction():
            output=cp.empty((m,m),dtype='d')
            for start in range(0,m,batch):
                count=min(batch,m-start)
                basis=cp.zeros((m,count),dtype='d');basis[start+cp.arange(count),cp.arange(count)]=1
                output[:,start:start+count]=action(basis)
                if progress:progress(start+count,m)
            return output
        correction,self.record['seconds']['correction']=timed(correction)
        def prior():
            norm2=(operator.ns**2).sum(0)
            rho=cp.where(norm2>0,operator.rho/cp.where(norm2>0,norm2,1),0)
            value=cp.empty((m,m),dtype='d')
            # Bounded difference-point storage; only the final M*M covariance
            # persists. Query bandwidth and training covariance are unchanged.
            for start in range(0,m,64):
                diff=(self.queries[start:start+64,None]-self.queries[None]).reshape(-1,3)
                value[start:start+64]=operator.evaluate(diff,rho).reshape(-1,m)
            return value
        self.prior,self.record['seconds']['prior']=timed(prior)
        covariance=self.prior-correction
        if residual_audit is not None:
            if solve is not residual_audit:raise ValueError('residual_audit must be the exact solve callable used above')
            covariance,self.record['seconds']['residual_correction']=timed(lambda:residual_audit.correct(covariance))
            self.record['residual_audit']=residual_audit.record
        (self.covariance,self.factor,psd),self.record['seconds']['factor']=timed(lambda:principal_root(covariance))
        self.record['psd']=psd
        self.record['retained_bytes']=sum(v.nbytes for v in [self.queries,self.mean,self.prior,self.covariance,self.factor])

    def draw(self, count, seed=0, white=None):
        if count!=int(count) or count<1:raise ValueError('positive integer draw count required')
        if white is None:white=cp.random.RandomState(seed).standard_normal((int(count),len(self.mean)),dtype='d')
        else:
            white=cp.asarray(white)
            if white.shape!=(count,len(self.mean)) or white.dtype!=cp.float64 or not bool(cp.isfinite(white).all()):
                raise ValueError('white noise shape/precision mismatch')
        return white@self.factor.T+self.mean
