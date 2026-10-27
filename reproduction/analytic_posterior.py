"""Stream coherent finite-scalar/full-analytic-vector posterior functions.

Finite scalar and vector priors share every mode, including the vector zero
mode. The independent omitted vector spectrum and observation noise are drawn
jointly through a fixed polynomial root. Conditional solves use the full
analytic covariance. Root, action and solve errors remain explicit.
"""
import time
from types import SimpleNamespace
import cupy as cp
import numpy as np
from geospsr.analytic_refinement import refine_analytic
from geospsr.analytic_tail import AnalyticTailAction
from geospsr.packed_spectral import PackedFourierKernel
from geospsr.periodic_leaf_cache import LeafCachedRangeKernel
from geospsr.spectral import FourierPoisson, FourierKernel
from geospsr.square_root import FixedChebyshevRoot, analytic_tail_spectral_bound
from geospsr.training_spectrum import FourierCovariance


def timed(fn):
    cp.cuda.get_current_stream().synchronize(); start=time.perf_counter()
    value=fn(); cp.cuda.get_current_stream().synchronize()
    return value,time.perf_counter()-start


class _AnalyticCG:
    def __init__(self, action):
        self.action=action; self.op=SimpleNamespace(dtype=np.dtype('float64'))
    def __call__(self, rhs): return self.action(rhs)
    solve=FourierKernel.solve


class AnalyticPosterior:
    def __init__(self, points, normals, cutoff=50, fit_cutoff=200,
                 lengthscale=.01, variance=.1, noise_variance=.02,
                 root_rtol=1e-7, solve_rtol=1e-8, solver='defect',
                 leaf_size=128, bound_grid=128, max_cache_bytes=2**30,
                 max_entries=100_000_000):
        x=np.asarray(points,dtype='f'); y=np.asarray(normals,dtype='d')
        if x.ndim!=2 or x.shape[1]!=3 or not len(x) or y.shape!=x.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('expected finite nonempty matching (N,3) points and normals')
        if solver not in ('defect','analytic_cg') or fit_cutoff<cutoff or solve_rtol<=0:
            raise ValueError('invalid solver, fit bandwidth or tolerance')
        self.x=cp.asarray(x); self.normals=cp.asarray(y); self.solver=solver
        self.solve_rtol=solve_rtol; self.noise_variance=noise_variance
        self.setup={}
        self.op,self.setup['scalar_spectrum_seconds']=timed(lambda:FourierPoisson(cutoff,lengthscale,variance,dtype='float64',eps=1e-12))
        self.analytic,self.setup['analytic_plan_seconds']=timed(lambda:LeafCachedRangeKernel(x,lengthscale,variance,noise_variance,leaf_size=leaf_size,max_cache_bytes=max_cache_bytes,max_entries=max_entries))
        self.bound,self.setup['bound_seconds']=timed(lambda:analytic_tail_spectral_bound(x,cutoff,lengthscale,variance,noise_variance,grid=bound_grid))
        self.polynomial,self.setup['polynomial_seconds']=timed(lambda:FixedChebyshevRoot(self.bound['lower'],self.bound['upper'],root_rtol))
        self.training=None
        if solver=='defect':
            self.training,self.setup['fit_spectrum_seconds']=timed(lambda:FourierCovariance(fit_cutoff,lengthscale,variance,upsampfac=1.25))
        self.batch_size=None; self.finite=self.fit=self.tail=self.source_plan=None
        _,self.setup['plans_seconds']=timed(lambda:self.prepare(1))
        (self.alpha,self.mean_solve),self.setup['mean_solve_seconds']=timed(lambda:self.solve(self.normals))
        self.mean_coeff,self.setup['mean_aggregation_seconds']=timed(lambda:self.op.aggregate(self.x,self.alpha))
        self.offset,self.setup['mean_centering_seconds']=timed(lambda:self.op.evaluate(self.x,self.mean_coeff).mean())

    def prepare(self, batch):
        if batch!=int(batch) or batch<1: raise ValueError('batch must be a positive integer')
        if self.batch_size==batch: return
        # Only one batch size is retained. Existing mean coefficients/weights
        # are independent of these plans; partial final batches remain valid.
        self.finite=self.fit=self.tail=self.source_plan=None
        self.finite=FourierKernel(self.x,self.op,0,rhs_count=3*batch)
        self.tail=AnalyticTailAction(self.analytic,self.finite)
        self.fit=(_AnalyticCG(self.analytic) if self.solver=='analytic_cg' else
                  PackedFourierKernel(self.x,self.training,self.noise_variance,rhs_count=3*batch))
        self.source_plan=self.op.plan(1,self.x,3*batch)
        self.batch_size=int(batch)

    def solve(self, rhs):
        if self.solver=='analytic_cg':
            weights,info=self.fit.solve(rhs,rtol=self.solve_rtol/2,maxiter=1000)
        else:
            approximate=lambda b:self.fit.solve(b,rtol=1e-5,maxiter=1000)
            initial,initial_info=approximate(rhs)
            weights,info=refine_analytic(self.analytic,approximate,rhs,initial,rtol=self.solve_rtol,max_corrections=8)
            info['initial_finite_solve']=initial_info
        if not info['converged']: raise RuntimeError(('analytic conditioning did not converge',info))
        return weights,info

    def aggregate(self, weights):
        modes=self.source_plan.execute(cp.ascontiguousarray(weights.T,dtype='complex128'))
        modes=modes.reshape(self.batch_size,3,*self.op.shape)
        return cp.ascontiguousarray(-1j*(modes*self.op.multiplier[None]).sum(1))

    def mean(self, query): return self.op.evaluate(query,self.mean_coeff)[0]-self.offset

    def draw_batches(self, count, seed=0, batch=1, retain_components=False):
        """Yield coefficients for complete correlated functions; bound live storage.

        Randomness is reproducible for a fixed batch schedule. A caller that
        retains blocks owns that additional storage. The zero mode contributes
        to vector observations but is excluded from the scalar Poisson prior.
        """
        if count!=int(count) or count<1 or batch!=int(batch) or batch<1:
            raise ValueError('count and batch must be positive integers')
        rng=cp.random.RandomState(seed); n=len(self.x)
        for start in range(0,count,batch):
            p=min(batch,count-start); record={}
            _,record['plan_seconds']=timed(lambda:self.prepare(p))
            def prior():
                xi=rng.standard_normal((p,3,*self.op.shape,2),dtype='d')
                vector,scalar=self.op.prior_coefficients(xi)
                values=self.finite.backward.execute(cp.ascontiguousarray(vector.reshape(3*p,*self.op.shape))).real.T
                white=rng.standard_normal((n,3*p),dtype='d')
                return scalar,values,white
            (scalar,values,white),record['finite_prior_seconds']=timed(prior)
            tail,record['tail_root_seconds']=timed(lambda:self.polynomial.apply(self.tail,white,xp=cp))
            rhs=cp.tile(self.normals,(1,p))-values-tail
            (weights,info),record['solve_seconds']=timed(lambda:self.solve(rhs))
            coefficients,record['aggregation_seconds']=timed(lambda:self.aggregate(weights)+scalar)
            record['solve']=info
            block={'start':start,'stop':start+p,'coefficients':coefficients,'record':record}
            if retain_components:
                block.update(prior=scalar,vector_values=values,tail_white=white,tail=tail,rhs=rhs,weights=weights)
            yield block
