"""Coupled scalar/vector Gaussian solves with a hierarchical normal block.

This finite-contact prototype stores the full cross block (O(N C)) and factors
the scalar PRIOR block (O(C^3)); it does not yet solve issue128's C~N regime.
It avoids forming the posterior scalar Schur complement or solving 3C cross
right-hand sides. Query and data moments share residual-corrected inverse
actions of the same augmented covariance.
"""
from time import perf_counter

import numpy as np


class CoupledObservationSystem:
    def __init__(self,normal_system,model,points,contacts,scalar_variance,*,preconditioner='prior'):
        import cupy as cp
        from reproduction.partial_direct_gpu import cross_rhs
        self.normal=normal_system;self.model=model
        self.points=np.asarray(points,dtype=float);self.contacts=np.asarray(contacts,dtype=float)
        self.n,self.nc=len(points),len(contacts)
        noise=np.broadcast_to(np.asarray(scalar_variance,dtype=float),(self.nc,))
        if not self.n or not self.nc or not np.isfinite(noise).all() or np.any(noise<=0):
            raise ValueError('nonempty observations and positive finite scalar noise required')
        if preconditioner not in ['prior','schur-diagonal']:raise ValueError('unknown coupled preconditioner')
        self.preconditioner_kind=preconditioner
        self.cross64=cp.empty((3*self.n,self.nc),dtype=cp.float64)
        for d in range(3):self.cross64[d*self.n:(d+1)*self.n]=cross_rhs(model,contacts,points,component=d)
        self.cross32=self.cross64.astype(cp.float32)
        self.scalar64=cp.asarray(model.scalar_covariance(contacts,contacts)+np.diag(noise))
        self.scalar32=self.scalar64.astype(cp.float32)
        self.preconditioner_info=dict(kind=preconditioner)
        if preconditioner=='prior':self.scalar_factor=cp.linalg.cholesky(self.scalar64)
        else:
            diagonal=cp.diag(self.scalar64).copy()
            for d in range(3):
                for first in range(0,self.nc,192):
                    last=min(first+192,self.nc);b=self.cross64[d*self.n:(d+1)*self.n,first:last]
                    mb=self.normal.precondition(b)
                    diagonal[first:last]-=cp.sum(b*mb,axis=0)
                    del mb
            floor=cp.asarray(noise)
            self.preconditioner_info.update(unfloored_min=float(diagonal.min().item()),
                floored_count=int(cp.count_nonzero(diagonal<floor).item()),
                scope='Noise floor stabilizes only the preconditioner; covariance and output variances are unchanged.')
            self.schur_diagonal=cp.maximum(diagonal,floor)
        self.extra_device_bytes=sum(a.nbytes for a in [self.cross64,self.cross32,self.scalar64,self.scalar32])
        self.extra_device_bytes+=(self.scalar_factor.nbytes if preconditioner=='prior' else self.schur_diagonal.nbytes)
        self.last_solve=None

    def _normal_layout(self,b):
        return b[:3*self.n].reshape(3,self.n,-1).transpose(1,0,2).reshape(self.n,-1)

    def _stack_layout(self,b):
        return b.reshape(self.n,3,-1).transpose(1,0,2).reshape(3*self.n,-1)

    def matmul(self,rhs):
        import cupy as cp
        b=cp.asarray(rhs,dtype=cp.float64);out=cp.empty_like(b)
        out[:3*self.n]=self._stack_layout(self.normal.matmul(self._normal_layout(b)))
        top=b[:3*self.n].astype(cp.float32);bottom=b[3*self.n:].astype(cp.float32)
        out[:3*self.n]+=self.cross32@bottom
        out[3*self.n:]=self.cross32.T@top+self.scalar32@bottom
        return out

    def accurate_matmul(self,rhs):
        import cupy as cp
        b=cp.asarray(rhs,dtype=cp.float64);out=cp.empty_like(b)
        out[:3*self.n]=self._stack_layout(self.normal.accurate_matmul(self._normal_layout(b)))+self.cross64@b[3*self.n:]
        out[3*self.n:]=self.cross64.T@b[:3*self.n]+self.scalar64@b[3*self.n:]
        return out

    def precondition(self,rhs):
        import cupy as cp
        from cupyx.scipy.linalg import solve_triangular
        b=cp.asarray(rhs,dtype=cp.float64);out=cp.empty_like(b)
        out[:3*self.n]=self._stack_layout(self.normal.precondition(self._normal_layout(b)))
        if self.preconditioner_kind=='schur-diagonal':
            out[3*self.n:]=(b[3*self.n:]-self.cross32.T@out[:3*self.n].astype(cp.float32))/self.schur_diagonal[:,None]
            correction=self.cross32@out[3*self.n:].astype(cp.float32)
            shaped=correction.reshape(3,self.n,-1).transpose(1,0,2).reshape(self.n,-1)
            out[:3*self.n]-=self._stack_layout(self.normal.precondition(shaped))
            return out
        w=solve_triangular(self.scalar_factor,b[3*self.n:],lower=True,check_finite=False)
        out[3*self.n:]=solve_triangular(self.scalar_factor,w,lower=True,trans='T',check_finite=False)
        return out

    def predict(self,vectors,queries,*,iterations=16,refinements=0,scalar_targets=None):
        import cupy as cp
        from reproduction.partial_direct_gpu import cross_rhs
        from .hierarchical import batched_pcg
        if iterations<1 or refinements<0:raise ValueError('positive iteration count and nonnegative refinements required')
        q=np.asarray(queries,dtype=float);nq=len(q)
        y=np.asarray(vectors,dtype=float)
        targets=np.zeros(self.nc) if scalar_targets is None else np.asarray(scalar_targets,dtype=float)
        if y.shape!=(self.n,3) or targets.shape!=(self.nc,) or not np.isfinite(y).all() or not np.isfinite(targets).all():
            raise ValueError('invalid vector or scalar observations')
        b=cp.empty((3*self.n+self.nc,nq+1),dtype=cp.float64)
        for d in range(3):
            b[d*self.n:(d+1)*self.n,:nq]=cross_rhs(self.model,q,self.points,component=d)
            b[d*self.n:(d+1)*self.n,-1]=cp.asarray(y[:,d])
        b[3*self.n:,:nq]=cp.asarray(self.model.scalar_covariance(self.contacts,q));b[3*self.n:,-1]=cp.asarray(targets)
        z=cp.zeros_like(b);residual=b.copy();history=[];start=perf_counter()
        norm=cp.linalg.norm(b,axis=0)
        for stage in range(refinements+1):
            delta,info=batched_pcg(self.matmul,self.precondition,residual,tolerance=1e-10,max_iterations=iterations,xp=cp)
            z+=delta;del delta
            residual=b-self.accurate_matmul(z)
            relative=cp.linalg.norm(residual,axis=0)/cp.where(norm>0,norm,1.)
            history.append(dict(stage=stage,inner=info,true_relative_residuals=cp.asnumpy(relative).tolist()))
        # B.T Z + Z.T(B-AZ) has second-order error in BOTH inverse actions.
        mean=b[:,:nq].T@z[:,-1]+z[:,:nq].T@residual[:,-1]
        variance=self.model.prior_variance-cp.sum(b[:,:nq]*z[:,:nq]+z[:,:nq]*residual[:,:nq],axis=0)
        self.last_solve=dict(history=history,seconds=perf_counter()-start,query_rhs=nq,observation_rhs=1,
            coupled_dimension=3*self.n+self.nc,normal_block='represented hierarchy with float64 residual action',
            cross_and_scalar_blocks='full kernels, float64 residual action',
            preconditioner=self.preconditioner_info,
            scope='O(N C) stored cross block and dense scalar prior factor; no C~N scalability claim')
        return cp.asnumpy(mean),cp.asnumpy(variance)
