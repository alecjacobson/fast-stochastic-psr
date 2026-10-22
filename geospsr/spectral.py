"""Optional cuFINUFFT fast sums for the *finite* GeoSPSR Fourier kernel.

This does not approximate or replace the training covariance implicitly. A separate
NUFFT tolerance controls transform error; cutoff controls spectral truncation.
CuPy/cufinufft are optional dependencies. Arrays stay on the GPU.
"""
import numpy as np
import cupy as cp
import cufinufft


def spectral_grid(cutoff, lengthscale=.01, variance=.1, dtype=np.float32):
    """Match ProductKernel(Matern32Kernel(...), ...) in upstream normalization."""
    n = np.arange(-cutoff, cutoff+1, dtype=np.float64)
    rt3 = np.sqrt(3.)
    w = rt3 / 2
    first = 2 * lengthscale + rt3 / np.tanh(w / lengthscale)
    norm = (1 + np.exp(-2*w/lengthscale)) * (np.pi**2*lengthscale/3) * (first-2*w*np.tanh(w/lengthscale))
    rho1 = (2*rt3*np.tanh(w/lengthscale)/((2*np.pi)**-2*lengthscale)) * (3/lengthscale**2+4*np.pi**2*n**2)**-2 / norm * variance
    rho = rho1[:,None,None]*rho1[None,:,None]*rho1[None,None,:]
    ns = np.stack(np.meshgrid(n,n,n,indexing='ij'))
    return cp.asarray(ns.astype(dtype)), cp.asarray(rho.astype(dtype))


class FourierPoisson:
    """Type 1 source aggregation + spectral multiplier + type 2 evaluation.

    Source alpha layout is (P,N,3), output (P,M). The real cross kernel is
    sum_n rho(n) n_i/|n|^2 sin(n.(x-y)); zero mode is exactly zero.
    """
    def __init__(self, cutoff=50, lengthscale=.01, variance=.1, eps=1e-6, dtype='float32'):
        self.dtype = np.dtype(dtype)
        self.cdtype = np.dtype('complex64' if self.dtype == np.float32 else 'complex128')
        self.eps = eps
        self.ns, self.rho = spectral_grid(cutoff,lengthscale,variance,self.dtype)
        self.shape = self.rho.shape
        norm2 = (self.ns**2).sum(0)
        self.multiplier = self.ns*self.rho/cp.where(norm2==0,1,norm2)

    def plan(self, kind, points, n_trans):
        points = cp.asarray(points,dtype=self.dtype)
        coords = [cp.ascontiguousarray(points[:,i]) for i in range(3)]
        plan = cufinufft.Plan(kind,self.shape,n_trans=n_trans,eps=self.eps,
                             isign=-1 if kind==1 else 1,dtype=self.cdtype.name)
        plan.setpts(*coords)
        # Retain coordinate memory for the plan's lifetime.
        plan._geospsr_coords = coords
        return plan

    def aggregate(self, points, alpha):
        alpha = cp.asarray(alpha,dtype=self.dtype)
        if alpha.ndim == 2: alpha = alpha[None]
        p,n,d = alpha.shape
        if d != 3 or n != len(points): raise ValueError('expected alpha (P,N,3)')
        plan = self.plan(1,points,3*p)
        weights = cp.ascontiguousarray(alpha.transpose(0,2,1).reshape(3*p,n),dtype=self.cdtype)
        modes = plan.execute(weights).reshape(p,3,*self.shape)
        return cp.ascontiguousarray(-1j*(modes*self.multiplier[None]).sum(1))

    def evaluate(self, points, coefficients):
        coefficients = cp.ascontiguousarray(coefficients,dtype=self.cdtype)
        if coefficients.ndim == 3: coefficients = coefficients[None]
        return self.plan(2,points,len(coefficients)).execute(coefficients).real

    def evaluate_batched(self, points, coefficients, batch=4):
        """Evaluate many fields with bounded workspace and one plan per batch size.

        Plans live for this call only: arbitrary new query geometry remains
        supported, and its setup is included in query timings.
        """
        if batch < 1:
            raise ValueError('batch must be positive')
        coefficients = cp.asarray(coefficients, dtype=self.cdtype)
        if coefficients.ndim == 3:
            coefficients = coefficients[None]
        plans, values = {}, []
        for start in range(0, len(coefficients), batch):
            block = cp.ascontiguousarray(coefficients[start:start + batch])
            count = len(block)
            if count not in plans:
                plans[count] = self.plan(2, points, count)
            values.append(plans[count].execute(block).real)
        return cp.concatenate(values)

    def cross_values(self, differences):
        coefficients = cp.ascontiguousarray(-1j*self.multiplier,dtype=self.cdtype)
        return self.evaluate(differences,coefficients).T

    def prior_coefficients(self, xi):
        """xi: (P,3,L,2), same independent real normals as upstream sampling."""
        xi = cp.asarray(xi,dtype=self.dtype).reshape(-1,3,*self.shape,2)
        v = cp.sqrt(self.rho)[None,None]*(xi[...,0]-1j*xi[...,1])
        norm2 = (self.ns**2).sum(0)
        f = -1j*(v*self.ns[None]).sum(1)/cp.where(norm2==0,1,norm2)
        return v, f


class FourierKernel:
    """Matrix-free truncated vector-field covariance, with reusable GPU plans.

    This is a different training covariance from the full analytic Matérn kernel.
    Always assess truncation error separately from CG residual and NUFFT error.
    """
    def __init__(self, points, op, noise_variance, rhs_count=3):
        self.op=op
        self.noise_variance=noise_variance
        self.forward=op.plan(1,points,rhs_count)
        self.backward=op.plan(2,points,rhs_count)
        self.shape=(len(points),len(points))

    def __call__(self, weights):
        weights=cp.asarray(weights,dtype=self.op.dtype)
        modes=self.forward.execute(cp.ascontiguousarray(weights.T,dtype=self.op.cdtype))
        values=self.backward.execute(cp.ascontiguousarray(modes*self.op.rho[None]))
        return values.real.T+self.noise_variance*weights

    def solve(self, rhs, rtol=1e-5, maxiter=500):
        """Independent CG recurrences, batched over RHS; true residual reported."""
        rhs=cp.asarray(rhs,dtype=self.op.dtype)
        x=cp.zeros_like(rhs);r=rhs.copy();p=r.copy()
        rr=(r*r).sum(0);bnorm=cp.sqrt(rr)
        threshold=rtol**2*rr
        converged=False
        for iteration in range(maxiter):
            ap=self(p);denom=(p*ap).sum(0)
            active=rr>threshold
            if not bool(cp.all((denom>0)|~active)):
                raise RuntimeError('CG encountered nonpositive curvature')
            step=cp.where(active,rr/cp.where(denom!=0,denom,1),0)
            x+=p*step;r-=ap*step
            newrr=(r*r).sum(0)
            if bool(cp.all(newrr<=threshold)):
                converged=True;break
            p=r+p*cp.where(active,newrr/cp.where(rr!=0,rr,1),0)
            rr=newrr
        actual=cp.sqrt(((self(x)-rhs)**2).sum(0))/cp.where(bnorm>0,bnorm,1)
        return x,{'iterations':iteration+1,'recurrence_converged':converged,
                  'relative_residual':cp.asnumpy(actual).tolist(),
                  'converged':bool(cp.all(actual<=rtol*2))}
