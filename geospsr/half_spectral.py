"""The same finite cube covariance using a positive half and a zero-axis plane.

Real source weights give conjugate negative frequencies. Their paired action
is twice the real positive-half action. The plane is evaluated by a 2D NUFFT.
No frequency is omitted relative to the full cube; analytic truncation remains.
"""
import cupy as cp
import cufinufft
import numpy as np
from geospsr.spectral import FourierKernel,spectral_axis


class HalfCubeSpectrum:
    def __init__(self,cutoff=200,lengthscale=.01,variance=.1,eps=1e-6,dtype='float32',
                 upsampfac=1.25,normalization='analytic'):
        if cutoff != int(cutoff) or cutoff < 0:
            raise ValueError('cutoff must be a nonnegative integer')
        if lengthscale <= 0 or variance <= 0:
            raise ValueError('lengthscale and variance must be positive')
        self.cutoff=int(cutoff)
        self.dtype=np.dtype(dtype)
        if self.dtype not in (np.dtype('float32'),np.dtype('float64')):
            raise ValueError('dtype must be float32 or float64')
        self.cdtype=np.dtype('complex64' if self.dtype==np.float32 else 'complex128')
        self.eps,self.upsampfac=eps,upsampfac
        self.lengthscale,self.variance,self.normalization=lengthscale,variance,normalization
        _,axis=spectral_axis(self.cutoff,lengthscale,variance,normalization=normalization)
        self.axis64=cp.asarray(axis)
        r=self.axis64.astype(self.dtype)
        c=self.cutoff
        self.plane_rho=r[c]*r[:,None]*r[None,:]
        self.positive_rho=r[c+1:,None,None]*r[None,:,None]*r[None,None,:]
        self.shift=1+c//2
        self.stored_modes=int(self.plane_rho.size+self.positive_rho.size)
        self.retained_modes=(2*c+1)**3

    def plan(self,kind,points,shape,n_trans):
        coords=[cp.ascontiguousarray(points[:,d],dtype=self.dtype) for d in range(points.shape[1])]
        plan=cufinufft.Plan(kind,shape,n_trans=n_trans,eps=self.eps,
                           isign=-1 if kind==1 else 1,dtype=self.cdtype.name,upsampfac=self.upsampfac)
        plan.setpts(*coords)
        plan._geospsr_coords=coords
        return plan


class HalfCubeKernel:
    """Reusable real-RHS cube action with 3D positive-half and 2D plane plans."""

    def solve(self,rhs,rtol=1e-5,maxiter=500,x0=None):
        if cp.iscomplexobj(rhs) or (x0 is not None and cp.iscomplexobj(x0)):
            raise ValueError('conjugate half-cube solves require real RHS and initial weights')
        return FourierKernel.solve(self,rhs,rtol=rtol,maxiter=maxiter,x0=x0)

    def __init__(self,points,op,noise_variance,rhs_count=3,axis=0):
        if axis not in (0,1,2) or axis!=int(axis):
            raise ValueError('axis must be 0, 1 or 2')
        self.op,self.noise_variance,self.rhs_count=op,noise_variance,rhs_count
        self.axis=int(axis)
        order=[self.axis]+[d for d in range(3) if d!=self.axis]
        self.order=order
        points=cp.asarray(points,dtype=op.dtype)
        if points.ndim!=2 or points.shape[1]!=3:
            raise ValueError('points must have shape (N,3)')
        self.points=cp.ascontiguousarray(points[:,order])
        self.shape=(len(points),len(points))
        self.plane_forward=op.plan(1,self.points[:,1:],op.plane_rho.shape,rhs_count)
        self.plane_backward=op.plan(2,self.points[:,1:],op.plane_rho.shape,rhs_count)
        self.phase=None
        if op.cutoff:
            self.phase=cp.exp(1j*(self.points[:,0].astype('d')*op.shift)).astype(op.cdtype)
            self.forward=op.plan(1,self.points,op.positive_rho.shape,rhs_count)
            self.backward=op.plan(2,self.points,op.positive_rho.shape,rhs_count)

    def __call__(self,weights):
        if cp.iscomplexobj(weights):
            raise ValueError('conjugate half-cube actions require real RHS')
        weights=cp.asarray(weights,dtype=self.op.dtype)
        if weights.shape!=(self.shape[0],self.rhs_count):
            raise ValueError('weights must have shape (N,rhs_count)')
        modes=self.plane_forward.execute(cp.ascontiguousarray(weights.T,dtype=self.op.cdtype))
        modes*=self.op.plane_rho[None]
        result=self.plane_backward.execute(modes).real.T+self.noise_variance*weights
        del modes
        if self.phase is not None:
            source=cp.ascontiguousarray(weights.T*self.phase.conj()[None])
            modes=self.forward.execute(source)
            modes*=self.op.positive_rho[None]
            values=self.backward.execute(modes)
            result+=2*(values*self.phase[None]).real.T
        return result

    def joint_prior(self,rng,scalar_op):
        """Coherent vector values and scalar coefficients for the same cube law.

        Positive-half coefficients have sqrt(2*rho) amplitude; the plane uses
        sqrt(rho). Scalar coefficients use the same draws on their low modes.
        Their negative folded half is zero: taking the real Fourier sum supplies
        its conjugate contribution. Seeded realizations differ from full cubes.
        """
        count,remainder=divmod(self.rhs_count,3)
        c=scalar_op.shape[0]//2
        if remainder or not count or c>self.op.cutoff:
            raise ValueError('need 3P RHS and scalar cutoff no larger than training cutoff')
        if scalar_op.normalization!=self.op.normalization or scalar_op.dtype!=self.op.dtype:
            raise ValueError('scalar and vector normalization/dtype must agree')
        r=self.op.axis64[self.op.cutoff-c:self.op.cutoff+c+1].astype(self.op.dtype)
        expected=r[:,None,None]*r[None,:,None]*r[None,None,:]
        if not bool(cp.allclose(scalar_op.rho,expected,rtol=8*np.finfo(self.op.dtype).eps,atol=0)):
            raise ValueError('scalar and vector kernel parameters must agree')
        scalar=cp.zeros((count,*scalar_op.shape),dtype=self.op.cdtype)
        folded=cp.moveaxis(scalar,self.axis+1,1)
        frequencies=cp.moveaxis(scalar_op.ns,self.axis+1,1)
        norm2=(frequencies**2).sum(0)
        pad=self.op.cutoff-c
        sl=slice(pad,pad+2*c+1)
        vector=rng.standard_normal((self.rhs_count,*self.op.plane_rho.shape),dtype=self.op.dtype).astype(self.op.cdtype)
        vector.imag=-rng.standard_normal(vector.shape,dtype=self.op.dtype)
        vector*=cp.sqrt(self.op.plane_rho)[None]
        low=vector.reshape(count,3,*self.op.plane_rho.shape)[:,:,sl,sl]
        folded[:,c]=-1j*(low*frequencies[None,:,c]).sum(1)/cp.maximum(norm2[c],1)
        values=self.plane_backward.execute(vector).real.T.copy()
        del low,vector
        if self.phase is not None:
            vector=rng.standard_normal((self.rhs_count,*self.op.positive_rho.shape),dtype=self.op.dtype).astype(self.op.cdtype)
            vector.imag=-rng.standard_normal(vector.shape,dtype=self.op.dtype)
            vector*=cp.sqrt(2*self.op.positive_rho)[None]
            if c:
                low=vector.reshape(count,3,*self.op.positive_rho.shape)[:,:,:c,sl,sl]
                folded[:,c+1:]=-1j*(low*frequencies[None,:,c+1:]).sum(1)/norm2[None,c+1:]
                del low
            values+=(self.backward.execute(vector)*self.phase[None]).real.T
        return values,scalar
