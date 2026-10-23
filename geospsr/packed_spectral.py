"""Pack two independently scaled real RHS into a full-cube complex transform.

For a real even spectrum K(a+i*b)=K(a)+i*K(b). The full cube is
required: the positive-half real-part action cannot be used here.
"""
import cupy as cp
import numpy as np
from geospsr.spectral import FourierKernel


class PackedFourierKernel:
    """Reusable same-model action; scaling prevents cross-channel leakage.

    Each input channel is divided by its maximum absolute value before
    packing, then restored independently. Zero channels remain exactly zero.
    This controls transform roundoff across unequal channels, not spectral
    truncation or the conditioning of the underlying linear system.
    """
    def __init__(self,points,op,noise_variance,rhs_count=3):
        if rhs_count!=int(rhs_count) or rhs_count<1:
            raise ValueError('rhs_count must be a positive integer')
        if noise_variance<0:
            raise ValueError('noise variance must be nonnegative')
        if np.dtype(op.dtype) not in (np.dtype('float32'),np.dtype('float64')):
            raise ValueError('dtype must be float32 or float64')
        if len(op.shape)!=3 or any(n%2!=1 for n in op.shape):
            raise ValueError('requires an odd centered full spectrum')
        if cp.iscomplexobj(op.rho) or not bool(cp.all(op.rho==op.rho[::-1,::-1,::-1])):
            raise ValueError('requires a real even spectrum')
        self.op,self.noise_variance,self.rhs_count=op,noise_variance,int(rhs_count)
        self.packed_count=(self.rhs_count+1)//2
        self.forward=op.plan(1,points,self.packed_count)
        self.backward=op.plan(2,points,self.packed_count)
        self.shape=(len(points),len(points))

    def __call__(self,weights):
        if cp.iscomplexobj(weights):
            raise ValueError('packed actions require real RHS')
        weights=cp.asarray(weights,dtype=self.op.dtype)
        if weights.shape!=(self.shape[0],self.rhs_count):
            raise ValueError('weights must have shape (N,rhs_count)')
        scales=cp.max(cp.abs(weights),axis=0)
        normalized=weights/cp.where(scales>0,scales,1)[None]
        source=cp.empty((self.packed_count,self.shape[0]),dtype=self.op.cdtype)
        source.real=normalized[:,::2].T
        source.imag=0
        source[:self.rhs_count//2].imag=normalized[:,1::2].T
        modes=self.forward.execute(source)
        modes*=self.op.rho[None]
        values=self.backward.execute(modes)
        result=cp.empty_like(weights)
        result[:,::2]=values.real.T*scales[None,::2]
        result[:,1::2]=values[:self.rhs_count//2].imag.T*scales[None,1::2]
        return result+self.noise_variance*weights

    def solve(self,rhs,rtol=1e-5,maxiter=500,x0=None,preconditioner=None):
        if cp.iscomplexobj(rhs) or (x0 is not None and cp.iscomplexobj(x0)):
            raise ValueError('packed solves require real RHS and initial weights')
        return FourierKernel.solve(self,rhs,rtol=rtol,maxiter=maxiter,x0=x0,preconditioner=preconditioner)
