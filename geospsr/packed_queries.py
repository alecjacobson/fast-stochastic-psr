"""Reusable ordinary type-2 plans for pairs of real Fourier fields.

This is a strong direct-query reference: it shares plans, projects arbitrary
coefficients onto their real field, and uses both channels of each complex
transform. It performs the ordinary NUFFT FFT on every evaluation call.
"""
import cupy as cp
import cufinufft
import numpy as np


class PackedDirectEvaluator:
    def __init__(self,points,shape,eps=1e-6,dtype='float32',upsampfac=1.25):
        self.shape=tuple(shape);self.dtype=np.dtype(dtype)
        if len(self.shape)!=3 or any(n<1 or n%2!=1 for n in self.shape):
            raise ValueError('expected three positive odd mode counts')
        if self.dtype not in [np.dtype('float32'),np.dtype('float64')]:
            raise ValueError('dtype must be float32 or float64')
        self.cdtype=np.dtype('complex64' if self.dtype==np.float32 else 'complex128')
        self.points=cp.asarray(points,dtype=self.dtype)
        if self.points.ndim!=2 or self.points.shape[1]!=3: raise ValueError('expected points (M,3)')
        self.coords=[cp.ascontiguousarray(self.points[:,i]) for i in range(3)]
        self.eps,self.upsampfac=eps,upsampfac
        self.plans={}

    def prepare(self,count):
        if count!=int(count) or count<1: raise ValueError('count must be a positive integer')
        pairs=(int(count)+1)//2
        if pairs not in self.plans:
            plan=cufinufft.Plan(2,self.shape,n_trans=pairs,eps=self.eps,isign=1,
                               dtype=self.cdtype.name,upsampfac=self.upsampfac)
            plan.setpts(*self.coords)
            self.plans[pairs]=plan
        return self.plans[pairs]

    def evaluate(self,coefficients):
        c=cp.asarray(coefficients,dtype=self.cdtype)
        if c.ndim==3: c=c[None]
        if c.ndim!=4 or tuple(c.shape[1:])!=self.shape or not len(c):
            raise ValueError('expected coefficient batch matching mode shape')
        # Independent scaling prevents a larger field leaking roundoff into a
        # smaller partner. Projecting Hermitian symmetry retains Re(f) exactly.
        scale=cp.max(cp.abs(c),axis=(1,2,3))
        normalized=c/cp.where(scale>0,scale,1)[:,None,None,None]
        hermitian=(normalized+normalized[:,::-1,::-1,::-1].conj())*.5
        packed=hermitian[::2].copy()
        packed[:len(hermitian[1::2])]+=1j*hermitian[1::2]
        values=self.prepare(len(c)).execute(cp.ascontiguousarray(packed))
        result=cp.empty((len(c),len(self.points)),dtype=self.dtype)
        result[::2]=values.real*scale[::2,None]
        result[1::2]=values[:len(c)//2].imag*scale[1::2,None]
        return result
