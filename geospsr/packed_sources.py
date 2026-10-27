"""Aggregate pairs of real source channels with one complex type-1 NUFFT."""
import cupy as cp


class PackedRealSourcePlan:
    """Return all individual spectra, preserving the odd centered full cube.

    F(a)=(F(a+i b)+conj(F(a+i b)(-k)))/2 and similarly for b.
    Independent maximum-absolute scaling limits cross-channel roundoff. The
    complete spectra are restored before any cross-kernel multiplication.
    """
    def __init__(self, operator, points, n_trans):
        if n_trans!=int(n_trans) or n_trans<1:
            raise ValueError('positive integer channel count required')
        if len(operator.shape)!=3 or any(n<1 or n%2!=1 for n in operator.shape):
            raise ValueError('requires an odd centered full Fourier cube')
        self.op,self.count,self.points=operator,int(n_trans),len(points)
        self.plan=operator.plan(1,points,(self.count+1)//2)

    def execute(self, strengths):
        strengths=cp.asarray(strengths)
        if cp.iscomplexobj(strengths) and bool(cp.any(strengths.imag!=0)):
            raise ValueError('packing requires real source strengths')
        values=cp.asarray(strengths.real,dtype=self.op.dtype)
        if values.shape!=(self.count,self.points) or not bool(cp.isfinite(values).all()):
            raise ValueError('invalid strengths shape or nonfinite values')
        scales=cp.max(cp.abs(values),axis=1)
        normalized=values/cp.where(scales>0,scales,1)[:,None]
        source=cp.empty(((self.count+1)//2,self.points),dtype=self.op.cdtype)
        source.real=normalized[::2];source.imag=0
        source[:self.count//2].imag=normalized[1::2]
        modes=self.plan.execute(source).reshape((self.count+1)//2,*self.op.shape)
        mirror=modes[:,::-1,::-1,::-1].conj()
        result=cp.empty((self.count,*self.op.shape),dtype=self.op.cdtype)
        result[::2]=(modes+mirror)*(.5*scales[::2,None,None,None])
        result[1::2]=(modes[:self.count//2]-mirror[:self.count//2])*(-.5j*scales[1::2,None,None,None])
        return result
