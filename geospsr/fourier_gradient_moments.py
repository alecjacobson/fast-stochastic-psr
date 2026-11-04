"""Stream squared gradients using one packed inverse FFT per potential.

For real u, |grad u|² = Δ(u²)/2 - u Δu. Products are formed without
aliasing; final differentiation is on the doubled Fourier support. This
changes the algebraic construction, not the represented marginal variance.
"""
import numpy as np
from geospsr.fourier_moments import FourierSquareAccumulator


class FourierGradientSquareAccumulator:
    def __init__(self, cutoff, xp=np, working_dtype='float64'):
        if working_dtype not in ('float32','float64'):
            raise ValueError('working_dtype must be float32 or float64')
        self.base=FourierSquareAccumulator(cutoff,xp=xp)
        self.xp=xp;self.cutoff=self.base.cutoff
        self.fft_width=self.base.fft_width
        self.square=self.base.grid
        self.product=xp.zeros_like(self.square)
        self.cdtype=xp.complex64 if working_dtype=='float32' else xp.complex128
        n=xp.arange(-self.cutoff,self.cutoff+1,dtype=xp.float64)
        self.k2=n[:,None,None]**2+n[None,:,None]**2+n[None,None,:]**2
        self.count=0
        self.reduce=None
        if xp.__name__=='cupy':
            self.reduce=xp.ElementwiseKernel(
                'raw T values, raw float64 su, raw float64 sl, int32 count, int64 stride, float64 old_square, float64 old_product',
                'float64 square, float64 product',
                '''double ss=0.,sp=0.;
                   for(int p=0;p<count;++p) {
                       T v=values[(long long)p*stride+i];
                       double u=(double)v.real()*su[p],lap=(double)v.imag()*sl[p];
                       ss+=u*u;sp+=u*lap;
                   }
                   square=old_square+ss;product=old_product+sp;''',
                'accumulate_gradient_products')

    def add(self, coefficients):
        xp=self.xp;c=xp.asarray(coefficients,dtype=xp.complex128)
        if c.ndim==3:c=c[None]
        if c.ndim!=4 or not len(c) or c.shape[1:]!=(2*self.cutoff+1,)*3 or not bool(xp.isfinite(c).all()):
            raise ValueError('expected finite nonempty (P,2K+1,2K+1,2K+1) coefficients')
        c=(c+c[:,::-1,::-1,::-1].conj())*.5
        # Gradient ignores constants. Removing them also avoids cancellation
        # between Δ(u²)/2 and uΔu for large DC components.
        c[:,self.cutoff,self.cutoff,self.cutoff]=0
        lap=-c*self.k2[None]
        def scaling(v):
            a=xp.max(xp.abs(v),axis=(1,2,3))
            return xp.exp2(xp.floor(xp.log2(xp.where(a>0,a,1))))
        su,sl=scaling(c),scaling(lap)
        packed=(c/su[:,None,None,None]+1j*lap/sl[:,None,None,None]).astype(self.cdtype)
        padded=xp.zeros((len(c),self.fft_width,self.fft_width,self.fft_width),dtype=self.cdtype)
        center=self.fft_width//2;s=slice(center-self.cutoff,center+self.cutoff+1)
        padded[:,s,s,s]=packed
        values=xp.fft.ifftn(xp.fft.ifftshift(padded,axes=(1,2,3)),axes=(1,2,3),norm='forward')
        if self.reduce is not None:
            self.reduce(values,su,sl,len(c),self.fft_width**3,self.square,self.product,
                        self.square,self.product)
        else:
            u=values.real.astype(xp.float64)*su[:,None,None,None]
            lap_u=values.imag.astype(xp.float64)*sl[:,None,None,None]
            self.square+=xp.sum(u*u,axis=0)
            self.product+=xp.sum(u*lap_u,axis=0)
        self.count+=len(c)

    def coefficients(self, constant=0., scale=1.):
        if not np.isfinite(constant) or not np.isfinite(scale):raise ValueError('finite scalars required')
        xp=self.xp;center=self.fft_width//2
        s=slice(center-2*self.cutoff,center+2*self.cutoff+1)
        a=xp.fft.fftshift(xp.fft.fftn(self.square,norm='forward'))[s,s,s]
        b=xp.fft.fftshift(xp.fft.fftn(self.product,norm='forward'))[s,s,s]
        n=xp.arange(-2*self.cutoff,2*self.cutoff+1,dtype=xp.float64)
        k2=n[:,None,None]**2+n[None,:,None]**2+n[None,None,:]**2
        result=scale*(-.5*k2*a-b)
        result[2*self.cutoff,2*self.cutoff,2*self.cutoff]+=constant
        return result
