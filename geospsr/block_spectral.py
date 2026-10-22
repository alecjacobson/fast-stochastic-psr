"""Positive Fourier core plus paired slabs, evaluated with shifted GPU NUFFTs.

This is a finite anisotropic covariance approximation, not the analytic kernel.
Conjugate pairing is valid for real RHS only. Prior draws use sqrt(2*rho) in
each positive slab; matrix actions use twice the real part instead.
"""
import cupy as cp
import numpy as np

from geospsr.spectral import FourierPoisson, FourierKernel, spectral_axis
from geospsr.spectral_blocks import one_high_axis_blocks


class FourierBand:
    plan = FourierPoisson.plan

    def __init__(self,bounds,axis,extent,dtype,eps,upsampfac,multiplicity):
        self.bounds = bounds
        self.shape = tuple(hi-lo+1 for lo,hi in bounds)
        self.shift = tuple(lo+size//2 for (lo,_),size in zip(bounds,self.shape))
        self.dtype = np.dtype(dtype)
        self.cdtype = np.dtype('complex64' if self.dtype == np.float32 else 'complex128')
        self.eps,self.upsampfac,self.multiplicity = eps,upsampfac,multiplicity
        axes = [cp.asarray(axis[lo+extent:hi+extent+1],dtype=self.dtype) for lo,hi in bounds]
        self.rho = axes[0][:,None,None]*axes[1][None,:,None]*axes[2][None,None,:]


class BlockSpectrum:
    def __init__(self,core,extent,lengthscale=.01,variance=.1,eps=1e-6,
                 dtype='float32',upsampfac=1.25,normalization='analytic'):
        boxes = one_high_axis_blocks(core,extent)
        if lengthscale <= 0 or variance <= 0:
            raise ValueError('lengthscale and variance must be positive')
        self.core,self.extent = int(core),int(extent)
        self.dtype = np.dtype(dtype)
        if self.dtype not in (np.dtype('float32'),np.dtype('float64')):
            raise ValueError('dtype must be float32 or float64')
        self.cdtype = np.dtype('complex64' if self.dtype == np.float32 else 'complex128')
        self.normalization = normalization
        self.lengthscale,self.variance = lengthscale,variance
        _,axis = spectral_axis(self.extent,lengthscale,variance,normalization=normalization)
        self.axis64 = axis
        selected = [boxes[0]]+boxes[2::2]
        self.bands = [FourierBand(box,axis,self.extent,self.dtype,eps,upsampfac,1 if i==0 else 2)
                      for i,box in enumerate(selected)]
        self.stored_modes = sum(int(np.prod(b.shape)) for b in self.bands)
        self.retained_modes = sum(b.multiplicity*int(np.prod(b.shape)) for b in self.bands)


class BlockKernel:
    """Reusable plans for one core and three conjugate-paired shifted slabs."""
    solve = FourierKernel.solve

    def __init__(self,points,spectrum,noise_variance,rhs_count=3):
        self.op = spectrum
        self.points = cp.asarray(points,dtype=spectrum.dtype)
        self.shape = (len(points),len(points))
        self.noise_variance,self.rhs_count = noise_variance,rhs_count
        self.plans = []
        for band in spectrum.bands:
            phase = None
            if any(band.shift):
                # Compute phase in double: multiplying float32 positions by a
                # large integer shift otherwise loses angular precision.
                angle = self.points.astype('d')@cp.asarray(band.shift,dtype='d')
                phase = cp.exp(1j*angle).astype(spectrum.cdtype)
            self.plans.append((band,phase,band.plan(1,self.points,rhs_count),band.plan(2,self.points,rhs_count)))

    def __call__(self,weights):
        if cp.iscomplexobj(weights):
            raise ValueError('conjugate-paired block actions require real RHS')
        weights = cp.asarray(weights,dtype=self.op.dtype)
        if weights.shape != (len(self.points),self.rhs_count):
            raise ValueError('weights must have shape (N,rhs_count)')
        result = self.noise_variance*weights
        for band,phase,forward,backward in self.plans:
            source = weights.T if phase is None else weights.T*phase.conj()[None]
            modes = forward.execute(cp.ascontiguousarray(source,dtype=self.op.cdtype))
            modes *= band.rho[None]
            values = backward.execute(modes)
            if phase is not None:
                values *= phase[None]
            result += band.multiplicity*values.real.T
        return result

    def joint_prior(self,rng,scalar_op):
        """Joint vector values at sources and scalar Fourier coefficients.

        Every scalar mode must lie in the core and uses the same Gaussian
        coefficient as its vector counterpart. Slabs are independent and have
        no scalar component. RNG realizations depend on band layout/batching.
        """
        count,remainder = divmod(self.rhs_count,3)
        cutoff = scalar_op.shape[0]//2
        if remainder or not count or cutoff > self.op.core:
            raise ValueError('need 3P RHS and scalar cutoff no larger than core')
        if scalar_op.normalization != self.op.normalization or scalar_op.dtype != self.op.dtype:
            raise ValueError('scalar and vector normalization/dtype must agree')
        # A matching grid alone is insufficient: its spectral weights must
        # describe the same kernel for a valid jointly conditioned prior.
        sl = slice(self.op.core-cutoff,self.op.core+cutoff+1)
        if not bool(cp.allclose(scalar_op.rho,self.op.bands[0].rho[sl,sl,sl],rtol=8*np.finfo(self.op.dtype).eps,atol=0)):
            raise ValueError('scalar and vector kernel parameters must agree')
        result = cp.zeros((len(self.points),self.rhs_count),dtype=self.op.dtype)
        scalar = None
        for index,(band,phase,_,backward) in enumerate(self.plans):
            shape = (self.rhs_count,*band.shape)
            vector = rng.standard_normal(shape,dtype=self.op.dtype).astype(self.op.cdtype)
            vector.imag = -rng.standard_normal(shape,dtype=self.op.dtype)
            vector *= cp.sqrt(band.multiplicity*band.rho)[None]
            if index == 0:
                pad = self.op.core-cutoff
                sl = slice(pad,pad+2*cutoff+1)
                low = vector.reshape(count,3,*band.shape)[:,:,sl,sl,sl]
                norm2 = (scalar_op.ns**2).sum(0)
                scalar = -1j*(low*scalar_op.ns[None]).sum(1)/cp.maximum(norm2,1)
            values = backward.execute(vector)
            if phase is not None:
                values *= phase[None]
            result += values.real.T
        return result,cp.ascontiguousarray(scalar)
