"""Finite-cube posterior with conjugacy-folded training and prior transforms."""
import cupy as cp
from contextlib import contextmanager
from geospsr.half_spectral import HalfCubeSpectrum,HalfCubeKernel
from geospsr.spectral import FourierPoisson
from reproduction.block_posterior import BlockPosterior
from reproduction.posterior import sync


class HalfCubePosterior(BlockPosterior):
    """Finite posterior with optional packed solves and streaming draws.

    draw() and draw_batches() accept pack_rhs=True for paired full-cube inverse
    transforms. solve_cutoff selects a narrower centered fluctuation inverse
    while retaining the training-band prior and mean; this hybrid is not an
    exact posterior at either bandwidth when the bands differ. Omit it for the
    consistently coupled finite-cube law. Folding/packing alone keep every mode.
    """
    def __init__(self,x,normals,training_cutoff=200,cutoff=50,lengthscale=.01,
                 variance=.1,noise_variance=.02,rtol=1e-5,eps=1e-6,dtype='float32',
                 upsampfac=1.25,normalization='analytic',axis=0):
        if cutoff>training_cutoff:
            raise ValueError('training cutoff must cover every scalar mode')
        self.x=cp.asarray(x,dtype=dtype)
        self.normals=cp.asarray(normals,dtype=dtype)
        self.noise_variance,self.rtol,self.axis=noise_variance,rtol,axis
        self.lengthscale,self.variance,self.cutoff=lengthscale,variance,cutoff
        self.normalization=normalization
        self.op=FourierPoisson(cutoff,lengthscale,variance,eps=eps,dtype=dtype,normalization=normalization)
        self.training=HalfCubeSpectrum(training_cutoff,lengthscale,variance,eps,dtype,upsampfac,normalization)
        self.K=self._make_kernel(3)
        self.alpha,self.mean_solve=self.K.solve(self.normals,rtol=rtol,maxiter=1000)
        if not self.mean_solve['converged']:
            raise RuntimeError(self.mean_solve)
        self.mean_coeff=self.op.aggregate(self.x,self.alpha)
        self.offset=self.op.evaluate(self.x,self.mean_coeff).mean()
        self.mean_cache=self.sample_cache=None
        self.sample_coeff=self.sample_weights=self.sample_prior_coeff=None
        self.cache_batch=16
        self._fluctuation_spectrum=self._fluctuation_kernel=None
        self._packed_spectrum=self._packed_kernel=None
        sync()

    def _make_kernel(self,rhs_count):
        return HalfCubeKernel(self.x,self.training,self.noise_variance,rhs_count,axis=self.axis)

    def _spectrum_record(self):
        record={'training_cutoff':self.training.cutoff,'half_cube':True,'fold_axis':self.axis}
        if self._packed_spectrum is not None:
            record['pack_rhs']=True
        if self._fluctuation_spectrum is not None:
            record.update(solve_cutoff=self._fluctuation_spectrum.cutoff,centered_fluctuations=True)
        return record

    def _solve_draw(self,kernel,rhs,initial_mean):
        if self._packed_spectrum is not None:
            from geospsr.packed_spectral import PackedFourierKernel
            from reproduction.spectral_posterior import solve_centered_fluctuations
            if self._packed_kernel is None or self._packed_kernel.rhs_count!=rhs.shape[1]:
                self._packed_kernel=PackedFourierKernel(self.x,self._packed_spectrum,
                                                        self.noise_variance,rhs.shape[1])
            if self._fluctuation_spectrum is not None:
                return solve_centered_fluctuations(self._packed_kernel,rhs,self.normals,self.alpha,rtol=self.rtol)
            return super()._solve_draw(self._packed_kernel,rhs,initial_mean)
        if self._fluctuation_spectrum is None:
            return super()._solve_draw(kernel,rhs,initial_mean)
        from reproduction.spectral_posterior import solve_centered_fluctuations
        if self._fluctuation_kernel is None or self._fluctuation_kernel.rhs_count!=rhs.shape[1]:
            self._fluctuation_kernel=HalfCubeKernel(self.x,self._fluctuation_spectrum,
                                                   self.noise_variance,rhs.shape[1],axis=self.axis)
        return solve_centered_fluctuations(self._fluctuation_kernel,rhs,self.normals,self.alpha,rtol=self.rtol)

    @contextmanager
    def _draw_context(self,initial_mean=False,solve_cutoff=None,pack_rhs=False):
        if solve_cutoff is not None:
            if solve_cutoff!=int(solve_cutoff) or not self.cutoff<=solve_cutoff<=self.training.cutoff:
                raise ValueError('solve cutoff must lie between scalar and prior cutoffs')
            if initial_mean:
                raise ValueError('centered fluctuations use a zero start; mean is added separately')
            self._fluctuation_spectrum=HalfCubeSpectrum(int(solve_cutoff),self.lengthscale,self.variance,
                self.training.eps,self.training.dtype,self.training.upsampfac,self.normalization)
        try:
            if pack_rhs:
                from geospsr.training_spectrum import FourierCovariance
                band=self.training.cutoff if solve_cutoff is None else int(solve_cutoff)
                self._packed_spectrum=FourierCovariance(band,self.lengthscale,self.variance,
                    self.training.eps,self.training.dtype,self.training.upsampfac,self.normalization)
            yield
        finally:
            self._fluctuation_kernel=self._fluctuation_spectrum=None
            self._packed_kernel=self._packed_spectrum=None
