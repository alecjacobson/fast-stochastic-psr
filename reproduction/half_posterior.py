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

    refine_solves=True checks mean and sample conditioning against serial
    float64 transforms of the same finite cube, repairing defects with at most
    two loose native solves. Acceptance uses an actual residual <= verification_rtol
    (default 2*rtol), independently of the native recurrence tolerance.
    This adds time and a large cached double FFT workspace; it does not validate
    truncation, prior sampling or Monte Carlo error. Default solves are unchanged.
    """
    def __init__(self,x,normals,training_cutoff=200,cutoff=50,lengthscale=.01,
                 variance=.1,noise_variance=.02,rtol=1e-5,eps=1e-6,dtype='float32',
                 upsampfac=1.25,normalization='analytic',axis=0,refine_solves=False,verification_rtol=None):
        if verification_rtol is not None and (not refine_solves or not 0<verification_rtol<1):
            raise ValueError('verification_rtol requires refinement and a value strictly between zero and one')
        if cutoff>training_cutoff:
            raise ValueError('training cutoff must cover every scalar mode')
        self.x=cp.asarray(x,dtype=dtype)
        self.normals=cp.asarray(normals,dtype=dtype)
        self.noise_variance,self.rtol,self.axis=noise_variance,rtol,axis
        self.lengthscale,self.variance,self.cutoff=lengthscale,variance,cutoff
        self.normalization=normalization
        self.op=FourierPoisson(cutoff,lengthscale,variance,eps=eps,dtype=dtype,normalization=normalization)
        self.training=HalfCubeSpectrum(training_cutoff,lengthscale,variance,eps,dtype,upsampfac,normalization)
        self.refine_solves=bool(refine_solves)
        self.verification_rtol=2*rtol if verification_rtol is None else float(verification_rtol)
        self._verification_kernel=None
        self.K=self._make_kernel(3)
        self.alpha,self.mean_solve=self._solve_system(self.K,self.normals,training_cutoff)
        if not self.mean_solve['converged']:
            raise RuntimeError(self.mean_solve)
        self.mean_coeff=self.op.aggregate(self.x,self.alpha)
        self.offset=self.op.evaluate(self.x,self.mean_coeff).mean()
        self.mean_cache=self.sample_cache=None
        self.sample_coeff=self.sample_weights=self.sample_prior_coeff=None
        self.cache_batch=16
        self._fluctuation_spectrum=self._fluctuation_kernel=None
        self._packed_spectrum=self._packed_kernel=None
        self.conditioning_preconditioner=None
        self._conditioning_preconditioner_key=None
        self._draw_preconditioner=None
        sync()

    def prepare_conditioning_preconditioner(self,rank=1024,seed=41,row_batch=8192,
                                           basis_dtype=None,recompute_features=False,apply_row_batch=None,fused_apply=False):
        """Build/cache a coarse space for this fixed fitted model.

        Setup is explicit so callers can time it separately. Draws use the
        cache only with preconditioned=True; default draws and fitted mean
        remain unchanged. Refit a new model after changing observations,
        covariance parameters or noise. The retained basis costs O(N*rank).
        Optional float16 basis storage is decoded in bounded blocks at the
        solve dtype; recompute_features avoids the full double setup array.
        fused_apply=True decodes in registers instead of materializing blocks.
        """
        if getattr(self,'_drawing',False):
            raise RuntimeError('cannot prepare a preconditioner during an active stream')
        key=(rank,seed,row_batch,basis_dtype,recompute_features,apply_row_batch,fused_apply)
        if self.conditioning_preconditioner is None or key!=self._conditioning_preconditioner_key:
            from geospsr.nystrom import LandmarkNystrom
            value=LandmarkNystrom(self.x,self.noise_variance,rank,self.lengthscale,self.variance,
                                  seed,row_batch,self.training.dtype,basis_dtype=basis_dtype,
                                  recompute_features=recompute_features,apply_row_batch=apply_row_batch,fused_apply=fused_apply)
            self.conditioning_preconditioner=value
            self._conditioning_preconditioner_key=key
            sync()
            cp.get_default_memory_pool().free_all_blocks()
        return self.conditioning_preconditioner

    def clear_conditioning_preconditioner(self):
        """Release the cached basis after all consumers have closed their stream."""
        if getattr(self,'_drawing',False):
            raise RuntimeError('cannot clear a preconditioner during an active stream')
        self.conditioning_preconditioner=self._conditioning_preconditioner_key=None

    def clear_verification_cache(self):
        """Release verifier plans; the next verified solve rebuilds them.

        CuPy may retain freed arrays in its allocator pool. Callers controlling
        GPU memory can trim that pool after releasing this cache.
        """
        if getattr(self,'_drawing',False):
            raise RuntimeError('cannot clear verification during an active stream')
        self._verification_kernel=None

    def _make_kernel(self,rhs_count):
        return HalfCubeKernel(self.x,self.training,self.noise_variance,rhs_count,axis=self.axis)

    def _spectrum_record(self):
        record={'training_cutoff':self.training.cutoff,'half_cube':True,'fold_axis':self.axis}
        if self._packed_spectrum is not None:
            record['pack_rhs']=True
        if self._fluctuation_spectrum is not None:
            record.update(solve_cutoff=self._fluctuation_spectrum.cutoff,centered_fluctuations=True)
        if self._draw_preconditioner is not None:
            record['conditioning_preconditioner']={'kind':'landmark_nystrom',
                'rank':self._draw_preconditioner.rank,'seed':self._draw_preconditioner.record['seed']}
        return record

    def _verification_action(self,weights,cutoff):
        """Serial double transforms of the same finite cube (not analytic K)."""
        if self._verification_kernel is None or self._verification_kernel.op.cutoff!=cutoff:
            # Drop the previous band's plans before allocating another large FFT.
            self._verification_kernel=None
            cp.get_default_memory_pool().free_all_blocks()
            spectrum=HalfCubeSpectrum(cutoff,self.lengthscale,self.variance,
                eps=1e-10,dtype='float64',upsampfac=2.,normalization=self.normalization)
            self._verification_kernel=HalfCubeKernel(self.x.astype('float64'),spectrum,
                self.noise_variance,rhs_count=1,axis=self.axis)
        return cp.concatenate([self._verification_kernel(weights[:,i:i+1])
                               for i in range(weights.shape[1])],axis=1)

    def _solve_system(self,kernel,rhs,cutoff,x0=None,preconditioner=None):
        options={} if preconditioner is None else {'preconditioner':preconditioner}
        weights,native=kernel.solve(rhs,rtol=self.rtol,maxiter=1000,x0=x0,**options)
        if not self.refine_solves:
            return weights,native
        from geospsr.refinement import refine_solution
        weights,check=refine_solution(
            lambda defect:kernel.solve(defect,rtol=.1,maxiter=1000,**options),
            lambda value:self._verification_action(value,cutoff),rhs,weights,
            target_rtol=self.verification_rtol,max_corrections=2,xp=cp)
        # Keep native recurrence/iteration fields explicitly scoped to the initial
        # solve; only independent verification determines overall convergence.
        return weights,{'converged':check['converged'],'native_solve':native,
                        'refinement':check,'verification_cutoff':cutoff}

    def solve(self,rhs):
        if rhs.shape!=self.normals.shape:
            raise ValueError('this solve interface expects three vector components')
        weights,info=self._solve_system(self.K,rhs,self.training.cutoff)
        if not info['converged']:
            raise RuntimeError(info)
        return weights

    def _solve_draw(self,kernel,rhs,initial_mean):
        band=self.training.cutoff
        if self._fluctuation_spectrum is not None:
            band=self._fluctuation_spectrum.cutoff
        if self._packed_spectrum is not None:
            from geospsr.packed_spectral import PackedFourierKernel
            if self._packed_kernel is None or self._packed_kernel.rhs_count!=rhs.shape[1]:
                self._packed_kernel=PackedFourierKernel(self.x,self._packed_spectrum,
                                                        self.noise_variance,rhs.shape[1])
            kernel=self._packed_kernel
        elif self._fluctuation_spectrum is not None:
            if self._fluctuation_kernel is None or self._fluctuation_kernel.rhs_count!=rhs.shape[1]:
                self._fluctuation_kernel=HalfCubeKernel(self.x,self._fluctuation_spectrum,
                                                       self.noise_variance,rhs.shape[1],axis=self.axis)
            kernel=self._fluctuation_kernel
        if self._fluctuation_spectrum is not None:
            centered=rhs-cp.tile(self.normals,(1,rhs.shape[1]//3))
            weights,info=self._solve_system(kernel,centered,band)
            info={**info,'centered_fluctuations':True,
                  'residual_normalization':'centered prior-plus-noise RHS'}
            return weights+cp.tile(self.alpha,(1,rhs.shape[1]//3)),info
        initial=cp.tile(self.alpha,(1,rhs.shape[1]//3)) if initial_mean else None
        return self._solve_system(kernel,rhs,band,x0=initial,preconditioner=self._draw_preconditioner)

    @contextmanager
    def _draw_context(self,initial_mean=False,solve_cutoff=None,pack_rhs=False,preconditioned=False):
        if preconditioned and self.conditioning_preconditioner is None:
            raise ValueError('prepare the conditioning preconditioner before drawing')
        if preconditioned and solve_cutoff is not None:
            raise ValueError('preconditioned drawing currently requires the full coupled cube')
        if solve_cutoff is not None:
            if solve_cutoff!=int(solve_cutoff) or not self.cutoff<=solve_cutoff<=self.training.cutoff:
                raise ValueError('solve cutoff must lie between scalar and prior cutoffs')
            if initial_mean:
                raise ValueError('centered fluctuations use a zero start; mean is added separately')
            self._fluctuation_spectrum=HalfCubeSpectrum(int(solve_cutoff),self.lengthscale,self.variance,
                self.training.eps,self.training.dtype,self.training.upsampfac,self.normalization)
        try:
            self._draw_preconditioner=self.conditioning_preconditioner if preconditioned else None
            if pack_rhs:
                from geospsr.training_spectrum import FourierCovariance
                band=self.training.cutoff if solve_cutoff is None else int(solve_cutoff)
                self._packed_spectrum=FourierCovariance(band,self.lengthscale,self.variance,
                    self.training.eps,self.training.dtype,self.training.upsampfac,self.normalization)
            yield
        finally:
            self._fluctuation_kernel=self._fluctuation_spectrum=None
            self._packed_kernel=self._packed_spectrum=None
            self._draw_preconditioner=None
