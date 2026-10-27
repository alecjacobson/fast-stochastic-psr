"""Matrix-free analytic Fourier-tail plus observation-noise covariance."""
import cupy as cp
import numpy as np


class AnalyticTailAction:
    """Subtract an unnoised finite action from a noisy analytic action.

Both actions must use identical points and kernel parameters. The constructor
checks the accessible noise/normalization/precision contracts; callers remain
responsible for matching coordinates and certifying the spectral interval.
Double precision reduces cancellation, but does not certify NUFFT error.
"""
    def __init__(self, analytic, finite):
        if finite.noise_variance != 0:
            raise ValueError('finite action must exclude observation noise')
        if finite.op.normalization != 'analytic':
            raise ValueError('finite action must use analytic normalization')
        if finite.op.dtype != np.dtype('float64'):
            raise ValueError('tail subtraction requires double-precision finite action')
        if not np.isfinite(analytic.noise_variance) or analytic.noise_variance <= 0:
            raise ValueError('analytic action must include positive observation noise')
        if finite.shape != (analytic.n, analytic.n):
            raise ValueError('action dimensions differ')
        self.analytic, self.finite = analytic, finite
        self.shape = finite.shape

    def __call__(self, weights):
        w = cp.asarray(weights)
        vector = w.ndim == 1
        if vector:
            w = w[:,None]
        if w.dtype != cp.float64 or w.ndim != 2 or w.shape[0] != self.shape[0] or not bool(cp.isfinite(w).all()):
            raise ValueError('expected finite double-precision RHS')
        result = self.analytic(w)-self.finite(w)
        return result[:,0] if vector else result
