"""Match vector-kernel half correlations, not posterior distributions.

GeoSPR Appendix C motivates matching interpolation lengthscales across kernels.
The half-correlation convention below is our explicit implementation choice.
"""
import numpy as np
from scipy.optimize import brentq


def half_correlation_constants():
    # Normalized cubic B-spline on a coordinate axis, with width parameter s.
    spline = brentq(lambda u: 1-1.5*u*u+.75*u**3-.5, 0, 1)
    matern = brentq(lambda u: (1+np.sqrt(3)*u)*np.exp(-np.sqrt(3)*u)-.5, 0, 4)
    return spline, matern


def matched_grid(lower, upper, lengthscale):
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if (lower.shape != (3,) or upper.shape != (3,) or
            not np.isfinite([*lower, *upper, lengthscale]).all() or
            np.any(upper <= lower) or lengthscale <= 0):
        raise ValueError('Expected a finite 3D box and positive lengthscale')
    spline, matern = half_correlation_constants()
    spacing = lengthscale*matern/spline/1.5
    shape = np.ceil((upper-lower)/spacing).astype(int)+1
    # Preserve exact isotropic spacing and center; pad by < one cell per axis.
    corner = (lower+upper-(shape-1)*spacing)/2
    count = int(np.prod(shape))
    return dict(shape=shape.tolist(), spacing=[float(spacing)]*3,
                corner=corner.tolist(), kernel_sigma=float(1.5*spacing),
                target_physical_lengthscale=float(lengthscale),
                half_correlation_distance=float(matern*lengthscale),
                matching_rule='axis normalized vector-kernel correlation = 1/2',
                box_policy='full scan bounds plus .1; symmetric outward rounding to isotropic cells',
                grid_points=count, one_dense_covariance_bytes=8*count**2)
