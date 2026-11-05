"""Physical parameter contract for the density-dependent Eagle experiment."""
import numpy as np


def lengthscale(n):
    if n <= 0: raise ValueError('positive observation count required')
    return .12*np.sqrt(8192/n)


def original_parameters(spacing, amplitude, noise_variance):
    """Express fixed vector amplitude/noise in the density-lumped native units.

    Let c=A/k(0). Then c*K2/(c*rho+eta) = K2/(rho+eta/c),
    and posterior covariance is c*(K1-K2.T/(rho+eta/c)*K2).
    The native routine also adds eta/c to scalar variance; the caller replaces
    that separate postprocessing floor with fixed eta+eps after the solve.
    """
    if amplitude <= 0 or noise_variance < 0 or min(spacing) <= 0:
        raise ValueError('invalid kernel/noise parameters')
    width = 1.5*min(spacing)
    peak = (2/3)**3/width**3
    scale = amplitude/peak
    return dict(kernel_peak=peak, covariance_scale=scale, sigma=np.sqrt(scale),
                sigma_n=np.sqrt(noise_variance/scale),
                target_vector_variance=amplitude, target_vector_noise_variance=noise_variance,
                native_noise_variance=noise_variance/scale,
                scalar_variance_floor=noise_variance+1e-6,
                convention='scaled density-lumped prior/noise; fixed scalar variance floor; distinct from published defaults')


def replace_original_floor(variance, parameters):
    return np.asarray(variance)-parameters['native_noise_variance']+parameters['target_vector_noise_variance']
