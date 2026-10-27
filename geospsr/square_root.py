"""Fixed positive-root polynomials for coherent Gaussian covariance draws.

Coefficients and degree are selected before seeing the white-noise RHS. Unlike
an adaptively stopped Krylov draw, this is a fixed linear map in exact arithmetic.
The inherited Clenshaw evaluator needs only degree matrix actions and bounded
workspace; its numerical action errors are not covered by the scalar bound.
"""
import numpy as np
from scipy.fft import dct
from geospsr.inverse_root import FixedChebyshevInverseRoot


class FixedChebyshevRoot(FixedChebyshevInverseRoot):
    def __init__(self, lower, upper, relative_tolerance=1e-4, max_degree=20000):
        if not (np.isfinite(lower) and np.isfinite(upper) and 0 < lower <= upper):
            raise ValueError('require finite 0 < lower <= upper')
        if not 0 < relative_tolerance < 1 or max_degree < 1 or max_degree != int(max_degree):
            raise ValueError('invalid tolerance or degree limit')
        self.lower, self.upper = float(lower), float(upper)
        self.center, self.radius = (upper+lower)/2, (upper-lower)/2
        if lower == upper:
            self.coefficients = np.array([np.sqrt(lower)])
            self.coefficients.setflags(write=False)
            self.degree = 0
            self.record = {'degree': 0, 'relative_scalar_error_bound': 0.,
                           'relative_covariance_error_bound': 0., 'lower': lower, 'upper': upper}
            return
        critical = (np.sqrt(upper)+np.sqrt(lower))/(np.sqrt(upper)-np.sqrt(lower))
        candidates = []
        for fraction in np.linspace(.1, .95, 64):
            log_rho = fraction*np.log(critical)
            left = self.center-self.radius*np.cosh(log_rho)
            right = self.center+self.radius*np.cosh(log_rho)
            if left <= 0:
                continue
            # sqrt(center+radius*z) is analytic inside this Bernstein ellipse.
            # Its modulus is <=sqrt(right). Lobatto interpolation error is
            # <=4*sqrt(right)*rho^-degree/(rho-1). Divide by sqrt(lower)
            # for a uniform relative root error on the real spectral interval.
            log_prefactor = np.log(4.)+.5*np.log(right/lower)-np.log(np.expm1(log_rho))
            degree = max(1, int(np.ceil((log_prefactor-np.log(relative_tolerance))/log_rho)))
            candidates.append((degree, log_prefactor-degree*log_rho, log_rho, left))
        if not candidates:
            raise ValueError('spectral interval is numerically unresolved')
        degree, log_error, log_rho, left = min(candidates)
        if degree > max_degree:
            raise ValueError(f'required degree {degree} exceeds limit {max_degree}')
        nodes = np.cos(np.pi*np.arange(degree+1)/degree)
        coefficients = dct(np.sqrt(self.center+self.radius*nodes), type=1)/degree
        coefficients[[0, -1]] *= .5
        coefficients.setflags(write=False)
        self.coefficients, self.degree = coefficients, degree
        error = float(np.exp(log_error))
        self.record = {'degree': degree, 'lower': lower, 'upper': upper,
                       'requested_relative_error': relative_tolerance,
                       'relative_scalar_error_bound': error,
                       'relative_covariance_error_bound': 2*error+error**2,
                       'ellipse_rho': float(np.exp(log_rho)),
                       'ellipse_left_real_part': float(left),
                       'qualification': 'exact-arithmetic interpolation bound; excludes coefficient rounding, matrix-action error and recurrence roundoff'}


def analytic_tail_spectral_bound(points, cutoff, lengthscale=.01, variance=.1,
                                 noise_variance=.02, grid=128, numerical_pad=1e-8,
                                 normalization='analytic'):
    """Enclose R=K_analytic-K_finite+noise*I without a dense matrix.

Positive omitted Fourier coefficients give Delta >= 0 and trace(Delta)=N*mass.
Also Delta <= K_analytic in PSD order, so the analytic box bound applies. Use
the smaller upper bound; observation noise is a lower bound. Coordinates and
spectrum must match both actions. No bound for NUFFT/roundoff errors is claimed.
"""
    from geospsr.inverse_root import spatial_box_spectral_bound
    from geospsr.spectral import spectral_axis
    if normalization != 'analytic':
        raise ValueError('tail bound requires analytic normalization')
    if cutoff != int(cutoff) or cutoff < 0 or not np.isfinite(lengthscale) or lengthscale <= 0 or not np.isfinite(variance) or variance <= 0:
        raise ValueError('invalid cutoff or kernel parameters')
    if not np.isfinite(noise_variance) or noise_variance <= 0 or not np.isfinite(numerical_pad) or numerical_pad < 0:
        raise ValueError('invalid noise or numerical pad')
    box = spatial_box_spectral_bound(points, noise_variance, grid, lengthscale, variance, numerical_pad)
    _, axis = spectral_axis(int(cutoff), lengthscale, variance)
    fraction = float(axis.sum()/variance)
    if not 0 < fraction < 1:
        raise ValueError('tail mass is not resolved at float64 precision')
    mass = float(variance**3*(-np.expm1(3*np.log(fraction))))
    # An absolute pad at the full diagonal scale also covers cancellation when
    # the tail is tiny. Like the box pad, this is not directed-rounding proof.
    trace = len(points)*(mass+numerical_pad*variance**3)
    return {'lower': float(noise_variance),
            'upper': float(min(noise_variance+trace, box['upper'])),
            'tail_diagonal': mass, 'tail_trace_upper': float(trace),
            'cutoff': int(cutoff), 'normalization': normalization, 'analytic_box': box,
            'qualification': 'PSD/trace enclosure in exact arithmetic; float64 padded evaluations, not interval arithmetic; numerical action errors separate'}
