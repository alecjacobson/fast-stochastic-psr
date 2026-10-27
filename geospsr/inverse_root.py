"""Fixed polynomial inverse-root actions for Gaussian correction sampling.

The polynomial is fixed before seeing any RHS. Exact arithmetic therefore
preserves Gaussianity for Gaussian inputs. Floating-point/NUFFT errors and the
validity of the supplied spectral interval are separate responsibilities.
"""
import numpy as np
from scipy.fft import dct, fft, rfft, rfftn, irfftn


class FixedChebyshevInverseRoot:
    def __init__(self, lower, upper, relative_tolerance=1e-4, max_degree=20000):
        if not (np.isfinite(lower) and np.isfinite(upper) and 0 < lower <= upper):
            raise ValueError('require finite 0 < lower <= upper')
        if not 0 < relative_tolerance < 1 or max_degree < 1 or max_degree != int(max_degree):
            raise ValueError('invalid tolerance or degree limit')
        self.lower, self.upper = float(lower), float(upper)
        self.center = (upper + lower) / 2
        self.radius = (upper - lower) / 2
        if lower == upper:
            self.coefficients = np.array([1/np.sqrt(lower)])
            self.degree = 0
            self.record = {'degree': 0, 'relative_scalar_error_bound': 0., 'lower': lower, 'upper': upper}
            return
        # f(z)=1/sqrt(center+radius*z) is analytic in every ellipse whose
        # leftmost mapped real part is positive. Optimize only the bound,
        # independently of matrix spectra estimates, probes or Gaussian RHS.
        critical = (np.sqrt(upper)+np.sqrt(lower))/(np.sqrt(upper)-np.sqrt(lower))
        candidates = []
        for fraction in np.linspace(.1, .95, 64):
            log_rho = fraction*np.log(critical)
            rho = np.exp(log_rho)
            left = self.center - self.radius*np.cosh(log_rho)
            if left <= 0:
                continue
            # Chebyshev-Lobatto interpolation error <=4 M rho^-degree/(rho-1).
            # M<=1/sqrt(left); multiply by sqrt(upper) for relative root error.
            log_prefactor = np.log(4.) + .5*np.log(upper/left) - np.log(np.expm1(log_rho))
            degree = max(1, int(np.ceil((log_prefactor-np.log(relative_tolerance))/log_rho)))
            candidates.append((degree, log_prefactor-degree*log_rho, rho, left))
        if not candidates:
            raise ValueError('spectral interval is numerically unresolved')
        degree, log_error, rho, left = min(candidates)
        if degree > max_degree:
            raise ValueError(f'required degree {degree} exceeds limit {max_degree}')
        nodes = np.cos(np.pi*np.arange(degree+1)/degree)
        values = 1/np.sqrt(self.center+self.radius*nodes)
        coefficients = dct(values, type=1)/degree
        coefficients[[0,-1]] *= .5
        coefficients.setflags(write=False)
        self.coefficients, self.degree = coefficients, degree
        self.record = {'degree': degree, 'lower': lower, 'upper': upper,
                       'requested_relative_error': relative_tolerance,
                       'relative_scalar_error_bound': float(np.exp(log_error)),
                       'ellipse_rho': float(rho), 'ellipse_left_real_part': float(left),
                       'qualification': 'exact-arithmetic interpolation bound; excludes coefficient rounding, matrix-action error and recurrence roundoff'}

    def apply(self, action, rhs, xp=np, progress=None):
        """Clenshaw evaluation, degree actions and O(rhs.size) live workspace.

        action must be a fixed self-adjoint SPD map with spectrum in the supplied
        interval. No per-RHS coefficients, adaptive iteration count or stopping.
        A callback receives completed/total action counts and cannot change them.
        """
        rhs = xp.asarray(rhs)
        if rhs.dtype not in (np.dtype('float32'), np.dtype('float64')) or rhs.ndim not in (1,2):
            raise ValueError('expected real float32/float64 vector or RHS matrix')
        if not bool(xp.all(xp.isfinite(rhs))):
            raise ValueError('nonfinite RHS')
        coefficients = self.coefficients.astype(rhs.dtype)
        if not self.degree:
            return coefficients[0]*rhs
        center, radius = rhs.dtype.type(self.center), rhs.dtype.type(self.radius)
        def scaled(value):
            applied = action(value)
            if applied.shape != rhs.shape or applied.dtype != rhs.dtype:
                raise ValueError('action changed RHS shape or precision')
            return (applied-center*value)/radius
        b1 = coefficients[-1]*rhs
        b2 = xp.zeros_like(rhs)
        count = 0
        for k in range(self.degree-1,0,-1):
            b0 = 2*scaled(b1)-b2+coefficients[k]*rhs
            b2,b1 = b1,b0
            count += 1
            if progress is not None:
                progress(count,self.degree)
        result = scaled(b1)-b2+coefficients[0]*rhs
        if progress is not None:
            progress(self.degree,self.degree)
        if not bool(xp.all(xp.isfinite(result))):
            raise FloatingPointError('matrix-function recurrence produced nonfinite values')
        return result


def periodic_matern_axis(distance, lengthscale=.01, variance=.1):
    """Analytic circle Matérn value at nonnegative shortest angular distance."""
    distance = np.asarray(distance, dtype=np.float64)
    if lengthscale <= 0 or variance <= 0 or np.any(distance < 0) or np.any(distance > np.pi):
        raise ValueError('invalid Matérn parameters or shortest angular distance')
    rt3 = np.sqrt(3.)
    w = rt3*(distance/(2*np.pi)-.5)
    first = 2*lengthscale+rt3/np.tanh(rt3/(2*lengthscale))
    norm = (1+np.exp(-rt3/lengthscale))*(first-rt3*np.tanh(rt3/(2*lengthscale)))
    return variance*(np.exp((w-rt3/2)/lengthscale)+np.exp((-w-rt3/2)/lengthscale))*(first-2*w*np.tanh(w/lengthscale))/norm


def spatial_box_spectral_bound(points, noise_variance=.02, grid=128,
                               lengthscale=.01, variance=.1, numerical_pad=1e-8):
    """Positive analytic row-sum envelope from separable periodic box convolution.

    The circle Matérn decreases with shortest distance. Minimum box distances
    therefore upper-bound all entries in each source/target box pair. The largest
    occupied-box row sum bounds the analytic spectral radius. Every correctly
    normalized finite positive Fourier truncation is <= the analytic matrix in
    PSD order, so it inherits this bound, even if some finite entries are negative.

    FFT and kernel evaluations use float64, with an explicit numerical pad.
    This is an exact-arithmetic enclosure evaluated numerically, NOT a directed-
    rounding certificate. NUFFT operator error is not bounded by this routine.
    """
    x = np.asarray(points)
    if x.ndim != 2 or x.shape[1] != 3 or not len(x) or not np.isfinite(x).all():
        raise ValueError('expected finite nonempty (N,3) source')
    if grid != int(grid) or grid < 2 or noise_variance <= 0 or numerical_pad < 0:
        raise ValueError('invalid grid, noise variance or pad')
    grid = int(grid)
    ids = np.minimum((np.remainder(x.astype('d'),2*np.pi)*(grid/(2*np.pi))).astype(np.int64),grid-1)
    flat = (ids[:,0]*grid+ids[:,1])*grid+ids[:,2]
    counts = np.bincount(flat,minlength=grid**3).reshape((grid,)*3).astype('d')
    separation = np.minimum(np.arange(grid),grid-np.arange(grid))
    minimum_distance = np.maximum(0,separation-1)*(2*np.pi/grid)
    axis = periodic_matern_axis(minimum_distance,lengthscale,variance)
    axis_fft, axis_rfft = fft(axis),rfft(axis)
    transformed = rfftn(counts)
    transformed *= axis_fft[:,None,None]
    transformed *= axis_fft[None,:,None]
    transformed *= axis_rfft[None,None,:]
    sums = irfftn(transformed,s=counts.shape)
    box_upper = float(sums[counts>0].max())*(1+numerical_pad)
    trace_upper = len(x)*variance**3
    upper = min(box_upper,trace_upper)+noise_variance
    return {'lower':float(noise_variance),'upper':float(upper),
            'grid':grid,'box_width':float(2*np.pi/grid),'occupied_boxes':int(np.count_nonzero(counts)),
            'max_box_count':int(counts.max()),'box_row_sum_upper':box_upper,
            'trace_upper':trace_upper,'numerical_pad':numerical_pad,
            'qualification':'analytic enclosure in exact arithmetic; float64 FFT/kernel evaluations padded heuristically, not interval arithmetic; finite analytic-normalized kernel only, NUFFT error separate'}
