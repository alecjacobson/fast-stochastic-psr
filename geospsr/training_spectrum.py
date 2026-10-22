"""Independent training bandwidth and direct finite-covariance reference.

The vector covariance spectrum is separable, so this avoids allocating the
three Poisson multipliers needed only for the scalar cross kernel. It remains
a truncated training model and must be checked against the analytic covariance.
"""
import cupy as cp
import numpy as np
from geospsr.spectral import FourierPoisson, spectral_axis


class FourierCovariance:
    def __init__(self, cutoff=100, lengthscale=.01, variance=.1, eps=1e-6,
                 dtype='float32', upsampfac=1.25):
        self.cutoff = cutoff
        self.dtype = np.dtype(dtype)
        self.cdtype = np.dtype('complex64' if self.dtype == np.float32 else 'complex128')
        self.eps, self.upsampfac = eps, upsampfac
        _, axis = spectral_axis(cutoff, lengthscale, variance)
        self.axis64 = cp.asarray(axis)
        r = self.axis64.astype(self.dtype)
        self.rho = r[:, None, None] * r[None, :, None] * r[None, None, :]
        self.shape = self.rho.shape

    plan = FourierPoisson.plan
    evaluate = FourierPoisson.evaluate
    evaluate_batched = FourierPoisson.evaluate_batched


_rows_source = r'''
extern "C" __global__ void finite_rows(
    const float* q, const float* x, const double* rho, float* out,
    const int nq, const int nx, const int cutoff) {
    const long long id = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= (long long)nq * nx) return;
    const int i = id / nx, j = id % nx;
    double value = 1.;
    for (int d = 0; d < 3; ++d) {
        const double angle = (double)q[3*i+d] - (double)x[3*j+d];
        const double cosine = cos(angle);
        double previous = 1., current = cosine;
        double sum = rho[cutoff];
        for (int n = 1; n <= cutoff; ++n) {
            sum += 2. * rho[cutoff+n] * current;
            const double next = 2. * cosine * current - previous;
            previous = current;
            current = next;
        }
        value *= sum;
    }
    out[id] = (float)value;
}
'''
_rows = cp.RawKernel(_rows_source, 'finite_rows')
_rows64 = cp.RawKernel(_rows_source.replace('float* out', 'double* out').replace(
    'out[id] = (float)value', 'out[id] = value'), 'finite_rows')


def finite_covariance_rows(query, points, spectrum, dtype='float32'):
    """Direct product of 1D cosine sums, independent of the NUFFT implementation.

    Intended for overlap validation, not massive N-by-N storage. Recurrence and
    accumulation use float64; fixed input positions use float32. Float64 output
    is available for subtracting the finite kernel from the analytic kernel.
    """
    q = cp.ascontiguousarray(cp.asarray(query, dtype=cp.float32))
    x = cp.ascontiguousarray(cp.asarray(points, dtype=cp.float32))
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype('float32'), np.dtype('float64')):
        raise ValueError('output dtype must be float32 or float64')
    result = cp.empty((len(q), len(x)), dtype=dtype)
    size = result.size
    kernel = _rows if dtype == np.float32 else _rows64
    kernel(((size + 127) // 128,), (128,),
          (q, x, spectrum.axis64, result, len(q), len(x), spectrum.cutoff))
    return result
