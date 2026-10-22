"""Diagnostics for the omitted product-Matérn training spectrum.

This estimates a covariance-error bound, not a posterior sampler or an FMM.
The frequency law includes the infinite discrete tail, without a second cutoff.
Monte Carlo uncertainty must be reported separately from the exact PSD bound.
"""
import cupy as cp
import numpy as np
from geospsr.spectral import spectral_axis


def sample_tail_frequencies(count, cutoff, lengthscale=.01, variance=.1, seed=0):
    """Sample iid integer frequencies from rho(n) outside [-cutoff,cutoff]^3.

    Partition the complement by its first out-of-band axis. The 1D tail uses
    rejection from rounded Pareto(shape=3, minimum=cutoff+1/2). Convexity of
    x^-4 gives q(n) >= 3*(cutoff+1/2)^3*n^-4, hence acceptance <= 1.
    """
    if count < 1 or cutoff < 0 or lengthscale <= 0 or variance <= 0:
        raise ValueError('need positive count, lengthscale, variance and nonnegative cutoff')
    axis, rho = spectral_axis(cutoff, lengthscale, variance)
    p = rho.sum() / variance
    if not 0 < p < 1:
        raise ValueError('tail mass is not resolved at float64 precision')
    mass = variance ** 3 * (-np.expm1(3 * np.log(p)))
    rng = np.random.default_rng(seed)
    lam = np.sqrt(3) / (2 * np.pi * lengthscale)
    low = cutoff + .5

    def tail(size):
        result, remaining = [], size
        while remaining:
            draws = max(64, 2 * remaining)
            x = low / np.cbrt(1 - rng.random(draws))
            n = np.floor(x + .5)
            # Stable exact mass of the rounding cell under the Pareto proposal.
            a, b = n - .5, n + .5
            q = (low / a) ** 3 * (-np.expm1(3 * np.log1p(-1 / b)))
            acceptance = 3 * low ** 3 / (lam * lam + n * n) ** 2 / q
            if np.any(acceptance > 1 + 1e-12):
                raise ArithmeticError('invalid rejection envelope')
            accepted = n[rng.random(draws) < acceptance][:remaining].astype(np.int64)
            accepted *= rng.choice([-1, 1], len(accepted))
            result.append(accepted)
            remaining -= len(accepted)
        return np.concatenate(result) if size else np.empty(0, dtype=np.int64)

    first = rng.choice(3, count, p=np.array([1., p, p * p]) / (1 + p + p * p))
    frequencies = np.empty((count, 3), dtype=np.int64)
    for d in range(3):
        outside = (first == d) | ((first < d) & (rng.random(count) >= p))
        frequencies[outside, d] = tail(int(outside.sum()))
        frequencies[~outside, d] = rng.choice(axis.astype(np.int64), int((~outside).sum()), p=rho / rho.sum())
    return frequencies, float(mass)


_phase = cp.RawKernel(r'''
extern "C" __global__ void phases(const float* x, const long long* n,
    float* cosine, float* sine, const int nx, const int nf) {
    const long long id = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= (long long)nx * nf) return;
    const int i = id / nf, j = id % nf;
    double phase = 0.;
    for (int d = 0; d < 3; ++d) phase += (double)x[3*i+d] * (double)n[3*j+d];
    double s, c;
    sincos(phase, &s, &c);
    cosine[id] = (float)c;
    sine[id] = (float)s;
}
''', 'phases')


def tail_quadratic_samples(points, weights, frequencies, mass, source_batch=65536,
                           frequency_batch=256, progress=None):
    """Unbiased samples of H=sum_d A_d.T Delta A_d, up to floating point.

    weights is (N,Q,3), typically B^-1 times scalar/vector cross covariance.
    Return (R,Q,Q), whose mean estimates H. Work is O(N*R*Q); memory is bounded
    in N*R. This diagnostic does not have the asymptotic cost of a fast sum.
    Phase accumulation uses float64, feature products float32, and block sums
    and outer products float64. Negative off-diagonals are valid.
    """
    x = cp.ascontiguousarray(cp.asarray(points, dtype='f'))
    a = cp.asarray(weights, dtype='f')
    n = cp.ascontiguousarray(cp.asarray(frequencies, dtype=cp.int64))
    if a.ndim != 3 or a.shape[0] != len(x) or a.shape[2] != 3 or n.ndim != 2 or n.shape[1] != 3:
        raise ValueError('expected points (N,3), weights (N,Q,3), frequencies (R,3)')
    if min(source_batch, frequency_batch) < 1 or mass <= 0:
        raise ValueError('batch sizes and mass must be positive')
    q = a.shape[1]
    output = []
    for start in range(0, len(n), frequency_batch):
        f = cp.ascontiguousarray(n[start:start + frequency_batch])
        real = cp.zeros((len(f), q * 3), dtype='d')
        imag = cp.zeros_like(real)
        for offset in range(0, len(x), source_batch):
            xb = x[offset:offset + source_batch]
            ab = cp.ascontiguousarray(a[offset:offset + source_batch].reshape(len(xb), q * 3))
            c = cp.empty((len(xb), len(f)), dtype='f')
            s = cp.empty_like(c)
            _phase(((c.size + 255) // 256,), (256,), (xb, f, c, s, len(xb), len(f)))
            real += (c.T @ ab).astype('d')
            imag += (s.T @ ab).astype('d')
        real, imag = real.reshape(-1, q, 3), imag.reshape(-1, q, 3)
        value = mass * (cp.einsum('rqd,rkd->rqk', real, real) + cp.einsum('rqd,rkd->rqk', imag, imag))
        output.append(cp.asnumpy(value))
        if progress:
            progress(min(start + len(f), len(n)))
    return np.concatenate(output)


_analytic = cp.RawKernel(r'''
extern "C" __global__ void analytic_rows(const float* q, const float* x,
    double* out, const int nq, const int nx, const double ell, const double variance) {
    const long long id = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= (long long)nq * nx) return;
    const int i = id / nx, j = id % nx;
    const double rt3 = sqrt(3.), pi = acos(-1.);
    const double first = 2*ell + rt3 / tanh(rt3 / (2*ell));
    const double norm = (1 + exp(-rt3/ell)) * (first - rt3*tanh(rt3/(2*ell)));
    double value = 1.;
    for (int d = 0; d < 3; ++d) {
        double distance = fmod(fabs((double)q[3*i+d] - (double)x[3*j+d]), 2*pi) / (2*pi);
        double w = rt3*(distance - .5);
        double factor = (exp((w-rt3/2)/ell) + exp((-w-rt3/2)/ell))
                      * (first - 2*w*tanh(w/ell));
        value *= variance * factor / norm;
    }
    out[id] = value;
}
''', 'analytic_rows')


def analytic_covariance_rows64(query, points, lengthscale=.01, variance=.1):
    """Float64 periodic analytic covariance, at float32 input coordinates.

    Same normalized circle Matérn formula as kernels.Matern32Kernel. Intended
    for dense overlap validation; this still has quadratic work/storage.
    """
    q, x = [cp.ascontiguousarray(cp.asarray(v, dtype='f')) for v in (query, points)]
    out = cp.empty((len(q), len(x)), dtype='d')
    _analytic(((out.size + 127) // 128,), (128,),
              (q, x, out, len(q), len(x), float(lengthscale), float(variance)))
    return out
