"""Cache a NUFFT's spatial grid for repeated evaluation of fixed functions.

Uses cuFINUFFT's public spread/interpolate-only interface. A spread unit impulse
calibrates the discrete gridding-kernel Fourier response. Division by that
response is exact on grid nodes; off-grid accuracy is limited by kernel aliasing
and floating point, and is verified independently against direct Fourier sums.
All requested modes are retained. This is not low-pass sample compression.
"""
import cupy as cp
import cufinufft
import numpy as np
from scipy.fft import next_fast_len


class CachedFourierEvaluator:
    """Single-precision prepared type-2 evaluation; points may change every call.

    Coefficients are centered, odd-sized, cubic 3D spectra. Two real fields are
    packed into each complex grid, requiring ceil(P/2) * grid_size**3 complex
    storage. Preparation and query costs must be
    reported separately. Interpolation-only plans are reused; each query call
    updates their point locations, so no fixed-query answer is cached.
    """
    def __init__(self, coefficients, eps=1e-6, oversampling=1.25, batch=4,
                 grid_oversampling=None):
        coefficients = cp.asarray(coefficients, dtype=cp.complex64)
        if coefficients.ndim == 3:
            coefficients = coefficients[None]
        if (coefficients.ndim != 4 or len(set(coefficients.shape[1:])) != 1
                or coefficients.shape[-1] % 2 != 1 or len(coefficients) == 0):
            raise ValueError('expected a nonempty batch of odd cubic centered spectra')
        if oversampling not in (1.25, 2.0) or batch < 1:
            raise ValueError('supported oversampling is 1.25 or 2; batch must be positive')
        self.eps, self.oversampling = eps, oversampling
        # The interpolation-only response is measured in working precision.
        # Extra padding limits cancellation for corner modes in 3D at float32.
        grid_oversampling = max(1.6, oversampling) if grid_oversampling is None else grid_oversampling
        if grid_oversampling < oversampling:
            raise ValueError('grid padding must be at least the kernel oversampling')
        self.grid_oversampling = grid_oversampling
        self.batch, self.count = batch, len(coefficients)
        self.packed_count = (self.count + 1) // 2
        width = coefficients.shape[-1]
        self.cutoff = width // 2
        # At least 32 cells avoids gridding support overlapping itself on tiny
        # test problems. Production grids are chosen by the spectral bandwidth.
        self.size = next_fast_len(max(32, int(np.ceil(width * grid_oversampling))))
        self.shape = (self.size,) * 3
        impulse = cp.ones(1, dtype=cp.complex64)
        origin = cp.zeros(1, dtype=cp.float32)
        spread = cufinufft.Plan(1, (self.size,), n_trans=1, eps=eps,
                               dtype='complex64', gpu_spreadinterponly=1,
                               upsampfac=oversampling, isign=1)
        spread.setpts(origin)
        kernel = spread.execute(impulse)
        # Interpolation at zero of exp(+2pi i k j / G) equals this response.
        response = cp.fft.ifft(kernel.astype(cp.complex128)) * self.size
        modes = cp.arange(-self.cutoff, self.cutoff + 1)
        response = response[modes % self.size]
        denominator = (response[:, None, None] * response[None, :, None]
                       * response[None, None, :]).astype(cp.complex64)
        if not bool(cp.all(cp.isfinite(denominator) & (cp.abs(denominator) > 1e-20))):
            raise ValueError('invalid gridding response in retained frequency band')
        self.grid = cp.empty((self.packed_count, *self.shape), dtype=cp.complex64)
        start = self.size // 2 - self.cutoff
        sl = slice(start, start + width)
        for i in range(0, self.packed_count, batch):
            raw = coefficients[2 * i:2 * (i + batch)]
            # Taking the Hermitian part preserves Re(sum c_n exp(i n x)).
            # Packing two resulting real functions into one complex FFT halves
            # cache storage and interpolation work without dropping any modes.
            hermitian = (raw + raw[:, ::-1, ::-1, ::-1].conj()) * .5
            block = hermitian[::2].copy()
            block[:len(hermitian[1::2])] += 1j * hermitian[1::2]
            padded = cp.zeros((len(block), *self.shape), dtype=cp.complex64)
            padded[:, sl, sl, sl] = block / denominator
            spatial = cp.fft.ifftn(cp.fft.ifftshift(padded, axes=(1, 2, 3)),
                                   axes=(1, 2, 3), norm='forward')
            self.grid[i:i + len(block)] = spatial
        self.cache_bytes = int(self.grid.nbytes)
        self._plans = {}

    def evaluate(self, points, batch=None):
        points = cp.asarray(points, dtype=cp.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError('expected query positions with shape (M, 3)')
        batch = self.batch if batch is None else batch
        if batch < 1:
            raise ValueError('batch must be positive')
        # Fold explicitly to avoid single-precision range-reduction error for
        # reasonable periodic translations; input precision still limits this.
        points = (points + np.pi) % (2 * np.pi) - np.pi
        coords = [cp.ascontiguousarray(points[:, k]) for k in range(3)]
        plans, values = {}, []
        for i in range(0, self.packed_count, batch):
            block = self.grid[i:i + batch]
            count = len(block)
            if count not in plans:
                if count not in self._plans:
                    self._plans[count] = cufinufft.Plan(2, self.shape, n_trans=count, eps=self.eps,
                        dtype='complex64', gpu_spreadinterponly=1,
                        upsampfac=self.oversampling, isign=1)
                plan = self._plans[count]
                plan.setpts(*coords)
                plans[count] = plan
            values.append(plans[count].execute(block))
        packed = cp.concatenate(values)
        result = cp.empty((self.count, len(points)), dtype=cp.float32)
        result[::2] = packed.real
        result[1::2] = packed[:self.count // 2].imag
        return result
