"""Collapse streamed squared finite Fourier functions into one Fourier series.

This accelerates repeated marginal-moment queries. It neither constructs a
posterior factor nor improves a sampling estimator's statistical accuracy.
"""
import numpy as np


class FourierSquareAccumulator:
    """Accumulate ``sum_j weight_j * Re(h_j(x))**2`` without spectral aliasing.

    Input coefficients are centered, odd cubic arrays with frequencies [-K,K].
    Products have frequencies [-2K,2K]; a grid at least (4K+1)^3 resolves every
    product mode. By default pad to a 2/3/5-smooth FFT width and crop the final
    spectrum, avoiding expensive prime-sized FFTs without discarding modes.
    Only one accumulated real grid and the current batch remain.
    Pass ``xp=cupy`` for GPU use. Accumulation and FFTs use double precision.
    """

    def __init__(self, cutoff, xp=np, fft_width=None):
        if cutoff != int(cutoff) or cutoff < 0:
            raise ValueError("cutoff must be a nonnegative integer")
        self.cutoff, self.xp = int(cutoff), xp
        self.width = 4 * self.cutoff + 1
        if fft_width is None:
            fft_width = self.width
            while True:
                remainder = fft_width
                for factor in (2, 3, 5):
                    while remainder % factor == 0:
                        remainder //= factor
                if remainder == 1:
                    break
                fft_width += 1
        if fft_width != int(fft_width) or fft_width < self.width:
            raise ValueError('FFT grid must resolve every product frequency')
        self.fft_width = int(fft_width)
        self.grid = xp.zeros((self.fft_width,) * 3, dtype=xp.float64)
        self.count = 0

    def add(self, coefficients, weights=1.):
        xp = self.xp
        c = xp.asarray(coefficients, dtype=xp.complex128)
        if c.ndim == 3:
            c = c[None]
        if c.ndim != 4 or not len(c) or c.shape[1:] != (2*self.cutoff+1,)*3:
            raise ValueError("expected nonempty (functions, 2K+1, 2K+1, 2K+1) coefficients")
        w = xp.asarray(weights, dtype=xp.float64)
        if w.ndim > 1 or (w.ndim == 1 and w.shape != (len(c),)):
            raise ValueError("weights must be scalar or one per function")
        if not bool(xp.isfinite(c).all()) or not bool(xp.isfinite(w).all()):
            raise ValueError("coefficients and weights must be finite")
        # Re(sum c_n exp(i n.x)) has these Hermitian coefficients, even when
        # the supplied coefficients do not themselves have conjugate symmetry.
        c = (c + c[:, ::-1, ::-1, ::-1].conj()) * .5
        padded = xp.zeros((len(c), self.fft_width, self.fft_width, self.fft_width), dtype=xp.complex128)
        center = self.fft_width // 2
        sl = slice(center-self.cutoff, center+self.cutoff+1)
        padded[:, sl, sl, sl] = c
        values = xp.fft.ifftn(xp.fft.ifftshift(padded, axes=(1, 2, 3)),
                             axes=(1, 2, 3), norm="forward").real
        weight = w if w.ndim == 0 else w[:, None, None, None]
        self.grid += xp.sum(weight * values**2, axis=0)
        self.count += len(c)

    def coefficients(self, constant=0., scale=1.):
        """Return the series for constant + scale * the accumulated squares.

        For an exact correction factor H=C A^-1/2 use constant=prior_variance,
        scale=-1. For S zero-mean posterior draws use scale=1/S. The latter is
        the same finite-sample moment estimator, now valid at arbitrary queries.
        No positivity clipping or statistical correction is applied.
        """
        if not np.isfinite(constant) or not np.isfinite(scale):
            raise ValueError("constant and scale must be finite scalars")
        xp = self.xp
        result = xp.fft.fftshift(xp.fft.fftn(self.grid, norm="forward")) * scale
        center = self.fft_width // 2
        sl = slice(center-2*self.cutoff, center+2*self.cutoff+1)
        result = result[sl, sl, sl].copy()
        result[2*self.cutoff, 2*self.cutoff, 2*self.cutoff] += constant
        return result
