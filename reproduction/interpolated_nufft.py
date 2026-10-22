"""NUFFT representation of the released piecewise-trilinear cross kernel.

The interpolation grid itself changes the finite-Fourier kernel. To reproduce
that approximation, integrate its periodic hat basis analytically in frequency
space, then truncate the interpolant's Fourier series at a separate cutoff.
"""
import cupy as cp
import cufinufft
import numpy as np
import jax.numpy as jnp
from geospsr.spectral import FourierPoisson


def periodic_hat_coefficients(knots, cutoff):
    """Fourier integrals (1/2pi) of periodic piecewise-linear nodal basis."""
    x = np.asarray(knots, dtype=np.float64)[:-1]
    left = x - np.roll(x, 1)
    left[0] += 2 * np.pi
    right = np.roll(x, -1) - x
    right[-1] += 2 * np.pi
    n = np.arange(-cutoff, cutoff + 1, dtype=np.float64)[:, None]
    safe_n = np.where(n == 0, 1, n)
    u, v = safe_n * left, -safe_n * right
    # Remove the analytically cancelling linear terms. expm1 preserves
    # accuracy for the strongly clustered fifth-power interpolation nodes.
    shape = -(np.expm1(1j * u) / left + np.expm1(1j * v) / right) / (2 * np.pi * safe_n ** 2)
    weights = np.exp(-1j * n * x) * shape
    weights[cutoff] = (left + right) / (4 * np.pi)
    return weights


class InterpolatedFourierPoisson(FourierPoisson):
    def __init__(self, cutoff=100, eps=1e-6):
        self.dtype = np.dtype('float32')
        self.cdtype = np.dtype('complex64')
        self.eps = eps
        self.shape = (2 * cutoff + 1,) * 3
        knots = np.asarray((jnp.mgrid[-1:1:50j] ** 5) * jnp.pi)
        points = np.stack(np.meshgrid(knots, knots, knots, indexing='ij'), -1).reshape(-1, 3)
        base = FourierPoisson(50)
        values = base.cross_values(points).reshape(50, 50, 50, 3)[:-1, :-1, :-1]
        w = cp.asarray(periodic_hat_coefficients(knots, cutoff))
        tmp = cp.einsum('ai,ijkd->ajkd', w, values, optimize=True)
        tmp = cp.einsum('bj,ajkd->abkd', w, tmp, optimize=True)
        self.coefficients = cp.ascontiguousarray(cp.einsum('ck,abkd->dabc', w, tmp, optimize=True), dtype=self.cdtype)

    def plan(self, kind, points, n_trans):
        points = cp.asarray(points, dtype=self.dtype)
        coords = [cp.ascontiguousarray(points[:, i]) for i in range(3)]
        # Small point sets are FFT-bound; larger sets benefit from the
        # narrower spreading stencil at the standard oversampling factor.
        upsampling = 1.25 if len(points) < 100000 else 2.0
        plan = cufinufft.Plan(kind, self.shape, n_trans=n_trans, eps=self.eps,
                             isign=-1 if kind == 1 else 1, dtype=self.cdtype.name,
                             upsampfac=upsampling)
        plan.setpts(*coords)
        plan._geospsr_coords = coords
        return plan

    def cross_values(self, differences):
        return self.evaluate(differences, self.coefficients).T

    def aggregate(self, points, alpha):
        alpha = cp.asarray(alpha, dtype=self.dtype)
        if alpha.ndim == 2:
            alpha = alpha[None]
        p, n, d = alpha.shape
        if d != 3 or n != len(points):
            raise ValueError('expected alpha (P,N,3)')
        weights = cp.ascontiguousarray(alpha.transpose(0, 2, 1).reshape(3 * p, n), dtype=self.cdtype)
        modes = self.plan(1, points, 3 * p).execute(weights).reshape(p, 3, *self.shape)
        return cp.ascontiguousarray((modes * self.coefficients[None]).sum(1))
