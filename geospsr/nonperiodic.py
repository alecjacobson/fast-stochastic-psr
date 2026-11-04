"""Experimental free-space vector Matérn -> Poisson scalar Gaussian process.

This is a float64 dense correctness oracle, not an implementation of the
original grid-based SPSR and not a scalable conditioning backend. No wrapping,
fitted isovalue, variance shift, jitter or variance clipping is applied.
"""

from dataclasses import dataclass
from math import factorial

import numpy as np
from scipy.linalg import cho_solve, cholesky, solve_triangular
from scipy.spatial.distance import cdist


def _points(value):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 3 or not np.isfinite(value).all():
        raise ValueError("points must be a finite (n, 3) array")
    return value


# h(z) = [8 - exp(-z) (8+5z+z²)] / z. Taylor coefficients
# through degree 18 avoid subtractive cancellation at coincident points.
_H = np.array([
    -sum(a * (-1.) ** (j + 1 - i) / factorial(j + 1 - i)
         for i, a in enumerate((8, 5, 1)) if i <= j + 1)
    for j in range(19)
])
_CROSS = -np.arange(2, len(_H)) * _H[2:]  # -h'(z)/z


@dataclass(frozen=True)
class FreeSpacePoissonMatern:
    """Independent vector components with k_v(r)=a(1+κr)exp(-κr).

    κ=sqrt(3)/lengthscale; amplitude is the variance of ONE vector component.
    f=Δ^{-1} div(v) is the stationary scalar solution in R³, with no added
    random constant. Its covariance tends to zero at infinity, but its
    marginal variance is spatially constant (3a/κ²) before conditioning.
    """

    lengthscale: float = 1.0
    amplitude: float = 1.0

    def __post_init__(self):
        if not all(np.isfinite(v) and v > 0
                   for v in (self.lengthscale, self.amplitude)):
            raise ValueError("lengthscale and amplitude must be finite and positive")

    @property
    def kappa(self):
        return np.sqrt(3.) / self.lengthscale

    @property
    def prior_variance(self):
        return 3 * self.amplitude / self.kappa**2

    def vector_covariance(self, x, y):
        z = self.kappa * cdist(_points(x), _points(y))
        return self.amplitude * (1 + z) * np.exp(-z)

    def scalar_covariance(self, x, y):
        z = self.kappa * cdist(_points(x), _points(y))
        h = np.empty_like(z)
        small = z < .5
        h[small] = np.polynomial.polynomial.polyval(z[small], _H)
        t = z[~small]
        h[~small] = (8 - np.exp(-t) * (8 + 5*t + t*t)) / t
        return self.amplitude / self.kappa**2 * h

    def cross_covariance(self, queries, observations):
        """Return Cov(f(q), v_d(x)) = -∂_{q_d}g(q-x), shape (m,n,3)."""
        delta = _points(queries)[:, None, :] - _points(observations)[None, :, :]
        z = self.kappa * np.linalg.norm(delta, axis=-1)
        radial = np.empty_like(z)
        small = z < .5
        radial[small] = np.polynomial.polynomial.polyval(z[small], _CROSS)
        t = z[~small]
        radial[~small] = (8 - np.exp(-t) * (8 + 8*t + 4*t*t + t**3)) / t**3
        return self.amplitude * radial[..., None] * delta


class DenseFreeSpacePosterior:
    """O(n³) setup, O(n² m) marginal queries; small-problem reference only."""

    def __init__(self, model, points, vectors, noise_variance):
        self.model = model
        self.points = _points(points).copy()
        vectors = _points(vectors)
        if not len(self.points) or vectors.shape != self.points.shape:
            raise ValueError("points and vectors must have the same nonempty shape")
        noise = np.asarray(noise_variance, dtype=float)
        if (noise.ndim > 1 or (noise.ndim == 1 and noise.shape != (len(points),))
                or not np.isfinite(noise).all() or np.any(noise < 0)):
            raise ValueError("noise_variance must be nonnegative, scalar or length n")
        a = model.vector_covariance(self.points, self.points)
        a[np.diag_indices_from(a)] += noise
        self.factor = cholesky(a, lower=True)
        self.alpha = cho_solve((self.factor, True), vectors)

    def whitened_cross(self, queries):
        """Stack the three L^{-1}c_d(q) blocks as a (3n,m) matrix."""
        c = self.model.cross_covariance(queries, self.points)
        if not len(c):
            return np.empty((3*len(self.points), 0))
        rhs = c.transpose(1, 2, 0).reshape(len(self.points), -1)
        w = solve_triangular(self.factor, rhs, lower=True)
        return w.reshape(len(self.points), 3, len(c)).transpose(1, 0, 2).reshape(-1, len(c))

    def predict(self, queries, batch_size=256):
        queries = _points(queries)
        if not isinstance(batch_size, (int, np.integer)) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        mean, variance = np.empty(len(queries)), np.empty(len(queries))
        for start in range(0, len(queries), batch_size):
            stop = min(start + batch_size, len(queries))
            q = queries[start:stop]
            c = self.model.cross_covariance(q, self.points)
            mean[start:stop] = np.einsum('mnd,nd->m', c, self.alpha)
            w = self.whitened_cross(q)
            variance[start:stop] = self.model.prior_variance - np.sum(w*w, axis=0)
        return mean, variance

    def covariance(self, queries):
        """Full joint scalar covariance for small validation panels."""
        queries = _points(queries)
        w = self.whitened_cross(queries)
        return self.model.scalar_covariance(queries, queries) - w.T @ w


class RegionalVarianceCache:
    """Tensor Chebyshev interpolation of the cross-covariance on one box.

    This prototype uses the dense oracle to build a reusable quadratic form.
    Its online work depends on r=order³, not n; setup is still O(n² r).
    Interpolation does not guarantee nonnegative variances: report errors
    against the oracle. No extrapolation outside the declared box is allowed.
    """

    def __init__(self, posterior, center, halfwidth, order):
        self.center = np.asarray(center, dtype=float)
        self.halfwidth = np.broadcast_to(np.asarray(halfwidth, dtype=float), (3,)).copy()
        if (self.center.shape != (3,) or not np.isfinite(self.center).all()
                or not np.isfinite(self.halfwidth).all() or np.any(self.halfwidth <= 0)):
            raise ValueError("center and positive halfwidth must describe a finite 3D box")
        if not isinstance(order, (int, np.integer)) or order < 2:
            raise ValueError("order must be an integer >= 2")
        self.order = order
        theta = np.pi * (np.arange(order) + .5) / order
        self.abscissae = np.cos(theta)
        self.weights = (-1.)**np.arange(order) * np.sin(theta)
        grid = np.stack(np.meshgrid(*([self.abscissae]*3), indexing='ij'), axis=-1).reshape(-1, 3)
        self.nodes = self.center + self.halfwidth * grid
        w = posterior.whitened_cross(self.nodes)
        self.correction = w.T @ w
        c = posterior.model.cross_covariance(self.nodes, posterior.points)
        self.mean_coefficients = np.einsum('mnd,nd->m', c, posterior.alpha)
        self.prior_variance = posterior.model.prior_variance

    def basis(self, queries):
        x = (_points(queries) - self.center) / self.halfwidth
        if np.any(np.abs(x) > 1 + 1e-12):
            raise ValueError("queries must lie inside the cache box")
        basis = []
        for d in range(3):
            delta = x[:, d, None] - self.abscissae
            exact = np.abs(delta) <= 8*np.finfo(float).eps
            b = self.weights / np.where(exact, 1., delta)
            b /= b.sum(axis=1, keepdims=True)
            hit = exact.any(axis=1)
            b[hit] = exact[hit].astype(float)
            basis.append(b)
        return np.einsum('mi,mj,mk->mijk', *basis).reshape(len(x), self.order**3)

    def predict(self, queries):
        phi = self.basis(queries)
        return (phi @ self.mean_coefficients,
                self.prior_variance - np.sum((phi @ self.correction)*phi, axis=1))
