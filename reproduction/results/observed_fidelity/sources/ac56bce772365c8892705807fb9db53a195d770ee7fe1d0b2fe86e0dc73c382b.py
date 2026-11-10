"""Noisy scalar surface observations after vector-field conditioning.

These observations set f(c)=target with declared independent scalar noise.
They are neither a fitted deterministic isovalue nor a random level subtraction.
The small contact Schur complement never requires a 3N-by-3N factorization.
"""
import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular

from .nonperiodic import _points, RegionalVarianceCache, compress_cross_covariance


class ScalarContactUpdate:
    def __init__(self, mean, covariance, targets, noise_variance):
        mean = np.asarray(mean, dtype=float)
        covariance = np.asarray(covariance, dtype=float)
        targets = np.asarray(targets, dtype=float)
        noise = np.asarray(noise_variance, dtype=float)
        if (mean.ndim != 1 or not len(mean) or targets.shape != mean.shape
                or covariance.shape != (len(mean), len(mean))
                or noise.ndim > 1 or (noise.ndim == 1 and noise.shape != mean.shape)
                or not all(np.isfinite(a).all() for a in (mean, covariance, targets, noise))
                or np.any(noise <= 0)):
            raise ValueError('finite contact moments/targets and positive scalar or per-contact noise required')
        if not np.allclose(covariance, covariance.T, rtol=1e-12, atol=1e-14):
            raise ValueError('contact covariance must be symmetric')
        system = (covariance+covariance.T)*.5
        system[np.diag_indices_from(system)] += noise
        self.factor = cholesky(system, lower=True)
        self.weights = cho_solve((self.factor, True), targets-mean)

    def apply(self, mean, variance, cross_covariance):
        cross = np.asarray(cross_covariance, dtype=float)
        mean, variance = np.asarray(mean, dtype=float), np.asarray(variance, dtype=float)
        if (mean.ndim != 1 or variance.shape != mean.shape
                or cross.shape != (len(mean), len(self.weights))
                or not all(np.isfinite(a).all() for a in (mean, variance, cross))):
            raise ValueError('incompatible finite query moments and contact cross covariance')
        whitened = solve_triangular(self.factor, cross.T, lower=True)
        return mean+cross@self.weights, variance-np.sum(whitened**2, axis=0)


class ContactConditionedDensePosterior:
    """Dense correctness oracle; query work remains quadratic in observations."""
    def __init__(self, posterior, contacts, targets, noise_variance):
        self.posterior = posterior
        self.contacts = _points(contacts).copy()
        self.whitened_contacts = posterior.whitened_cross(self.contacts)
        c = posterior.model.cross_covariance(self.contacts, posterior.points)
        mean = np.einsum('mnd,nd->m', c, posterior.alpha)
        covariance = (posterior.model.scalar_covariance(self.contacts, self.contacts)
                      - self.whitened_contacts.T@self.whitened_contacts)
        self.update = ScalarContactUpdate(mean, covariance, targets, noise_variance)

    def predict(self, queries, batch_size=128):
        q = _points(queries)
        if not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
            raise ValueError('positive integer batch size required')
        mean, variance = np.empty(len(q)), np.empty(len(q))
        for first in range(0, len(q), batch_size):
            last = min(first+batch_size, len(q)); part = q[first:last]
            p = self.posterior
            cross = p.model.cross_covariance(part, p.points)
            m = np.einsum('mnd,nd->m', cross, p.alpha)
            w = p.whitened_cross(part)
            v = p.model.prior_variance-np.sum(w*w, axis=0)
            contact_cross = p.model.scalar_covariance(part, self.contacts)-w.T@self.whitened_contacts
            mean[first:last], variance[first:last] = self.update.apply(m, v, contact_cross)
        return mean, variance


def conditioned_regional_cache(model, points, vectors, contacts, targets,
                               contact_noise_variance, project, center, halfwidth,
                               order=4, compression_tolerance=1e-9,
                               cross_block_size=None):
    """Build marginal polynomials including coherent scalar-contact conditioning.

    Jointly compress vector cross covariances for region nodes and contacts.
    ``project(U, y)`` returns U.T A^-1 U and U.T A^-1 y from any backend.
    Contact covariance and query/contact covariance share the same projection.
    The scalar prior cross covariance is interpolated too, so the resulting
    field needs independent off-node checks, especially near contacts.
    """
    points, vectors, contacts = _points(points), _points(vectors), _points(contacts)
    if points.shape != vectors.shape or not len(points):
        raise ValueError('nonempty matching observations required')
    cache = RegionalVarianceCache.__new__(RegionalVarianceCache)
    cache._initialize_box(center, halfwidth, order)
    nodes = np.concatenate((cache.nodes, contacts))
    basis, coefficients, error = compress_cross_covariance(
        model, points, nodes, compression_tolerance, cross_block_size)
    gram, projected = project(basis, vectors)
    gram = .5*(gram+gram.T)
    width = len(nodes); r = len(cache.nodes)
    blocks = [coefficients[:, d*width:(d+1)*width] for d in range(3)]
    correction = sum(c.T@gram@c for c in blocks)
    mean = sum(c.T@projected[:, d] for d, c in enumerate(blocks))
    contact_covariance = model.scalar_covariance(contacts, contacts)-correction[r:, r:]
    update = ScalarContactUpdate(mean[r:], contact_covariance, targets, contact_noise_variance)
    cross = model.scalar_covariance(cache.nodes, contacts)-correction[:r, r:]
    w = solve_triangular(update.factor, cross.T, lower=True)
    cache.mean_coefficients = mean[:r]+cross@update.weights
    cache.correction = correction[:r, :r]+w.T@w
    cache.prior_variance = model.prior_variance
    cache.solve_rank = basis.shape[1]
    cache.cross_relative_frobenius_error = error
    return cache
