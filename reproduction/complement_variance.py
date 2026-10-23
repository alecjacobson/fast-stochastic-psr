"""Marginal variance estimates from independent posterior/correction components.

For an exactly conditioned Gaussian model, centered posterior g and correction c
are independent, with Var(g)+Var(c)=V0. Means are known to be zero: these are
second moments, not sample variances with an estimated mean. This module cannot
verify the coupling or solver accuracy. It makes no claim for hybrid inverses,
correlated draws, continuous collision probabilities or covariance PSD.
"""
import numpy as np
from scipy.stats import chi2


def _components(posterior, correction):
    g, c = np.asarray(posterior, dtype='d'), np.asarray(correction, dtype='d')
    if g.ndim != 2 or c.shape != g.shape or min(g.shape) < 1:
        raise ValueError('require matching nonempty (draw, query) arrays')
    if not np.isfinite(g).all() or not np.isfinite(c).all():
        raise ValueError('components must be finite')
    return g, c


def pilot_weights(posterior, correction):
    """Freeze weights from independent pilot draws; do not reuse them for scoring.

    The return value weights the posterior estimate. The complementary weight
    multiplies V0-mean(c**2). Any fixed weights are unbiased under the exact law;
    these pilot weights need not attain the unattainable oracle variance.
    """
    g, c = _components(posterior, correction)
    v, w = np.mean(g*g, axis=0), np.mean(c*c, axis=0)
    scale = np.maximum(v, w)
    v = np.divide(v, scale, out=np.zeros_like(v), where=scale > 0)
    w = np.divide(w, scale, out=np.zeros_like(w), where=scale > 0)
    return np.divide(w*w, v*v+w*w, out=np.full_like(w, .5), where=scale > 0)


def estimate(posterior, correction, prior_variance, weights, delta=.05):
    """Raw estimates and simultaneous exact-law component interval intersection.

    Weights must have been fixed independently of these scoring draws. For Q
    queries, Bonferroni allocates delta/(2Q) to each of the two component variance
    intervals. Spatial independence is unnecessary. Independence across draws,
    Gaussianity, exact centering and the consistent conditional law are required.
    Negative raw estimates and empty intersections are retained, never repaired.
    """
    g, c = _components(posterior, correction)
    n, q = g.shape
    v0 = np.broadcast_to(np.asarray(prior_variance, dtype='d'), (q,))
    beta = np.broadcast_to(np.asarray(weights, dtype='d'), (q,))
    if not np.isfinite(v0).all() or np.any(v0 <= 0):
        raise ValueError('prior variance must be finite and positive')
    if not np.isfinite(beta).all() or np.any((beta < 0) | (beta > 1)):
        raise ValueError('weights must lie in [0,1]')
    if not 0 < delta < 1:
        raise ValueError('delta must lie in (0,1)')
    v, w = np.mean(g*g, axis=0), np.mean(c*c, axis=0)
    lower_chi, upper_chi = chi2.ppf([delta/(4*q), 1-delta/(4*q)], n)
    lower = np.maximum.reduce([n*v/upper_chi, v0-n*w/lower_chi, np.zeros(q)])
    upper = np.minimum.reduce([n*v/lower_chi, v0-n*w/upper_chi, v0])
    return {'posterior': v, 'correction': v0-w,
            'combined': beta*v+(1-beta)*(v0-w), 'weights': beta.copy(),
            'lower': lower, 'upper': upper, 'empty_interval': lower > upper,
            'posterior_correction_second_moment': np.mean(g*c, axis=0)}
