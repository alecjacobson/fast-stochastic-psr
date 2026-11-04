import numpy as np
import pytest
from scipy.special import ndtr

from geospsr.nonperiodic import FreeSpacePoissonMatern
from reproduction.audit_hierarchical_variance import corrected_reference_moments, probability_allowance


@pytest.mark.parametrize('perturbation', [1e-3, 1e-5])
def test_reference_corrections_bound_independent_dense_posterior(perturbation):
    rng = np.random.default_rng(49)
    x = rng.normal(size=(70, 3))*.2
    y = rng.normal(size=x.shape)
    q = rng.uniform(-.01, .01, size=(9, 3))
    model = FreeSpacePoissonMatern(.15, .7)
    noise = .05
    a = model.vector_covariance(x, x)+noise*np.eye(len(x))
    c = model.cross_covariance(q, x).transpose(1, 2, 0).reshape(len(x), -1)
    rhs = np.column_stack((c, y))
    exact = np.linalg.solve(a, rhs)
    approximate = exact+perturbation*rng.normal(size=exact.shape)
    residual = rhs-a@approximate
    mean, variance, mb, vb = corrected_reference_moments(rhs, approximate, residual, model.prior_variance, noise)
    mref = sum(c[:, d*len(q):(d+1)*len(q)].T@exact[:, -3+d] for d in range(3))
    vref = model.prior_variance-np.sum(c*exact[:, :-3], axis=0).reshape(3, len(q)).sum(axis=0)
    assert np.all(abs(mean-mref) <= mb+1e-14)
    assert np.all(vref <= variance+1e-14)
    assert np.all(vref >= variance-vb-1e-14)
    p = ndtr((mean[-1]-mean)/np.sqrt(variance))
    pref = ndtr((mref[-1]-mref)/np.sqrt(vref))
    allowance = probability_allowance(mean, variance, mb, vb)
    assert np.all(abs(p-pref) <= allowance+1e-13)
