import numpy as np
import pytest
from scipy.linalg import block_diag, cho_solve

from geospsr.nonperiodic import FreeSpacePoissonMatern, DenseFreeSpacePosterior
from geospsr.scalar_contacts import (
    ScalarContactUpdate, ContactConditionedDensePosterior, conditioned_regional_cache)
from geospsr.regional_field import PolynomialMomentField


def problem():
    rng = np.random.default_rng(41)
    x, y = rng.normal(size=(18, 3)), rng.normal(size=(18, 3))
    c, q = rng.normal(size=(4, 3))*.2, rng.normal(size=(13, 3))*.05
    model = FreeSpacePoissonMatern(.7, .4)
    base = DenseFreeSpacePosterior(model, x, y, .03)
    targets, noise = rng.normal(size=4)*.1, np.array([.01, .02, .03, .04])
    return model, x, y, c, q, base, targets, noise


def test_sequential_conditioning_matches_full_augmented_gaussian():
    model, x, y, c, q, base, targets, noise = problem()
    posterior = ContactConditionedDensePosterior(base, c, targets, noise)
    a = model.vector_covariance(x, x)+.03*np.eye(len(x))
    cc = model.cross_covariance(c, x).transpose(0, 2, 1).reshape(len(c), -1)
    joint = np.block([[block_diag(a, a, a), cc.T],
                      [cc, model.scalar_covariance(c, c)+np.diag(noise)]])
    cross = np.column_stack((model.cross_covariance(q, x).transpose(0, 2, 1).reshape(len(q), -1),
                             model.scalar_covariance(q, c)))
    observations = np.r_[y.T.ravel(), targets]
    truth_mean = cross@np.linalg.solve(joint, observations)
    truth_var = model.prior_variance-np.sum(cross*np.linalg.solve(joint, cross.T).T, axis=1)
    m, v = posterior.predict(q, batch_size=5)
    np.testing.assert_allclose(m, truth_mean, atol=1e-14)
    np.testing.assert_allclose(v, truth_var, atol=1e-14)
    assert np.all(v > 0)
    assert np.all(v <= base.predict(q)[1])


def test_contact_cache_matches_direct_off_node_predictions():
    model, x, y, c, q, base, targets, noise = problem()
    def project(u, vectors):
        return u.T@cho_solve((base.factor, True), u), u.T@base.alpha
    cache = conditioned_regional_cache(model, x, y, c, targets, noise, project,
                                      [0, 0, 0], .2, order=12, compression_tolerance=1e-13)
    truth = ContactConditionedDensePosterior(base, c, targets, noise)
    expected = truth.predict(q)
    actual = PolynomialMomentField(cache).predict(q)
    np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=0)


def test_contact_validation_and_no_jitter():
    with pytest.raises(ValueError):
        ScalarContactUpdate([0], [[1]], [0], 0)
    with pytest.raises(np.linalg.LinAlgError):
        ScalarContactUpdate([0], [[-2]], [0], 1)
    u = ScalarContactUpdate([0], [[1]], [0], .1)
    with pytest.raises(ValueError):
        u.apply([0, 1], [1, 1], [[1]])
