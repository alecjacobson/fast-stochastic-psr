import numpy as np
import pytest
from reproduction.complement_variance import estimate, pilot_weights


def test_dense_gaussian_decomposition_and_wrong_inverse_control():
    # Independent real-feature joint GP; no NUFFT or posterior implementation.
    rng = np.random.default_rng(301)
    source = rng.normal(size=(9, 17))
    query = rng.normal(size=(7, 17))
    noise = np.geomspace(.01, 2., 9)
    a = source@source.T+np.diag(noise)
    cross = query@source.T
    b = np.linalg.solve(a, cross.T).T
    latent = np.concatenate([query, np.zeros((7, 9))], axis=1)
    observations = np.concatenate([source, np.diag(np.sqrt(noise))], axis=1)
    correction = b@observations
    posterior = latent-correction
    v0 = np.sum(latent*latent, axis=1)
    v, w = np.sum(posterior**2, axis=1), np.sum(correction**2, axis=1)
    np.testing.assert_allclose(posterior@correction.T, 0, atol=2e-14)
    np.testing.assert_allclose(v+w, v0, rtol=2e-15)
    # A changed inverse breaks the identity even with the same joint prior.
    wrong = np.linalg.solve(a+np.eye(9)*2, cross.T).T@observations
    assert np.linalg.norm((latent-wrong)@wrong.T) > .1
    draws = rng.standard_normal((32768, 26))
    g, c = draws@posterior.T, draws@correction.T
    beta = pilot_weights(g[:256], c[:256])
    result = estimate(g[256:], c[256:], v0, beta)
    se = np.sqrt(2*(beta**2*v**2+(1-beta)**2*w**2)/(len(g)-256))
    assert np.max(np.abs(result['combined']-v)/se) < 6
    assert not result['empty_interval'].any()
    assert np.all((result['lower'] <= v) & (v <= result['upper']))


def test_variance_efficiency_in_both_extreme_regimes():
    # Pilot and confirmation are independent; strongly/weakly constrained points
    # require opposite weights. Count only independent Gaussian draws.
    rng = np.random.default_rng(905)
    v = np.array([.001, .999]); w = 1-v
    pg, pc = rng.normal(size=(2, 64, 2))*np.sqrt([v, w])[:, None, :]
    beta = pilot_weights(pg, pc)
    assert beta[0] > .99 and beta[1] < .01
    g, c = rng.normal(size=(2, 4096, 2))*np.sqrt([v, w])[:, None, :]
    result = estimate(g, c, 1., beta)
    se = np.sqrt(2*(beta**2*v*v+(1-beta)**2*w*w)/len(g))
    assert np.all(np.abs(result['combined']-v) < 6*se)
    assert result['upper'][1]-result['lower'][1] < .001


def test_negative_estimates_empty_intervals_and_validation_are_explicit():
    g = np.ones((32, 2)); c = np.full((32, 2), 3.)
    result = estimate(g, c, 1, 0)
    np.testing.assert_array_equal(result['combined'], -8)
    assert result['empty_interval'].all()
    np.testing.assert_array_equal(pilot_weights(g*0, c*0), .5)
    with pytest.raises(ValueError): estimate(g, c[:4], 1, 0)
    with pytest.raises(ValueError): estimate(g, c, 0, 0)
    with pytest.raises(ValueError): estimate(g, c, 1, 2)
    with pytest.raises(ValueError): estimate(g, c, 1, 0, delta=1)
    with pytest.raises(ValueError): pilot_weights(g, c*np.nan)
