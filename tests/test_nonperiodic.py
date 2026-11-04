import numpy as np
import pytest
from scipy.integrate import quad
from scipy.linalg import block_diag

from geospsr.nonperiodic import (
    DenseFreeSpacePosterior, FreeSpacePoissonMatern, RegionalVarianceCache,
)


@pytest.mark.parametrize('lengthscale,amplitude', [(np.sqrt(3), 1.), (.27, 2.3)])
def test_free_space_green_integral_and_cross_sign(lengthscale, amplitude):
    model = FreeSpacePoissonMatern(lengthscale, amplitude)
    # Independent radial Newton potential: r^-1 int_0^r s² k(s) ds
    # + int_r^infinity s k(s) ds. Its negative gradient is the cross kernel.
    for z in [0., 1e-8, 1e-4, .4999, .5001, 1., 3., 10., 80.]:
        r = z / model.kappa
        k = lambda s: amplitude*(1 + model.kappa*s)*np.exp(-model.kappa*s)
        mass = quad(lambda s: s*s*k(s), 0, r, epsabs=1e-13)[0]
        expected = quad(lambda s: s*k(s), r, np.inf, epsabs=1e-13)[0]
        if r:
            expected += mass/r
        actual = model.scalar_covariance([[r, 0, 0]], [[0, 0, 0]])[0, 0]
        np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=1e-14)
        cross = model.cross_covariance([[r, 0, 0]], [[0, 0, 0]])[0, 0]
        np.testing.assert_allclose(cross, [mass/r**2 if r else 0, 0, 0],
                                   rtol=2e-11, atol=1e-15)


def test_poisson_identity_and_cross_derivative():
    model = FreeSpacePoissonMatern(.8, 1.7)
    origin = np.zeros((1, 3))
    for r in [.03, .4, 2.]:
        q = np.array([[r, .13, -.21]])
        step = 2e-4
        base = model.scalar_covariance(q, origin)[0, 0]
        laplacian = 0.
        gradient = []
        for direction in np.eye(3):
            plus = model.scalar_covariance(q + step*direction, origin)[0, 0]
            minus = model.scalar_covariance(q - step*direction, origin)[0, 0]
            laplacian += (plus - 2*base + minus)/step**2
            gradient.append((plus-minus)/(2*step))
        np.testing.assert_allclose(-laplacian, model.vector_covariance(q, origin)[0, 0], rtol=2e-6)
        np.testing.assert_allclose(-np.array(gradient), model.cross_covariance(q, origin)[0, 0], rtol=2e-6)


def fixture_data():
    rng = np.random.default_rng(107)
    x = rng.normal(size=(23, 3))
    y = rng.normal(size=x.shape)
    q = rng.normal(size=(17, 3))
    return x, y, q


def test_joint_psd_and_independent_block_conditioning():
    x, y, q = fixture_data()
    model = FreeSpacePoissonMatern(.7, 2.)
    noise = np.linspace(.001, .02, len(x))
    a = block_diag(*([model.vector_covariance(x, x) + np.diag(noise)]*3))
    c = model.cross_covariance(q, x).transpose(0, 2, 1).reshape(len(q), -1)
    g = model.scalar_covariance(q, q)
    joint = np.block([[a, c.T], [c, g]])
    assert np.linalg.eigvalsh(joint).min() > 0
    expected_cov = g - c @ np.linalg.solve(a, c.T)
    expected_mean = c @ np.linalg.solve(a, y.T.ravel())
    posterior = DenseFreeSpacePosterior(model, x, y, noise)
    mean, var = posterior.predict(q, batch_size=5)
    np.testing.assert_allclose(mean, expected_mean, atol=2e-14)
    np.testing.assert_allclose(var, np.diag(expected_cov), atol=2e-14)
    np.testing.assert_allclose(posterior.covariance(q), expected_cov, atol=2e-14)
    assert np.linalg.eigvalsh(expected_cov).min() > 0
    assert np.all(var <= model.prior_variance)


def test_translation_rotation_and_no_periodic_wrap():
    x, y, q = fixture_data()
    model = FreeSpacePoissonMatern(.6)
    posterior = DenseFreeSpacePosterior(model, x, y, .01)
    rotation = np.linalg.qr(np.random.default_rng(9).normal(size=(3, 3)))[0]
    shift = np.array([7., -21., 13.])
    transformed = DenseFreeSpacePosterior(model, x @ rotation + shift, y @ rotation, .01)
    for a, b in zip(posterior.predict(q), transformed.predict(q @ rotation + shift)):
        np.testing.assert_allclose(a, b, atol=2e-14)
    k = model.vector_covariance([[0, 0, 0]], [[0, 0, 0], [2*np.pi, 0, 0]])
    assert k[0, 1] < k[0, 0]*1e-5
    far = model.scalar_covariance([[1e5, 0, 0]], [[0, 0, 0]])[0, 0]
    np.testing.assert_allclose(far*1e5, 8*model.amplitude/model.kappa**3)


def test_coincident_observations_require_declared_noise():
    model = FreeSpacePoissonMatern()
    x = np.zeros((2, 3))
    with pytest.raises(np.linalg.LinAlgError):
        DenseFreeSpacePosterior(model, x, x, 0.)
    p = DenseFreeSpacePosterior(model, x, x, .01)
    m, v = p.predict(x)
    np.testing.assert_array_equal(m, 0)
    np.testing.assert_allclose(v, model.prior_variance)


def test_cache_nodes_heldout_convergence_and_energy_bound():
    x, y, _ = fixture_data()
    p = DenseFreeSpacePosterior(FreeSpacePoissonMatern(.7), x, y, .01)
    rng = np.random.default_rng(15)
    q = np.array([4., 0, 0]) + rng.uniform(-.2, .2, (50, 3))
    truth_mean, truth_var = p.predict(q)
    errors = []
    for order in [2, 4, 6]:
        cache = RegionalVarianceCache(p, [4., 0, 0], .2, order)
        np.testing.assert_allclose(cache.basis(cache.nodes), np.eye(order**3), atol=1e-14)
        at_nodes = cache.predict(cache.nodes)
        for a, b in zip(at_nodes, p.predict(cache.nodes)):
            np.testing.assert_allclose(a, b, atol=1e-14)
        mean, var = cache.predict(q)
        errors.append(np.max(np.abs(var-truth_var)))
        w_approx = p.whitened_cross(cache.nodes) @ cache.basis(q).T
        e = p.whitened_cross(q) - w_approx
        en = np.linalg.norm(e, axis=0)
        bound = 2*np.linalg.norm(w_approx, axis=0)*en + en*en
        assert np.all(np.abs(var-truth_var) <= bound + 1e-14)
    assert errors[2] < errors[0]*1e-5
    np.testing.assert_allclose(mean, truth_mean, atol=1e-9)
    with pytest.raises(ValueError, match='inside'):
        cache.predict([[10., 0, 0]])


@pytest.mark.parametrize('kwargs', [{'lengthscale':0}, {'amplitude':-1}, {'lengthscale':np.nan}])
def test_invalid_model(kwargs):
    with pytest.raises(ValueError):
        FreeSpacePoissonMatern(**kwargs)


def test_invalid_data_and_noise():
    model = FreeSpacePoissonMatern()
    with pytest.raises(ValueError):
        model.scalar_covariance([[0, 0]], [[0, 0, 0]])
    for noise in [-1., [0., 1.], np.nan]:
        with pytest.raises(ValueError):
            DenseFreeSpacePosterior(model, [[0, 0, 0]], [[0, 0, 1]], noise)


def test_empty_queries():
    p = DenseFreeSpacePosterior(FreeSpacePoissonMatern(), [[0, 0, 0]], [[0, 0, 1]], .01)
    q = np.empty((0, 3))
    assert p.covariance(q).shape == (0, 0)
    assert all(value.shape == (0,) for value in p.predict(q))
