import cupy as cp
import numpy as np
from geospsr.spectral import FourierKernel
from geospsr.training_spectrum import FourierCovariance, finite_covariance_rows


def test_independent_finite_covariance_and_nufft_action():
    rng = np.random.default_rng(48)
    x = rng.uniform(-np.pi, np.pi, (29, 3)).astype('f')
    op = FourierCovariance(4, lengthscale=.05)
    modes = np.stack(np.meshgrid(*[np.arange(-4, 5)] * 3, indexing='ij'), -1).reshape(-1, 3)
    phi = np.exp(1j * x @ modes.T)
    expected = ((phi * cp.asnumpy(op.rho).ravel()) @ phi.conj().T).real
    direct = finite_covariance_rows(x, x, op)
    np.testing.assert_allclose(cp.asnumpy(direct), expected, rtol=2e-5, atol=1e-10)
    b = cp.asarray(rng.normal(size=(len(x), 3)), dtype='f')
    kernel = FourierKernel(x, op, .002)
    np.testing.assert_allclose(cp.asnumpy(kernel(b)), (expected + .002 * np.eye(len(x))) @ cp.asnumpy(b),
                               rtol=2e-5, atol=2e-8)
    alpha, info = kernel.solve(b, rtol=1e-5)
    assert info['converged']
    np.testing.assert_allclose(cp.asnumpy(direct @ alpha + .002 * alpha), cp.asnumpy(b), rtol=1e-4, atol=3e-5)


def test_separate_bandwidth_sampler_has_correct_conditional_law():
    from reproduction.spectral_posterior import SpectralPosterior
    rng = np.random.default_rng(47)
    x = rng.uniform(-.7, .7, (9, 3)).astype('f')
    y = rng.normal(size=x.shape).astype('f')
    q = rng.uniform(-.7, .7, (4, 3)).astype('f')
    model = SpectralPosterior(x, y, training_cutoff=4, cutoff=2,
                              lengthscale=.1, noise_variance=.001)
    K = finite_covariance_rows(x, x, model.training) + .001 * cp.eye(len(x), dtype='f')
    cross = model.op.cross_values((q[:, None] - x[None]).reshape(-1, 3)).reshape(len(q), len(x), 3)
    alpha = cp.linalg.solve(K, cp.asarray(y))
    mean = cp.einsum('qnd,nd->q', cross, alpha) - model.offset
    norm2 = (model.op.ns ** 2).sum(0)
    rho_f = cp.where(norm2 > 0, model.op.rho / cp.maximum(norm2, 1), 0)
    covariance = model.op.evaluate((q[:, None] - q[None]).reshape(-1, 3), rho_f)[0].reshape(len(q), len(q))
    for d in range(3):
        covariance -= cross[:, :, d] @ cp.linalg.solve(K, cross[:, :, d].T)
    model.draw(4096, seed=91, batch=64, retain_components=True, validation_rows=[0, 3], retain_rhs=True)
    residual = cp.einsum('rn,pnd->prd', K, model.sample_weights) - model.sample_rhs
    assert float(cp.linalg.norm(residual) / cp.linalg.norm(model.sample_rhs)) < 2e-5
    np.testing.assert_array_equal(cp.asnumpy(model.sample_rhs[:, [0, 3]]), cp.asnumpy(model.sample_validation_rhs))
    samples = cp.asnumpy(model.samples(q, batch=64))
    mean, covariance = cp.asnumpy(mean), cp.asnumpy(covariance)
    standard_error = np.sqrt(np.diag(covariance) / len(samples))
    assert np.all(np.abs(samples.mean(0) - mean) < 5 * standard_error + 2e-6)
    assert np.linalg.norm(np.cov(samples.T) - covariance) / np.linalg.norm(covariance) < .09
    assert all(r['converged'] for r in model.sample_solve_records)
