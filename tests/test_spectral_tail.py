import cupy as cp
import numpy as np
import pytest
from geospsr.spectral import spectral_axis
from geospsr.spectral_tail import (sample_tail_frequencies, tail_quadratic_samples,
                                   analytic_covariance_rows64)
from geospsr.training_spectrum import FourierCovariance, finite_covariance_rows


@pytest.mark.parametrize('ell,cutoff', [(.04,20),(1.,2),(10.,1)])
def test_tail_sampler_and_quadratic_against_infinite_analytic_kernel(ell,cutoff):
    rng = np.random.default_rng(42)
    x = rng.uniform(-1, 1, (13, 3)).astype('f')
    # Clustered points exercise off-diagonal tail covariance too.
    x[1:5] = x[0] + rng.normal(0, .003, (4, 3))
    a = rng.normal(size=(13, 3, 3)).astype('f')
    n, mass = sample_tail_frequencies(65536, cutoff, ell, seed=20)
    assert np.all(np.max(np.abs(n), axis=1) > cutoff)
    p = spectral_axis(cutoff, ell)[1].sum() / .1
    # Factor the complement: 1-p**3 loses relative precision for tiny tails.
    np.testing.assert_allclose(mass, .001 * (1-p)*(1+p+p*p), rtol=1e-12)
    first = (np.abs(n) > cutoff).argmax(1)
    expected = np.array([1, p, p*p]) / (1+p+p*p)
    np.testing.assert_allclose(np.bincount(first, minlength=3)/len(n), expected, atol=.007)
    op = FourierCovariance(cutoff, ell)
    delta = cp.asnumpy(analytic_covariance_rows64(x, x, ell) - finite_covariance_rows(x, x, op, dtype='d'))
    h = np.einsum('iqd,ij,jkd->qk', a.astype('d'), delta, a.astype('d'))
    samples = tail_quadratic_samples(x, a, n, mass, source_batch=5, frequency_batch=2048)
    se = samples.std(0, ddof=1) / np.sqrt(len(samples))
    assert np.all(np.abs(samples.mean(0) - h) < 6 * se + 1e-11)
    # Independent direct complex feature calculation, including a high mode.
    n = np.array([[30, 2, -4], [1000003, 3, 21], [-70, 31, 5]])
    phi = np.exp(1j * (x.astype('d') @ n.T))
    f = np.einsum('ir,iqd->rqd', phi, a)
    expected = mass * np.einsum('rqd,rkd->rqk', f.conj(), f).real
    actual = tail_quadratic_samples(x, a, n, mass, source_batch=5, frequency_batch=2)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-10)
    with pytest.raises(ValueError,match='in-band defect'):
        sample_tail_frequencies(100,cutoff,ell,normalization='released')


def test_psd_posterior_bracket_and_analytic_normalization():
    from reproduction.posterior import kernel_matrix
    rng = np.random.default_rng(10)
    x = rng.uniform(-1, 1, (35, 3)).astype('f')
    ell, noise = .06, .0001
    k = cp.asnumpy(analytic_covariance_rows64(x, x, ell))
    upstream = kernel_matrix(x, ell)
    upstream.block_until_ready()
    np.testing.assert_allclose(k, np.asarray(upstream), atol=1e-9, rtol=2e-5)
    op = FourierCovariance(5, ell)
    finite = cp.asnumpy(finite_covariance_rows(x, x, op, dtype='d'))
    # Independent explicit finite Fourier sum.
    ns = np.stack(np.meshgrid(*[np.arange(-5, 6)]*3, indexing='ij'), -1).reshape(-1, 3)
    phi = np.exp(1j*x.astype('d')@ns.T)
    axis = spectral_axis(5, ell)[1]
    rho = (axis[:, None, None]*axis[None, :, None]*axis[None, None, :]).ravel()
    np.testing.assert_allclose(finite, ((phi*rho) @ phi.conj().T).real, atol=2e-17, rtol=2e-10)
    delta = k - finite
    assert np.linalg.eigvalsh(delta).min() > -1e-14
    b, full = finite + noise*np.eye(len(x)), k + noise*np.eye(len(x))
    cross = rng.normal(size=(len(x), 4))
    a = np.linalg.solve(b, cross)
    gap = cross.T @ (a - np.linalg.solve(full, cross))
    h = a.T @ delta @ a
    scale = np.linalg.norm(h)
    assert np.linalg.eigvalsh(gap).min() > -1e-12*scale
    assert np.linalg.eigvalsh(h-gap).min() > -1e-12*scale
    # Full vector prior + finite inverse gives V_finite+H, not V_full.
    prior = np.eye(4)*np.linalg.norm(cross.T@a)*2
    hybrid = prior - cross.T@a - a.T@cross + a.T@full@a
    np.testing.assert_allclose(hybrid, prior-cross.T@a+h, rtol=1e-12)
    # Conversely, a finite joint prior with the full inverse understates the
    # full covariance, but lies above the consistently finite posterior.
    full_weights = np.linalg.solve(full,cross)
    shared_prior_full_solve = (prior-cross.T@full_weights-full_weights.T@cross
                              +full_weights.T@b@full_weights)
    vfull = prior-cross.T@full_weights
    vfinite = prior-cross.T@a
    hfull = full_weights.T@delta@full_weights
    np.testing.assert_allclose(shared_prior_full_solve,vfull-hfull,rtol=1e-12)
    assert np.linalg.eigvalsh(shared_prior_full_solve-vfinite).min() > -1e-12*scale
    assert np.linalg.eigvalsh(vfull-shared_prior_full_solve).min() > -1e-12*scale
