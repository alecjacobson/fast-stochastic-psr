"""Compare streamed exact variances with the untouched released implementation."""
from functools import partial
import cupy as cp
import jax
import jax.numpy as jnp
import numpy as np
from reproduction.matched_teaser import (original, conditional_variance, own_jax_array,
                                         Matern32Kernel, ProductKernel, trunc_Zd)
from geospsr.spectral import FourierPoisson


def test_streamed_variance_matches_original_api():
    rng = np.random.default_rng(21)
    x = rng.uniform(-.6, .6, (513, 3)).astype('f')
    q = rng.uniform(-.7, .7, (65, 3)).astype('f')
    kv = ProductKernel(*[Matern32Kernel(.05, .1)] * 3)
    freq = trunc_Zd(2)
    ev = jax.vmap(kv.spectral_density)(freq) ** .5
    cross = partial(original.poisson_cross_covariances, eigenvectors=freq, eigenvalues=ev)
    kfv = lambda a, b: cross(a - b)
    op = FourierPoisson(2, .05, .1)
    f_ev = original.f_eigenvalues_from_v_eigenvalues(freq, ev)
    prior = float((f_ev ** 2).sum())
    sigma = .002
    expected = np.asarray(original.compute_variance(jnp.asarray(q), jnp.asarray(x), kv, kfv,
                                                    prior, sigma, bs=16))
    K = jax.jit(jax.vmap(jax.vmap(kv, (None, 0)), (0, None)))(jnp.asarray(x), jnp.asarray(x))
    Kcp = own_jax_array(K)
    Kcp.flat[::len(x) + 1] += sigma
    chol = cp.linalg.cholesky(Kcp)
    rows = jax.jit(jax.vmap(jax.vmap(kfv, (None, 0)), (0, None)))
    def reference_rows(query):
        return own_jax_array(rows(jnp.asarray(query), jnp.asarray(x)))
    def nufft_rows(query):
        diff = (cp.asarray(query)[:, None] - cp.asarray(x)[None]).reshape(-1, 3)
        return op.cross_values(diff).reshape(len(query), len(x), 3)
    for callback in [reference_rows, nufft_rows]:
        actual = cp.asnumpy(conditional_variance(chol, prior, q, callback, batch=16))
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-11)


def test_periodic_hat_fourier_coefficients():
    from reproduction.interpolated_nufft import periodic_hat_coefficients
    knots = np.linspace(-np.pi, np.pi, 11)
    weights = periodic_hat_coefficients(knots, 8)
    expected = np.zeros(17, dtype=complex)
    expected[8] = 1
    np.testing.assert_allclose(weights.sum(axis=1), expected, atol=2e-15)
    grid = np.linspace(-np.pi, np.pi, 20000, endpoint=False)
    nodal = np.sin(2 * knots) + .4 * np.cos(3 * knots)
    values = np.interp(grid, knots, nodal)
    frequencies = np.arange(-8, 9)
    quadrature = (np.exp(-1j * frequencies[:, None] * grid) * values).mean(axis=1)
    np.testing.assert_allclose(weights @ nodal[:-1], quadrature, atol=2e-8)
