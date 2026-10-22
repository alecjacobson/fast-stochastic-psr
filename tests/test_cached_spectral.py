"""Independent direct-sum checks for cached interpolation-only Fourier fields."""
import cupy as cp
import numpy as np
import pytest

from geospsr.cached_spectral import CachedFourierEvaluator
from geospsr.spectral import FourierPoisson


@pytest.mark.parametrize('oversampling', [1.25, 2.0])
@pytest.mark.parametrize('cutoff', [3, 10])
def test_cached_grid_matches_direct_sum_with_periodic_queries(oversampling, cutoff):
    rng = np.random.default_rng(172)
    shape = (2 * cutoff + 1,) * 3
    coefficients = (rng.normal(size=(5, *shape)) + 1j * rng.normal(size=(5, *shape))).astype('F')
    q = rng.uniform(-np.pi, np.pi, (29, 3)).astype('f')
    q = np.concatenate([q, q + np.array([2 * np.pi, -2 * np.pi, 0]),
                        [[0, 0, 0], [-np.pi, np.pi, 0]]]).astype('f')
    n = np.stack(np.meshgrid(*[np.arange(-cutoff, cutoff + 1)] * 3, indexing='ij'), -1).reshape(-1, 3)
    expected = (coefficients.reshape(5, -1) @ np.exp(1j * n @ q.T)).real
    cached = CachedFourierEvaluator(coefficients, oversampling=oversampling, batch=2)
    actual = cp.asnumpy(cached.evaluate(q))
    assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 2e-5
    # Translating then rounding float32 coordinates perturbs the actual query;
    # compare in a field-scale norm rather than relative to a near-zero value.
    assert np.linalg.norm(actual[:, :29] - actual[:, 29:58]) / np.linalg.norm(expected[:, :29]) < 2e-5
    q2 = q[:13] * .37
    expected2 = (coefficients.reshape(5, -1) @ np.exp(1j * n @ q2.T)).real
    np.testing.assert_allclose(cp.asnumpy(cached.evaluate(q2, batch=3)), expected2, rtol=2e-4, atol=3e-4)


@pytest.mark.parametrize('cutoff', [50, 100])
def test_cached_highest_modes_and_production_bandwidth(cutoff):
    rng = np.random.default_rng(173)
    shape = (2 * cutoff + 1,) * 3
    coeff = np.zeros((2, *shape), dtype='F')
    modes = np.array([[cutoff, cutoff, cutoff], [-cutoff, cutoff - 1, 3], [0, 0, 0], [23, -31, 45]])
    weights = (rng.normal(size=(2, len(modes))) + 1j * rng.normal(size=(2, len(modes)))).astype('F')
    for index, mode in enumerate(modes):
        coeff[(slice(None), *(mode + cutoff))] = weights[:, index]
    q = rng.uniform(-np.pi, np.pi, (137, 3)).astype('f')
    expected = (weights @ np.exp(1j * modes @ q.T)).real
    cached = CachedFourierEvaluator(coeff, batch=1)
    actual = cp.asnumpy(cached.evaluate(q))
    # Single-precision coordinates imply larger phase error at higher modes.
    tolerance = 4e-5 * cutoff / 50
    assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < tolerance
    ordinary = cp.asnumpy(FourierPoisson(cutoff).evaluate(q, cp.asarray(coeff)))
    assert np.linalg.norm(actual - ordinary) / np.linalg.norm(expected) < tolerance
