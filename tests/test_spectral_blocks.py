import itertools
import numpy as np
import pytest

from geospsr.spectral_blocks import axis_weights, block_audit, one_high_axis_blocks


def test_positive_block_partition_and_explicit_mass():
    for core, extent in [(0, 3), (1, 4), (3, 3)]:
        frequencies = []
        for block in one_high_axis_blocks(core, extent):
            frequencies.extend(itertools.product(*(range(lo, hi+1) for lo, hi in block)))
        unique = set(frequencies)
        assert len(unique) == len(frequencies)
        assert all(tuple(-n for n in f) in unique for f in unique)
        assert all(f in unique for f in itertools.product(range(-core, core+1), repeat=3))
        expected = {f for f in itertools.product(range(-extent, extent+1), repeat=3)
                    if sum(abs(n) > core for n in f) <= 1}
        assert unique == expected
        record = block_audit(core, extent, lengthscale=.2)
        mass = np.prod(axis_weights(np.array(frequencies), lengthscale=.2), axis=1).sum()
        np.testing.assert_allclose(record['retained_prior_diagonal'], mass, rtol=1e-14)
        full = axis_weights(np.arange(-extent, extent+1), lengthscale=.2).sum()**3
        np.testing.assert_allclose(record['omitted_inside_cube_diagonal'], full-mass, atol=1e-18)
        np.testing.assert_allclose(mass+record['omitted_inside_cube_diagonal']+
                                   record['omitted_outside_cube_diagonal'], .1**3, rtol=1e-14)
        assert record['retained_modes'] == len(unique)
    for bounds in [(-1, 3), (4, 3), (1.5, 4)]:
        with pytest.raises(ValueError):
            one_high_axis_blocks(*bounds)


def test_independent_axis_normalization_matches_production():
    from geospsr.spectral import spectral_axis
    for lengthscale in [.005, .01, .05, .1, .5, 1., 10.]:
        n, expected = spectral_axis(200, lengthscale=lengthscale)
        np.testing.assert_allclose(axis_weights(n, lengthscale), expected, rtol=2e-12)
        _, released = spectral_axis(200, lengthscale=lengthscale, normalization='released')
        ratio = 1/(1+np.exp(-np.sqrt(3)/lengthscale))
        np.testing.assert_allclose(released, expected*ratio, rtol=2e-12)


def test_normalized_spectrum_reconstructs_analytic_periodic_kernel():
    angles = np.array([0., .01, .1, 1., np.pi])
    n = np.arange(-20000, 20001)
    for lengthscale in [.005, .1, 1., 10.]:
        rt3 = np.sqrt(3.)
        def helper(angle):
            w = rt3*(np.abs(angle/(2*np.pi))-.5)
            z = rt3/2
            return (np.exp((w-z)/lengthscale)+np.exp((-w-z)/lengthscale))*(
                2*lengthscale+rt3/np.tanh(z/lengthscale)-2*w*np.tanh(w/lengthscale))
        analytic = .1*helper(angles)/helper(0.)
        finite = axis_weights(n, lengthscale) @ np.cos(n[:, None]*angles)
        np.testing.assert_allclose(finite, analytic, atol=1e-9, rtol=1e-8)
