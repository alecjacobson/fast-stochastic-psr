import numpy as np
import pytest

from reproduction.matched_lengthscale import matched_grid


@pytest.mark.parametrize('ell', [.03, .06, .12])
def test_original_kernel_and_matern_have_same_half_correlation(ell):
    from gpytoolbox import compactly_supported_normal
    lo, hi = np.array([-1., -.7, -.4]), np.array([1., .7, .4])
    grid = matched_grid(lo, hi, ell)
    q = np.zeros((2, 3)); q[1, 0] = grid['half_correlation_distance']
    k = compactly_supported_normal(q, n=3, sigma=1.5*min(grid['spacing']))
    np.testing.assert_allclose(k[1]/k[0], .5, atol=1e-12)
    r = q[1, 0]/ell
    np.testing.assert_allclose((1+np.sqrt(3)*r)*np.exp(-np.sqrt(3)*r), .5, atol=1e-12)
    corner = np.array(grid['corner']); h = np.array(grid['spacing'])
    end = corner+(np.array(grid['shape'])-1)*h
    assert np.all(corner <= lo) and np.all(end >= hi)
    assert np.all(lo-corner < h/2+1e-14) and np.all(end-hi < h/2+1e-14)


def test_grid_count_cannot_depend_on_training_point_count():
    # Bounds come from the full scan, never its training subset.
    g = matched_grid([0, 0, 0], [1, 2, 3], .12)
    assert g['one_dense_covariance_bytes'] == 8*np.prod(g['shape'])**2
    with pytest.raises(ValueError):
        matched_grid([0, 0, 0], [1, 2, 3], 0)
