import numpy as np
import pytest
from reproduction.contour_distance import curve_distance, curve_statistics, sample_curves


def test_parallel_segments_bound_known_distance():
    a = [np.array([[0., 0.], [1., 0.]])]
    b = [np.array([[0., .2], [.37, .2], [1., .2]])]
    r = curve_distance(a, b, spacing=.03)
    assert r['hausdorff_lower'] <= .2 <= r['hausdorff_upper']
    assert r['hausdorff_upper'] - r['hausdorff_lower'] <= .03 + 1e-14


def test_different_tessellations_of_same_curve_include_zero():
    a = [np.array([[0., 0.], [1., 0.], [1., 1.]])]
    b = [np.array([[0., 0.], [.31, 0.], [1., 0.], [1., .63], [1., 1.]])]
    r = curve_distance(a, b, spacing=.07)
    assert r['hausdorff_lower'] == 0
    assert r['hausdorff_upper'] <= .07


def test_missing_component_is_not_averaged_away():
    a = [np.array([[0., 0.], [1., 0.]])]
    b = a + [np.array([[0., 2.], [1., 2.]])]
    r = curve_distance(a, b, spacing=.01)
    assert r['hausdorff_lower'] <= 2 <= r['hausdorff_upper']
    assert r['hausdorff_lower'] > 1.99


def test_open_endpoints_and_empty_input():
    a = [np.array([[0., 0.], [0., .2]])]
    values, radius = sample_curves(a, .07)
    np.testing.assert_array_equal(values[[0, -1]], a[0])
    assert radius <= .035
    assert curve_statistics(a)['open_components'] == 1
    with pytest.raises(ValueError, match='empty level set'):
        sample_curves([], .01)
