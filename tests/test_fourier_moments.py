import numpy as np
import pytest

from geospsr.fourier_moments import FourierSquareAccumulator


def evaluate(c, q):
    k = c.shape[-1] // 2
    modes = np.stack(np.meshgrid(*([np.arange(-k, k+1)]*3), indexing='ij'), -1).reshape(-1, 3)
    return np.real(c.reshape(-1, len(modes)) @ np.exp(1j*modes @ q.T))


@pytest.mark.parametrize('cutoff', [0, 1, 3, 4])
def test_streamed_moments_match_independent_nonuniform_sums(cutoff):
    rng = np.random.default_rng(812)
    c = rng.normal(size=(5,)+(2*cutoff+1,)*3) + 1j*rng.normal(size=(5,)+(2*cutoff+1,)*3)
    w = np.array([1., .2, 0., -1., 2.])
    q = rng.uniform(-np.pi, np.pi, (31, 3))
    acc = FourierSquareAccumulator(cutoff)
    acc.add(c[:2], w[:2]); acc.add(c[2:], w[2:])
    expected = .7 - (w[:, None]*evaluate(c, q)**2).sum(0)
    np.testing.assert_allclose(evaluate(acc.coefficients(.7, -1), q)[0], expected,
                               atol=3e-11, rtol=3e-12)
    np.testing.assert_allclose(evaluate(acc.coefficients(.7, -1), q+2*np.pi)[0], expected,
                               atol=5e-11, rtol=3e-12)
    assert acc.count == 5


def test_corner_modes_survive_squaring_without_aliasing():
    c = np.zeros((5,)*3, dtype=complex)
    c[-1, -1, -1] = 1.
    acc = FourierSquareAccumulator(2)
    acc.add(c)
    expected = np.zeros((9,)*3, dtype=complex)
    expected[4,4,4] = .5
    expected[0,0,0] = expected[-1,-1,-1] = .25
    np.testing.assert_allclose(acc.coefficients(), expected, atol=2e-16)


def test_empty_accumulator_and_input_validation():
    acc = FourierSquareAccumulator(1)
    assert np.count_nonzero(acc.coefficients(2)) == 1
    with pytest.raises(ValueError): acc.add(np.zeros((2,2,2)))
    with pytest.raises(ValueError): acc.add(np.full((3,3,3), np.nan))
    with pytest.raises(ValueError): acc.add(np.zeros((2,3,3,3)), [1,2,3])
    with pytest.raises(ValueError): FourierSquareAccumulator(2, fft_width=8)
